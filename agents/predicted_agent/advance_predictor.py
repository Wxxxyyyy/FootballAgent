# -*- coding: utf-8 -*-
"""
赛前预测流水线 (Advance Predictor)

完整流程:
  1. 向 OpenClaw (旧电脑) 请求赛前分析数据 (赔率 + 双方近5场)
  2. 提取赔率 -> 喂入 ML 模型 -> 获取 WDL / OU 基准概率
  3. 查询 Neo4j 知识图谱 -> 获取两队历史交锋记录
  4. 汇总所有信息 -> 调用 Kimi 2.5 LLM -> 输出最终预测

输入: 主队名(英文), 客队名(英文), 比赛日期(可选)
输出: 结构化预测结果
"""

import os
import sys
import re
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Optional

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from neo4j import GraphDatabase

# ═══════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════

RELAY_URL = "http://localhost:15000"
OPENCLAW_TIMEOUT = 120

NEO4J_URI = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


class PreMatchPredictor:
    """赛前预测流水线"""

    def __init__(self):
        self._model = None
        self._neo4j_driver = None

    @property
    def model(self):
        if self._model is None:
            from agents.predicted_agent.models.statistical_model import OddsModel
            self._model = OddsModel.load()
            print("[赛前预测] ML 模型已加载")
        return self._model

    @property
    def neo4j(self):
        if self._neo4j_driver is None:
            self._neo4j_driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            print(f"[赛前预测] Neo4j 已连接: {NEO4J_URI}")
        return self._neo4j_driver

    def close(self):
        if self._neo4j_driver:
            self._neo4j_driver.close()

    # ═══════════════════════════════════════════════════════════════
    #  主流程
    # ═══════════════════════════════════════════════════════════════

    def predict(self, home_team: str, away_team: str, date: str = None) -> dict:
        """
        完整赛前预测

        Args:
            home_team: 主队英文名 (如 "Arsenal")
            away_team: 客队英文名 (如 "Chelsea")
            date: 比赛日期 YYYY-MM-DD (可选)

        Returns:
            结构化预测结果
        """
        ts_start = time.time()
        print("=" * 60)
        print(f"[赛前预测] {home_team} vs {away_team} ({date or '最近'})")
        print("=" * 60)

        # ── Step 1: 请求 OpenClaw 赛前数据 ──
        print("\n[1/4] 向 OpenClaw 请求赛前分析数据...")
        openclaw_data = self._request_openclaw(home_team, away_team, date)

        # ── Step 2: ML 模型预测 ──
        print("\n[2/4] 运行 ML 模型...")
        ml_result = self._run_ml_model(openclaw_data, home_team, away_team)

        # ── Step 3: Neo4j 历史交锋 ──
        print("\n[3/5] 查询 Neo4j 历史交锋...")
        h2h_records = self._query_h2h(home_team, away_team)

        # ── Step 4: 爆冷信号分析 ──
        print("\n[4/5] 分析爆冷信号...")
        home_last_5, away_last_5 = [], []
        if openclaw_data:
            hs = openclaw_data.get("home_last_5", {})
            if hs.get("found"):
                home_last_5 = hs.get("last_5", [])
            aws = openclaw_data.get("away_last_5", {})
            if aws.get("found"):
                away_last_5 = aws.get("last_5", [])

        upset_signals = self._analyze_upset_signals(
            ml_result, home_team, away_team,
            home_last_5, away_last_5, h2h_records,
        )

        # ── Step 5: LLM 综合分析 ──
        print("\n[5/5] 调用 LLM 进行综合分析...")
        llm_result = self._call_llm(
            home_team, away_team, date,
            ml_result, openclaw_data, h2h_records,
            upset_signals,
        )

        elapsed = time.time() - ts_start
        print(f"\n[赛前预测] 完成，耗时 {elapsed:.1f}s")

        return {
            "home_team": home_team,
            "away_team": away_team,
            "date": date,
            "ml_prediction": ml_result,
            "h2h_records": h2h_records,
            "upset_signals": upset_signals,
            "llm_analysis": llm_result,
            "openclaw_summary": self._summarize_openclaw(openclaw_data),
            "elapsed_seconds": round(elapsed, 1),
        }

    # ═══════════════════════════════════════════════════════════════
    #  Step 1: 请求 OpenClaw 数据
    # ═══════════════════════════════════════════════════════════════

    def _request_openclaw(self, home_team: str, away_team: str, date: str) -> Optional[dict]:
        """
        向 OpenClaw 请求赛前分析数据

        数据通过正向通道 (/receive_data) 回传，
        使用 pre_match_state 模块进行跨线程同步等待。
        """
        from api.pre_match_state import wait_for_pre_match

        payload = {
            "task_id": str(uuid.uuid4()),
            "task_type": "pre_match_analysis",
            "params": {
                "home_team": home_team,
                "away_team": away_team,
            },
            "async_mode": False,
            "timestamp": datetime.now().isoformat(),
        }
        if date:
            payload["params"]["date"] = date

        # 发送请求到 OpenClaw
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(f"{RELAY_URL}/relay_to_openclaw", json=payload)
                direct = resp.json()
                print(f"  [OpenClaw] 直接响应: {direct.get('status', 'unknown')}")

                # 有些 OpenClaw 实现会在 HTTP 响应中直接返回完整数据
                oc_resp = direct.get("openclaw_response", direct)
                if isinstance(oc_resp, dict):
                    result_data = oc_resp.get("result", {})
                    if isinstance(result_data, dict) and result_data.get("odds"):
                        print("  [OpenClaw] 从直接响应获取到完整数据")
                        return result_data
        except Exception as e:
            print(f"  [OpenClaw] 发送请求失败: {e}")

        # 等待正向通道回传
        print(f"  [OpenClaw] 等待正向通道数据（超时 {OPENCLAW_TIMEOUT}s）...")
        result = wait_for_pre_match(home_team, away_team, timeout=OPENCLAW_TIMEOUT)
        if result:
            print("  [OpenClaw] 成功收到赛前分析数据")
        else:
            print("  [OpenClaw] 超时或未收到数据")
        return result

    # ═══════════════════════════════════════════════════════════════
    #  Step 2: ML 模型预测
    # ═══════════════════════════════════════════════════════════════

    def _run_ml_model(self, openclaw_data: Optional[dict],
                      home_team: str, away_team: str) -> dict:
        """从赔率数据提取特征并运行 ML 模型"""
        if not openclaw_data:
            return {"error": "无 OpenClaw 数据，无法运行 ML 模型"}

        odds = self._extract_odds(openclaw_data, home_team, away_team)
        if odds is None:
            return {"error": "无法提取有效赔率"}

        print(f"  [ML] 赔率: H={odds['B365H']}, D={odds['B365D']}, A={odds['B365A']}")
        print(f"       Over={odds.get('B365>2.5', 'N/A')}, "
              f"Under={odds.get('B365<2.5', 'N/A')}, AHh={odds.get('AHh', 0)}")

        try:
            result = self.model.predict_from_odds(
                b365h=odds["B365H"],
                b365d=odds["B365D"],
                b365a=odds["B365A"],
                b365_over25=odds.get("B365>2.5", 1.90),
                b365_under25=odds.get("B365<2.5", 1.90),
                ahh=odds.get("AHh", 0.0),
            )
            result["odds_source"] = odds.get("_source", "unknown")
            print(f"  [ML] WDL: 主胜={result['home_win_prob']:.1%}, "
                  f"平={result['draw_prob']:.1%}, 客胜={result['away_win_prob']:.1%}")
            print(f"  [ML] OU : 大球={result['over25_prob']:.1%}, "
                  f"小球={result['under25_prob']:.1%}")
            return result
        except Exception as e:
            print(f"  [ML] 预测失败: {e}")
            return {"error": str(e)}

    # ── 赔率提取（兼容多种数据格式）──

    def _extract_odds(self, data: dict, home_team: str, away_team: str) -> Optional[dict]:
        """
        从 OpenClaw 数据中提取赔率

        优先级:
          B365 直接字段 -> full_result 中的 B365 -> Pinnacle -> 市场平均 -> bookmakers
        """
        odds_section = data.get("odds", {})
        if not odds_section or not odds_section.get("found"):
            return None

        # 路径 1: OpenClaw 已提取好的 B365 赔率
        direct_odds = odds_section.get("odds", {})
        if direct_odds and direct_odds.get("B365H"):
            return {
                "B365H": float(direct_odds["B365H"]),
                "B365D": float(direct_odds["B365D"]),
                "B365A": float(direct_odds["B365A"]),
                "B365>2.5": float(direct_odds.get("B365>2.5", 1.90)),
                "B365<2.5": float(direct_odds.get("B365<2.5", 1.90)),
                "AHh": float(direct_odds.get("AHh", 0)),
                "_source": "Bet365",
            }

        # 路径 2: full_result (football-data.co.uk CSV 行)
        full = odds_section.get("full_result", {})
        if full:
            for prefix, label in [("B365", "Bet365"), ("PS", "Pinnacle"), ("Avg", "市场平均")]:
                result = self._extract_from_csv_row(full, prefix)
                if result:
                    result["_source"] = label
                    return result

        # 路径 3: bookmakers 字典 (即将开赛的比赛 live 赔率)
        bookmakers = odds_section.get("bookmakers", {})
        if bookmakers:
            return self._extract_from_bookmakers(bookmakers, home_team, away_team)

        return None

    @staticmethod
    def _extract_from_csv_row(row: dict, prefix: str) -> Optional[dict]:
        """从 football-data.co.uk 格式的完整行中提取指定庄家赔率"""
        try:
            h = float(row.get(f"{prefix}H", 0))
            d = float(row.get(f"{prefix}D", 0))
            a = float(row.get(f"{prefix}A", 0))
            if not all([h, d, a]):
                return None

            over_key = f"{prefix}>2.5" if f"{prefix}>2.5" in row else "B365>2.5"
            under_key = f"{prefix}<2.5" if f"{prefix}<2.5" in row else "B365<2.5"

            return {
                "B365H": h, "B365D": d, "B365A": a,
                "B365>2.5": float(row.get(over_key, row.get("P>2.5", 1.90))),
                "B365<2.5": float(row.get(under_key, row.get("P<2.5", 1.90))),
                "AHh": float(row.get("AHh", 0)),
            }
        except (ValueError, TypeError):
            return None

    def _extract_from_bookmakers(self, bookmakers: dict,
                                 home_team: str, away_team: str) -> Optional[dict]:
        """从 bookmakers 字典中提取赔率（用于即将开赛的比赛）"""
        preferred = ["pinnacle", "williamhill", "betway", "betfair",
                     "unibet", "ladbrokes"]
        for bm_key in list(bookmakers.keys()):
            if bm_key not in preferred:
                preferred.append(bm_key)

        for bm_key in preferred:
            bm = bookmakers.get(bm_key, {})
            markets = bm.get("markets", {})
            h2h = markets.get("h2h", {})
            if not h2h:
                continue

            home_odds = self._fuzzy_get(h2h, home_team)
            away_odds = self._fuzzy_get(h2h, away_team)
            draw_odds = h2h.get("Draw")
            if not all([home_odds, away_odds, draw_odds]):
                continue

            result = {
                "B365H": float(home_odds),
                "B365D": float(draw_odds),
                "B365A": float(away_odds),
                "_source": bm.get("title", bm_key),
            }

            # 大小球
            totals = markets.get("totals", {})
            for key, val in totals.items():
                if "Over" in key:
                    result["B365>2.5"] = float(val)
                elif "Under" in key:
                    result["B365<2.5"] = float(val)

            # 亚盘
            spreads = markets.get("spreads", {})
            result["AHh"] = self._extract_ahh(spreads, home_team)

            result.setdefault("B365>2.5", 1.90)
            result.setdefault("B365<2.5", 1.90)

            print(f"  [赔率] 使用 {result['_source']} 赔率作为模型输入")
            return result

        return None

    @staticmethod
    def _fuzzy_get(d: dict, target: str):
        """模糊匹配字典 key"""
        if target in d:
            return d[target]
        tl = target.lower()
        for key, val in d.items():
            if key == "Draw":
                continue
            kl = key.lower()
            if kl == tl or tl in kl or kl in tl:
                return val
        return None

    @staticmethod
    def _extract_ahh(spreads: dict, home_team: str) -> float:
        """从 spreads 提取主队亚盘让球数"""
        for key in spreads:
            if home_team.lower() in key.lower():
                m = re.search(r'\(([-\d.]+)\)', key)
                if m:
                    return float(m.group(1))
        return 0.0

    # ═══════════════════════════════════════════════════════════════
    #  Step 3: Neo4j 历史交锋
    # ═══════════════════════════════════════════════════════════════

    def _query_h2h(self, home_team: str, away_team: str, limit: int = 5) -> list[dict]:
        """查询两队历史交锋记录"""
        cypher = """
        MATCH (a:Team {name: $team_a})-[r:PLAYED_AGAINST]-(b:Team {name: $team_b})
        RETURN r.match_date       AS date,
               r.season           AS season,
               r.match_result     AS result,
               r.total_goals      AS total_goals,
               r.odds_info        AS odds,
               r.over_under_odds  AS over_under
        ORDER BY r.match_date DESC
        LIMIT $limit
        """
        try:
            with self.neo4j.session(database=NEO4J_DATABASE) as session:
                result = session.run(
                    cypher, team_a=home_team, team_b=away_team, limit=limit
                )
                records = result.data()
            if records:
                print(f"  [Neo4j] 找到 {len(records)} 条交锋记录")
                for r in records[:3]:
                    print(f"    {r.get('date', '?')} | {r.get('result', '?')}")
            else:
                print("  [Neo4j] 未找到交锋记录")
            return records
        except Exception as e:
            print(f"  [Neo4j] 查询失败: {e}")
            return []

    # ═══════════════════════════════════════════════════════════════
    #  Step 4: 爆冷信号分析
    # ═══════════════════════════════════════════════════════════════

    def _analyze_upset_signals(
        self, ml_result: dict, home_team: str, away_team: str,
        home_last_5: list, away_last_5: list, h2h_records: list,
    ) -> dict:
        """
        基于多维数据分析爆冷可能性

        检测维度:
          1. 近况反差: 被看衰的一方近期战绩远好于被看好方
          2. 交锋克制: 历史交锋中弱势方反而占优
          3. 赛程疲劳: 热门方密集赛程导致体能隐患
          4. 状态断崖: 热门方近期连败或频繁丢球

        Returns:
            {
              "has_risk": bool,
              "signals": [{"type": "...", "desc": "...", "severity": "高/中"}, ...],
              "underdog": "home 或 away",
              "underdog_name": "...",
            }
        """
        signals = []

        if not ml_result or "error" in ml_result:
            return {"has_risk": False, "signals": []}

        home_prob = ml_result.get("home_win_prob", 0.33)
        away_prob = ml_result.get("away_win_prob", 0.33)

        # 确定热门/冷门方
        if home_prob > away_prob:
            fav_side, und_side = "home", "away"
            fav_name, und_name = home_team, away_team
            fav_last5, und_last5 = home_last_5, away_last_5
            fav_prob = home_prob
        else:
            fav_side, und_side = "away", "home"
            fav_name, und_name = away_team, home_team
            fav_last5, und_last5 = away_last_5, home_last_5
            fav_prob = away_prob

        # ── 维度 1: 近况反差 ──
        fav_form = self._calc_form(fav_last5, fav_name)
        und_form = self._calc_form(und_last5, und_name)

        if und_form["wins"] >= 4 and fav_form["wins"] <= 2:
            signals.append({
                "type": "近况反差",
                "desc": (f"冷门方 {und_name} 近5场 {und_form['wins']}胜{und_form['draws']}平"
                         f"{und_form['losses']}负，状态火热；"
                         f"热门方 {fav_name} 近5场仅 {fav_form['wins']}胜"),
                "severity": "高",
            })
        elif und_form["wins"] >= 3 and fav_form["wins"] <= 1:
            signals.append({
                "type": "近况反差",
                "desc": (f"冷门方 {und_name} 近5场 {und_form['wins']}胜，"
                         f"热门方 {fav_name} 近5场仅 {fav_form['wins']}胜"),
                "severity": "中",
            })

        # ── 维度 2: 状态断崖 — 热门方近期连败或频繁丢球 ──
        if fav_form["losses"] >= 3:
            signals.append({
                "type": "状态断崖",
                "desc": f"热门方 {fav_name} 近5场 {fav_form['losses']}负，状态严重下滑",
                "severity": "高",
            })
        elif fav_form["losses"] >= 2 and fav_form["goals_conceded"] >= 8:
            signals.append({
                "type": "防线漏洞",
                "desc": (f"热门方 {fav_name} 近5场丢 {fav_form['goals_conceded']} 球"
                         f"（场均 {fav_form['goals_conceded']/max(fav_form['played'],1):.1f}），防守不稳"),
                "severity": "中",
            })

        # ── 维度 3: 交锋克制 ──
        if h2h_records and len(h2h_records) >= 3:
            h2h_stats = self._calc_h2h_dominance(h2h_records, home_team, away_team)
            if h2h_stats:
                h2h_winner_side = h2h_stats["dominant_side"]
                if h2h_winner_side == und_side and h2h_stats["win_pct"] >= 0.6:
                    signals.append({
                        "type": "交锋克制",
                        "desc": (f"冷门方 {und_name} 在近 {h2h_stats['total']} 次交锋中"
                                 f"赢了 {h2h_stats['dominant_wins']} 场"
                                 f"（胜率 {h2h_stats['win_pct']:.0%}），具有心理优势"),
                        "severity": "高" if h2h_stats["win_pct"] >= 0.75 else "中",
                    })

        # ── 维度 4: 赛程疲劳 ──
        fav_fatigue = self._calc_fatigue(fav_last5)
        und_fatigue = self._calc_fatigue(und_last5)

        if fav_fatigue["is_fatigued"]:
            fatigue_detail = []
            if fav_fatigue["days_span"] and fav_fatigue["days_span"] <= 15:
                fatigue_detail.append(
                    f"{fav_fatigue['played']}场比赛压缩在{fav_fatigue['days_span']}天内"
                )
            if fav_fatigue["has_european"]:
                fatigue_detail.append("期间有欧战")
            signals.append({
                "type": "赛程疲劳",
                "desc": f"热门方 {fav_name} {'，'.join(fatigue_detail)}，体能存在隐患",
                "severity": "中",
            })

        # ── 维度 5: 冷门方冲击力 — 弱队进攻凶猛 ──
        if und_form["goals_scored"] >= 10 and fav_form["goals_conceded"] >= 6:
            signals.append({
                "type": "火力冲击",
                "desc": (f"冷门方 {und_name} 近5场攻入 {und_form['goals_scored']} 球，"
                         f"而热门方 {fav_name} 近5场丢 {fav_form['goals_conceded']} 球"),
                "severity": "中",
            })

        has_risk = len(signals) > 0
        if has_risk:
            high_count = sum(1 for s in signals if s["severity"] == "高")
            print(f"  [爆冷] 检测到 {len(signals)} 个信号（{high_count} 个高危）")
            for s in signals:
                print(f"    [{s['severity']}] {s['type']}: {s['desc']}")
        else:
            print("  [爆冷] 未检测到爆冷信号")

        return {
            "has_risk": has_risk,
            "signals": signals,
            "underdog": und_side,
            "underdog_name": und_name,
            "favorite_name": fav_name,
        }

    # ── 近5场战绩统计 ──

    @staticmethod
    def _calc_form(matches: list, team_name: str) -> dict:
        """统计球队近5场胜平负、进失球"""
        wins, draws, losses = 0, 0, 0
        goals_scored, goals_conceded = 0, 0
        played = 0

        for m in matches:
            try:
                hg = int(m.get("home_goals", 0))
                ag = int(m.get("away_goals", 0))
            except (ValueError, TypeError):
                continue

            played += 1
            ht = m.get("home_team", "")
            is_home = team_name.lower() in ht.lower() or ht.lower() in team_name.lower()

            if is_home:
                goals_scored += hg
                goals_conceded += ag
                if hg > ag:
                    wins += 1
                elif hg == ag:
                    draws += 1
                else:
                    losses += 1
            else:
                goals_scored += ag
                goals_conceded += hg
                if ag > hg:
                    wins += 1
                elif ag == hg:
                    draws += 1
                else:
                    losses += 1

        return {
            "played": played, "wins": wins, "draws": draws, "losses": losses,
            "goals_scored": goals_scored, "goals_conceded": goals_conceded,
        }

    # ── 交锋优势分析 ──

    @staticmethod
    def _calc_h2h_dominance(records: list, home_team: str, away_team: str) -> Optional[dict]:
        """分析历史交锋中哪方占优"""
        home_wins, away_wins, draws = 0, 0, 0
        for r in records:
            result_str = r.get("result", "")
            # 格式: "TeamA X:Y TeamB" — 从比分解析
            score_m = re.search(r'(\d+):(\d+)', result_str)
            if not score_m:
                continue
            g1, g2 = int(score_m.group(1)), int(score_m.group(2))
            if g1 == g2:
                draws += 1
            elif home_team.lower() in result_str.lower().split(str(g1))[0].lower():
                if g1 > g2:
                    home_wins += 1
                else:
                    away_wins += 1
            else:
                if g1 > g2:
                    away_wins += 1
                else:
                    home_wins += 1

        total = home_wins + away_wins + draws
        if total == 0:
            return None

        if home_wins >= away_wins:
            return {
                "dominant_side": "home", "dominant_wins": home_wins,
                "total": total, "win_pct": home_wins / total,
            }
        else:
            return {
                "dominant_side": "away", "dominant_wins": away_wins,
                "total": total, "win_pct": away_wins / total,
            }

    # ── 赛程疲劳检测 ──

    @staticmethod
    def _calc_fatigue(matches: list) -> dict:
        """分析赛程密集度和是否有欧战"""
        if not matches or len(matches) < 3:
            return {"is_fatigued": False}

        dates = []
        has_european = False
        for m in matches:
            d = m.get("date", "")
            lg = m.get("league", "")
            if d:
                try:
                    dates.append(datetime.strptime(d, "%Y-%m-%d"))
                except ValueError:
                    pass
            if any(kw in lg.lower() for kw in
                   ["champions league", "europa league", "conference league",
                    "欧冠", "欧联", "欧协联"]):
                has_european = True

        days_span = None
        if len(dates) >= 2:
            dates.sort(reverse=True)
            days_span = (dates[0] - dates[-1]).days

        played = len(matches)
        is_fatigued = False
        if days_span is not None:
            if played >= 5 and days_span <= 18:
                is_fatigued = True
            elif played >= 4 and days_span <= 12:
                is_fatigued = True
        if has_european and played >= 4:
            is_fatigued = True

        return {
            "is_fatigued": is_fatigued,
            "played": played,
            "days_span": days_span,
            "has_european": has_european,
        }

    # ═══════════════════════════════════════════════════════════════
    #  Step 5: LLM 综合分析
    # ═══════════════════════════════════════════════════════════════

    def _call_llm(self, home_team, away_team, date,
                  ml_result, openclaw_data, h2h_records,
                  upset_signals=None) -> dict:
        """调用 LLM 综合分析"""
        from agents.predicted_agent.models.llm_predictor import predict_with_llm

        home_last_5, away_last_5 = [], []
        odds_info = {}

        if openclaw_data:
            home_sec = openclaw_data.get("home_last_5", {})
            if home_sec.get("found"):
                home_last_5 = home_sec.get("last_5", [])

            away_sec = openclaw_data.get("away_last_5", {})
            if away_sec.get("found"):
                away_last_5 = away_sec.get("last_5", [])

            odds_sec = openclaw_data.get("odds", {})
            if odds_sec.get("bookmakers"):
                top = {}
                for k in list(odds_sec["bookmakers"].keys())[:5]:
                    top[k] = odds_sec["bookmakers"][k]
                odds_info = {"bookmakers": top}
            elif odds_sec.get("odds"):
                odds_info = odds_sec["odds"]

        return predict_with_llm(
            home_team=home_team,
            away_team=away_team,
            date=date or "未指定",
            ml_result=ml_result,
            home_last_5=home_last_5,
            away_last_5=away_last_5,
            h2h_records=h2h_records,
            odds_info=odds_info,
            upset_signals=upset_signals,
        )

    # ═══════════════════════════════════════════════════════════════
    #  辅助
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _summarize_openclaw(data: Optional[dict]) -> dict:
        """生成 OpenClaw 数据摘要（不暴露完整原始数据）"""
        if not data:
            return {"status": "no_data"}

        summary = {"status": "ok"}
        odds = data.get("odds", {})
        summary["odds_found"] = odds.get("found", False)
        if odds.get("league"):
            summary["league"] = odds["league"]

        for side in ["home_last_5", "away_last_5"]:
            sec = data.get(side, {})
            summary[side] = {
                "found": sec.get("found", False),
                "count": len(sec.get("last_5", [])),
                "team": sec.get("team", ""),
            }
        return summary


# ═══════════════════════════════════════════════════════════════
#  模块级单例
# ═══════════════════════════════════════════════════════════════

_predictor: Optional[PreMatchPredictor] = None


def get_predictor() -> PreMatchPredictor:
    global _predictor
    if _predictor is None:
        _predictor = PreMatchPredictor()
    return _predictor
