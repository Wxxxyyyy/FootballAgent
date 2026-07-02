# -*- coding: utf-8 -*-
"""
预测 API 服务

接收 OpenClaw 推送的预测请求，执行完整预测流程，返回预测结果。

端点:
  POST /api/predict — 接收 OpenClaw 推送的赔率+比赛信息，执行预测
  GET  /health     — 健康检查

启动:
  python -m api.prediction_api
"""

import os
import sys
import json
import logging
from datetime import datetime

from flask import Flask, request, jsonify

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prediction_api")

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "prediction_api",
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    接收 OpenClaw 推送的预测请求

    请求体:
    {
        "match_id": "2906946",
        "home_team": "瑞士",
        "away_team": "加拿大",
        "date": "06-25",
        "tier": 1,
        "kickoff_time": "2026-06-25T22:00:00",
        "odds": {
            "B365H": 2.0, "B365D": 3.3, "B365A": 3.6,
            "B365CH": 2.63, "B365CD": 3.0, "B365CA": 3.0
        },
        "trigger_time": "2026-06-25T14:30:00"
    }

    返回:
    {
        "status": "ok",
        "result": {
            "ml_prediction": {...},
            "monte_carlo": {...},
            "recent_matches": {...},
            "h2h_records": [...],
            "llm_analysis": {...},
            "prediction_time": "..."
        }
    }
    """
    data = request.json or {}
    match_id = data.get("match_id", "")
    home_team = data.get("home_team", "")
    away_team = data.get("away_team", "")
    date = data.get("date", "")
    tier = data.get("tier", 1)
    odds = data.get("odds", {})

    logger.info(f"收到预测请求: {home_team} vs {away_team} (tier={tier}, match_id={match_id})")

    result = {
        "status": "ok",
        "match_id": match_id,
        "home_team": home_team,
        "away_team": away_team,
        "tier": tier,
        "prediction_time": datetime.now().isoformat(),
    }

    # ══════ Step 1: ML 模型预测 ══════
    try:
        from agents.predicted_agent.models.statistical_model import OddsModel
        model = OddsModel.load()

        b365h = odds.get("B365H", 2.0)
        b365d = odds.get("B365D", 3.2)
        b365a = odds.get("B365A", 3.0)
        b365ch = odds.get("B365CH") or b365h
        b365cd = odds.get("B365CD") or b365d
        b365ca = odds.get("B365CA") or b365a

        ml_result = model.predict_from_odds(
            b365h=b365h, b365d=b365d, b365a=b365a,
            b365ch=b365ch, b365cd=b365cd, b365ca=b365ca,
        )
        result["ml_prediction"] = ml_result
        logger.info(f"  [ML] 主胜={ml_result['home_win_prob']:.1%} 平={ml_result['draw_prob']:.1%} 客胜={ml_result['away_win_prob']:.1%}")
    except Exception as e:
        logger.error(f"  [ML] 预测失败: {e}")
        result["ml_prediction"] = {"error": str(e)}

    # ══════ Step 2: 蒙特卡洛模拟 ══════
    try:
        from agents.predicted_agent.models.monte_carlo_simulator import MonteCarloSimulator
        sim = MonteCarloSimulator(n_simulations=10000)
        mc_result = sim.simulate(
            odds.get("B365CH", 2.0), odds.get("B365CD", 3.2), odds.get("B365CA", 3.0)
        )
        result["monte_carlo"] = mc_result
        logger.info(f"  [MC] 最可能比分: {mc_result['most_likely_score']}")
    except Exception as e:
        logger.error(f"  [MC] 模拟失败: {e}")
        result["monte_carlo"] = {"error": str(e)}

    # ══════ Step 3: MySQL 查近5场 ══════
    try:
        import pymysql
        conn = pymysql.connect(
            host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", "football123"),
            database=os.getenv("MYSQL_DATABASE", "football_agent"),
            charset="utf8mb4",
        )
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        home_last5 = []
        away_last5 = []
        for team, key in [(home_team, "home_last5"), (away_team, "away_last5")]:
            # 标准化队名
            from agents.predicted_agent.scouters.national_team_config import resolve_national_team
            team_en = resolve_national_team(team) or team
            cursor.execute(
                """SELECT match_date, home_team, away_team, home_goals, away_goals,
                          result, competition
                   FROM intl_matches
                   WHERE home_team = %s OR away_team = %s
                   ORDER BY match_date_sorted DESC LIMIT 5""",
                (team_en, team_en),
            )
            rows = cursor.fetchall()
            for r in rows:
                r["date"] = str(r.pop("match_date"))
            if key == "home_last5":
                home_last5 = rows
            else:
                away_last5 = rows

        result["recent_matches"] = {
            "home": home_last5,
            "away": away_last5,
        }
        cursor.close()
        conn.close()
        logger.info(f"  [MySQL] {home_team} {len(home_last5)}场, {away_team} {len(away_last5)}场")
    except Exception as e:
        logger.error(f"  [MySQL] 查询失败: {e}")
        result["recent_matches"] = {"home": [], "away": []}

    # ══════ Step 4: Neo4j 查历史交锋 ══════
    try:
        from neo4j import GraphDatabase
        from agents.predicted_agent.scouters.national_team_config import resolve_national_team

        home_en = resolve_national_team(home_team) or home_team
        away_en = resolve_national_team(away_team) or away_team

        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URL", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USERNAME", "neo4j"),
                  os.getenv("NEO4J_PASSWORD", "football123")),
        )
        with driver.session() as s:
            records = s.run("""
                MATCH (a)-[r:PLAYED_AGAINST]-(b)
                WHERE (a:NationalTeam OR a:Team) AND (b:NationalTeam OR b:Team)
                  AND a.name = $home AND b.name = $away
                RETURN r.match_date as date, r.season as season,
                       r.match_result as result, r.competition as comp,
                       r.home_goals as hg, r.away_goals as ag
                ORDER BY r.match_date DESC LIMIT 5
            """, home=home_en, away=away_en).data()

        result["h2h_records"] = records
        driver.close()
        logger.info(f"  [Neo4j] 交锋记录: {len(records)}条")
    except Exception as e:
        logger.error(f"  [Neo4j] 查询失败: {e}")
        result["h2h_records"] = []

    # ══════ Step 4.5: 赛前情报采集（伤停/首发/新闻/教练风格）══════
    # 采集结果以摘要文本注入 LLM prompt，让预测考量赛前新闻/战意/伤停。
    # 任意数据源失败都不阻断预测。
    intel_summary = ""
    try:
        from agents.predicted_agent.scouters import PreMatchIntel

        hours_to_kickoff = _hours_to_kickoff(data.get("kickoff_time"), tier)
        intel = PreMatchIntel().gather(
            home_team, away_team, date or None, hours_to_kickoff=hours_to_kickoff
        )
        intel_summary = intel.get("summary", "") if isinstance(intel, dict) else ""
        result["pre_match_intel"] = {
            "tier": intel.get("tier") if isinstance(intel, dict) else tier,
            "summary": intel_summary,
            "elapsed_seconds": intel.get("elapsed_seconds") if isinstance(intel, dict) else None,
        }
        logger.info(f"  [情报] 采集完成 ({len(intel_summary)} 字符, 距开赛 {hours_to_kickoff:.1f}h)")
    except Exception as e:
        logger.error(f"  [情报] 采集失败（跳过，不阻断预测）: {e}")
        result["pre_match_intel"] = {"error": str(e), "summary": ""}

    # ══════ Step 5: LLM 综合分析 ══════
    try:
        from agents.predicted_agent.models.llm_predictor import predict_with_llm

        llm_result = predict_with_llm(
            home_team=home_team,
            away_team=away_team,
            date=date or "未指定",
            ml_result=result.get("ml_prediction", {}),
            home_last_5=result["recent_matches"]["home"],
            away_last_5=result["recent_matches"]["away"],
            h2h_records=result.get("h2h_records", []),
            odds_info=odds,
            upset_signals=None,
            pre_match_intel_summary=intel_summary,
            monte_carlo_result=result.get("monte_carlo"),
        )
        result["llm_analysis"] = llm_result
        logger.info(f"  [LLM] 分析完成")
    except Exception as e:
        logger.error(f"  [LLM] 分析失败: {e}")
        # 即便 LLM 模块整体不可用，也用 ML+蒙特卡洛 合成可推送结果
        try:
            from agents.predicted_agent.models.llm_predictor import _fallback_from_models

            result["llm_analysis"] = _fallback_from_models(
                result.get("ml_prediction", {}), result.get("monte_carlo"), str(e)
            )
        except Exception:
            result["llm_analysis"] = {"error": str(e)}

    # 保存预测结果
    _save_result(match_id, home_team, away_team, tier, result)

    logger.info(f"预测完成: {home_team} vs {away_team}")
    return jsonify(result)


def _hours_to_kickoff(kickoff_time, tier) -> float:
    """估算距开赛小时数：优先解析 kickoff_time（ISO），否则按 tier 推断。

    tier=2 表示赛前 1h 内（第二档官方首发），tier=1 表示更早（第一档缺阵预判）。
    """
    if kickoff_time:
        try:
            ko = datetime.fromisoformat(str(kickoff_time))
            return max(0.0, (ko - datetime.now()).total_seconds() / 3600.0)
        except (ValueError, TypeError):
            pass
    return 0.5 if tier == 2 else 24.0


def _save_result(match_id, home, away, tier, result):
    """保存预测结果到文件"""
    output_dir = os.path.join(PROJECT_ROOT, "data", "predictions")
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    home_safe = home.replace(" ", "_")
    away_safe = away.replace(" ", "_")
    filename = f"{date_str}_{home_safe}_vs_{away_safe}_tier{tier}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"  预测结果保存: {filepath}")


if __name__ == "__main__":
    port = int(os.getenv("PREDICTOR_PORT", "8000"))
    logger.info(f"预测API服务启动，端口 {port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
