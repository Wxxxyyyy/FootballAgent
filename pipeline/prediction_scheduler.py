# -*- coding: utf-8 -*-
"""
预测调度模块（Prediction Scheduler）

功能:
  1. 每小时检查当日比赛列表
  2. 赛前24h触发第一档预测（缺阵预判 + 惯用阵型）
  3. 赛前1h触发第二档预测（官方首发 + 实际阵型）
  4. 赛后2h拉取结果入库
  5. 预测结果保存为 JSON 文件

比赛列表来源:
  - 优先从 titan007 比分页面爬取（2026.titan007.com）
  - 备选从 OpenClaw 获取

用法:
  from pipeline.prediction_scheduler import start_prediction_loop
  start_prediction_loop()  # 启动后台定时任务

  # 手动触发单场预测
  from pipeline.prediction_scheduler import predict_single_match
  predict_single_match("Portugal", "Uzbekistan", "2026-06-24", tier=1)
"""

import os
import json
import time
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTION_OUTPUT_DIR = os.getenv(
    "PREDICTION_OUTPUT_DIR",
    str(PROJECT_ROOT / "data" / "predictions"),
)

# titan007 比分页面
TITAN007_SCORE_URL = os.getenv(
    "TITAN007_BASE_URL", "https://2026.titan007.com/"
)

# 预测触发时间窗口
TIER1_BEFORE_HOURS = 24   # 赛前24h触发第一档
TIER1_WINDOW = 2          # 触发窗口：23~25h
TIER2_BEFORE_HOURS = 1    # 赛前1h触发第二档
TIER2_WINDOW = 1          # 触发窗口：0.5~1.5h
POST_MATCH_DELAY_HOURS = 2  # 赛后2h拉取结果


# ═══════════════════════════════════════════════════════════════
#  比赛列表获取
# ═══════════════════════════════════════════════════════════════

def fetch_today_matches() -> list[dict]:
    """
    从 titan007 比分页面获取今日比赛列表

    返回: [{"match_id", "home_team", "away_team", "date", "kickoff_time", "hours_to_kickoff"}, ...]
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    }

    try:
        resp = httpx.get(TITAN007_SCORE_URL, headers=headers, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            logger.warning(f"titan007 比分页返回 {resp.status_code}")
            return []
    except Exception as e:
        logger.error(f"获取比赛列表失败: {e}")
        return []

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")

    matches = []
    now = datetime.now()

    for tr in soup.find_all("tr", id=re.compile(r"^(tr\d+_|near_tr_)\d{7}$")):
        tr_id = tr.get("id", "")
        # 提取 match_id（兼容 tr1_xxx 和 near_tr_xxx 两种格式）
        match_id = re.sub(r"^(tr\d+_|near_tr_)", "", tr_id)
        if not match_id.isdigit():
            continue

        tds = tr.find_all("td")
        if len(tds) < 5:
            continue

        date_str = ""
        home_team = ""
        away_team = ""
        finished = False

        # 找到 score td，它的前一个是主队，后一个是客队
        score_idx = None
        for i, td in enumerate(tds):
            cls = td.get("class", [""])[0] if td.get("class") else ""
            text = td.get_text(strip=True)
            if cls == "gamedate":
                date_str = text
            if cls == "score":
                score_idx = i
            if "完" in text:
                finished = True

        # 通过 score td 定位主客队
        if score_idx is not None:
            if score_idx > 0:
                home_team = tds[score_idx - 1].get_text(strip=True)
            if score_idx + 1 < len(tds):
                away_team = tds[score_idx + 1].get_text(strip=True)

        # 备选：用 class 名匹配
        if not home_team or not away_team:
            for td in tds:
                cls = td.get("class", [""])[0] if td.get("class") else ""
                text = td.get_text(strip=True)
                if cls == "home" and not home_team:
                    home_team = text
                elif cls == "guest" and not away_team:
                    away_team = text

        if not home_team or not away_team:
            continue

        # 解析开赛时间（titan007 格式：MM-DD HH:MM）
        kickoff_time = None
        hours_to_kickoff = None
        try:
            # 格式如 "06-24 01:00"
            year = now.year
            kickoff_time = datetime.strptime(f"{year}-{date_str}", "%Y-%m-%d %H:%M")
            hours_to_kickoff = (kickoff_time - now).total_seconds() / 3600
        except (ValueError, TypeError):
            pass

        matches.append({
            "match_id": match_id,
            "home_team": home_team,
            "away_team": away_team,
            "date": date_str,
            "kickoff_time": kickoff_time.isoformat() if kickoff_time else None,
            "hours_to_kickoff": hours_to_kickoff,
            "finished": finished,
        })

    logger.info(f"获取到 {len(matches)} 场比赛")
    return matches


# ═══════════════════════════════════════════════════════════════
#  单场预测
# ═══════════════════════════════════════════════════════════════

def predict_single_match(
    home_team: str,
    away_team: str,
    date: str = None,
    tier: int = 1,
    match_id: str = None,
) -> dict:
    """
    触发单场比赛的完整预测流程

    Args:
        home_team: 主队名（中文/英文均可）
        away_team: 客队名
        date: 比赛日期 YYYY-MM-DD
        tier: 1=第一档(赛前>1h), 2=第二档(赛前≤1h)
        match_id: titan007 match_id

    Returns:
        完整预测结果 dict
    """
    from agents.predicted_agent.advance_predictor import PreMatchPredictor

    hours_to_kickoff = 24 if tier == 1 else 0.5

    # 如果有 match_id，先确保赔率快照已更新
    if match_id:
        try:
            from pipeline.odds_snapshot_manager import get_manager
            mgr = get_manager()
            mgr.update_snapshot(match_id)
        except Exception as e:
            logger.warning(f"赔率快照更新失败: {e}")

    predictor = PreMatchPredictor()
    try:
        result = predictor.predict(home_team, away_team, date)

        # 如果有 match_id，用赔率快照覆盖 ML 预测的赔率信息
        if match_id:
            try:
                from pipeline.odds_snapshot_manager import get_manager
                mgr = get_manager()
                odds = mgr.get_odds(match_id)
                if odds and result.get("ml_prediction"):
                    result["ml_prediction"]["input_odds"] = {
                        "B365H": odds["B365H"],
                        "B365D": odds["B365D"],
                        "B365A": odds["B365A"],
                        "B365CH": odds["B365CH"],
                        "B365CD": odds["B365CD"],
                        "B365CA": odds["B365CA"],
                    }
            except Exception as e:
                logger.warning(f"读取赔率快照失败: {e}")

        # 加入蒙特卡洛模拟
        try:
            from agents.predicted_agent.models.monte_carlo_simulator import MonteCarloSimulator
            sim = MonteCarloSimulator(n_simulations=10000)

            # 优先从赔率快照获取赔率，否则用 ML 预测的赔率
            b365h, b365d, b365a = 2.0, 3.2, 3.0
            if match_id:
                from pipeline.odds_snapshot_manager import get_manager
                odds = get_manager().get_odds(match_id)
                if odds:
                    b365h = odds.get("B365CH") or odds.get("B365H") or 2.0
                    b365d = odds.get("B365CD") or odds.get("B365D") or 3.2
                    b365a = odds.get("B365CA") or odds.get("B365A") or 3.0
            else:
                input_odds = result.get("ml_prediction", {}).get("input_odds", {})
                b365h = input_odds.get("B365H", 2.0)
                b365d = input_odds.get("B365D", 3.2)
                b365a = input_odds.get("B365A", 3.0)

            mc_result = sim.simulate(b365h, b365d, b365a)
            result["monte_carlo"] = mc_result
        except Exception as e:
            logger.warning(f"蒙特卡洛模拟失败: {e}")
            result["monte_carlo"] = None

        # 加入元信息
        result["tier"] = tier
        result["match_id"] = match_id
        result["prediction_time"] = datetime.now().isoformat()

        # 保存到文件
        _save_prediction(result)

        return result
    finally:
        predictor.close()


# ═══════════════════════════════════════════════════════════════
#  预测结果保存
# ═══════════════════════════════════════════════════════════════

def _save_prediction(result: dict) -> str:
    """
    保存预测结果到 JSON 文件

    文件命名: {date}_{home}_vs_{away}_tier{N}.json
    返回: 保存的文件路径
    """
    os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)

    date_str = result.get("date", datetime.now().strftime("%Y-%m-%d"))
    home = result.get("home_team", "unknown").replace(" ", "_")
    away = result.get("away_team", "unknown").replace(" ", "_")
    tier = result.get("tier", 1)

    filename = f"{date_str}_{home}_vs_{away}_tier{tier}.json"
    filepath = os.path.join(PREDICTION_OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"预测结果已保存: {filepath}")
    return filepath


def load_prediction(home_team: str, away_team: str, date: str, tier: int = 1) -> Optional[dict]:
    """加载已保存的预测结果"""
    home = home_team.replace(" ", "_")
    away = away_team.replace(" ", "_")
    filename = f"{date}_{home}_vs_{away}_tier{tier}.json"
    filepath = os.path.join(PREDICTION_OUTPUT_DIR, filename)

    if not os.path.exists(filepath):
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
#  Agent Loop 主循环
# ═══════════════════════════════════════════════════════════════

def prediction_loop_tick():
    """
    预测调度的一次 tick（每小时调用一次）

    逻辑:
      1. 获取今日比赛列表
      2. 更新即将开赛比赛的赔率快照（初盘首次记录，后续更新终盘）
      3. 遍历每场比赛，根据距开赛时间判断是否触发预测
      4. 赛后拉取结果
    """
    logger.info("=== 预测调度 tick 开始 ===")
    matches = fetch_today_matches()

    if not matches:
        logger.info("今日无比赛")
        return

    # ── 更新即将开赛比赛的赔率快照 ──
    try:
        from pipeline.odds_snapshot_manager import get_manager
        mgr = get_manager()
        mgr.update_all_upcoming(matches)
    except Exception as e:
        logger.error(f"赔率快照更新失败: {e}")

    now = datetime.now()

    for m in matches:
        hours = m.get("hours_to_kickoff")
        if hours is None:
            continue

        home = m["home_team"]
        away = m["away_team"]
        match_id = m["match_id"]
        date_str = m.get("date", now.strftime("%m-%d"))

        # 赛前24h → 第一档预测
        if (TIER1_BEFORE_HOURS - TIER1_WINDOW) < hours < (TIER1_BEFORE_HOURS + TIER1_WINDOW):
            # 检查是否已预测过
            existing = load_prediction(home, away, date_str, tier=1)
            if existing:
                logger.info(f"跳过（已预测）: {home} vs {away} 第一档")
                continue

            logger.info(f"触发第一档预测: {home} vs {away} (距开赛 {hours:.1f}h)")
            try:
                predict_single_match(home, away, date_str, tier=1, match_id=match_id)
            except Exception as e:
                logger.error(f"第一档预测失败 {home} vs {away}: {e}")

        # 赛前1h → 第二档预测
        elif (TIER2_BEFORE_HOURS - 0.5) < hours < (TIER2_BEFORE_HOURS + 0.5):
            existing = load_prediction(home, away, date_str, tier=2)
            if existing:
                logger.info(f"跳过（已预测）: {home} vs {away} 第二档")
                continue

            logger.info(f"触发第二档预测: {home} vs {away} (距开赛 {hours:.1f}h)")
            try:
                predict_single_match(home, away, date_str, tier=2, match_id=match_id)
            except Exception as e:
                logger.error(f"第二档预测失败 {home} vs {away}: {e}")

        # 赛后2h → 拉取结果入库
        elif hours < -POST_MATCH_DELAY_HOURS and m.get("finished"):
            logger.info(f"比赛已结束: {home} vs {away}，结果将通过 daily_openclaw_sync 入库")

    logger.info("=== 预测调度 tick 结束 ===")


# ═══════════════════════════════════════════════════════════════
#  启动定时调度
# ═══════════════════════════════════════════════════════════════

def start_prediction_loop():
    """
    启动预测调度循环（接入 APScheduler）

    每小时执行一次 prediction_loop_tick
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger

    scheduler = BackgroundScheduler(timezone=os.getenv("TZ", "Asia/Shanghai"))
    scheduler.add_job(
        prediction_loop_tick,
        IntervalTrigger(hours=1),
        id="prediction_loop",
        replace_existing=True,
        next_run_time=datetime.now(),  # 立即执行一次
    )
    scheduler.start()
    logger.info("预测调度已启动: 每小时检查比赛并触发预测")
    return scheduler


# ═══════════════════════════════════════════════════════════════
#  命令行入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "list":
        # 列出今日比赛
        matches = fetch_today_matches()
        print(f"\n今日比赛 ({len(matches)} 场):")
        for m in matches:
            hours = m.get("hours_to_kickoff")
            status = "已完赛" if m["finished"] else (f"{hours:.1f}h后开赛" if hours else "未知")
            print(f"  {m['match_id']} | {m['home_team']} vs {m['away_team']} | {status}")

    elif len(sys.argv) > 3:
        # 手动预测: python prediction_scheduler.py Portugal Uzbekistan 2026-06-24 [tier]
        home = sys.argv[1]
        away = sys.argv[2]
        date = sys.argv[3]
        tier = int(sys.argv[4]) if len(sys.argv) > 4 else 1

        print(f"手动触发预测: {home} vs {away} ({date}) 第{tier}档")
        result = predict_single_match(home, away, date, tier=tier)
        print(f"\n预测完成，结果已保存")

    else:
        # 单次 tick 测试
        print("执行单次预测调度 tick...")
        prediction_loop_tick()
