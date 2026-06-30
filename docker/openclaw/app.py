# -*- coding: utf-8 -*-
"""
OpenClaw 爬虫服务 — Flask + APScheduler

三大功能:
  1. 每日15:00（北京时间）同步已完赛比赛到 MySQL + Neo4j
  2. 每小时爬取即将开赛的赔率快照（初盘首次记录，后续更新终盘）
  3. 赛前7.5h/1h/30m 自动触发预测

API 端点:
  GET  /health           — 健康检查
  POST /task             — 手动触发任务
  GET  /matches/today    — 获取今日比赛列表
  GET  /odds/{match_id}  — 获取赔率快照
  POST /predict          — 手动触发单场预测
"""

import os
import logging
from datetime import datetime

from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# 配置日志
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("openclaw")

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════
#  调度器
# ═══════════════════════════════════════════════════════════════

scheduler = BackgroundScheduler(timezone=os.getenv("TZ", "Asia/Shanghai"))


def start_scheduler():
    """启动定时任务"""
    # 功能1: 每日15:00同步比赛结果
    scheduler.add_job(
        func=_daily_sync_job,
        trigger=CronTrigger(hour=15, minute=0),
        id="daily_sync",
        replace_existing=True,
    )
    logger.info("已注册: daily_sync 每天15:00")

    # 功能2: 每小时更新赔率快照
    scheduler.add_job(
        func=_odds_update_job,
        trigger=IntervalTrigger(hours=1),
        id="odds_update",
        replace_existing=True,
        next_run_time=datetime.now(),  # 启动时立即执行一次
    )
    logger.info("已注册: odds_update 每小时")

    # 功能3: 每30分钟检查预测触发
    # max_instances=1 确保不会并发执行（预测可能耗时超30分钟）
    scheduler.add_job(
        func=_prediction_trigger_job,
        trigger=IntervalTrigger(minutes=30),
        id="prediction_trigger",
        replace_existing=True,
        next_run_time=datetime.now(),
        max_instances=1,
        coalesce=True,  # 如果错过多次调度，只执行一次
    )
    logger.info("已注册: prediction_trigger 每30分钟 (max_instances=1)")

    scheduler.start()
    logger.info("APScheduler 已启动")


def _daily_sync_job():
    """每日比赛结果同步"""
    try:
        from match_sync import daily_sync
        daily_sync()
    except Exception as e:
        logger.exception(f"daily_sync 执行失败: {e}")


def _odds_update_job():
    """赔率快照更新"""
    try:
        from odds_scraper import fetch_match_list, get_snapshot_manager
        matches = fetch_match_list()
        if matches:
            mgr = get_snapshot_manager()
            count = mgr.update_all_upcoming(matches)
            logger.info(f"赔率快照更新: {count}场")
    except Exception as e:
        logger.exception(f"odds_update 执行失败: {e}")


def _prediction_trigger_job():
    """预测触发检查"""
    try:
        from prediction_trigger import prediction_loop
        prediction_loop()
    except Exception as e:
        logger.exception(f"prediction_trigger 执行失败: {e}")


# ═══════════════════════════════════════════════════════════════
#  API 端点
# ═══════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "service": "openclaw",
        "timestamp": datetime.now().isoformat(),
        "scheduler_running": scheduler.running,
        "jobs": [j.id for j in scheduler.get_jobs()],
    })


@app.route("/matches/today", methods=["GET"])
def get_matches():
    """获取今日比赛列表"""
    from odds_scraper import fetch_match_list
    matches = fetch_match_list()
    return jsonify({
        "count": len(matches),
        "matches": matches,
    })


@app.route("/odds/<match_id>", methods=["GET"])
def get_odds(match_id):
    """获取赔率快照"""
    from odds_scraper import get_snapshot_manager
    mgr = get_snapshot_manager()
    odds = mgr.get_odds(match_id)
    if odds:
        return jsonify(odds)
    return jsonify({"error": "赔率快照不存在"}), 404


@app.route("/task", methods=["POST"])
def handle_task():
    """
    手动触发任务

    请求体:
      {"task_type": "fetch_worldcup_odds" | "pre_match_analysis" | "fetch_daily_matches",
       "params": {...}}
    """
    data = request.json or {}
    task_type = data.get("task_type")
    params = data.get("params", {})

    logger.info(f"收到任务: {task_type}, params: {params}")

    if task_type == "fetch_worldcup_odds":
        from odds_scraper import fetch_match_list
        matches = fetch_match_list()
        return jsonify({"status": "ok", "count": len(matches), "matches": matches})

    elif task_type == "fetch_daily_matches":
        from match_sync import fetch_finished_matches
        matches = fetch_finished_matches()
        return jsonify({"status": "ok", "count": len(matches), "matches": matches})

    elif task_type == "pre_match_analysis":
        # 手动触发赛前分析
        from prediction_trigger import trigger_prediction
        match_id = params.get("match_id", "")
        home = params.get("home_team", "")
        away = params.get("away_team", "")
        tier = params.get("tier", 1)
        match = {
            "match_id": match_id,
            "home_team": home,
            "away_team": away,
            "date": params.get("date", ""),
            "kickoff_time": params.get("kickoff_time"),
        }
        result = trigger_prediction(match, tier=tier)
        if result:
            return jsonify({"status": "ok", "result": result})
        return jsonify({"status": "error", "message": "预测失败"}), 500

    elif task_type == "update_odds":
        # 手动更新赔率快照
        from odds_scraper import get_snapshot_manager
        match_id = params.get("match_id", "")
        if match_id:
            mgr = get_snapshot_manager()
            mgr.update(match_id, params.get("kickoff_time"))
            odds = mgr.get_odds(match_id)
            return jsonify({"status": "ok", "odds": odds})
        return jsonify({"error": "缺少 match_id"}), 400

    else:
        return jsonify({"error": f"未知任务类型: {task_type}"}), 400


@app.route("/predict", methods=["POST"])
def manual_predict():
    """
    手动触发单场预测

    请求体:
      {"match_id": "2906973", "home_team": "葡萄牙", "away_team": "乌兹别克斯坦",
       "date": "06-24", "tier": 1}
    """
    from prediction_trigger import trigger_prediction
    data = request.json or {}
    match = {
        "match_id": data.get("match_id", ""),
        "home_team": data.get("home_team", ""),
        "away_team": data.get("away_team", ""),
        "date": data.get("date", ""),
        "kickoff_time": data.get("kickoff_time"),
    }
    tier = data.get("tier", 1)
    result = trigger_prediction(match, tier=tier)
    if result:
        return jsonify({"status": "ok", "result": result})
    return jsonify({"status": "error", "message": "预测失败"}), 500


# ═══════════════════════════════════════════════════════════════
#  启动
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.getenv("OPENCLAW_PORT", "9000"))
    start_scheduler()
    logger.info(f"OpenClaw 服务启动，端口 {port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
