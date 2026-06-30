# -*- coding: utf-8 -*-
"""
定时任务调度（APScheduler BackgroundScheduler）
每日 OpenClaw 同步 + 周期性 Redis 缓存清理
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_scheduler: Optional[BackgroundScheduler] = None


def daily_openclaw_sync() -> None:
    """每日触发：拉取昨日比赛并入库"""
    try:
        from pipeline.openclaw_sync import sync_daily_matches

        sync_daily_matches()
    except Exception as e:
        logger.exception("daily_openclaw_sync 执行失败: %s", e)


def cache_cleanup() -> None:
    """每小时：清理/整理应用侧 Redis 缓存键（无 TTL 的键补 TTL，避免长期堆积）"""
    load_dotenv()
    # 从各独立环境变量拼接 Redis URL，兼容 REDIS_URL 直接配置
    _redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
    _redis_port = os.getenv("REDIS_PORT", "6379")
    _redis_password = os.getenv("REDIS_PASSWORD", "")
    _redis_db = os.getenv("REDIS_DB", "0")
    _redis_auth = f":{_redis_password}@" if _redis_password else ""
    url = os.getenv("REDIS_URL", f"redis://{_redis_auth}{_redis_host}:{_redis_port}/{_redis_db}")
    try:
        import redis

        from common.constants import (
            REDIS_PREFIX_CACHE_MATCH,
            REDIS_PREFIX_CACHE_ODDS,
            REDIS_PREFIX_LLM_CACHE,
        )

        r = redis.from_url(url, decode_responses=False)
        prefixes = (REDIS_PREFIX_CACHE_MATCH, REDIS_PREFIX_CACHE_ODDS, REDIS_PREFIX_LLM_CACHE)
        patched = 0
        for prefix in prefixes:
            for key in r.scan_iter(f"{prefix}*", count=200):
                ttl = r.ttl(key)
                if ttl == -1:
                    r.expire(key, 4 * 3600)
                    patched += 1
        logger.info("cache_cleanup: 已为无 TTL 的缓存键设置 4h 过期, patched=%s", patched)
    except Exception as e:
        logger.warning("cache_cleanup 跳过或失败: %s", e)


def start_scheduler() -> BackgroundScheduler:
    """启动后台调度器（幂等：若已启动则直接返回）"""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        return _scheduler
    sched = BackgroundScheduler(timezone=os.getenv("TZ", "Asia/Shanghai"))
    sched.add_job(
        daily_openclaw_sync,
        CronTrigger(hour=9, minute=0),
        id="daily_openclaw_sync",
        replace_existing=True,
    )
    sched.add_job(
        cache_cleanup,
        IntervalTrigger(hours=1),
        id="cache_cleanup",
        replace_existing=True,
    )

    # 预测调度已统一由 Docker OpenClaw 侧的 prediction_trigger.py 负责
    # 主应用侧不再独立调度预测，避免两套系统同时触发导致重复推送
    # 如需恢复主应用侧调度，取消以下注释:
    # try:
    #     from pipeline.prediction_scheduler import prediction_loop_tick
    #     sched.add_job(
    #         prediction_loop_tick,
    #         IntervalTrigger(hours=1),
    #         id="prediction_loop",
    #         replace_existing=True,
    #     )
    #     logger.info("已注册预测调度: prediction_loop 每小时检查比赛")
    # except ImportError:
    #     logger.warning("prediction_scheduler 未安装，跳过预测调度")
    logger.info("预测调度由 Docker OpenClaw 侧统一管理")

    sched.start()
    _scheduler = sched
    logger.info("APScheduler 已启动: daily_openclaw_sync 每天 09:00, cache_cleanup 每小时")
    return sched


def stop_scheduler() -> None:
    """停止调度器"""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("APScheduler 已停止")
