# -*- coding: utf-8 -*-
"""
容器内统一调度器（替代 systemd timer，全容器化后由 compose scheduler 服务拉起）：

  - accuracy_tracker：每天 15:30 跑一次（对齐原 football-accuracy.timer，
    在 OpenClaw 15:00 完赛结果同步之后评估昨日预测）
  - alert_rules：每 10 分钟跑一次（对齐原 football-alert.timer，
    巡检 llm_calls 台账 + Bark 告警）

启动方式：python -m scripts.scheduler_runner（见 docker-compose.yml scheduler 服务）
"""

import logging
import sys
from pathlib import Path

# 保证可以 import 到 scripts 包下的任务脚本
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from scripts import accuracy_tracker, alert_rules

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("scheduler_runner")


def _run(name: str, fn) -> None:
    """统一执行入口：吞掉异常与 SystemExit，保证调度进程本身永不退出。"""
    logger.info(f"[scheduler] 开始执行 {name}")
    try:
        fn()
    except SystemExit:
        # 脚本 __main__ 里的 sys.exit(main()) 不会走到这里（我们直接调 main），
        # 但脚本内部若主动 sys.exit 也不应拖垮调度器
        pass
    except Exception:
        logger.exception(f"[scheduler] {name} 执行异常")


def main() -> None:
    scheduler = BlockingScheduler(timezone="Asia/Shanghai")
    # 准确率追踪：每日 15:30（OpenClaw 15:00 同步完赛结果之后）
    scheduler.add_job(
        _run,
        CronTrigger(hour=15, minute=30),
        args=["accuracy_tracker", accuracy_tracker.main],
        id="accuracy_tracker",
        max_instances=1,
    )
    # LLM 调用告警巡检：每 10 分钟
    scheduler.add_job(
        _run,
        IntervalTrigger(minutes=10),
        args=["alert_rules", alert_rules.main],
        id="alert_rules",
        max_instances=1,
    )
    logger.info("[scheduler] 调度器启动：accuracy_tracker(每日15:30) + alert_rules(每10分钟)")
    scheduler.start()


if __name__ == "__main__":
    main()
