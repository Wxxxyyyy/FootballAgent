# -*- coding: utf-8 -*-
"""
FootballAgent MQ 消费者主进程

架构（第一性原理）:
  - OpenClaw 容器只负责采集 + 发布消息（轻量）
  - 本进程运行在宿主机，订阅 3 个队列，复用已有业务函数处理消息
  - 独立进程 = 独立故障域，consumer 挂了不影响 OpenClaw 采集

队列职责:
  q.mysql_writer   ← match_result: 调 match_sync.sync_to_mysql
                   ← odds_update:  写赔率快照到 MySQL odds_snapshots 表
  q.neo4j_writer   ← match_result: 调 match_sync.sync_to_neo4j
  q.predict_worker ← predict_request: 调 prediction_trigger.trigger_prediction（HTTP直连预测服务）

幂等性保证:
  - match_result: INSERT IGNORE，重复消息不会产生重复行
  - odds_update:  UPSERT（ON DUPLICATE KEY UPDATE），重复消息覆盖为最新值
  - predict_request: 由 prediction_trigger 的去重机制保证（Redis + 文件双写）

运行方式:
  python pipeline/mq_consumer.py                    # 单进程串行消费所有队列
  python pipeline/mq_consumer.py --queue q.mysql_writer  # 只消费指定队列
  python pipeline/mq_consumer.py --worker-per-queue      # 每队列一个线程（推荐）

部署:
  systemd 服务 deploy/football-mq-consumer.service 管理本进程
"""

import os
import sys
import logging
import argparse
import threading
from datetime import datetime

# 确保能导入 common/ 和 docker/openclaw/ 下的模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "docker", "openclaw"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# pika 连接日志太啰嗦，降到 WARNING
logging.getLogger("pika").setLevel(logging.WARNING)
logger = logging.getLogger("mq_consumer")

from common.mq_client import consume, QUEUE_BINDINGS


# ═══════════════════════════════════════════════════════════════
#  Handler: MySQL 写入
# ═══════════════════════════════════════════════════════════════

def handle_mysql_writer(msg_type: str, data: dict) -> bool:
    """处理 q.mysql_writer 队列的消息"""
    try:
        if msg_type == "match_result":
            return _handle_match_result_to_mysql(data)
        elif msg_type == "odds_update":
            return _handle_odds_update_to_mysql(data)
        else:
            logger.warning(f"[mysql_writer] 未知的消息类型: {msg_type}")
            return True  # 未知类型直接 ack，避免毒消息阻塞队列
    except Exception as e:
        logger.exception(f"[mysql_writer] 处理失败: {e}")
        return False


def _handle_match_result_to_mysql(data: dict) -> bool:
    """比赛结果写 MySQL（复用 match_sync.sync_to_mysql，INSERT IGNORE 幂等）"""
    from match_sync import sync_to_mysql
    matches = data.get("matches", [])
    if not matches:
        return True
    count = sync_to_mysql(matches)
    logger.info(f"[mysql_writer] 比赛结果写入 MySQL: {count} 场")
    return True


def _handle_odds_update_to_mysql(data: dict) -> bool:
    """赔率快照写 MySQL（UPSERT 幂等）"""
    import pymysql

    match_id = data.get("match_id", "")
    home_team = data.get("home_team", "")
    away_team = data.get("away_team", "")
    odds = data.get("odds", {})
    kickoff_time = data.get("kickoff_time")

    if not match_id or not odds:
        logger.warning(f"[mysql_writer] 赔率消息缺少 match_id 或 odds，跳过")
        return True

    conn = pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "football123"),
        database=os.getenv("MYSQL_DATABASE", "football_agent"),
        charset="utf8mb4",
    )
    cursor = conn.cursor()

    try:
        # 建表（幂等）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS odds_snapshots (
              id INT AUTO_INCREMENT PRIMARY KEY,
              match_id VARCHAR(32) NOT NULL,
              home_team VARCHAR(64),
              away_team VARCHAR(64),
              kickoff_time DATETIME NULL,
              bet365_home FLOAT,
              bet365_draw FLOAT,
              bet365_away FLOAT,
              snapshot_time DATETIME NOT NULL,
              UNIQUE KEY uk_match_time (match_id, snapshot_time),
              KEY idx_match (match_id),
              KEY idx_kickoff (kickoff_time)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """)

        # 提取 Bet365 赔率（字段名与 odds_scraper.get_odds 返回一致）
        # 优先即时赔率（B365C*），回退初盘（B365*）
        h = odds.get("B365CH") or odds.get("B365H")
        d = odds.get("B365CD") or odds.get("B365D")
        a = odds.get("B365CA") or odds.get("B365A")

        cursor.execute("""
            INSERT INTO odds_snapshots
              (match_id, home_team, away_team, kickoff_time,
               bet365_home, bet365_draw, bet365_away, snapshot_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
              bet365_home=VALUES(bet365_home),
              bet365_draw=VALUES(bet365_draw),
              bet365_away=VALUES(bet365_away)
        """, (
            match_id, home_team, away_team, kickoff_time,
            h, d, a, datetime.now(),
        ))
        conn.commit()
        logger.info(f"[mysql_writer] 赔率快照写入: {home_team} vs {away_team} ({match_id})")
        return True
    finally:
        cursor.close()
        conn.close()


# ═══════════════════════════════════════════════════════════════
#  Handler: Neo4j 写入
# ═══════════════════════════════════════════════════════════════

def handle_neo4j_writer(msg_type: str, data: dict) -> bool:
    """处理 q.neo4j_writer 队列的消息"""
    try:
        if msg_type == "match_result":
            from match_sync import sync_to_neo4j
            matches = data.get("matches", [])
            if not matches:
                return True
            count = sync_to_neo4j(matches)
            logger.info(f"[neo4j_writer] 比赛结果写入 Neo4j: {count} 场")
            return True
        else:
            logger.warning(f"[neo4j_writer] 未知的消息类型: {msg_type}")
            return True
    except Exception as e:
        logger.exception(f"[neo4j_writer] 处理失败: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
#  Handler: 预测请求
# ═══════════════════════════════════════════════════════════════

def handle_predict_worker(msg_type: str, data: dict) -> bool:
    """处理 q.predict_worker 队列的消息
    
    复用 prediction_trigger.trigger_prediction：
      - HTTP 调预测服务 API
      - 推送预测结果到用户手机
      - 失败时发系统告警
    """
    try:
        if msg_type != "predict_request":
            logger.warning(f"[predict_worker] 未知的消息类型: {msg_type}")
            return True

        match = data.get("match", {})
        tier = data.get("tier", 1)
        trigger_point = data.get("trigger_point", 0)

        if not match.get("match_id"):
            logger.warning("[predict_worker] 消息缺少 match_id，跳过")
            return True

        from prediction_trigger import trigger_prediction
        result = trigger_prediction(match, tier=tier, trigger_point=trigger_point)

        if result is not None:
            logger.info(f"[predict_worker] 预测完成: {match.get('home_team')} vs {match.get('away_team')}")
            return True
        else:
            logger.warning(f"[predict_worker] 预测失败: {match.get('home_team')} vs {match.get('away_team')}")
            return False  # 触发 MQ 重试
    except Exception as e:
        logger.exception(f"[predict_worker] 处理失败: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
#  链路追踪 wrapper: 每条消息处理开启一个独立 trace
# ═══════════════════════════════════════════════════════════════

def _with_trace(handler, queue_name):
    """包裹 handler，每条消息处理时开启一个 trace。

    作用: 一条消息从 "收到" 到 "处理完成" 形成一个完整链路，
    handler 内部的 LLM 调用/DB 写入等 @traced span 自动归属此 trace。
    """
    from common.tracer import start_trace, record_span
    import time as _time

    def wrapped(msg_type, data):
        trace_id = start_trace(f"mq.consume.{queue_name}", service="consumer")
        t0 = _time.time()
        try:
            ok = handler(msg_type, data)
            record_span(
                name=f"mq.handle.{msg_type}",
                service="consumer",
                attributes={
                    "queue": queue_name,
                    "msg_type": msg_type,
                    "match_id": data.get("match_id", ""),
                    "home_team": data.get("home_team", ""),
                },
                duration_ms=int((_time.time() - t0) * 1000),
                status="ok" if ok else "error",
            )
            return ok
        except Exception as e:
            record_span(
                name=f"mq.handle.{msg_type}",
                service="consumer",
                attributes={"queue": queue_name, "msg_type": msg_type},
                duration_ms=int((_time.time() - t0) * 1000),
                status="error",
                error=str(e),
            )
            raise

    return wrapped


# 队列 → handler 映射
QUEUE_HANDLERS = {
    "q.mysql_writer":   handle_mysql_writer,
    "q.neo4j_writer":   handle_neo4j_writer,
    "q.predict_worker": handle_predict_worker,
}


def main():
    parser = argparse.ArgumentParser(description="FootballAgent MQ 消费者")
    parser.add_argument(
        "--queue",
        choices=list(QUEUE_HANDLERS.keys()),
        help="只消费指定队列（不指定则每队列一个线程）",
    )
    parser.add_argument(
        "--worker-per-queue",
        action="store_true",
        default=True,
        help="每个队列一个线程并行消费（默认启用）",
    )
    args = parser.parse_args()

    # 加载 .env（宿主机运行时需要）
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

    logger.info("=" * 60)
    logger.info("FootballAgent MQ 消费者启动")
    logger.info(f"MQ: {os.getenv('MQ_HOST', '127.0.0.1')}:{os.getenv('MQ_PORT', '5672')}")
    logger.info("=" * 60)

    if args.queue:
        # 单队列模式
        handler = QUEUE_HANDLERS[args.queue]
        handler = _with_trace(handler, args.queue)  # 注入链路追踪
        logger.info(f"单队列模式: {args.queue}")
        consume(args.queue, handler)
    else:
        # 多线程模式：每队列一个线程
        threads = []
        for queue_name, handler in QUEUE_HANDLERS.items():
            traced_handler = _with_trace(handler, queue_name)  # 注入链路追踪
            t = threading.Thread(
                target=consume,
                args=(queue_name, traced_handler),
                name=f"consumer-{queue_name}",
                daemon=True,
            )
            t.start()
            threads.append(t)
            logger.info(f"启动消费者线程: {queue_name}")

        # 主线程等待（daemon 线程会随主线程退出）
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            logger.info("收到中断信号，退出")


if __name__ == "__main__":
    main()
