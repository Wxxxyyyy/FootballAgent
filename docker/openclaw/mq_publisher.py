# -*- coding: utf-8 -*-
"""
OpenClaw 容器内 MQ Publisher（轻量级，自包含）

设计说明:
  - OpenClaw 容器只 COPY docker/openclaw/ 下的文件，没有 common/ 目录
  - 这里实现一个独立的轻量 publisher，只做"发布"，不做消费
  - 消费逻辑在宿主机的 pipeline/mq_consumer.py（独立进程）
  - 与 common/mq_client.py 保持拓扑一致（exchange/queue/routing_key 相同）

第一性原理:
  - publisher 是 best-effort 的：MQ 不可用时返回 False，调用方走降级
  - 不做重试（下次 tick 自然重发），不做连接池（每次新建短连接）
  - 消息格式与 common/mq_client.py 完全一致（consumer 端不区分来源）
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  配置（从环境变量读取，docker-compose 注入）
# ═══════════════════════════════════════════════════════════════

MQ_HOST = os.getenv("MQ_HOST", "127.0.0.1")
MQ_PORT = int(os.getenv("MQ_PORT", "5672"))
MQ_USER = os.getenv("MQ_USER", "football")
MQ_PASSWORD = os.getenv("MQ_PASSWORD", "football123")

# 拓扑常量（与 common/mq_client.py 保持一致）
EXCHANGE_NAME = "football.events"
EXCHANGE_TYPE = "topic"

QUEUE_BINDINGS = {
    "q.mysql_writer":   ["football.match_result", "football.odds_update"],
    "q.neo4j_writer":   ["football.match_result"],
    "q.predict_worker": ["football.predict_request"],
}

MSG_TYPE_ROUTING = {
    "match_result":    "football.match_result",
    "odds_update":     "football.odds_update",
    "predict_request": "football.predict_request",
}


# ═══════════════════════════════════════════════════════════════
#  连接 + 拓扑声明（幂等）
# ═══════════════════════════════════════════════════════════════

def _get_connection():
    """获取 MQ 连接（调用方负责关闭）"""
    import pika
    creds = pika.PlainCredentials(MQ_USER, MQ_PASSWORD)
    params = pika.ConnectionParameters(
        host=MQ_HOST,
        port=MQ_PORT,
        credentials=creds,
        heartbeat=30,
        blocked_connection_timeout=10,
        connection_attempts=2,
        retry_delay=2,
    )
    return pika.BlockingConnection(params)


def _declare_topology(channel):
    """声明 exchange + 队列 + 绑定 + DLQ（幂等）"""
    import pika

    # 死信 exchange
    channel.exchange_declare(
        exchange="football.dlx", exchange_type="fanout", durable=True,
    )

    # 主 exchange
    channel.exchange_declare(
        exchange=EXCHANGE_NAME, exchange_type=EXCHANGE_TYPE, durable=True,
    )

    # 各主队列 + 对应 DLQ
    for queue_name in QUEUE_BINDINGS:
        dlq_name = f"{queue_name}.dlq"
        channel.queue_declare(queue=dlq_name, durable=True)
        channel.queue_bind(queue=dlq_name, exchange="football.dlx")

        args = {
            "x-dead-letter-exchange": "football.dlx",
            "x-dead-letter-routing-key": dlq_name,
        }
        channel.queue_declare(queue=queue_name, durable=True, arguments=args)

        for routing_key in QUEUE_BINDINGS[queue_name]:
            channel.queue_bind(
                queue=queue_name, exchange=EXCHANGE_NAME, routing_key=routing_key,
            )


# ═══════════════════════════════════════════════════════════════
#  发布函数
# ═══════════════════════════════════════════════════════════════

def _build_envelope(msg_type: str, data: dict) -> dict:
    """构造统一消息信封（与 common/mq_client.py 格式一致）"""
    return {
        "msg_id": str(uuid.uuid4()),
        "msg_type": msg_type,
        "source": "openclaw",
        "timestamp": datetime.now().isoformat(),
        "data": data,
    }


def _publish(msg_type: str, data: dict) -> bool:
    """发布消息（best-effort，失败返回 False 不抛异常）"""
    import pika

    routing_key = MSG_TYPE_ROUTING.get(msg_type)
    if not routing_key:
        logger.error(f"[MQ] 未知消息类型: {msg_type}")
        return False

    envelope = _build_envelope(msg_type, data)
    try:
        body = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as e:
        logger.error(f"[MQ] 消息序列化失败: {e}")
        return False

    try:
        conn = _get_connection()
        ch = conn.channel()
        _declare_topology(ch)
        ch.basic_publish(
            exchange=EXCHANGE_NAME,
            routing_key=routing_key,
            body=body,
            properties=pika.BasicProperties(
                delivery_mode=2,  # 持久化
                content_type="application/json",
                message_id=envelope["msg_id"],
                timestamp=int(datetime.now().timestamp()),
            ),
        )
        ch.close()
        conn.close()
        logger.info(f"[MQ] 发布成功: {msg_type} (msg_id={envelope['msg_id'][:8]})")
        return True
    except Exception as e:
        logger.warning(f"[MQ] 发布失败（降级，下次tick重发）: {e}")
        return False


def publish_match_result(matches: list[dict]) -> bool:
    """发布比赛结果消息（消费者: MySQL + Neo4j 写入）"""
    return _publish("match_result", {"matches": matches})


def publish_odds_update(match_id: str, home_team: str, away_team: str,
                        odds: Optional[dict] = None,
                        kickoff_time: Optional[str] = None) -> bool:
    """发布赔率更新消息（消费者: MySQL 赔率快照写入）"""
    return _publish("odds_update", {
        "match_id": match_id,
        "home_team": home_team,
        "away_team": away_team,
        "odds": odds or {},
        "kickoff_time": kickoff_time,
    })


def publish_predict_request(match: dict, tier: int = 1,
                             trigger_point: float = 0) -> bool:
    """发布预测请求消息（消费者: 调预测API + 推送结果）"""
    return _publish("predict_request", {
        "match": match,
        "tier": tier,
        "trigger_point": trigger_point,
    })
