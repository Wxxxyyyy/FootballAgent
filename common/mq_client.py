# -*- coding: utf-8 -*-
"""
RabbitMQ 客户端 — 事件驱动的核心基础设施

设计原则（第一性原理）:
  1. 解耦: OpenClaw 只管采集+发布，不关心谁消费、怎么消费
  2. 可靠: 消息持久化 + DLQ 兜底，消费者挂了不丢消息
  3. 降级: MQ 不可用时 publisher 静默失败（采集不阻断），下次 tick 自然重发
  4. 幂等: 消费者必须幂等处理（同一条消息重复消费不产生副作用）

拓扑结构:
  Exchange: football.events (topic)
    ├─ q.mysql_writer   ← football.match_result, football.odds_update
    ├─ q.neo4j_writer   ← football.match_result
    └─ q.predict_worker ← football.predict_request
  每个主队列绑定一个 DLQ（x-dead-letter-exchange），失败3次进死信。

消息格式（统一 envelope）:
  {
    "msg_id": "uuid4",
    "msg_type": "match_result | odds_update | predict_request",
    "source": "openclaw",
    "timestamp": "ISO8601",
    "data": { ... }   # 类型相关，见各 publish_* 函数文档
  }
"""

import os
import json
import logging
import uuid
from datetime import datetime
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════

MQ_HOST = os.getenv("MQ_HOST", "127.0.0.1")
MQ_PORT = int(os.getenv("MQ_PORT", "5672"))
MQ_USER = os.getenv("MQ_USER", "football")
MQ_PASSWORD = os.getenv("MQ_PASSWORD", "football123")
MQ_VHOST = os.getenv("MQ_VHOST", "/")

# 拓扑常量
EXCHANGE_NAME = "football.events"
EXCHANGE_TYPE = "topic"

# 队列 → 订阅的 routing_key 模式
QUEUE_BINDINGS = {
    "q.mysql_writer":   ["football.match_result", "football.odds_update"],
    "q.neo4j_writer":   ["football.match_result"],
    "q.predict_worker": ["football.predict_request"],
}

# 消息类型 → routing_key
MSG_TYPE_ROUTING = {
    "match_result":    "football.match_result",
    "odds_update":     "football.odds_update",
    "predict_request": "football.predict_request",
}

# 重试配置
MAX_RETRIES = 3


# ═══════════════════════════════════════════════════════════════
#  连接管理
# ═══════════════════════════════════════════════════════════════

def _get_connection():
    """获取 MQ 连接（惰性创建，调用方负责关闭）"""
    import pika
    creds = pika.PlainCredentials(MQ_USER, MQ_PASSWORD)
    params = pika.ConnectionParameters(
        host=MQ_HOST,
        port=MQ_PORT,
        virtual_host=MQ_VHOST,
        credentials=creds,
        heartbeat=30,
        blocked_connection_timeout=10,
        connection_attempts=2,
        retry_delay=2,
    )
    return pika.BlockingConnection(params)


def _declare_topology(channel):
    """声明 exchange + 队列 + 绑定 + DLQ（幂等，重复调用无副作用）"""
    import pika

    # 死信 exchange（fanout，所有失败消息都进这里）
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
        # DLQ 队列
        channel.queue_declare(queue=dlq_name, durable=True)
        channel.queue_bind(queue=dlq_name, exchange="football.dlx")

        # 主队列（失败转 DLQ）
        args = {
            "x-dead-letter-exchange": "football.dlx",
            "x-dead-letter-routing-key": dlq_name,
        }
        channel.queue_declare(queue=queue_name, durable=True, arguments=args)

        # 绑定到主 exchange
        for routing_key in QUEUE_BINDINGS[queue_name]:
            channel.queue_bind(
                queue=queue_name, exchange=EXCHANGE_NAME, routing_key=routing_key,
            )


# ═══════════════════════════════════════════════════════════════
#  Publisher
# ═══════════════════════════════════════════════════════════════

def _build_envelope(msg_type: str, data: dict) -> dict:
    """构造统一消息信封"""
    return {
        "msg_id": str(uuid.uuid4()),
        "msg_type": msg_type,
        "source": "openclaw",
        "timestamp": datetime.now().isoformat(),
        "data": data,
    }


def publish(msg_type: str, data: dict) -> bool:
    """
    发布消息到 MQ（best-effort，失败静默返回 False，不阻断采集主链路）。

    Args:
        msg_type: 消息类型（match_result / odds_update / predict_request）
        data: 消息体（dict，会被 JSON 序列化）

    Returns:
        True=发布成功, False=发布失败（MQ不可达/序列化失败）
    """
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
                delivery_mode=2,            # 持久化
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
        # 第一性原理: MQ 是提效层不是正确性依赖，挂了不能阻断采集
        logger.warning(f"[MQ] 发布失败（降级，下次tick重发）: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
#  Consumer 基类
# ═══════════════════════════════════════════════════════════════

def consume(
    queue_name: str,
    handler: Callable[[dict, dict], bool],
    prefetch_count: int = 1,
):
    """
    阻塞式消费指定队列。

    Args:
        queue_name: 队列名（q.mysql_writer / q.neo4j_writer / q.predict_worker）
        handler: 处理函数，签名 (msg_type, data) -> bool
                 返回 True=处理成功（ack），False=处理失败（nack，重试）
        prefetch_count: 预取数量（1=逐条处理，避免单消费者堆积）

    注意:
      - 消息失败会重试，超过 MAX_RETRIES 次后进 DLQ
      - 此函数阻塞当前线程，由调用方决定如何运行（多线程/多进程）
    """
    import pika

    def _on_message(ch, method, properties, body):
        try:
            envelope = json.loads(body.decode("utf-8"))
            msg_type = envelope.get("msg_type", "")
            data = envelope.get("data", {})
            msg_id = envelope.get("msg_id", "?")[:8]

            # ─── 重试计数（自维护 header，不依赖 x-death）───
            # 旧方案依赖 x-death header，但 basic_nack(requeue=True) 不经过 DLQ，
            # x-death 永远不会被添加 → retry_count 永远为 0 → 无限循环 bug。
            # 新方案：在消息 headers 中自行维护 x-retry-count 字段。
            headers = properties.headers or {}
            retry_count = int(headers.get("x-retry-count", 0))

            logger.info(f"[MQ] 收到消息: {msg_type} (msg_id={msg_id}, retry={retry_count})")

            try:
                ok = handler(msg_type, data)
            except Exception as e:
                logger.exception(f"[MQ] 处理异常: {e}")
                ok = False

            if ok:
                ch.basic_ack(delivery_tag=method.delivery_tag)
                logger.info(f"[MQ] 处理成功: {msg_type} (msg_id={msg_id})")
            else:
                if retry_count >= MAX_RETRIES:
                    # 超过重试上限 → reject 进 DLQ
                    logger.error(f"[MQ] 超过重试上限 {MAX_RETRIES}，消息进 DLQ: {msg_type} (msg_id={msg_id})")
                    ch.basic_reject(
                        delivery_tag=method.delivery_tag,
                        requeue=False,  # 不重新入队 → 走 DLQ
                    )
                else:
                    # 退避重试：先 ack 当前消息，延迟后重新发布（带递增的 retry_count）
                    # 不用 basic_nack(requeue=True)，因为那会导致立即重投递形成死循环
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    retry_delay = min(5 * (2 ** retry_count), 60)  # 指数退避: 5s, 10s, 20s... 最大60s
                    logger.warning(
                        f"[MQ] 处理失败，{retry_delay}s 后重试 ({retry_count+1}/{MAX_RETRIES}): {msg_type}"
                    )
                    import time as _time
                    import pika as _pika
                    _time.sleep(retry_delay)
                    # 重新发布，header 中递增 retry_count
                    new_headers = dict(headers)
                    new_headers["x-retry-count"] = retry_count + 1
                    try:
                        ch.basic_publish(
                            exchange=EXCHANGE_NAME,
                            routing_key=method.routing_key,
                            body=body,
                            properties=_pika.BasicProperties(
                                delivery_mode=2,
                                content_type="application/json",
                                headers=new_headers,
                            ),
                        )
                    except Exception as pub_err:
                        logger.error(f"[MQ] 重发消息失败: {pub_err}")
        except json.JSONDecodeError as e:
            logger.error(f"[MQ] 消息体非合法 JSON，直接丢弃: {e}")
            ch.basic_ack(delivery_tag=method.delivery_tag)

    while True:
        try:
            conn = _get_connection()
            ch = conn.channel()
            _declare_topology(ch)
            ch.basic_qos(prefetch_count=prefetch_count)
            ch.basic_consume(queue=queue_name, on_message_callback=_on_message)
            logger.info(f"[MQ] 消费者启动: {queue_name} (等待消息...)")
            ch.start_consuming()
        except Exception as e:
            logger.error(f"[MQ] 消费连接断开，5秒后重连: {e}")
            import time
            time.sleep(5)


# ═══════════════════════════════════════════════════════════════
#  便捷发布函数（OpenClaw 调用）
# ═══════════════════════════════════════════════════════════════

def publish_match_result(matches: list[dict]) -> bool:
    """
    发布"比赛结果"消息。

    data 结构: {"matches": [...]}  # 已完赛比赛列表
    消费者: q.mysql_writer (写MySQL), q.neo4j_writer (写Neo4j)
    """
    return publish("match_result", {"matches": matches})


def publish_odds_update(match_id: str, home_team: str, away_team: str,
                        odds: Optional[dict] = None,
                        kickoff_time: Optional[str] = None) -> bool:
    """
    发布"赔率更新"消息。

    data 结构: {"match_id", "home_team", "away_team", "odds", "kickoff_time"}
    消费者: q.mysql_writer (写赔率快照到MySQL)
    """
    return publish("odds_update", {
        "match_id": match_id,
        "home_team": home_team,
        "away_team": away_team,
        "odds": odds or {},
        "kickoff_time": kickoff_time,
    })


def publish_predict_request(match: dict, tier: int = 1,
                             trigger_point: float = 0) -> bool:
    """
    发布"预测请求"消息。

    data 结构: {"match": {...}, "tier", "trigger_point"}
    消费者: q.predict_worker (调预测API + 推送结果)
    """
    return publish("predict_request", {
        "match": match,
        "tier": tier,
        "trigger_point": trigger_point,
    })


def is_mq_available() -> bool:
    """探测 MQ 是否可用（用于降级判断，不抛异常）"""
    try:
        conn = _get_connection()
        ch = conn.channel()
        ch.close()
        conn.close()
        return True
    except Exception:
        return False
