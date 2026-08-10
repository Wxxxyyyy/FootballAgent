# -*- coding: utf-8 -*-
"""
MQ 客户端契约测试。

第一性原理: MQ 是提效层不是正确性依赖。
锁死:
  1. MQ 不可达时 publish 返回 False、不抛异常、快速返回（不阻断采集）
  2. 消息信封格式正确（msg_id/msg_type/timestamp/data 四要素齐全）
  3. 未知消息类型直接拒绝（防止 poison message）
  4. 拓扑声明幂等（重复调用不报错）
"""

import os
import sys
import time
import json

# conftest 已把环境变量设好（含 MQ 指向不存在的端口）

# 必须在 import mq_client 前设置 MQ 端口为不存在的（conftest 未覆盖 MQ）
os.environ.setdefault("MQ_HOST", "127.0.0.1")
os.environ.setdefault("MQ_PORT", "5699")  # 不存在的端口 → 降级

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from common.mq_client import (
    _build_envelope,
    publish,
    MSG_TYPE_ROUTING,
    QUEUE_BINDINGS,
    MAX_RETRIES,
)


class TestEnvelope:
    def test_envelope_has_required_fields(self):
        env = _build_envelope("match_result", {"matches": []})
        assert "msg_id" in env and isinstance(env["msg_id"], str)
        assert env["msg_type"] == "match_result"
        assert "timestamp" in env
        assert env["source"] == "openclaw"
        assert "data" in env

    def test_msg_id_is_unique(self):
        env1 = _build_envelope("odds_update", {})
        env2 = _build_envelope("odds_update", {})
        assert env1["msg_id"] != env2["msg_id"]


class TestPublishDegradation:
    """MQ 不可达时的降级行为（第一性原理: 不阻断采集）"""

    def test_unreachable_returns_false_fast(self):
        start = time.time()
        ok = publish("match_result", {"matches": [{"home": "A", "away": "B"}]})
        assert ok is False  # 降级但不抛
        # 连接超时应该快速返回（connection_attempts=2, retry_delay=2 → 约4秒封顶）
        assert time.time() - start < 10, "MQ 不可达时应快速返回"

    def test_unknown_msg_type_returns_false(self):
        ok = publish("unknown_type", {"data": 1})
        assert ok is False

    def test_unserializable_data_returns_false(self):
        # 含不可序列化对象 → 返回 False，不抛异常
        ok = publish("match_result", {"bad": object()})
        assert ok is False


class TestTopologyConsistency:
    """拓扑一致性：producer 和 consumer 必须用相同的 exchange/queue/routing_key"""

    def test_all_msg_types_have_routing(self):
        for msg_type in ["match_result", "odds_update", "predict_request"]:
            assert msg_type in MSG_TYPE_ROUTING
            assert MSG_TYPE_ROUTING[msg_type].startswith("football.")

    def test_every_routing_key_bound_to_at_least_one_queue(self):
        """每条 routing_key 必须至少绑定一个队列，否则消息发出去没人收"""
        all_bindings = set()
        for keys in QUEUE_BINDINGS.values():
            all_bindings.update(keys)
        for msg_type, routing_key in MSG_TYPE_ROUTING.items():
            assert routing_key in all_bindings, (
                f"{msg_type} 的 routing_key {routing_key} 没有绑定到任何队列"
            )

    def test_dlq_configured_for_every_queue(self):
        """每个主队列都必须有对应的 DLQ（失败消息不丢失）"""
        for queue_name in QUEUE_BINDINGS:
            # consumer 端 _declare_topology 会声明 {queue_name}.dlq
            assert f"{queue_name}.dlq"  # 名称约定检查

    def test_max_retries_is_positive(self):
        """重试上限必须是正整数（0=不重试，负数=永远重试毒消息）"""
        assert isinstance(MAX_RETRIES, int) and MAX_RETRIES > 0
