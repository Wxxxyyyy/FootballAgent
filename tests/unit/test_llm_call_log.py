# -*- coding: utf-8 -*-
"""
llm_call_log 旁路契约测试。

锁死: LLM 调用日志是纯旁路 — MySQL 不可达时返回 False、不抛异常、
不能拖慢预测主链路（connect_timeout=3 封顶）。
"""

import time

import common.llm_call_log as cl


class TestSideChannelDegradation:
    def test_mysql_unreachable_returns_false_fast(self, monkeypatch):
        monkeypatch.setenv("MYSQL_HOST", "127.0.0.1")
        monkeypatch.setenv("MYSQL_PORT", "3399")  # 不存在的端口
        start = time.time()
        ok = cl.log_llm_call(model="test-model", home_team="A", away_team="B",
                             latency_ms=100, success=True)
        assert ok is False  # 降级但不抛
        assert time.time() - start < 4, "连接失败应快速返回（connect_timeout=3）"
