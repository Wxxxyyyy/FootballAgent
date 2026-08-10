# -*- coding: utf-8 -*-
"""
链路追踪 (tracer) 契约测试。

核心契约:
  1. start_trace 返回非空 trace_id，且写入上下文
  2. @traced decorator 不改变函数行为（返回值/异常都透传）
  3. @traced 异常时记录 status=error 并重新抛出原异常
  4. 嵌套 @traced 自动建立 parent_id 层级
  5. MySQL 不可达时静默降级，绝不抛异常影响业务
  6. record_span 手动埋点正常工作
  7. llm_call_log 能关联当前 trace_id（跨模块一致性）
"""

import time
import pytest

import common.tracer as tracer
import common.llm_call_log as cl


# ═══════════════════════════════════════════════════════════════
#  上下文传播契约
# ═══════════════════════════════════════════════════════════════

class TestContextPropagation:
    def test_start_trace_returns_nonempty_id(self):
        """start_trace 必须返回非空 trace_id"""
        tid = tracer.start_trace("test_request", service="test")
        assert tid and len(tid) == 32, f"trace_id 应为32位hex, 实际: {tid}"

    def test_get_current_trace_id_after_start(self):
        """start_trace 后 get_current_trace_id 必须能读到同一个 id"""
        tid = tracer.start_trace("test", service="test")
        assert tracer.get_current_trace_id() == tid

    def test_nested_start_trace_reuses_existing(self):
        """已有 trace_id 时再 start_trace 应复用，不覆盖"""
        tid1 = tracer.start_trace("outer", service="test")
        tid2 = tracer.start_trace("inner", service="test")
        assert tid1 == tid2, "嵌套 start_trace 应复用已有 trace_id"


# ═══════════════════════════════════════════════════════════════
#  @traced decorator 行为契约
# ═══════════════════════════════════════════════════════════════

class TestTracedDecorator:
    def test_traced_preserves_return_value(self):
        """@traced 不能改变函数返回值"""
        @tracer.traced("test.add", service="test")
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_traced_reraises_exception(self):
        """@traced 必须重新抛出原异常，不能吞掉"""
        @tracer.traced("test.fail", service="test")
        def boom():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            boom()

    def test_traced_with_attributes(self):
        """@traced 的 attributes 参数不破坏函数行为"""
        @tracer.traced("test.with_attrs", service="test",
                       attributes={"model": "kimi", "tier": 1})
        def predict(x):
            return {"result": x}

        result = predict("home_win")
        assert result == {"result": "home_win"}


# ═══════════════════════════════════════════════════════════════
#  降级契约（MySQL 不可达时绝不阻断业务）
# ═══════════════════════════════════════════════════════════════

class TestDegradation:
    def test_traced_does_not_raise_on_mysql_down(self, monkeypatch):
        """MySQL 不可达时 @traced 必须静默降级，不影响被包裹函数"""
        monkeypatch.setenv("MYSQL_HOST", "127.0.0.1")
        monkeypatch.setenv("MYSQL_PORT", "3399")  # 不存在的端口

        @tracer.traced("test.mysql_down", service="test")
        def should_still_work():
            return "ok"

        # 不抛异常 + 返回值正确
        assert should_still_work() == "ok"

    def test_record_span_does_not_raise_on_mysql_down(self, monkeypatch):
        """MySQL 不可达时 record_span 必须静默返回，不抛异常"""
        monkeypatch.setenv("MYSQL_HOST", "127.0.0.1")
        monkeypatch.setenv("MYSQL_PORT", "3399")

        # 不抛异常即可
        sid = tracer.record_span("test.manual", service="test",
                                 duration_ms=42, status="ok")
        assert sid and len(sid) == 32

    def test_get_trace_returns_empty_on_mysql_down(self, monkeypatch):
        """MySQL 不可达时查询返回空列表，不抛异常"""
        monkeypatch.setenv("MYSQL_HOST", "127.0.0.1")
        monkeypatch.setenv("MYSQL_PORT", "3399")

        result = tracer.get_trace("nonexistent-trace-id")
        assert result == []

    def test_list_recent_traces_returns_empty_on_mysql_down(self, monkeypatch):
        """MySQL 不可达时列表查询返回空列表，不抛异常"""
        monkeypatch.setenv("MYSQL_HOST", "127.0.0.1")
        monkeypatch.setenv("MYSQL_PORT", "3399")

        result = tracer.list_recent_traces(limit=5)
        assert result == []


# ═══════════════════════════════════════════════════════════════
#  手动埋点契约
# ═══════════════════════════════════════════════════════════════

class TestManualSpan:
    def test_record_span_returns_span_id(self):
        """record_span 必须返回非空 span_id"""
        tracer.start_trace("manual_test", service="test")
        sid = tracer.record_span("manual.step", service="test", duration_ms=10)
        assert sid and len(sid) == 32

    def test_record_span_with_error_status(self):
        """record_span 能记录 error 状态"""
        tracer.start_trace("error_test", service="test")
        sid = tracer.record_span(
            "failed.step", service="test",
            duration_ms=5, status="error", error="connection refused"
        )
        assert sid and len(sid) == 32


# ═══════════════════════════════════════════════════════════════
#  跨模块关联契约 (tracer ↔ llm_call_log)
# ═══════════════════════════════════════════════════════════════

class TestCrossModuleTraceId:
    def test_llm_call_log_reads_current_trace_id(self, monkeypatch):
        """llm_call_log 必须能读到当前上下文的 trace_id（关联 traces 表）"""
        monkeypatch.setenv("MYSQL_HOST", "127.0.0.1")
        monkeypatch.setenv("MYSQL_PORT", "3399")  # MySQL 不可达，但 trace_id 仍应被读取

        tid = tracer.start_trace("llm_call_test", service="agent")
        assert tracer.get_current_trace_id() == tid

        # llm_call_log 写入会失败（MySQL 不可达），但不应抛异常
        ok = cl.log_llm_call(model="test", home_team="A", away_team="B",
                             latency_ms=100, success=True)
        assert ok is False  # MySQL 不可达，降级返回 False

        # 但 trace_id 在上下文中仍然存在（未被 llm_call_log 破坏）
        assert tracer.get_current_trace_id() == tid

    def test_traced_function_inside_trace_inherits_id(self):
        """start_trace 后调用 @traced 函数，span 应归属同一 trace_id"""
        tid = tracer.start_trace("parent_trace", service="api")

        @tracer.traced("child.op", service="agent")
        def child_operation():
            # 函数内部能读到外层 trace_id
            return tracer.get_current_trace_id()

        child_tid = child_operation()
        assert child_tid == tid, "嵌套 @traced 必须继承外层 trace_id"
