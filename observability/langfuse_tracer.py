# -*- coding: utf-8 -*-
"""
LangFuse 链路追踪集成
从环境变量初始化；未配置时装饰器为空操作，不影响主流程
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
import time
from typing import Any, Callable, Optional, TypeVar

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_langfuse_client: Any = None
_langfuse_enabled: bool = False


def init_langfuse() -> Optional[Any]:
    """
    从 .env 读取 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST（可选），
    初始化 Langfuse 客户端。未配置或导入失败则返回 None。
    """
    global _langfuse_client, _langfuse_enabled
    load_dotenv()
    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    sk = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    if not pk or not sk:
        _langfuse_enabled = False
        _langfuse_client = None
        return None
    try:
        from langfuse import Langfuse

        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()
        _langfuse_client = Langfuse(public_key=pk, secret_key=sk, host=host)
        _langfuse_enabled = True
        return _langfuse_client
    except Exception as e:
        logger.debug("LangFuse 初始化跳过: %s", e)
        _langfuse_enabled = False
        _langfuse_client = None
        return None


def _safe_repr(obj: Any, limit: int = 4000) -> Any:
    try:
        s = repr(obj)
        return s if len(s) <= limit else s[:limit] + "...(truncated)"
    except Exception:
        return "<unreprable>"


def trace_agent_call(func: F) -> F:
    """记录 Agent 节点入参、返回值与耗时（同步/异步均支持）"""

    def _wrap_sync(*args: Any, **kwargs: Any) -> Any:
        if not _langfuse_enabled or _langfuse_client is None:
            return func(*args, **kwargs)
        t0 = time.perf_counter()
        trace = _langfuse_client.trace(name=getattr(func, "__name__", "agent_node"))
        try:
            trace.update(input={"args": _safe_repr(args), "kwargs": _safe_repr(kwargs)})
            out = func(*args, **kwargs)
            trace.update(output=_safe_repr(out))
            return out
        finally:
            trace.update(metadata={"latency_ms": (time.perf_counter() - t0) * 1000})
            try:
                _langfuse_client.flush()
            except Exception:
                pass

    async def _wrap_async(*args: Any, **kwargs: Any) -> Any:
        if not _langfuse_enabled or _langfuse_client is None:
            return await func(*args, **kwargs)
        t0 = time.perf_counter()
        trace = _langfuse_client.trace(name=getattr(func, "__name__", "agent_node"))
        try:
            trace.update(input={"args": _safe_repr(args), "kwargs": _safe_repr(kwargs)})
            out = await func(*args, **kwargs)
            trace.update(output=_safe_repr(out))
            return out
        finally:
            trace.update(metadata={"latency_ms": (time.perf_counter() - t0) * 1000})
            try:
                _langfuse_client.flush()
            except Exception:
                pass

    if inspect.iscoroutinefunction(func):
        return functools.wraps(func)(_wrap_async)  # type: ignore[return-value]
    return functools.wraps(func)(_wrap_sync)  # type: ignore[return-value]


def trace_llm_generation(func: F) -> F:
    """记录 LLM 调用详情（名称取自被装饰函数）"""

    def _wrap_sync(*args: Any, **kwargs: Any) -> Any:
        if not _langfuse_enabled or _langfuse_client is None:
            return func(*args, **kwargs)
        t0 = time.perf_counter()
        trace = _langfuse_client.trace(name=f"llm:{getattr(func, '__name__', 'gen')}")
        gen = trace.generation(name=getattr(func, "__name__", "generation"))
        try:
            gen.update(input={"args": _safe_repr(args), "kwargs": _safe_repr(kwargs)})
            out = func(*args, **kwargs)
            gen.update(output=_safe_repr(out), metadata={"latency_ms": (time.perf_counter() - t0) * 1000})
            return out
        finally:
            try:
                _langfuse_client.flush()
            except Exception:
                pass

    async def _wrap_async(*args: Any, **kwargs: Any) -> Any:
        if not _langfuse_enabled or _langfuse_client is None:
            return await func(*args, **kwargs)
        t0 = time.perf_counter()
        trace = _langfuse_client.trace(name=f"llm:{getattr(func, '__name__', 'gen')}")
        gen = trace.generation(name=getattr(func, "__name__", "generation"))
        try:
            gen.update(input={"args": _safe_repr(args), "kwargs": _safe_repr(kwargs)})
            out = await func(*args, **kwargs)
            gen.update(output=_safe_repr(out), metadata={"latency_ms": (time.perf_counter() - t0) * 1000})
            return out
        finally:
            try:
                _langfuse_client.flush()
            except Exception:
                pass

    if inspect.iscoroutinefunction(func):
        return functools.wraps(func)(_wrap_async)  # type: ignore[return-value]
    return functools.wraps(func)(_wrap_sync)  # type: ignore[return-value]
