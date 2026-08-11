# -*- coding: utf-8 -*-
"""
聊天路由：同步调用 LangGraph；SSE 流式输出 token/节点更新。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# 项目根目录（backend/api -> footballAgent）
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

router = APIRouter(tags=["聊天"])

_graph = None


def _get_graph():
    """懒编译 LangGraph，避免重复构建。"""
    global _graph
    if _graph is None:
        from agents.graph_builder import build_graph

        _graph = build_graph()
    return _graph


class ChatBody(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str = Field(default="default-thread")


@router.post("/chat")
async def chat_sync(body: ChatBody):
    """同步整轮对话，返回最后一条助手消息文本。"""
    from langchain_core.messages import HumanMessage
    from common.langfuse_client import get_client as _get_langfuse, get_langchain_handler

    try:
        graph = _get_graph()
    except Exception as e:
        raise HTTPException(503, f"图加载失败: {e}") from e

    # Langfuse 链路追踪: 挂 LangChain 回调后，图内所有节点/LLM 调用自动嵌套进一条 trace
    # （未配置密钥时 handler/client 均为 None，自动跳过，业务不受影响）
    config = {"configurable": {"thread_id": body.thread_id}}
    handler = get_langchain_handler()
    if handler:
        config["callbacks"] = [handler]

    langfuse = _get_langfuse()
    trace_id = None
    if langfuse:
        with langfuse.start_as_current_trace(name="chat.request"):
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=body.message)]},
                config,
            )
            try:
                trace_id = langfuse.get_current_trace_id()
            except Exception:  # noqa: BLE001
                trace_id = None
    else:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=body.message)]},
            config,
        )
    msgs = result.get("messages") or []

    if not msgs:
        return {"reply": "", "thread_id": body.thread_id, "trace_id": trace_id}
    final = msgs[-1]
    text = getattr(final, "content", str(final))
    return {"reply": text, "thread_id": body.thread_id, "trace_id": trace_id}


@router.post("/chat/stream")
async def chat_stream(body: ChatBody):
    """SSE：流式推送 astream 的增量状态（JSON 行）。"""

    async def gen():
        from langchain_core.messages import HumanMessage
        from common.langfuse_client import get_langchain_handler

        try:
            graph = _get_graph()
        except Exception as e:
            yield {"data": json.dumps({"error": str(e)}, ensure_ascii=False)}
            return
        # Langfuse 链路追踪（未配置密钥时 handler 为 None，自动跳过）
        config = {"configurable": {"thread_id": body.thread_id}}
        handler = get_langchain_handler()
        if handler:
            config["callbacks"] = [handler]
        try:
            async for chunk in graph.astream(
                {"messages": [HumanMessage(content=body.message)]},
                config,
                stream_mode="updates",
            ):
                line = json.dumps(chunk, ensure_ascii=False, default=str)
                yield {"data": line}
        except Exception as e:
            yield {"data": json.dumps({"error": str(e)}, ensure_ascii=False)}
        yield {"data": "[DONE]"}

    return EventSourceResponse(gen())
