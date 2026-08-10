# -*- coding: utf-8 -*-
"""
记忆管理节点 (Memory Manager Node)
══════════════════════════════════════
在 Summary Agent 之后、END 之前执行。

职责：
  1. 上下文压缩检查（L3 Micro-Compact / L5 Auto-Compact，见 context_manager）
  2. 后台提取动态记忆（extractMemories，守护线程异步执行，不阻塞响应）

未触发压缩时返回空 dict（0ms 开销）。
"""

import logging
import threading

from agents.states import AgentState
from agents.memory_manager.context_manager import maybe_compact

logger = logging.getLogger(__name__)


def _recent_conversation_text(state: AgentState, limit: int = 6) -> str:
    """取最近几条对话，格式化为纯文本，供记忆提取分析"""
    messages = state.get("messages", [])
    lines = []
    for m in messages[-limit:]:
        role = "用户" if getattr(m, "type", "") == "human" else "助手"
        content = m.content if hasattr(m, "content") else str(m)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _extract_bg(conversation_text: str) -> None:
    """后台线程入口：提取并写入动态记忆，失败静默降级"""
    try:
        from agents.memory_manager.memory_store import extract_and_write
        extract_and_write(conversation_text)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[Memory] 后台提取记忆失败: {e}")


def memory_manager_node(state: AgentState) -> dict:
    """
    LangGraph 节点：上下文压缩 + 动态记忆提取。

    放在 Summary Agent → END 之间。记忆提取用守护线程异步执行，
    不占用主响应链路耗时。
    """
    # 1. 上下文压缩检查（L3/L5）
    updates = maybe_compact(state)

    # 2. 后台提取动态记忆（本地小模型，零 API 成本；守护线程不阻塞响应）
    text = _recent_conversation_text(state)
    if text:
        threading.Thread(target=_extract_bg, args=(text,), daemon=True).start()

    return updates
