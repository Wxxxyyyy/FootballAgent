# -*- coding: utf-8 -*-
"""
记忆管理节点 (Memory Manager Node)
══════════════════════════════════════
在 Summary Agent 之后、END 之前执行。

职责：
  - 检查 messages 是否超过窗口上限（WINDOW_MAX=20）
  - 超过则触发 Compaction（Flush → 摘要 → 向量化 → 裁剪）
  - 未超过则直接跳过（0ms 开销）
"""

from agents.states import AgentState
from agents.memory_manager.compactor import should_compact, compact


def memory_manager_node(state: AgentState) -> dict:
    """
    LangGraph 节点：记忆管理。

    放在 Summary Agent → END 之间，每轮对话结束后检查是否需要压缩。
    大部分情况下（消息 ≤ 20 条）直接返回空 dict，不做任何操作。
    """
    messages = state.get("messages", [])

    if not should_compact(messages):
        return {}

    thread_id = "unknown"

    print(f"\n{'=' * 50}")
    print(f"[MemoryManager] 消息数 {len(messages)} > 窗口上限，触发 Compaction")
    print(f"{'=' * 50}")

    trimmed_messages, flush_result = compact(
        messages=messages,
        thread_id=thread_id,
    )

    print(f"[MemoryManager] Compaction 完成，消息数 {len(messages)} → {len(trimmed_messages)}")

    return {
        "messages": trimmed_messages,
        "memory_metadata": flush_result,
    }
