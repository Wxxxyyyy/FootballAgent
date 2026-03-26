# -*- coding: utf-8 -*-
"""
记忆检索器 (Memory Retriever)
═══════════════════════════════
从 ChromaDB conversation_memory 中按语义检索历史摘要。

使用条件触发策略：
  - 检测到代词/回指表达 → 先在 messages 窗口内找
  - 窗口内消解失败    → 才检索 ChromaDB
  - 90%+ 请求不会触发检索
"""

import re
from typing import Sequence, Optional

from langchain_core.messages import BaseMessage

from agents.memory_manager.compactor import _get_memory_collection

# ═══════════════════════════════════════════════════════════════
#  回指/代词检测
# ═══════════════════════════════════════════════════════════════

_MEMORY_TRIGGER_RE = re.compile(
    r'上次|上回|之前|前面|刚才|还记得|我问过|我说过|'
    r'那支球队|那场比赛|那个结果|那个预测|'
    r'第一次|一开始|最早|最开始'
)

_PRONOUN_RE = re.compile(
    r'它的?|他们的?|她们的?|'
    r'这支球队|那支球队|该队|这个队|那个队|'
    r'上述球队|前面那个|前面那支'
)


# ═══════════════════════════════════════════════════════════════
#  条件判断
# ═══════════════════════════════════════════════════════════════

def needs_memory_retrieval(user_msg: str) -> bool:
    """
    判断用户输入是否需要检索长期记忆。

    触发条件（满足任一）：
      1. 包含回指表达（"上次/之前/那支球队"等）
      2. 包含代词且暗示需要历史上下文
    """
    if _MEMORY_TRIGGER_RE.search(user_msg):
        return True
    if _PRONOUN_RE.search(user_msg):
        return True
    return False


def _can_resolve_in_window(user_msg: str, recent_messages: Sequence[BaseMessage]) -> bool:
    """
    检查近期消息窗口内是否包含足够的上下文来消解代词。
    简单启发式：窗口内是否提到过球队名。
    """
    window_text = ""
    for msg in recent_messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        window_text += content + " "

    from common.team_mapping import TeamMapper
    try:
        mapper = TeamMapper()
        for team_zh, team_en in mapper._zh_to_en.items():
            if team_zh in window_text or team_en.lower() in window_text.lower():
                return True
    except Exception:
        pass

    return len(window_text.strip()) > 100


# ═══════════════════════════════════════════════════════════════
#  核心检索
# ═══════════════════════════════════════════════════════════════

def retrieve_memory(
    query: str,
    thread_id: Optional[str] = None,
    top_k: int = 3,
) -> list[dict]:
    """
    从 ChromaDB conversation_memory 中语义检索历史摘要。

    Args:
        query:     用户当前问题
        thread_id: 当前会话 ID（可选，用于过滤同会话历史）
        top_k:     返回最相关的 K 条摘要

    Returns:
        list[dict]: 每个 dict 包含 {"summary": str, "metadata": dict, "distance": float}
    """
    collection = _get_memory_collection()

    if collection.count() == 0:
        return []

    where_filter = None
    if thread_id:
        where_filter = {"thread_id": thread_id}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            where=where_filter,
        )
    except Exception as e:
        print(f"[MemoryRetriever] 检索失败: {e}")
        return []

    hits = []
    if results and results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            hit = {
                "summary": doc,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0,
            }
            hits.append(hit)

    if hits:
        print(f"[MemoryRetriever] 检索到 {len(hits)} 条历史摘要")
        for h in hits:
            print(f"  distance={h['distance']:.3f}: {h['summary'][:80]}...")

    return hits


def format_memory_context(hits: list[dict]) -> str:
    """将检索到的历史摘要格式化为可注入 Prompt 的文本"""
    if not hits:
        return ""

    lines = ["[历史记忆]"]
    for i, hit in enumerate(hits, 1):
        lines.append(f"- {hit['summary']}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  上层接口：条件触发检索
# ═══════════════════════════════════════════════════════════════

def maybe_retrieve_memory(
    user_msg: str,
    recent_messages: Sequence[BaseMessage],
    thread_id: Optional[str] = None,
) -> str:
    """
    条件触发的记忆检索入口。

    决策流程：
      1. 无代词/回指 → 返回空字符串（不检索）
      2. 有代词但窗口内能消解 → 返回空字符串（不检索）
      3. 有代词且窗口内消解失败 → 检索 ChromaDB → 返回格式化文本

    Returns:
        str: 历史记忆上下文文本（空字符串表示不需要）
    """
    if not needs_memory_retrieval(user_msg):
        return ""

    if _can_resolve_in_window(user_msg, recent_messages):
        print("[MemoryRetriever] 窗口内可消解，跳过长期记忆检索")
        return ""

    print("[MemoryRetriever] 窗口内消解失败，检索长期记忆...")
    hits = retrieve_memory(query=user_msg, thread_id=thread_id, top_k=3)
    return format_memory_context(hits)
