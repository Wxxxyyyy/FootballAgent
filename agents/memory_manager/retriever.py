# -*- coding: utf-8 -*-
"""
记忆检索器 (Memory Retriever)
═══════════════════════════════
从 data/memory/*.md 动态记忆中召回相关历史（cc 做法：索引常驻 + 小模型选择题，
不用向量库）。

使用条件触发策略：
  - 检测到代词/回指表达 → 先在 messages 窗口内找
  - 窗口内消解失败    → 才召回长期记忆（memory_store.recall_memories）
  - 90%+ 请求不会触发召回
"""

import re
import logging
from typing import Sequence, Optional

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

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
    判断用户输入是否需要召回长期记忆。
    触发条件（满足任一）：包含回指表达 / 包含代词且暗示需要历史上下文。
    """
    if _MEMORY_TRIGGER_RE.search(user_msg):
        return True
    if _PRONOUN_RE.search(user_msg):
        return True
    return False


def _can_resolve_in_window(user_msg: str, recent_messages: Sequence[BaseMessage]) -> bool:
    """
    检查近期消息窗口内是否包含足够的上下文来消解代词。
    启发式：窗口内是否提到过任意已知球队名（中/英/别名）。
    """
    window_text = ""
    for msg in recent_messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        window_text += content + " "

    try:
        from common import team_mapping
        lower_window = window_text.lower()
        for en in team_mapping.all_teams():
            if en.lower() in lower_window:
                return True
            zh = team_mapping.to_chinese(en)
            alias = team_mapping.to_chinese(en, alias=True)
            if (zh and zh in window_text) or (alias and alias in window_text):
                return True
    except Exception:  # noqa: BLE001
        pass

    return len(window_text.strip()) > 100


# ═══════════════════════════════════════════════════════════════
#  上层接口：条件触发召回
# ═══════════════════════════════════════════════════════════════

def maybe_retrieve_memory(
    user_msg: str,
    recent_messages: Sequence[BaseMessage],
    thread_id: Optional[str] = None,
) -> str:
    """
    条件触发的记忆召回入口。

    决策流程：
      1. 无代词/回指 → 返回空字符串（不召回）
      2. 有代词但窗口内能消解 → 返回空字符串（不召回）
      3. 有代词且窗口内消解失败 → 召回长期记忆（md 索引 + 小模型选择题）→ 返回正文

    Returns:
        str: 召回的记忆正文（空字符串表示不需要）
    """
    if not needs_memory_retrieval(user_msg):
        return ""

    if _can_resolve_in_window(user_msg, recent_messages):
        logger.info("[MemoryRetriever] 窗口内可消解，跳过长期记忆召回")
        return ""

    logger.info("[MemoryRetriever] 窗口内消解失败，召回长期记忆（md 索引）...")
    from agents.memory_manager.memory_store import recall_memories
    return recall_memories(user_msg, top_k=3)
