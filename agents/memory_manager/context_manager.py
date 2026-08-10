# -*- coding: utf-8 -*-
"""
上下文压缩引擎（Context Manager）
═══════════════════════════════════════════════════════════════
参照 Claude Code 的五层压缩金字塔裁剪版（去掉 L2 砍远古、L4 读时投影），
保留三层，按 token 预算管理：

  L1 大结果存磁盘     单条工具结果 > L1_RESULT_THRESHOLD → 写磁盘，留预览+路径
  L3 Micro-Compact   双触发：① 上下文 ≥ L3_TOKEN_TRIGGER(500K)
                             ② 距上次压缩 ≥ L3_TIME_TRIGGER(6h)
                     动作：清掉"可重新获取"的工具结果（mysql/neo4j/RAG 查询），
                          只保留最近 5 条"不可重现"的结果（LLM 预测/分析）
  L5 Auto-Compact    上下文 ≥ L5_TRIGGER(1M-33K) → 全部消息送 LLM 全量重写，
                     摘要作为 SystemMessage 留在上下文（不存任何外部库），
                     旧消息丢弃

token 计数用 tiktoken 近似（Qwen tokenizer 略有差异，做预算管理够用）。
"""

import os
import time
import hashlib
import logging
from typing import Sequence, Optional

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    ToolMessage,
)

from common.llm_select import llm_call, LLM_MODEL_KIMI_NAME
from agents.memory_manager.prompts import get_flush_prompt, get_compaction_prompt

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  参数（对齐 cc，1M 窗口）
# ═══════════════════════════════════════════════════════════════

MODEL_WINDOW = 1_000_000          # 模型上下文窗口（Qwen 1M）
BUFFER = 33_000                   # 20K 摘要输出预留 + 13K 安全线
L5_TRIGGER = MODEL_WINDOW - BUFFER  # 967K：全量重写触发线
L3_TOKEN_TRIGGER = 500_000        # 50% 窗口：Micro-Compact token 触发线
L3_TIME_TRIGGER = 6 * 3600        # 6 小时：Micro-Compact 时间触发线
L1_RESULT_THRESHOLD = 50_000      # 单条工具结果超 50KB 存磁盘
L5_KEEP_TAIL = 6                  # L5 全量重写后保留的最近消息条数

# L3 时清掉的"可重新获取"工具（结果可重查，清了不丢信息）
_RETRIEVABLE_TOOLS = {"mysql_query", "neo4j_query", "search_knowledge_base"}

# ═══════════════════════════════════════════════════════════════
#  token 计数（tiktoken 近似，失败按字符数/3 兜底）
# ═══════════════════════════════════════════════════════════════

_counter = None


def _get_counter():
    global _counter
    if _counter is None:
        try:
            import tiktoken
            _counter = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _counter = False
    return _counter


def count_tokens(text) -> int:
    """近似 token 计数"""
    if not text:
        return 0
    s = str(text)
    enc = _get_counter()
    if enc:
        try:
            return len(enc.encode(s))
        except Exception:
            pass
    return len(s) // 3  # 兜底估算


def count_messages_tokens(messages: Sequence[BaseMessage]) -> int:
    """统计消息列表总 token（含每条消息的元数据开销）"""
    total = 0
    for m in messages:
        content = m.content if hasattr(m, "content") else str(m)
        total += count_tokens(content) + 4
    return total


# ═══════════════════════════════════════════════════════════════
#  L1 · 大结果存磁盘
# ═══════════════════════════════════════════════════════════════

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "context_cache",
)


def spill_to_disk(text, label: str = "tool_result"):
    """
    单条工具结果超过阈值时，全量写入磁盘，消息里只留预览 + 文件路径。
    完整内容不丢，需要时按路径 Read 回来（offset/limit 分段）。
    """
    s = str(text)
    if len(s) <= L1_RESULT_THRESHOLD:
        return text

    os.makedirs(_CACHE_DIR, exist_ok=True)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:12]
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_CACHE_DIR, f"{label}_{ts}_{h}.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(s)
    except Exception as e:
        logger.warning(f"[L1] 存磁盘失败，保留原文: {e}")
        return text

    preview = s[:2000]
    return (
        f"{preview}\n\n"
        f"[⋯ 内容过长（{len(s)} 字符），完整结果已存磁盘: {path} ⋯]\n"
        f"[需要细节时可用 Read 按路径 + offset/limit 分段取回]"
    )


# ═══════════════════════════════════════════════════════════════
#  L3 · Micro-Compact
# ═══════════════════════════════════════════════════════════════

def _classify(msg) -> str:
    """
    分类消息：
      retrievable     = 可重新获取的结果（L3 清掉，需要时再查/重算）
      non_retrievable = 不可重现的工具结果（L3 保留最近 5 条）
      other           = 普通对话（用户/助手文本，始终保留）
    """
    name = getattr(msg, "name", "") or ""
    if isinstance(msg, ToolMessage):
        return "retrievable" if name in _RETRIEVABLE_TOOLS else "non_retrievable"
    # 预测结果：可重新预测 + 磁盘 data/predictions/ 有存档 → 视为可重新获取
    if name == "prediction_result":
        return "retrievable"
    return "other"


def micro_compact(messages: Sequence[BaseMessage]) -> list:
    """
    L3：清掉可重新获取的工具结果，不可重现的结果只保留最近 5 条。
    普通对话（用户/助手文本）全部保留，保持原有顺序。
    """
    # 找出 non_retrievable 的索引，只留最近 5 条
    non_retr_idx = [i for i, m in enumerate(messages) if _classify(m) == "non_retrievable"]
    keep_non_retr = set(non_retr_idx[-5:]) if len(non_retr_idx) > 5 else set(non_retr_idx)

    out = []
    cleared = 0
    for i, m in enumerate(messages):
        cls = _classify(m)
        if cls == "retrievable":
            cleared += 1
            continue
        if cls == "non_retrievable" and i not in keep_non_retr:
            cleared += 1
            continue
        out.append(m)

    if cleared > 0:
        out.insert(0, SystemMessage(
            content=f"[Micro-Compact] 已清理 {cleared} 条可重新获取/过旧的中间结果"
                    f"（数据库/知识库查询结果，需要时可重新查询）。"
        ))
    logger.info(f"[L3] Micro-Compact 清理 {cleared} 条，剩余 {len(out)} 条")
    return out


# ═══════════════════════════════════════════════════════════════
#  L5 · Auto-Compact 全量重写
# ═══════════════════════════════════════════════════════════════

def _format_messages(messages: Sequence[BaseMessage]) -> str:
    """消息列表格式化为纯文本"""
    lines = []
    for msg in messages:
        role = "用户" if getattr(msg, "type", "") == "human" else "助手"
        content = msg.content if hasattr(msg, "content") else str(msg)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _flush_key_info(messages_text: str) -> dict:
    """Memory Flush：LLM 提取关键信息（实体/事实/偏好/决策），失败返回空结构"""
    import json as _json
    import re as _re
    prompt = get_flush_prompt(messages_text)
    try:
        response = llm_call(prompt, model=LLM_MODEL_KIMI_NAME, force_fallback=True)
        raw = response.content.strip()
        if "```" in raw:
            m = _re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', raw, _re.DOTALL)
            if m:
                raw = m.group(1).strip()
        return _json.loads(raw)
    except Exception as e:
        logger.warning(f"[L5] Flush 提取失败，用空结构: {e}")
        return {"entities": [], "key_facts": [], "user_preferences": [], "decisions": []}


def _generate_summary(messages_text: str, flush_result: dict) -> str:
    """全量重写：LLM 生成结构化摘要"""
    import json as _json
    flush_json = _json.dumps(flush_result, ensure_ascii=False, indent=2)
    prompt = get_compaction_prompt(messages_text, flush_json)
    try:
        response = llm_call(prompt, model=LLM_MODEL_KIMI_NAME, force_fallback=True)
        return response.content.strip()
    except Exception as e:
        logger.warning(f"[L5] 摘要生成失败，用原文截断兜底: {e}")
        return messages_text[:200] + "..."


def auto_compact(messages: Sequence[BaseMessage]) -> list:
    """
    L5：全部消息送 LLM 全量重写，摘要作为 SystemMessage 留在上下文（不存外部库），
    保留最近 L5_KEEP_TAIL 条消息作为"尾巴"，旧消息丢弃。
    """
    total = len(messages)
    tail = list(messages[-L5_KEEP_TAIL:]) if total > L5_KEEP_TAIL else list(messages)
    to_compress = list(messages[:-L5_KEEP_TAIL]) if total > L5_KEEP_TAIL else []

    if not to_compress:
        # 消息极少却超 token：说明单条超长，截断即可
        logger.warning(f"[L5] 消息仅 {total} 条却超触发线，跳过全量重写")
        return list(messages)

    text = _format_messages(to_compress)
    flush_result = _flush_key_info(text)
    summary = _generate_summary(text, flush_result)

    new_msgs = [SystemMessage(
        content=(
            f"[Auto-Compact] 本会话此前 {total} 条消息已压缩为摘要。"
            f"本段是接续而非从头开始，顺着摘要中的当前进展继续。\n\n"
            f"历史摘要：\n{summary}"
        )
    )] + tail

    logger.info(
        f"[L5] Auto-Compact: {total} 条 → {len(new_msgs)} 条，摘要 {len(summary)} 字符"
    )
    return new_msgs


# ═══════════════════════════════════════════════════════════════
#  消息替换（适配 add_messages reducer：先 RemoveMessage 旧的，再加新的）
# ═══════════════════════════════════════════════════════════════

def _build_replacement(old_messages: Sequence[BaseMessage], new_messages: list) -> list:
    """
    构造消息更新：用 RemoveMessage 移除旧消息，再追加新消息。
    兼容 add_messages reducer（该 reducer 只按 id 更新/追加，不会自动删）。
    消息无 id 时退化为直接返回新列表（依赖上层 reducer 行为）。
    """
    try:
        from langgraph.graph.message import RemoveMessage
        updates = []
        for m in old_messages:
            mid = getattr(m, "id", None)
            if mid:
                updates.append(RemoveMessage(id=mid))
        updates.extend(new_messages)
        return updates
    except Exception:
        return new_messages


# ═══════════════════════════════════════════════════════════════
#  编排：每轮处理前调用
# ═══════════════════════════════════════════════════════════════

def maybe_compact(state: dict) -> dict:
    """
    上下文压缩编排入口。在每轮处理前（intent_node 开头）调用。

    触发优先级：L5 > L3。
      - L5：tokens ≥ 967K → 全量重写
      - L3：tokens ≥ 500K 或 距上次压缩 ≥ 6h → Micro-Compact

    Returns:
        dict: state 更新（含 messages / last_compact_ts）；无需压缩返回空 dict。
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    now = time.time()
    last_ts = state.get("last_compact_ts", 0)
    tokens = count_messages_tokens(messages)

    # ── L5：到全量重写触发线 ──
    if tokens >= L5_TRIGGER:
        logger.warning(f"[Context] tokens={tokens} ≥ L5 触发线 {L5_TRIGGER}，执行 Auto-Compact")
        new_msgs = auto_compact(messages)
        return {
            "messages": _build_replacement(messages, new_msgs),
            "last_compact_ts": now,
        }

    # ── L3：到 500K 或 闲置 6h ──
    idle_too_long = last_ts and (now - last_ts) >= L3_TIME_TRIGGER
    if tokens >= L3_TOKEN_TRIGGER or idle_too_long:
        reason = f"tokens={tokens}≥{L3_TOKEN_TRIGGER}" if tokens >= L3_TOKEN_TRIGGER else f"闲置 {int(now - last_ts)}s"
        logger.info(f"[Context] L3 触发（{reason}），执行 Micro-Compact")
        new_msgs = micro_compact(messages)
        return {
            "messages": _build_replacement(messages, new_msgs),
            "last_compact_ts": now,
        }

    return {}
