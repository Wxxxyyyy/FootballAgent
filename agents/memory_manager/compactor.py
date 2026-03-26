# -*- coding: utf-8 -*-
"""
记忆压缩器 (Compactor)
═══════════════════════
当 messages 超过窗口上限时执行：
  1. Memory Flush —— 提取关键信息（实体、事实、偏好）
  2. Compaction   —— 生成摘要（严格保留关键信息）
  3. 向量化存储   —— Embedding 后存入 ChromaDB conversation_memory
  4. 裁剪 messages —— 移除已压缩的旧消息
"""

import json
from typing import Sequence
from datetime import datetime

from langchain_core.messages import BaseMessage

from common.llm_select import llm_call, LLM_MODEL_QWEN_SIMPLE_NAME
from agents.memory_manager.prompts import get_flush_prompt, get_compaction_prompt
from agents.tools.vector_tools.config import (
    MEMORY_COLLECTION,
    CHROMA_DB_PATH,
    BGE_M3_MODEL_PATH,
)

# ═══════════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════════

WINDOW_MAX = 20
COMPRESS_BATCH = 10

# ═══════════════════════════════════════════════════════════════
#  ChromaDB 记忆库单例
# ═══════════════════════════════════════════════════════════════

_memory_client = None
_memory_collection = None


def _get_memory_collection():
    """获取或创建 conversation_memory Collection（单例）"""
    global _memory_client, _memory_collection
    if _memory_collection is not None:
        return _memory_collection

    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    _memory_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name=BGE_M3_MODEL_PATH,
    )
    _memory_collection = _memory_client.get_or_create_collection(
        name=MEMORY_COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"[MemoryManager] ChromaDB collection '{MEMORY_COLLECTION}' ready, "
          f"count={_memory_collection.count()}")
    return _memory_collection


# ═══════════════════════════════════════════════════════════════
#  核心函数
# ═══════════════════════════════════════════════════════════════

def _format_messages(messages: Sequence[BaseMessage]) -> str:
    """将消息列表格式化为纯文本"""
    lines = []
    for msg in messages:
        role = "用户" if getattr(msg, "type", "") == "human" else "助手"
        content = msg.content if hasattr(msg, "content") else str(msg)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _flush_key_info(messages_text: str) -> dict:
    """
    Memory Flush —— 调用 LLM 提取关键信息。
    失败时返回空结构，不阻断压缩流程。
    """
    prompt = get_flush_prompt(messages_text)
    try:
        response = llm_call(prompt, model=LLM_MODEL_QWEN_SIMPLE_NAME, force_fallback=True)
        raw = response.content.strip()
        if "```" in raw:
            import re
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', raw, re.DOTALL)
            if match:
                raw = match.group(1).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[MemoryManager] Flush 提取失败: {e}，使用空结构")
        return {"entities": [], "key_facts": [], "user_preferences": [], "decisions": []}


def _generate_summary(messages_text: str, flush_result: dict) -> str:
    """Compaction —— 调用 LLM 生成摘要"""
    flush_json = json.dumps(flush_result, ensure_ascii=False, indent=2)
    prompt = get_compaction_prompt(messages_text, flush_json)
    try:
        response = llm_call(prompt, model=LLM_MODEL_QWEN_SIMPLE_NAME, force_fallback=True)
        return response.content.strip()
    except Exception as e:
        print(f"[MemoryManager] 摘要生成失败: {e}，使用原文截断兜底")
        return messages_text[:200] + "..."


def _store_to_chromadb(
    summary: str,
    flush_result: dict,
    thread_id: str,
    turn_range: str,
) -> None:
    """将摘要 Embedding 后存入 ChromaDB"""
    collection = _get_memory_collection()

    doc_id = f"{thread_id}_turns_{turn_range}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    metadata = {
        "thread_id": thread_id,
        "turn_range": turn_range,
        "timestamp": datetime.now().isoformat(),
        "entities": json.dumps(flush_result.get("entities", []), ensure_ascii=False),
        "user_preferences": json.dumps(flush_result.get("user_preferences", []), ensure_ascii=False),
    }

    collection.add(
        documents=[summary],
        metadatas=[metadata],
        ids=[doc_id],
    )
    print(f"[MemoryManager] 摘要已存入 ChromaDB: {doc_id} (collection count={collection.count()})")


# ═══════════════════════════════════════════════════════════════
#  对外接口
# ═══════════════════════════════════════════════════════════════

def should_compact(messages: Sequence[BaseMessage]) -> bool:
    """判断是否需要触发压缩"""
    return len(messages) > WINDOW_MAX


def compact(
    messages: Sequence[BaseMessage],
    thread_id: str = "unknown",
) -> tuple[list[BaseMessage], dict]:
    """
    执行完整的 Compaction 流程。

    Args:
        messages:  当前完整的消息列表
        thread_id: 当前会话 ID

    Returns:
        (裁剪后的消息列表, flush 提取的关键信息 dict)
    """
    total = len(messages)
    old_messages = list(messages[:COMPRESS_BATCH])
    keep_messages = list(messages[COMPRESS_BATCH:])

    turn_range = f"1-{COMPRESS_BATCH}"
    print(f"[MemoryManager] 触发 Compaction: 总消息 {total} 条，"
          f"压缩前 {COMPRESS_BATCH} 条，保留后 {len(keep_messages)} 条")

    messages_text = _format_messages(old_messages)

    print("[MemoryManager] Step 1/3: Memory Flush 提取关键信息...")
    flush_result = _flush_key_info(messages_text)
    print(f"  entities: {flush_result.get('entities', [])}")
    print(f"  key_facts: {flush_result.get('key_facts', [])}")

    print("[MemoryManager] Step 2/3: 生成摘要...")
    summary = _generate_summary(messages_text, flush_result)
    print(f"  摘要长度: {len(summary)} 字符")

    print("[MemoryManager] Step 3/3: 向量化存入 ChromaDB...")
    _store_to_chromadb(summary, flush_result, thread_id, turn_range)

    return keep_messages, flush_result
