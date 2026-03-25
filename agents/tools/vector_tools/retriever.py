# -*- coding: utf-8 -*-
"""
向量知识检索工具 · 检索器 + 两道防线
═════════════════════════════════════
防线1 (距离阈值拦截) : 丢弃 L2 距离 > DISTANCE_THRESHOLD 的结果
防线2 (强制 Top-K)   : 检索时 n_results 固定为 MAX_RESULTS，防 Token 爆炸

流程:
  1. 连接 ChromaDB (PersistentClient) 并获取 collection
  2. 使用 bge-m3 Embedding 对 query 向量化
  3. 执行 collection.query 获取 Top-K 候选
  4. 防线1: 遍历 distances，丢弃高于阈值的结果
  5. 返回过滤后的有效结果列表
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from agents.tools.vector_tools.config import (
    CHROMA_DB_PATH,
    BGE_M3_MODEL_PATH,
    COLLECTION_NAME,
    DISTANCE_THRESHOLD,
    MAX_RESULTS,
)

# ═══════════════════════════════════════════════════════════════
#  模块级单例（避免重复加载模型和数据库连接）
# ═══════════════════════════════════════════════════════════════

_embedding_fn = None
_collection = None


def _get_embedding_fn():
    """获取 bge-m3 Embedding Function 单例"""
    global _embedding_fn
    if _embedding_fn is None:
        print(f"[Vector] 加载 Embedding 模型: {BGE_M3_MODEL_PATH}")
        _embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=BGE_M3_MODEL_PATH,
            trust_remote_code=True,
        )
        print("[Vector] Embedding 模型加载完成")
    return _embedding_fn


def _get_collection():
    """获取 ChromaDB Collection 单例"""
    global _collection
    if _collection is None:
        print(f"[Vector] 连接 ChromaDB: {CHROMA_DB_PATH}")
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        embedding_fn = _get_embedding_fn()
        _collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
        doc_count = _collection.count()
        print(f"[Vector] Collection '{COLLECTION_NAME}' 已就绪，文档数: {doc_count}")
    return _collection


# ═══════════════════════════════════════════════════════════════
#  核心检索函数（含两道防线）
# ═══════════════════════════════════════════════════════════════

def search_team_profiles(query: str) -> list[dict]:
    """
    在球队简介向量库中检索与 query 最相关的文档。

    两道防线:
      防线1: 距离阈值拦截 — 丢弃 L2 距离 > DISTANCE_THRESHOLD 的结果
      防线2: 强制 Top-K    — n_results 固定为 MAX_RESULTS（在 config 中配置）

    Args:
        query: 用户的自然语言查询问题

    Returns:
        list[dict]: 通过阈值过滤后的有效结果列表，每个 dict 包含:
            - club_name:    球队英文名
            - club_name_zh: 球队中文名
            - alias_zh:     中文别名
            - league:       所属联赛
            - intro:        球队简介文本
            - distance:     与 query 的 L2 距离（越小越相关）
    """
    collection = _get_collection()

    # ── 防线2: 强制 Top-K，n_results 不超过 MAX_RESULTS ──
    print(f"[Vector] 检索 query: '{query[:50]}...' (Top-{MAX_RESULTS})")
    results = collection.query(
        query_texts=[query],
        n_results=MAX_RESULTS,
    )

    # Chroma 返回格式: results["ids"][0], results["documents"][0],
    #                   results["metadatas"][0], results["distances"][0]
    ids_list = results.get("ids", [[]])[0]
    docs_list = results.get("documents", [[]])[0]
    metas_list = results.get("metadatas", [[]])[0]
    dists_list = results.get("distances", [[]])[0]

    # ── 防线1: 距离阈值拦截 ──
    filtered = []
    for doc_id, doc, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
        if dist > DISTANCE_THRESHOLD:
            print(f"[Vector] 防线1拦截: {meta.get('ClubNameZh', doc_id)} "
                  f"(距离={dist:.4f} > 阈值={DISTANCE_THRESHOLD})")
            continue

        filtered.append({
            "club_name": meta.get("ClubName", ""),
            "club_name_zh": meta.get("ClubNameZh", ""),
            "alias_zh": meta.get("AliasZh", ""),
            "league": meta.get("League", ""),
            "intro": doc,
            "distance": round(dist, 4),
        })

    print(f"[Vector] 检索结果: {len(filtered)}/{len(ids_list)} 条通过阈值过滤")
    return filtered

