# -*- coding: utf-8 -*-
"""
向量知识检索工具 · 检索器（Milvus）+ 两道防线
═══════════════════════════════════════════════════════════════
防线1 (相似度阈值拦截) : 丢弃 COSINE 相似度 < SIM_THRESHOLD 的结果
防线2 (强制 Top-K)   : 检索 limit 固定为 MAX_RESULTS，防 Token 爆炸

流程:
  1. 连接 Milvus 并加载 collection
  2. 使用 bge-m3 对 query 向量化（归一化，COSINE = 内积）
  3. 执行 search 获取 Top-K 候选
  4. 防线1: 遍历相似度，丢弃低于阈值的结果
  5. 返回过滤后的有效结果列表
"""

import logging

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

from agents.tools.vector_tools.config import (
    MILVUS_URI,
    BGE_M3_MODEL_PATH,
    COLLECTION_NAME,
    SIM_THRESHOLD,
    MAX_RESULTS,
    METRIC_TYPE,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  模块级单例（避免重复加载模型和数据库连接）
# ═══════════════════════════════════════════════════════════════

_client = None
_encoder = None


def _get_client() -> MilvusClient:
    """获取 Milvus 客户端单例"""
    global _client
    if _client is None:
        logger.info(f"[Vector] 连接 Milvus: {MILVUS_URI}")
        _client = MilvusClient(uri=MILVUS_URI)
        # 确认 collection 存在
        if not _client.has_collection(COLLECTION_NAME):
            raise RuntimeError(
                f"Milvus collection '{COLLECTION_NAME}' 不存在，"
                f"请先运行 pipeline/vector_loader.py 灌入数据"
            )
        _client.load_collection(COLLECTION_NAME)
        logger.info(f"[Vector] Collection '{COLLECTION_NAME}' 已加载")
    return _client


def _get_encoder() -> SentenceTransformer:
    """获取 bge-m3 编码器单例"""
    global _encoder
    if _encoder is None:
        logger.info(f"[Vector] 加载 Embedding 模型: {BGE_M3_MODEL_PATH}")
        _encoder = SentenceTransformer(BGE_M3_MODEL_PATH, trust_remote_code=True)
        logger.info("[Vector] Embedding 模型加载完成")
    return _encoder


def _embed(query: str) -> list:
    """对查询文本向量化（L2 归一化，COSINE 等价于内积）"""
    vec = _get_encoder().encode(query, normalize_embeddings=True)
    return vec.tolist()


# ═══════════════════════════════════════════════════════════════
#  核心检索函数（含两道防线）
# ═══════════════════════════════════════════════════════════════

def search_team_profiles(query: str) -> list[dict]:
    """
    在球队简介向量库中检索与 query 最相关的文档。

    两道防线:
      防线1: 相似度阈值拦截 — 丢弃 COSINE 相似度 < SIM_THRESHOLD 的结果
      防线2: 强制 Top-K    — limit 固定为 MAX_RESULTS（在 config 中配置）

    Args:
        query: 用户的自然语言查询问题

    Returns:
        list[dict]: 通过阈值过滤后的有效结果列表，每个 dict 包含:
            - club_name:    球队英文名
            - club_name_zh: 球队中文名
            - alias_zh:     中文别名
            - league:       所属联赛
            - intro:        球队简介文本
            - distance:     与 query 的 COSINE 相似度（越大越相关）
    """
    client = _get_client()
    query_vec = _embed(query)

    # ── 防线2: 强制 Top-K ──
    logger.info(f"[Vector] 检索 query: '{query[:50]}...' (Top-{MAX_RESULTS})")
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        limit=MAX_RESULTS,
        output_fields=["ClubName", "League", "ClubNameZh", "AliasZh", "intro"],
        search_params={"metric_type": METRIC_TYPE, "params": {"ef": 64}},
    )

    # MilvusClient.search 返回 list[list[hit]]，取第一条 query 的结果
    hits = results[0] if results else []

    # ── 防线1: 相似度阈值拦截（COSINE：低于阈值丢弃）──
    filtered = []
    for hit in hits:
        sim = hit.get("distance", 0.0)
        entity = hit.get("entity", hit)
        if sim < SIM_THRESHOLD:
            logger.info(
                f"[Vector] 防线1拦截: {entity.get('ClubNameZh', '?')} "
                f"(相似度={sim:.4f} < 阈值={SIM_THRESHOLD})"
            )
            continue

        filtered.append({
            "club_name": entity.get("ClubName", ""),
            "club_name_zh": entity.get("ClubNameZh", ""),
            "alias_zh": entity.get("AliasZh", ""),
            "league": entity.get("League", ""),
            "intro": entity.get("intro", ""),
            "distance": round(sim, 4),
        })

    logger.info(f"[Vector] 检索结果: {len(filtered)}/{len(hits)} 条通过阈值过滤")
    return filtered
