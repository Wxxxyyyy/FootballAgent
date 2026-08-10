# -*- coding: utf-8 -*-
"""
向量知识检索工具 · 配置文件
═══════════════════════════
存放 Milvus 连接配置及两道防线阈值。

RAG 知识库（球队简介）使用 Milvus 管理；
对话长期记忆（conversation_memory）走 md 方案，不在本配置内。
"""

import os

# ─── 项目根目录 ───────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))

# ─── Milvus 连接配置 ─────────────────────────────────────────
# 容器间走服务名 football_milvus:19530；宿主机调试走 127.0.0.1:19530
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_URI = os.getenv("MILVUS_URI", f"http://{MILVUS_HOST}:{MILVUS_PORT}")

# ─── Embedding 模型本地路径 ───────────────────────────────────
BGE_M3_MODEL_PATH = os.path.join(PROJECT_ROOT, "bge-m3")

# bge-m3 稠密向量维度
EMBEDDING_DIM = 1024

# ─── Collection 名称（必须与 vector_loader.py 写入时一致） ────
COLLECTION_NAME = "team_profiles"

# ─── 防线1: 相似度阈值拦截 ────────────────────────────────────
#     Milvus 用 COSINE 相似度（越大越相关，范围 [-1, 1]）。
#     低于此值的结果视为"无关"，直接丢弃。
#     注：原 ChromaDB 用 L2 距离 0.6（越小越相关）；
#         对 L2 归一化向量 L2²=2(1-cos)，L2=0.6 ≈ cos=0.82。
#         迁库后建议重跑 100 条评测重新校准该阈值。
SIM_THRESHOLD = 0.82

# ─── 防线2: 强制 Top-K 上限（防 Token 爆炸） ─────────────────
#     每次检索最多返回 K 条结果，避免将过多 chunk 塞入 Prompt
MAX_RESULTS = 3

# 检索度量类型
METRIC_TYPE = "COSINE"
INDEX_TYPE = "HNSW"
