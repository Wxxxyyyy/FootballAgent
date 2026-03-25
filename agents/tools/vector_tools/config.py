# -*- coding: utf-8 -*-
"""
向量知识检索工具 · 配置文件
═══════════════════════════
存放本地路径配置及两道防线阈值。
"""

import os

# ─── 项目根目录 ───────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))

# ─── ChromaDB 持久化路径 ──────────────────────────────────────
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")

# ─── Embedding 模型本地路径 ───────────────────────────────────
BGE_M3_MODEL_PATH = os.path.join(PROJECT_ROOT, "bge-m3")

# ─── Collection 名称（必须与 vector_loader.py 写入时一致） ────
COLLECTION_NAME = "team_profiles"

# ─── 防线1: 距离阈值拦截（Chroma 默认 L2 距离） ──────────────
#     大于此值的结果视为"无关"，直接丢弃，防止向量检索强行匹配
DISTANCE_THRESHOLD = 0.6

# ─── 防线2: 强制 Top-K 上限（防 Token 爆炸） ─────────────────
#     每次检索最多返回 K 条结果，避免将过多 chunk 塞入 Prompt
MAX_RESULTS = 3

