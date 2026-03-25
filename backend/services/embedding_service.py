# -*- coding: utf-8 -*-
"""
向量嵌入服务
- 使用 SentenceTransformer 加载 bge-m3
- 单条与批量编码，模型首次调用时再加载
"""
from __future__ import annotations

import os
from threading import Lock
from typing import Sequence

import numpy as np


class EmbeddingService:
    """封装 BAAI/bge-m3，提供文本向量。"""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-m3"
        )
        self._model = None
        self._lock = Lock()

    def _ensure_model(self):
        with self._lock:
            if self._model is not None:
                return self._model
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """单条文本 -> 向量（numpy）。"""
        m = self._ensure_model()
        vec = m.encode(text, normalize_embeddings=True)
        return np.asarray(vec)

    def embed_batch(
        self, texts: Sequence[str], batch_size: int = 32
    ) -> np.ndarray:
        """批量编码，适合检索库构建。"""
        m = self._ensure_model()
        return np.asarray(
            m.encode(
                list(texts),
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
