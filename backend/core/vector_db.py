# -*- coding: utf-8 -*-
"""
向量数据库连接管理

当前使用 ChromaDB 本地持久化模式。
后续如需扩展到分布式（Milvus），在此模块内切换实现即可，
上层调用方无需感知底层变化。
"""

import os
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

_client: Optional[chromadb.ClientAPI] = None

# 默认路径：项目根目录 data/chroma_db
_DEFAULT_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "chroma_db",
)


def get_chroma_client(persist_dir: str = None) -> chromadb.ClientAPI:
    """
    获取 ChromaDB 客户端（单例）

    Args:
        persist_dir: 持久化目录路径，默认为 data/chroma_db
    """
    global _client
    if _client is not None:
        return _client

    path = persist_dir or _DEFAULT_PERSIST_DIR
    os.makedirs(path, exist_ok=True)

    _client = chromadb.PersistentClient(
        path=path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    print(f"[vector_db] ChromaDB 已连接: {path}")
    return _client


def get_collection(name: str = "team_profiles") -> chromadb.Collection:
    """获取指定 Collection"""
    client = get_chroma_client()
    return client.get_or_create_collection(name=name)


def health_check() -> dict:
    """健康检查：返回集合数量和心跳"""
    try:
        client = get_chroma_client()
        heartbeat = client.heartbeat()
        collections = client.list_collections()
        return {
            "status": "ok",
            "heartbeat": heartbeat,
            "collections": len(collections),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def close_chroma():
    """释放客户端（PersistentClient 无需显式关闭，但保持接口统一）"""
    global _client
    _client = None
