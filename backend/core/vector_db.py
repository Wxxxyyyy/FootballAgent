# -*- coding: utf-8 -*-
"""
向量数据库连接管理（Milvus）

RAG 知识库（球队简介）使用 Milvus 管理。
上层调用方通过 get_client / get_collection 获取连接，无需感知底层细节。

Milvus 依赖 etcd + minio，见 docker-compose.yml。
"""

from typing import Optional

from pymilvus import MilvusClient

from agents.tools.vector_tools.config import (
    MILVUS_URI,
    COLLECTION_NAME,
)

_client: Optional[MilvusClient] = None


def get_client() -> MilvusClient:
    """获取 Milvus 客户端（单例）"""
    global _client
    if _client is not None:
        return _client
    _client = MilvusClient(uri=MILVUS_URI)
    print(f"[vector_db] Milvus 已连接: {MILVUS_URI}")
    return _client


def get_collection(name: str = COLLECTION_NAME) -> str:
    """返回 collection 名称（Milvus 用名称操作，无需返回对象）"""
    client = get_client()
    if not client.has_collection(name):
        raise RuntimeError(
            f"Milvus collection '{name}' 不存在，请先运行 pipeline/vector_loader.py"
        )
    return name


def health_check() -> dict:
    """健康检查：返回 collection 数量和连通性"""
    try:
        client = get_client()
        collections = client.list_collections()
        return {
            "status": "ok",
            "uri": MILVUS_URI,
            "collections": len(collections),
            "collection_names": collections,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def close():
    """释放客户端"""
    global _client
    _client = None
