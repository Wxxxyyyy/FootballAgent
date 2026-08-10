# -*- coding: utf-8 -*-
"""
球队底蕴数据导入 Milvus 向量库
- 读取 data/team_profiles/ 下的球队介绍 JSON 文件
- 使用本地 bge-m3 模型生成向量（L2 归一化）
- 通过 pymilvus 写入 Milvus（HNSW + COSINE 索引）
- 供 RAG 检索使用

使用前确保 Milvus 已启动：
  docker compose up -d etcd minio milvus
然后运行：
  python pipeline/vector_loader.py
"""

import os
import json
import glob

from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

# ─── 路径配置 ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAM_PROFILES_DIR = os.path.join(PROJECT_ROOT, "data", "team_profiles")
BGE_M3_PATH = os.path.join(PROJECT_ROOT, "bge-m3")

# ─── Milvus 配置（与 vector_tools/config.py 保持一致） ────────
from agents.tools.vector_tools.config import (
    MILVUS_URI,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    METRIC_TYPE,
    INDEX_TYPE,
)

COLLECTION_DESC = "五大联赛球队历史底蕴简介向量库"
BATCH_SIZE = 50


def load_team_profiles() -> tuple[list[str], list[str], list[dict]]:
    """
    读取 data/team_profiles/ 下所有 JSON 文件，
    返回 (ids, documents, metadatas) 三元组。
    """
    ids = []
    documents = []
    metadatas = []

    json_files = sorted(glob.glob(os.path.join(TEAM_PROFILES_DIR, "*.json")))

    if not json_files:
        raise RuntimeError(f"未找到 JSON 文件: {TEAM_PROFILES_DIR}")

    for filepath in json_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            teams = json.load(f)

        print(f"  [读取] {filename:20s}  →  {len(teams):>3d} 支球队")

        for team in teams:
            club_name = team["ClubName"]
            league = team["League"]
            club_name_zh = team.get("ClubNameZh", "")
            alias_zh = team.get("AliasZh", "")
            intro_zh = team.get("IntroZh", "")

            if not intro_zh:
                continue

            # id: 联赛_球队名（唯一标识）
            doc_id = f"{league}_{club_name}".replace(" ", "_")

            ids.append(doc_id)
            documents.append(intro_zh)
            metadatas.append({
                "ClubName": club_name,
                "League": league,
                "ClubNameZh": club_name_zh,
                "AliasZh": alias_zh,
            })

    print(f"\n[合计] 共加载 {len(ids)} 支球队简介")
    return ids, documents, metadatas


def create_collection(client: MilvusClient):
    """创建 Milvus collection（幂等：已存在则删除重建）"""
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"[清理] 已删除旧 collection: {COLLECTION_NAME}")

    # 定义 schema
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, max_length=128, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field("ClubName", DataType.VARCHAR, max_length=128)
    schema.add_field("League", DataType.VARCHAR, max_length=64)
    schema.add_field("ClubNameZh", DataType.VARCHAR, max_length=128)
    schema.add_field("AliasZh", DataType.VARCHAR, max_length=256)
    schema.add_field("intro", DataType.VARCHAR, max_length=8192)

    # HNSW 索引（向量字段）
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type=INDEX_TYPE,
        metric_type=METRIC_TYPE,
        params={"M": 16, "efConstruction": 200},
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
        description=COLLECTION_DESC,
    )
    print(f"[✓] Collection '{COLLECTION_NAME}' 已创建（{INDEX_TYPE}/{METRIC_TYPE}）")


def import_to_milvus(client: MilvusClient, encoder, ids, documents, metadatas):
    """批量向量化并写入 Milvus"""
    total = len(ids)
    for i in range(0, total, BATCH_SIZE):
        end = min(i + BATCH_SIZE, total)
        batch_docs = documents[i:end]
        batch_metas = metadatas[i:end]
        batch_ids = ids[i:end]

        # bge-m3 向量化（归一化，COSINE = 内积）
        vectors = encoder.encode(batch_docs, normalize_embeddings=True).tolist()

        # 组装插入数据
        data = []
        for doc_id, vec, meta in zip(batch_ids, vectors, batch_metas):
            data.append({
                "id": doc_id,
                "vector": vec,
                "ClubName": meta["ClubName"],
                "League": meta["League"],
                "ClubNameZh": meta["ClubNameZh"],
                "AliasZh": meta["AliasZh"],
                "intro": documents[ids.index(doc_id)],
            })

        client.insert(collection_name=COLLECTION_NAME, data=data)
        print(f"  [写入] {end:>4d} / {total}")

    print(f"[✓] 全部写入完成: {total} 条文档")


def verify(client: MilvusClient, encoder):
    """验证写入结果：统计数量并做示例查询"""
    stats = client.get_collection_stats(COLLECTION_NAME)
    count = stats.get("row_count", 0)
    print(f"\n{'=' * 55}")
    print(f"  Collection '{COLLECTION_NAME}' 文档总数: {count}")
    print(f"{'=' * 55}")

    test_queries = [
        "哪支球队曾经联赛不败夺冠？",
        "意大利哪支球队有欧冠三连冠的历史？",
        "法国最成功的俱乐部是哪家？",
    ]

    print("\n[示例查询]")
    for query in test_queries:
        vec = encoder.encode(query, normalize_embeddings=True).tolist()
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[vec],
            limit=3,
            output_fields=["ClubName", "ClubNameZh", "League"],
            search_params={"metric_type": METRIC_TYPE, "params": {"ef": 64}},
        )
        print(f"\n  Q: {query}")
        hits = results[0] if results else []
        for j, hit in enumerate(hits):
            entity = hit.get("entity", hit)
            sim = hit.get("distance", 0.0)
            print(f"    Top-{j+1}: {entity.get('ClubNameZh')}({entity.get('ClubName')}) "
                  f"[{entity.get('League')}]  相似度={sim:.4f}")


def main():
    print("=" * 55)
    print("  Football Agent · Milvus 向量数据库导入工具")
    print("=" * 55)

    # 1. 连接 Milvus
    print(f"\n[1/4] 连接 Milvus: {MILVUS_URI}")
    client = MilvusClient(uri=MILVUS_URI)
    print("[✓] Milvus 连接成功")

    # 2. 加载 bge-m3 嵌入模型
    print(f"\n[2/4] 加载嵌入模型: {BGE_M3_PATH}")
    encoder = SentenceTransformer(BGE_M3_PATH, trust_remote_code=True)
    print("[✓] bge-m3 模型加载完成")

    # 3. 读取 JSON 并写入
    print(f"\n[3/4] 读取球队简介 → 创建 Collection → 写入向量")
    ids, documents, metadatas = load_team_profiles()
    create_collection(client)
    import_to_milvus(client, encoder, ids, documents, metadatas)

    # 4. 验证
    print(f"\n[4/4] 验证写入结果")
    verify(client, encoder)

    print(f"\n[✓] 全部完成！向量数据已写入 Milvus: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
