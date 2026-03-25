# -*- coding: utf-8 -*-
"""
球队底蕴数据导入向量库
- 读取 data/team_profiles/ 下的球队介绍 JSON 文件
- 使用本地 bge-m3 模型生成向量
- 通过 ChromaDB PersistentClient 持久化到本地文件夹
- 供 RAG 检索使用
"""

import os
import json
import glob
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ─── 路径配置 ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAM_PROFILES_DIR = os.path.join(PROJECT_ROOT, "data", "team_profiles")
CHROMA_PERSIST_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_db")
BGE_M3_PATH = os.path.join(PROJECT_ROOT, "bge-m3")

# ─── 集合名称 ─────────────────────────────────────────────────
COLLECTION_NAME = "team_profiles"


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


def create_collection(embedding_fn):
    """创建或获取 ChromaDB collection"""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # 如果 collection 已存在则先删除再重建，保证幂等
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"[清理] 已删除旧 collection: {COLLECTION_NAME}")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"description": "五大联赛球队历史底蕴简介向量库"},
    )
    print(f"[✓] Collection '{COLLECTION_NAME}' 已创建")
    return client, collection


def import_to_chroma(collection, ids, documents, metadatas):
    """将数据批量写入 ChromaDB"""
    batch_size = 50
    total = len(ids)

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )
        print(f"  [写入] {end:>4d} / {total}")

    print(f"[✓] 全部写入完成: {total} 条文档")


def verify(collection):
    """验证写入结果：统计数量并做示例查询"""
    count = collection.count()
    print(f"\n{'=' * 55}")
    print(f"  Collection '{COLLECTION_NAME}' 文档总数: {count}")
    print(f"{'=' * 55}")

    # 示例：用中文查询测试语义检索
    test_queries = [
        "哪支球队曾经联赛不败夺冠？",
        "意大利哪支球队有欧冠三连冠的历史？",
        "法国最成功的俱乐部是哪家？",
    ]

    print("\n[示例查询]")
    for query in test_queries:
        results = collection.query(query_texts=[query], n_results=3)
        print(f"\n  Q: {query}")
        for j, (doc_id, meta, dist) in enumerate(zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            print(f"    Top-{j+1}: {meta['ClubNameZh']}({meta['ClubName']}) "
                  f"[{meta['League']}]  距离={dist:.4f}")


def main():
    print("=" * 55)
    print("  Football Agent · 向量数据库导入工具")
    print("=" * 55)

    # 1. 加载 bge-m3 嵌入模型
    print(f"\n[1/4] 加载嵌入模型: {BGE_M3_PATH}")
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=BGE_M3_PATH,
        trust_remote_code=True,
    )
    print("[✓] bge-m3 模型加载完成")

    # 2. 读取 JSON 文件
    print(f"\n[2/4] 读取球队简介文件")
    ids, documents, metadatas = load_team_profiles()

    # 3. 创建 Collection 并写入
    print(f"\n[3/4] 创建 ChromaDB Collection & 写入向量")
    client, collection = create_collection(embedding_fn)
    import_to_chroma(collection, ids, documents, metadatas)

    # 4. 验证
    print(f"\n[4/4] 验证写入结果")
    verify(collection)

    print(f"\n[✓] 全部完成！向量数据已持久化到: {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()
