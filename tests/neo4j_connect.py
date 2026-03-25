import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# 加载 .env 文件中的环境变量
load_dotenv()

def test_connection():
    print("⏳ 正在尝试连接 Neo4j...")
    try:
        uri = os.getenv("NEO4J_URL", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        database = os.getenv("NEO4J_DATABASE", "neo4j")

        driver = GraphDatabase.driver(uri, auth=(user, password))

        # 验证连通性
        driver.verify_connectivity()
        print("✅ 成功连接到 Neo4j 数据库！\n")

        with driver.session(database=database) as session:
            # 查询版本
            result = session.run("CALL dbms.components() YIELD name, versions RETURN name, versions[0] AS version")
            record = result.single()
            print(f"🖥️  Neo4j 版本: {record['name']} {record['version']}")

            # 查询 Team 节点数
            result = session.run("MATCH (t:Team) RETURN count(t) AS count")
            team_count = result.single()["count"]
            print(f"📊 Team 节点数量: {team_count}")

            # 查询 PLAYED_AGAINST 关系数
            result = session.run("MATCH ()-[r:PLAYED_AGAINST]->() RETURN count(r) AS count")
            rel_count = result.single()["count"]
            print(f"📊 PLAYED_AGAINST 关系数量: {rel_count}")

        driver.close()
        print("\n🎉 测试完美通过，Neo4j 随时待命！")

    except Exception as e:
        print(f"\n❌ 连接失败: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_connection()
