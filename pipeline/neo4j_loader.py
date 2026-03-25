# -*- coding: utf-8 -*-
"""
数据导入 Neo4j
- 读取 data/processed 下所有联赛 CSV 文件
- 构建 Team 节点（属性: name, league）
- 创建 PLAYED_AGAINST 有向关系（HomeTeam → AwayTeam）
- 关系属性全部拼接为包含球队名称的直白文本，方便 LLM Agent 精准读取
- 使用 MERGE 防止重复导入
"""

import os
import glob
import math
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ─── 路径 ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

# ─── 联赛映射（文件名前缀 → 联赛标签）────────────────────────
LEAGUE_MAP = {
    "England": "England",
    "Spain": "Spain",
    "Italy": "Italy",
    "Germany": "Germany",
    "France": "France",
}


# ═══════════════════════════════════════════════════════════════
#  1. 读取并整合 CSV
# ═══════════════════════════════════════════════════════════════

def load_all_csv() -> pd.DataFrame:
    """读取 data/processed 下全部联赛 CSV，拼合为一张 DataFrame"""
    all_frames = []
    csv_pattern = os.path.join(PROCESSED_DIR, "*.csv")

    for filepath in sorted(glob.glob(csv_pattern)):
        filename = os.path.basename(filepath)
        if filename in ("ClubName.csv", ".gitkeep"):
            continue

        parts = filename.replace(".csv", "").split("_", 1)
        if len(parts) != 2:
            continue
        league_key, season = parts[0], parts[1]
        if league_key not in LEAGUE_MAP:
            continue

        df = pd.read_csv(filepath)
        df["league"] = LEAGUE_MAP[league_key]
        df["season"] = season
        all_frames.append(df)
        print(f"  [读取] {filename:40s}  →  {len(df):>4d} 条记录")

    if not all_frames:
        raise RuntimeError("未找到任何联赛 CSV 文件")

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\n[合计] 共读取 {len(combined)} 条比赛记录")
    return combined


# ═══════════════════════════════════════════════════════════════
#  2. 数据清洗 & 字段拼接
# ═══════════════════════════════════════════════════════════════

def safe_float(val, default="N/A"):
    """安全地将值转为保留两位小数的字符串，NaN 返回默认值"""
    try:
        f = float(val)
        if math.isnan(f):
            return default
        return f"{f:.2f}"
    except (ValueError, TypeError):
        return default


def build_match_records(df: pd.DataFrame) -> list[dict]:
    """
    将 DataFrame 逐行转换为 Neo4j 需要的结构化字典列表。
    所有文本字段都显式包含球队名称，绝不使用 "主胜/客胜" 等模糊字眼。
    """
    records = []
    for _, row in df.iterrows():
        home = str(row["HomeTeam"]).strip()
        away = str(row["AwayTeam"]).strip()

        # --- 比分 ---
        try:
            fthg = int(row["FTHG"])
            ftag = int(row["FTAG"])
        except (ValueError, TypeError):
            # 比分缺失的行跳过
            continue

        match_result = f"{home} {fthg}:{ftag} {away}"
        total_goals = fthg + ftag

        # --- 胜平负赔率（Bet365 初盘）---
        h_odds = safe_float(row.get("B365H"))
        d_odds = safe_float(row.get("B365D"))
        a_odds = safe_float(row.get("B365A"))
        odds_info = (
            f"{home} 胜赔率: {h_odds} | "
            f"平局赔率: {d_odds} | "
            f"{away} 胜赔率: {a_odds}"
        )

        # --- 大小球赔率（Bet365 初盘 2.5球）---
        over_val = safe_float(row.get("B365>2.5"))
        under_val = safe_float(row.get("B365<2.5"))
        over_under_odds = f"大球赔率: {over_val} | 小球赔率: {under_val}"

        # --- 终盘赔率（Bet365 终盘）---
        ch_odds = safe_float(row.get("B365CH"))
        cd_odds = safe_float(row.get("B365CD"))
        ca_odds = safe_float(row.get("B365CA"))
        closing_odds_info = (
            f"{home} 终盘胜赔率: {ch_odds} | "
            f"终盘平局赔率: {cd_odds} | "
            f"{away} 终盘胜赔率: {ca_odds}"
        )

        # --- 终盘大小球赔率 ---
        c_over_val = safe_float(row.get("B365C>2.5"))
        c_under_val = safe_float(row.get("B365C<2.5"))
        closing_over_under = f"终盘大球赔率: {c_over_val} | 终盘小球赔率: {c_under_val}"

        # --- 亚盘 ---
        ahh = safe_float(row.get("AHh"))
        ah_home = safe_float(row.get("B365AHH"))
        ah_away = safe_float(row.get("B365AHA"))
        asian_handicap_info = (
            f"让球盘口: {ahh} | "
            f"{home} 亚盘赔率: {ah_home} | "
            f"{away} 亚盘赔率: {ah_away}"
        )

        records.append({
            "home_team": home,
            "away_team": away,
            "league": str(row["league"]),
            "season": str(row["season"]),
            "match_date": str(row["Date"]),
            "match_result": match_result,
            "total_goals": total_goals,
            "odds_info": odds_info,
            "over_under_odds": over_under_odds,
            "closing_odds_info": closing_odds_info,
            "closing_over_under": closing_over_under,
            "asian_handicap_info": asian_handicap_info,
        })

    print(f"[构建] 共生成 {len(records)} 条关系记录")
    return records


# ═══════════════════════════════════════════════════════════════
#  3. Neo4j 写入
# ═══════════════════════════════════════════════════════════════

# --- Cypher 语句 ---
CYPHER_CONSTRAINT_TEAM = (
    "CREATE CONSTRAINT IF NOT EXISTS "
    "FOR (t:Team) REQUIRE (t.name, t.league) IS UNIQUE"
)

CYPHER_MERGE_MATCH = """
MERGE (home:Team {name: $home_team, league: $league})
MERGE (away:Team {name: $away_team, league: $league})
MERGE (home)-[r:PLAYED_AGAINST {
    match_date: $match_date,
    season: $season,
    match_result: $match_result
}]->(away)
SET r.total_goals      = $total_goals,
    r.odds_info        = $odds_info,
    r.over_under_odds  = $over_under_odds,
    r.closing_odds_info   = $closing_odds_info,
    r.closing_over_under  = $closing_over_under,
    r.asian_handicap_info = $asian_handicap_info,
    r.league           = $league,
    r.season           = $season
"""


class Neo4jLoader:
    """Neo4j 数据导入器"""

    def __init__(self):
        load_dotenv(ENV_PATH)
        uri = os.getenv("NEO4J_URL", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        database = os.getenv("NEO4J_DATABASE", "neo4j")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        print(f"[连接] Neo4j @ {uri}  数据库: {database}")

    def close(self):
        self.driver.close()

    def create_constraints(self):
        """创建唯一性约束，确保 Team 节点不重复"""
        with self.driver.session(database=self.database) as session:
            session.run(CYPHER_CONSTRAINT_TEAM)
        print("[✓] 唯一性约束已创建 (Team: name + league)")

    def clear_all(self):
        """清空图数据库中所有 Team 节点和 PLAYED_AGAINST 关系（可选）"""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (t:Team)-[r:PLAYED_AGAINST]->() "
                "DELETE r RETURN count(r) AS deleted_rels"
            )
            rels = result.single()["deleted_rels"]
            result2 = session.run(
                "MATCH (t:Team) DELETE t RETURN count(t) AS deleted_nodes"
            )
            nodes = result2.single()["deleted_nodes"]
        print(f"[清空] 删除 {rels} 条关系, {nodes} 个节点")

    def import_batch(self, records: list[dict], batch_size: int = 500):
        """分批写入比赛数据"""
        total = len(records)
        imported = 0

        with self.driver.session(database=self.database) as session:
            for i in range(0, total, batch_size):
                batch = records[i: i + batch_size]

                # 在事务中批量执行
                tx = session.begin_transaction()
                try:
                    for rec in batch:
                        tx.run(CYPHER_MERGE_MATCH, **rec)
                    tx.commit()
                except Exception as e:
                    tx.rollback()
                    raise e

                imported += len(batch)
                pct = imported / total * 100
                print(f"  [写入] {imported:>6d} / {total}  ({pct:5.1f}%)")

        print(f"[✓] 全部写入完成: {imported} 条关系")

    def verify(self):
        """验证写入结果"""
        with self.driver.session(database=self.database) as session:
            # 统计节点
            r1 = session.run("MATCH (t:Team) RETURN count(t) AS cnt")
            team_count = r1.single()["cnt"]

            # 按联赛统计关系
            r2 = session.run(
                "MATCH ()-[r:PLAYED_AGAINST]->() "
                "RETURN r.league AS league, r.season AS season, count(r) AS cnt "
                "ORDER BY league, season"
            )
            rows = r2.data()

        print(f"\n{'=' * 55}")
        print(f"  Team 节点总数: {team_count}")
        print(f"{'=' * 55}")
        print(f"{'联赛':<12}{'赛季':<15}{'比赛数':>8}")
        print(f"{'-' * 55}")
        total = 0
        for row in rows:
            print(f"{row['league']:<12}{row['season']:<15}{row['cnt']:>8d}")
            total += row["cnt"]
        print(f"{'-' * 55}")
        print(f"{'总计':<27}{total:>8d}")
        print(f"{'=' * 55}")


# ═══════════════════════════════════════════════════════════════
#  4. 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  Football Agent · Neo4j 数据导入工具")
    print("=" * 55)

    # 1. 读取 CSV
    print(f"\n[1/4] 读取 CSV 文件")
    df = load_all_csv()

    # 2. 构建记录
    print(f"\n[2/4] 构建关系数据")
    records = build_match_records(df)

    # 3. 连接 Neo4j 并写入
    print(f"\n[3/4] 写入 Neo4j")
    loader = Neo4jLoader()
    try:
        loader.create_constraints()
        loader.clear_all()
        loader.import_batch(records, batch_size=500)

        # 4. 验证
        print(f"\n[4/4] 验证写入结果")
        loader.verify()
    finally:
        loader.close()

    print("\n[✓] 全部完成！数据已导入 Neo4j 图数据库")


if __name__ == "__main__":
    main()
