# -*- coding: utf-8 -*-
"""
OpenClaw 每日比赛数据增量入库

功能:
  1. 解析 OpenClaw 推送的 JSON 数据
  2. 转换为与 match_master 表完全一致的 DataFrame 格式
  3. 去重检查（Date + HomeTeam + AwayTeam 唯一标识）
  4. 增量追加写入 MySQL match_master 表
  5. 增量写入 Neo4j (Team 节点 + PLAYED_AGAINST 关系)

调用方式:
  - 被 server_api.py 在收到 daily_matches 数据后自动调用
  - 也可手动执行: python pipeline/openclaw_ingestion.py data/openclaw_received/xxx.json
"""

import os
import sys
import json
import math
from datetime import datetime
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ─── 路径 ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

# ─── 联赛代码 → 联赛名（与 mysql_loader.py / neo4j_loader.py 一致）────
LEAGUE_CODE_MAP = {
    "E0": "England",
    "D1": "Germany",
    "I1": "Italy",
    "SP1": "Spain",
    "F1": "France",
}

# ─── 列名清洗（与 mysql_loader.py 的 clean_columns 完全一致）────
COLUMN_RENAME_MAP = {
    "B365>2.5": "B365_Over25",
    "B365<2.5": "B365_Under25",
    "P>2.5": "P_Over25",
    "P<2.5": "P_Under25",
    "Max>2.5": "Max_Over25",
    "Max<2.5": "Max_Under25",
    "Avg>2.5": "Avg_Over25",
    "Avg<2.5": "Avg_Under25",
    "B365C>2.5": "B365C_Over25",
    "B365C<2.5": "B365C_Under25",
    "PC>2.5": "PC_Over25",
    "PC<2.5": "PC_Under25",
    "MaxC>2.5": "MaxC_Over25",
    "MaxC<2.5": "MaxC_Under25",
    "AvgC>2.5": "AvgC_Over25",
    "AvgC<2.5": "AvgC_Under25",
}


# ═══════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════

def safe_float_str(val, default="N/A") -> str:
    """安全转换为保留两位小数的字符串（Neo4j 文本属性用）"""
    if val is None or val == "":
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f"{f:.2f}"
    except (ValueError, TypeError):
        return default


def convert_season(season_str: str) -> str:
    """将 '2526' 转换为 '2025-2026'"""
    if season_str and len(season_str) == 4 and season_str.isdigit():
        return f"20{season_str[:2]}-20{season_str[2:]}"
    return season_str


# ═══════════════════════════════════════════════════════════════
#  数据解析: OpenClaw JSON → DataFrame（与 match_master 格式一致）
# ═══════════════════════════════════════════════════════════════

def parse_to_dataframe(data: dict) -> pd.DataFrame:
    """
    解析 OpenClaw JSON，生成与 match_master 表列名完全一致的 DataFrame

    保持原始 CSV 列名（如 Date=DD/MM/YYYY, HomeTeam, FTHG, B365H 等）
    并添加 league 和 season 列
    """
    content = data.get("content", {})
    inner_data = content.get("data", {})
    leagues = inner_data.get("leagues", {})
    raw_season = inner_data.get("season", "")
    season = convert_season(raw_season)

    rows = []
    for league_code, league_info in leagues.items():
        match_list = league_info.get("matches", [])
        league_name = LEAGUE_CODE_MAP.get(league_code, league_code)

        for m in match_list:
            home = str(m.get("HomeTeam", "")).strip()
            away = str(m.get("AwayTeam", "")).strip()
            if not home or not away:
                continue
            if m.get("FTHG", "") == "" or m.get("FTAG", "") == "":
                continue

            row = dict(m)
            row["league"] = league_name
            row["season"] = season
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 日期格式转换: DD/MM/YYYY → YYYY-MM-DD（与 match_master 现有数据一致）
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce").dt.strftime("%Y-%m-%d")

    # 将空字符串替换为 NaN（与原始 CSV 加载行为一致）
    df.replace("", pd.NA, inplace=True)

    # 列名清洗（与 mysql_loader.py 的 clean_columns 完全一致）
    df.rename(columns=COLUMN_RENAME_MAP, inplace=True)

    return df


# ═══════════════════════════════════════════════════════════════
#  MySQL 入库: 写入 match_master 表
# ═══════════════════════════════════════════════════════════════

class MySQLIngestion:
    """MySQL 增量入库 → match_master 表"""

    def __init__(self):
        load_dotenv(ENV_PATH)
        host = os.getenv("MYSQL_HOST", "localhost")
        port = os.getenv("MYSQL_PORT", "3306")
        user = os.getenv("MYSQL_USER", "root")
        password = os.getenv("MYSQL_PASSWORD", "")
        database = os.getenv("MYSQL_DATABASE", "football_agent")
        encoded_pw = quote_plus(password)
        url = f"mysql+pymysql://{user}:{encoded_pw}@{host}:{port}/{database}?charset=utf8mb4"
        self.engine = create_engine(url, echo=False)

    def close(self):
        self.engine.dispose()

    def get_existing_keys(self) -> set:
        """查询 match_master 中已有的 (Date, HomeTeam, AwayTeam) 组合"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT `Date`, HomeTeam, AwayTeam FROM match_master"
                ))
                return {(row[0], row[1], row[2]) for row in result}
        except Exception:
            return set()

    def get_max_id(self) -> int:
        """获取 match_master 中当前最大 id"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT MAX(id) FROM match_master"))
                row = result.fetchone()
                return int(row[0]) if row and row[0] is not None else 0
        except Exception:
            return 0

    def ingest(self, df: pd.DataFrame) -> dict:
        """
        增量写入 match_master 表

        返回: {"inserted": int, "skipped": int}
        """
        if df.empty:
            return {"inserted": 0, "skipped": 0}

        # 查询已有数据的唯一键
        existing_keys = self.get_existing_keys()
        print(f"[MySQL] match_master 已有 {len(existing_keys)} 条记录")

        # 过滤出新数据
        new_rows = []
        skipped = 0
        for _, row in df.iterrows():
            key = (str(row.get("Date", "")), str(row.get("HomeTeam", "")), str(row.get("AwayTeam", "")))
            if key in existing_keys:
                skipped += 1
                print(f"  [跳过] 已存在: {key[0]} {key[1]} vs {key[2]}")
            else:
                new_rows.append(row)

        if not new_rows:
            print(f"[MySQL] 无新数据需要插入（全部 {skipped} 条已存在）")
            return {"inserted": 0, "skipped": skipped}

        new_df = pd.DataFrame(new_rows)

        # 分配 id（从当前最大 id 递增）
        max_id = self.get_max_id()
        new_df.insert(0, "id", range(max_id + 1, max_id + 1 + len(new_df)))

        # 确保列顺序与 match_master 一致（获取现有列名）
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'match_master' "
                    "ORDER BY ORDINAL_POSITION"
                ))
                existing_cols = [row[0] for row in result]

            # 只保留 match_master 中存在的列，缺少的列补 None
            for col in existing_cols:
                if col not in new_df.columns:
                    new_df[col] = None
            # 只保留 match_master 表中存在的列
            cols_to_insert = [c for c in existing_cols if c in new_df.columns]
            new_df = new_df[cols_to_insert]
        except Exception as e:
            print(f"  [警告] 读取 match_master 列信息失败: {e}，将尝试直接追加")

        # 追加写入
        try:
            new_df.to_sql(
                name="match_master",
                con=self.engine,
                if_exists="append",
                index=False,
                chunksize=500,
                method="multi",
            )
            print(f"[MySQL] 成功插入 {len(new_df)} 条新记录到 match_master")
        except Exception as e:
            print(f"[MySQL] 写入失败: {e}")
            return {"inserted": 0, "skipped": skipped, "error": str(e)}

        return {"inserted": len(new_df), "skipped": skipped}


# ═══════════════════════════════════════════════════════════════
#  Neo4j 入库（与 neo4j_loader.py 完全一致的 PLAYED_AGAINST 模式）
# ═══════════════════════════════════════════════════════════════

CYPHER_MERGE_MATCH = """
MERGE (home:Team {name: $home_team, league: $league})
MERGE (away:Team {name: $away_team, league: $league})
MERGE (home)-[r:PLAYED_AGAINST {
    match_date: $match_date,
    season: $season,
    match_result: $match_result
}]->(away)
SET r.total_goals         = $total_goals,
    r.odds_info           = $odds_info,
    r.over_under_odds     = $over_under_odds,
    r.closing_odds_info   = $closing_odds_info,
    r.closing_over_under  = $closing_over_under,
    r.asian_handicap_info = $asian_handicap_info,
    r.league              = $league,
    r.season              = $season
"""


class Neo4jIngestion:
    """Neo4j 增量入库（与 neo4j_loader.py 完全一致的模式）"""

    def __init__(self):
        load_dotenv(ENV_PATH)
        uri = os.getenv("NEO4J_URL", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def ensure_constraints(self):
        """确保唯一性约束存在"""
        with self.driver.session(database=self.database) as session:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (t:Team) REQUIRE (t.name, t.league) IS UNIQUE"
            )

    def ingest(self, df: pd.DataFrame) -> dict:
        """
        增量写入 Neo4j（MERGE 天然去重）

        从 DataFrame 中提取数据，拼接文本属性后写入
        """
        if df.empty:
            return {"processed": 0, "errors": 0}

        processed = 0
        error_count = 0

        with self.driver.session(database=self.database) as session:
            for _, row in df.iterrows():
                try:
                    home = str(row["HomeTeam"])
                    away = str(row["AwayTeam"])
                    league = str(row["league"])

                    fthg = int(row["FTHG"])
                    ftag = int(row["FTAG"])
                    match_result = f"{home} {fthg}:{ftag} {away}"
                    total_goals = fthg + ftag

                    # 赔率文本拼接（与 neo4j_loader.py build_match_records 一致）
                    h_odds = safe_float_str(row.get("B365H"))
                    d_odds = safe_float_str(row.get("B365D"))
                    a_odds = safe_float_str(row.get("B365A"))
                    odds_info = (
                        f"{home} 胜赔率: {h_odds} | "
                        f"平局赔率: {d_odds} | "
                        f"{away} 胜赔率: {a_odds}"
                    )

                    over_val = safe_float_str(row.get("B365>2.5", row.get("B365_Over25")))
                    under_val = safe_float_str(row.get("B365<2.5", row.get("B365_Under25")))
                    over_under_odds = f"大球赔率: {over_val} | 小球赔率: {under_val}"

                    ch = safe_float_str(row.get("B365CH"))
                    cd = safe_float_str(row.get("B365CD"))
                    ca = safe_float_str(row.get("B365CA"))
                    closing_odds_info = (
                        f"{home} 终盘胜赔率: {ch} | "
                        f"终盘平局赔率: {cd} | "
                        f"{away} 终盘胜赔率: {ca}"
                    )

                    c_over = safe_float_str(row.get("B365C>2.5", row.get("B365C_Over25")))
                    c_under = safe_float_str(row.get("B365C<2.5", row.get("B365C_Under25")))
                    closing_over_under = f"终盘大球赔率: {c_over} | 终盘小球赔率: {c_under}"

                    ahh = safe_float_str(row.get("AHh"))
                    ah_home = safe_float_str(row.get("B365AHH"))
                    ah_away = safe_float_str(row.get("B365AHA"))
                    asian_handicap_info = (
                        f"让球盘口: {ahh} | "
                        f"{home} 亚盘赔率: {ah_home} | "
                        f"{away} 亚盘赔率: {ah_away}"
                    )

                    # match_date 保持原始格式 DD/MM/YYYY
                    match_date = str(row.get("Date", ""))

                    session.run(CYPHER_MERGE_MATCH, {
                        "home_team": home,
                        "away_team": away,
                        "league": league,
                        "season": str(row["season"]),
                        "match_date": match_date,
                        "match_result": match_result,
                        "total_goals": total_goals,
                        "odds_info": odds_info,
                        "over_under_odds": over_under_odds,
                        "closing_odds_info": closing_odds_info,
                        "closing_over_under": closing_over_under,
                        "asian_handicap_info": asian_handicap_info,
                    })
                    processed += 1

                except Exception as e:
                    error_count += 1
                    print(f"  [Neo4j] 写入失败: {row.get('HomeTeam')} vs {row.get('AwayTeam')} - {e}")

        stats = {"processed": processed, "errors": error_count}
        print(f"[Neo4j] 入库完成: 处理 {processed}, 失败 {error_count}")
        return stats


# ═══════════════════════════════════════════════════════════════
#  统一入库入口
# ═══════════════════════════════════════════════════════════════

def ingest_openclaw_data(data: dict) -> dict:
    """
    OpenClaw 数据增量入库（MySQL match_master + Neo4j）

    Args:
        data: OpenClaw 推送的原始 JSON 字典
    Returns:
        {"total_matches": int, "mysql": {...}, "neo4j": {...}}
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. 解析为 DataFrame
    df = parse_to_dataframe(data)
    if df.empty:
        print(f"[{ts}] 无有效比赛数据需要入库")
        return {"total_matches": 0, "mysql": None, "neo4j": None}

    print(f"[{ts}] 解析到 {len(df)} 场比赛，开始入库...")
    for _, row in df.iterrows():
        print(f"  {row.get('Date')} | {row.get('Div')} | {row.get('HomeTeam')} {row.get('FTHG')}-{row.get('FTAG')} {row.get('AwayTeam')}")

    # 2. MySQL 入库 → match_master 表
    mysql_stats = None
    try:
        mysql_db = MySQLIngestion()
        mysql_stats = mysql_db.ingest(df)
        mysql_db.close()
    except Exception as e:
        print(f"[MySQL] 入库异常: {e}")
        mysql_stats = {"error": str(e)}

    # 3. Neo4j 入库
    neo4j_stats = None
    try:
        neo4j_db = Neo4jIngestion()
        neo4j_db.ensure_constraints()
        neo4j_stats = neo4j_db.ingest(df)
        neo4j_db.close()
    except Exception as e:
        print(f"[Neo4j] 入库异常: {e}")
        neo4j_stats = {"error": str(e)}

    result = {
        "total_matches": len(df),
        "mysql": mysql_stats,
        "neo4j": neo4j_stats,
    }
    print(f"[{ts}] 入库完成: {json.dumps(result, ensure_ascii=False)}")
    return result


# ═══════════════════════════════════════════════════════════════
#  命令行入口（手动导入 JSON 文件）
# ═══════════════════════════════════════════════════════════════

def main():
    """用法: python pipeline/openclaw_ingestion.py <json文件路径>"""
    if len(sys.argv) < 2:
        print("用法: python pipeline/openclaw_ingestion.py <json文件路径>")
        print("  示例: python pipeline/openclaw_ingestion.py data/openclaw_received/daily_matches_20260317_200756.json")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        sys.exit(1)

    print("=" * 55)
    print("  OpenClaw 数据增量入库工具")
    print("=" * 55)
    print(f"  文件: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = ingest_openclaw_data(data)
    print(f"\n结果: {json.dumps(result, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
