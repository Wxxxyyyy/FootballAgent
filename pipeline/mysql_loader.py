# -*- coding: utf-8 -*-
"""
数据导入 MySQL
- 读取 data/processed 下所有联赛 CSV 文件
- 统一追加写入 football_agent.match_master 表
- 通过 league 列区分联赛，season 列区分赛季
- 支持重复执行：每次先清空表再重建，保证幂等
"""

import os
import glob
from urllib.parse import quote_plus
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ─── 路径 ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

# ─── 联赛映射 ────────────────────────────────────────────────
LEAGUE_MAP = {
    "England": "England",
    "Spain": "Spain",
    "Italy": "Italy",
    "Germany": "Germany",
    "France": "France",
}


def get_mysql_engine():
    """根据 .env 配置创建 SQLAlchemy Engine"""
    load_dotenv(ENV_PATH)
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    database = os.getenv("MYSQL_DATABASE", "football_agent")

    # 对密码进行 URL 编码，避免特殊字符（如 ! @ # 等）导致解析失败
    encoded_password = quote_plus(password)
    url = f"mysql+pymysql://{user}:{encoded_password}@{host}:{port}/{database}?charset=utf8mb4"
    engine = create_engine(url, echo=False)
    return engine, database


def ensure_database(engine, database):
    """如果 football_agent 数据库不存在则自动创建"""
    # 用不带数据库名的连接来创建数据库
    load_dotenv(ENV_PATH)
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    user = os.getenv("MYSQL_USER", "root")
    password = quote_plus(os.getenv("MYSQL_PASSWORD", ""))
    url_no_db = f"mysql+pymysql://{user}:{password}@{host}:{port}/?charset=utf8mb4"
    tmp_engine = create_engine(url_no_db, echo=False)
    with tmp_engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{database}` "
                          f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
        conn.commit()
    tmp_engine.dispose()
    print(f"[✓] 数据库 `{database}` 已就绪")


def load_all_csv() -> pd.DataFrame:
    """读取 data/processed 下全部联赛 CSV，拼合为一张 DataFrame"""
    all_frames = []
    csv_pattern = os.path.join(PROCESSED_DIR, "*.csv")

    for filepath in sorted(glob.glob(csv_pattern)):
        filename = os.path.basename(filepath)

        # 跳过非联赛文件
        if filename in ("ClubName.csv", ".gitkeep"):
            continue

        # 从文件名解析联赛和赛季，例如 England_2021-2022.csv
        parts = filename.replace(".csv", "").split("_", 1)
        if len(parts) != 2:
            print(f"  [跳过] 无法解析文件名: {filename}")
            continue

        league_key, season = parts[0], parts[1]

        if league_key not in LEAGUE_MAP:
            print(f"  [跳过] 未知联赛: {filename}")
            continue

        df = pd.read_csv(filepath)
        df["league"] = LEAGUE_MAP[league_key]
        df["season"] = season
        all_frames.append(df)
        print(f"  [读取] {filename:40s}  →  {len(df):>4d} 条记录")

    if not all_frames:
        raise RuntimeError("未找到任何联赛 CSV 文件，请检查 data/processed 目录")

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\n[合计] 共读取 {len(combined)} 条比赛记录")
    return combined


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """清洗列名，将特殊字符替换为 MySQL 友好的名称"""
    rename_map = {
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
    df = df.rename(columns=rename_map)
    return df


def write_to_mysql(df: pd.DataFrame, engine):
    """将 DataFrame 写入 match_master 表（先删后建，保证幂等）"""
    table_name = "match_master"

    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))
        conn.commit()
    print(f"\n[...] 正在写入 `{table_name}` 表，共 {len(df)} 条...")

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=500,
        method="multi",
    )

    # 写入后添加索引以加速常见查询
    with engine.connect() as conn:
        conn.execute(text(f"ALTER TABLE `{table_name}` ADD PRIMARY KEY (`id`)"))
        conn.commit()
    print(f"[✓] `{table_name}` 写入完成")


def add_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """添加自增 id 列作为主键"""
    df.insert(0, "id", range(1, len(df) + 1))
    return df


def verify(engine):
    """写入后验证：按联赛统计行数"""
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT league, season, COUNT(*) AS cnt "
            "FROM match_master GROUP BY league, season ORDER BY league, season"
        ))
        rows = result.fetchall()

    print("\n" + "=" * 55)
    print(f"{'联赛':<12}{'赛季':<15}{'记录数':>8}")
    print("-" * 55)
    total = 0
    for league, season, cnt in rows:
        print(f"{league:<12}{season:<15}{cnt:>8d}")
        total += cnt
    print("-" * 55)
    print(f"{'总计':<27}{total:>8d}")
    print("=" * 55)


def main():
    print("=" * 55)
    print("  Football Agent · MySQL 数据导入工具")
    print("=" * 55)

    # 1. 创建引擎 & 确保数据库存在
    engine, database = get_mysql_engine()
    ensure_database(engine, database)

    # 2. 读取所有 CSV
    print(f"\n[1/3] 读取 CSV 文件 ({PROCESSED_DIR})")
    df = load_all_csv()

    # 3. 清洗列名 & 添加 id
    print("\n[2/3] 清洗列名 & 准备写入")
    df = clean_columns(df)
    df = add_id_column(df)

    # 4. 写入 MySQL
    print("\n[3/3] 写入 MySQL")
    write_to_mysql(df, engine)

    # 5. 验证
    verify(engine)

    engine.dispose()
    print("\n[✓] 全部完成！数据已导入 football_agent.match_master")


if __name__ == "__main__":
    main()
