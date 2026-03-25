# -*- coding: utf-8 -*-
"""
CSV 数据清洗 & 预处理
- 读取 data/ori_data/ 下的原始 CSV（五大联赛 2021-2026）
- 提取：比赛日期、球队、比分、胜平负赔率、大小球赔率、初盘和终盘赔率
- 输出到 data/processed/（25个文件一一对应）
- 汇总所有球队名称到 ClubName.csv
"""

import os
import pandas as pd
from pathlib import Path


# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORI_DATA_DIR = PROJECT_ROOT / "data" / "ori_data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 联赛代码 → 联赛名称映射
LEAGUE_MAP = {
    "E0": "England",
    "F1": "France",
    "D1": "Germany",
    "I1": "Italy",
    "SP1": "Spain",
}

# 文件名前缀 → 联赛名称映射
FILE_PREFIX_TO_LEAGUE = {
    "England": "England",
    "France": "France",
    "Germany": "Germany",
    "Italy": "Italy",
    "Spain": "Spain",
}


# ============================================================
# 需要提取的列定义（按类别分组）
# ============================================================

# 基础比赛信息
BASE_COLS = [
    "Div",          # 联赛代码
    "Date",         # 比赛日期
    "HomeTeam",     # 主队
    "AwayTeam",     # 客队
]

# 比分信息
SCORE_COLS = [
    "FTHG",         # 全场主队进球
    "FTAG",         # 全场客队进球
    "FTR",          # 全场结果 (H=主胜, D=平, A=客胜)
    "HTHG",         # 半场主队进球
    "HTAG",         # 半场客队进球
    "HTR",          # 半场结果
]

# 初盘 - 胜平负赔率 (1X2 Opening Odds)
OPENING_1X2_COLS = [
    "B365H", "B365D", "B365A",       # Bet365 主胜/平/客胜
    "PSH", "PSD", "PSA",             # Pinnacle（最精准庄家）
    "MaxH", "MaxD", "MaxA",          # 市场最高赔率
    "AvgH", "AvgD", "AvgA",         # 市场平均赔率
]

# 初盘 - 大小球赔率 (Over/Under 2.5 Opening Odds)
OPENING_OU_COLS = [
    "B365>2.5", "B365<2.5",          # Bet365 大2.5/小2.5
    "P>2.5", "P<2.5",                # Pinnacle
    "Max>2.5", "Max<2.5",            # 市场最高
    "Avg>2.5", "Avg<2.5",           # 市场平均
]

# 初盘 - 亚洲盘口 (Asian Handicap Opening)
OPENING_AH_COLS = [
    "AHh",                            # 亚盘盘口值
    "B365AHH", "B365AHA",            # Bet365 亚盘主/客
    "PAHH", "PAHA",                   # Pinnacle 亚盘
    "MaxAHH", "MaxAHA",              # 市场最高
    "AvgAHH", "AvgAHA",             # 市场平均
]

# 终盘 - 胜平负赔率 (1X2 Closing Odds)
CLOSING_1X2_COLS = [
    "B365CH", "B365CD", "B365CA",    # Bet365 终盘
    "PSCH", "PSCD", "PSCA",          # Pinnacle 终盘
    "MaxCH", "MaxCD", "MaxCA",       # 市场最高终盘
    "AvgCH", "AvgCD", "AvgCA",      # 市场平均终盘
]

# 终盘 - 大小球赔率 (Over/Under 2.5 Closing Odds)
CLOSING_OU_COLS = [
    "B365C>2.5", "B365C<2.5",        # Bet365 终盘大小球
    "PC>2.5", "PC<2.5",              # Pinnacle 终盘
    "MaxC>2.5", "MaxC<2.5",          # 市场最高终盘
    "AvgC>2.5", "AvgC<2.5",         # 市场平均终盘
]

# 终盘 - 亚洲盘口 (Asian Handicap Closing)
CLOSING_AH_COLS = [
    "AHCh",                           # 终盘亚盘盘口值
    "B365CAHH", "B365CAHA",          # Bet365 终盘亚盘
    "PCAHH", "PCAHA",                # Pinnacle 终盘亚盘
    "MaxCAHH", "MaxCAHA",            # 市场最高终盘
    "AvgCAHH", "AvgCAHA",           # 市场平均终盘
]

# 汇总所有需要提取的列
ALL_TARGET_COLS = (
    BASE_COLS
    + SCORE_COLS
    + OPENING_1X2_COLS
    + OPENING_OU_COLS
    + OPENING_AH_COLS
    + CLOSING_1X2_COLS
    + CLOSING_OU_COLS
    + CLOSING_AH_COLS
)


# ============================================================
# 核心处理函数
# ============================================================

def normalize_date(date_str: str) -> str:
    """
    统一日期格式为 YYYY-MM-DD
    输入格式可能为: DD/MM/YYYY 或 DD/MM/YY
    """
    if pd.isna(date_str):
        return date_str
    date_str = str(date_str).strip()
    try:
        # 尝试 DD/MM/YYYY 格式
        return pd.to_datetime(date_str, format="%d/%m/%Y").strftime("%Y-%m-%d")
    except ValueError:
        try:
            # 尝试 DD/MM/YY 格式
            return pd.to_datetime(date_str, format="%d/%m/%y").strftime("%Y-%m-%d")
        except ValueError:
            # 兜底：让 pandas 自动识别
            return pd.to_datetime(date_str, dayfirst=True).strftime("%Y-%m-%d")


def get_output_filename(ori_filename: str) -> str:
    """
    清理文件名（去掉重复的 .csv.csv → .csv）
    """
    name = ori_filename
    # 处理 .csv.csv 的情况
    while name.endswith(".csv.csv"):
        name = name[:-4]  # 去掉一个 .csv
    if not name.endswith(".csv"):
        name = name + ".csv"
    return name


def extract_league_from_filename(filename: str) -> str:
    """
    从文件名提取联赛名称
    例如: England_2024-2025.csv → England
    """
    for prefix in FILE_PREFIX_TO_LEAGUE:
        if filename.startswith(prefix):
            return FILE_PREFIX_TO_LEAGUE[prefix]
    return "Unknown"


def process_single_file(filepath: Path) -> pd.DataFrame:
    """
    处理单个 CSV 文件：
    1. 读取原始数据
    2. 提取目标列（缺失的列用 NaN 填充）
    3. 统一日期格式
    4. 清洗空行
    """
    print(f"  📖 读取: {filepath.name}")

    # 读取 CSV
    df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")

    # 过滤空行（如果 HomeTeam 为空则跳过）
    if "HomeTeam" in df.columns:
        df = df.dropna(subset=["HomeTeam"])

    # 提取目标列（不存在的列用 NaN 填充）
    existing_cols = [col for col in ALL_TARGET_COLS if col in df.columns]
    missing_cols = [col for col in ALL_TARGET_COLS if col not in df.columns]

    if missing_cols:
        print(f"    ⚠️  缺失 {len(missing_cols)} 列: {missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}")

    # 构建结果 DataFrame
    result = df[existing_cols].copy()

    # 添加缺失列（填充 NaN）
    for col in missing_cols:
        result[col] = pd.NA

    # 按照目标列顺序重排
    result = result[ALL_TARGET_COLS]

    # 统一日期格式
    if "Date" in result.columns:
        result["Date"] = result["Date"].apply(normalize_date)

    # 重置索引
    result = result.reset_index(drop=True)

    print(f"    ✅ 提取完成: {len(result)} 条记录, {len(existing_cols)} 列有效数据")
    return result


def collect_club_names(processed_dir: Path) -> pd.DataFrame:
    """
    从所有已处理的 CSV 中收集球队名称
    按联赛分类，生成 ClubName.csv
    """
    club_records = []

    for csv_file in sorted(processed_dir.glob("*.csv")):
        if csv_file.name == "ClubName.csv":
            continue

        df = pd.read_csv(csv_file, encoding="utf-8")
        league = extract_league_from_filename(csv_file.name)

        # 提取赛季信息
        # 文件名格式: England_2024-2025.csv
        season = csv_file.stem.split("_", 1)[-1] if "_" in csv_file.stem else "Unknown"

        # 收集主队和客队名称
        if "HomeTeam" in df.columns:
            home_teams = df["HomeTeam"].dropna().unique()
            away_teams = df["AwayTeam"].dropna().unique() if "AwayTeam" in df.columns else []
            all_teams = set(home_teams) | set(away_teams)

            for team in all_teams:
                club_records.append({
                    "ClubName": team,
                    "League": league,
                    "Season": season,
                })

    # 构建 DataFrame
    clubs_df = pd.DataFrame(club_records)

    if clubs_df.empty:
        return clubs_df

    # 去重：同一联赛同一球队只保留一条（保留最新赛季）
    clubs_df = clubs_df.sort_values("Season", ascending=False)
    clubs_unique = clubs_df.drop_duplicates(subset=["ClubName", "League"], keep="first")

    # 按联赛和球队名排序
    league_order = ["England", "France", "Germany", "Italy", "Spain"]
    clubs_unique = clubs_unique.copy()
    clubs_unique["LeagueOrder"] = clubs_unique["League"].map(
        {l: i for i, l in enumerate(league_order)}
    )
    clubs_unique = clubs_unique.sort_values(["LeagueOrder", "ClubName"])
    clubs_unique = clubs_unique.drop(columns=["LeagueOrder", "Season"])

    # 重置索引
    clubs_unique = clubs_unique.reset_index(drop=True)

    return clubs_unique


# ============================================================
# 主流程
# ============================================================

def main():
    """数据预处理主入口"""
    print("=" * 60)
    print("⚽ Football Agent 数据预处理")
    print("=" * 60)

    # 确保输出目录存在
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 获取所有原始 CSV 文件
    ori_files = sorted([
        f for f in ORI_DATA_DIR.iterdir()
        if f.is_file() and f.name.endswith(".csv")
    ])

    print(f"\n📁 找到 {len(ori_files)} 个原始 CSV 文件\n")

    # ---- 第一步：逐个处理 CSV 文件 ----
    print("-" * 60)
    print("📊 第一步：提取目标列并清洗数据")
    print("-" * 60)

    processed_count = 0
    for ori_file in ori_files:
        output_name = get_output_filename(ori_file.name)
        output_path = PROCESSED_DIR / output_name

        # 处理数据
        df_processed = process_single_file(ori_file)

        # 保存
        df_processed.to_csv(output_path, index=False, encoding="utf-8")
        print(f"    💾 保存: processed/{output_name}")
        print()
        processed_count += 1

    print(f"✅ 共处理 {processed_count} 个文件\n")

    # ---- 第二步：生成球队名称汇总表 ----
    print("-" * 60)
    print("🏟️  第二步：生成球队名称汇总表 ClubName.csv")
    print("-" * 60)

    clubs_df = collect_club_names(PROCESSED_DIR)
    club_output_path = PROCESSED_DIR / "ClubName.csv"
    clubs_df.to_csv(club_output_path, index=False, encoding="utf-8")

    # 打印统计信息
    for league in ["England", "France", "Germany", "Italy", "Spain"]:
        count = len(clubs_df[clubs_df["League"] == league])
        print(f"  🏆 {league:10s}: {count} 支球队")

    print(f"\n  📋 总计: {len(clubs_df)} 支球队（含降级/升级球队）")
    print(f"  💾 保存: processed/ClubName.csv")

    print("\n" + "=" * 60)
    print("🎉 数据预处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
