# -*- coding: utf-8 -*-
"""
赔率特征工程模块（13维 · 仅胜平负）

核心功能:
  1. 从 processed CSV / OpenClaw JSON 中提取 Bet365 初盘 + 终盘赔率
  2. 赔率 → 隐含概率转换（含庄家利润去除）
  3. 构建赔率变化特征（终盘 - 初盘）
  4. 构建 WDL（胜平负）标签
  5. 统一的特征列定义，供训练和预测复用

特征列 (3 原始 + 10 衍生 = 13 维):
  原始初盘: B365H, B365D, B365A
  衍生:
    prob_h, prob_d, prob_a      —— 初盘隐含概率
    overround                    —— 初盘庄家利润率
    odds_move_h, odds_move_d, odds_move_a —— 赔率变化（终盘 - 初盘）
    prob_h_c, prob_d_c, prob_a_c —— 终盘隐含概率

说明:
  - 不再使用大小球（OU）特征，专注于胜平负预测
  - 训练数据使用联赛 CSV（含 B365H 初盘 + B365CH 终盘）
  - 预测时 OpenClaw 推送当前赔率视为终盘，初盘从 titan007 爬取；
    若无初盘数据，初盘 = 终盘（赔率变化 = 0），模型仍可工作
"""

import os
import glob
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# ═══════════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════════

# 初盘原始赔率列（CSV 中存在）
OPENING_ODDS_COLS = ["B365H", "B365D", "B365A"]

# 终盘原始赔率列（CSV 中存在，带 C 后缀 = Closing）
CLOSING_ODDS_COLS = ["B365CH", "B365CD", "B365CA"]

# 模型输入的原始特征列（3 维初盘赔率）
RAW_FEATURE_COLS = ["B365H", "B365D", "B365A"]

# 衍生特征列（16 维）
DERIVED_FEATURE_COLS = [
    "prob_h", "prob_d", "prob_a",        # 初盘隐含概率
    "overround",                          # 初盘 overround
    "odds_move_h", "odds_move_d", "odds_move_a",  # 赔率变化
    "prob_h_c", "prob_d_c", "prob_a_c",  # 终盘隐含概率
    "odds_spread",                        # 初盘赔率离散度（越小越可能平局）
    "odds_cv",                            # 初盘赔率变异系数
    "top2_gap",                           # 初盘最低两赔率差距
    "move_x_prob_h", "move_x_prob_d", "move_x_prob_a",  # 交互特征（赔率变化×概率）
]

# 模型输入的完整特征列（19 维）
ALL_FEATURE_COLS = RAW_FEATURE_COLS + DERIVED_FEATURE_COLS

# 特征总数
FEATURE_DIM = len(ALL_FEATURE_COLS)  # 19


# ═══════════════════════════════════════════════════════════════
#  赔率 → 隐含概率
# ═══════════════════════════════════════════════════════════════

def odds_to_probs(h_odds: float, d_odds: float, a_odds: float):
    """
    胜平负赔率 → 去除庄家利润后的真实隐含概率

    博彩公司赔率的倒数之和（overround）通常 > 1（如 1.05），
    将各赔率倒数除以 overround 即可得到归一化概率

    Returns:
        (prob_h, prob_d, prob_a, overround)
    """
    inv_h = 1.0 / h_odds
    inv_d = 1.0 / d_odds
    inv_a = 1.0 / a_odds
    overround = inv_h + inv_d + inv_a
    return inv_h / overround, inv_d / overround, inv_a / overround, overround


# ═══════════════════════════════════════════════════════════════
#  特征构建
# ═══════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从包含初盘 + 终盘赔率列的 DataFrame 构建 13 维特征

    输入 df 需包含:
      - OPENING_ODDS_COLS: B365H, B365D, B365A（初盘）
      - CLOSING_ODDS_COLS: B365CH, B365CD, B365CA（终盘）

    若终盘缺失，会用初盘填充（赔率变化 = 0）

    输出 df 新增 DERIVED_FEATURE_COLS 中的列
    返回: 含全部 ALL_FEATURE_COLS + 原始列的 DataFrame（已删除含 NaN 的行）
    """
    df = df.copy()

    # 确保原始特征列为 float
    for col in OPENING_ODDS_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除初盘赔率有 NaN 的行
    df.dropna(subset=OPENING_ODDS_COLS, inplace=True)

    # 终盘赔率：若缺失则用初盘填充（赔率变化 = 0）
    for open_col, close_col in zip(OPENING_ODDS_COLS, CLOSING_ODDS_COLS):
        df[close_col] = pd.to_numeric(df.get(close_col), errors="coerce")
        df[close_col] = df[close_col].fillna(df[open_col])

    # ── 初盘隐含概率 + overround ──
    opening_probs = df.apply(
        lambda r: odds_to_probs(r["B365H"], r["B365D"], r["B365A"]),
        axis=1, result_type="expand",
    )
    df["prob_h"] = opening_probs[0]
    df["prob_d"] = opening_probs[1]
    df["prob_a"] = opening_probs[2]
    df["overround"] = opening_probs[3]

    # ── 赔率变化（终盘 - 初盘）──
    # 正值 = 赔率上升 = 资金流出该结果
    # 负值 = 赔率下降 = 资金流入该结果
    df["odds_move_h"] = df["B365CH"] - df["B365H"]
    df["odds_move_d"] = df["B365CD"] - df["B365D"]
    df["odds_move_a"] = df["B365CA"] - df["B365A"]

    # ── 终盘隐含概率 ──
    closing_probs = df.apply(
        lambda r: odds_to_probs(r["B365CH"], r["B365CD"], r["B365CA"]),
        axis=1, result_type="expand",
    )
    df["prob_h_c"] = closing_probs[0]
    df["prob_d_c"] = closing_probs[1]
    df["prob_a_c"] = closing_probs[2]

    # ── 赔率离散度特征（帮助识别平局场景）──
    # 三队赔率越接近（离散度越低），平局可能性越高
    odds_matrix = df[["B365H", "B365D", "B365A"]]
    df["odds_spread"] = (odds_matrix.max(axis=1) - odds_matrix.min(axis=1)) / odds_matrix.mean(axis=1)
    df["odds_cv"] = odds_matrix.std(axis=1) / odds_matrix.mean(axis=1)
    # 最低赔率和第二低赔率的差距
    df["top2_gap"] = odds_matrix.apply(
        lambda r: sorted(r)[1] - sorted(r)[0], axis=1
    )

    # ── 交互特征（赔率变化 × 概率）──
    # 捕捉"高概率结果 + 赔率变化"的组合效应
    df["move_x_prob_h"] = df["odds_move_h"] * df["prob_h"]
    df["move_x_prob_d"] = df["odds_move_d"] * df["prob_d"]
    df["move_x_prob_a"] = df["odds_move_a"] * df["prob_a"]

    return df


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建训练标签（仅胜平负）

    WDL 标签: FTR 字段 → 数值编码 (H=0, D=1, A=2)
    """
    df = df.copy()

    # 确保比分列为数值（用于可能的扩展分析）
    df["FTHG"] = pd.to_numeric(df.get("FTHG"), errors="coerce")
    df["FTAG"] = pd.to_numeric(df.get("FTAG"), errors="coerce")

    # 删除 FTR 有缺失的行
    df.dropna(subset=["FTR"], inplace=True)

    # WDL 标签: H=0, D=1, A=2
    wdl_map = {"H": 0, "D": 1, "A": 2}
    df["label_wdl"] = df["FTR"].map(wdl_map)

    # 删除标签 NaN
    df.dropna(subset=["label_wdl"], inplace=True)
    df["label_wdl"] = df["label_wdl"].astype(int)

    return df


# ═══════════════════════════════════════════════════════════════
#  加载训练数据（含训练/验证集划分）
# ═══════════════════════════════════════════════════════════════

# 联赛数据划分界限（按赛季）
# 训练集: 2021-2024 赛季
# 验证集: 2024-2025 赛季（holdout，用于早停/调参）
TRAIN_SEASONS = {"2021-2022", "2022-2023", "2023-2024"}
VAL_SEASONS = {"2024-2025"}


def _season_from_filename(filename: str) -> str:
    """从文件名提取赛季字符串，如 'England_2024-2025.csv' → '2024-2025'"""
    import re
    m = re.search(r"(\d{4}-\d{4})", filename)
    return m.group(1) if m else ""


def load_training_data(return_split: bool = True) -> tuple:
    """
    读取 data/processed 下全部联赛 CSV，构建特征 + 标签

    Args:
        return_split: True 时按赛季划分返回 (X_train, y_train, X_val, y_val)
                      False 时返回全部合并 (X, y)

    Returns:
        return_split=True: (X_train, y_train, X_val, y_val)
        return_split=False: (X, y)
    """
    csv_pattern = os.path.join(PROCESSED_DIR, "*.csv")
    frames = []

    for filepath in sorted(glob.glob(csv_pattern)):
        filename = os.path.basename(filepath)
        if filename in ("ClubName.csv", ".gitkeep"):
            continue
        df = pd.read_csv(filepath)
        df["_season"] = _season_from_filename(filename)
        frames.append(df)

    if not frames:
        raise RuntimeError(f"未找到 CSV 文件: {PROCESSED_DIR}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"[特征工程] 读取 {len(frames)} 个文件，共 {len(combined)} 行")

    # 构建特征 + 标签
    combined = build_features(combined)
    combined = build_labels(combined)

    print(f"[特征工程] 清洗后有效样本: {len(combined)}")

    if return_split:
        # 按赛季划分
        train_df = combined[combined["_season"].isin(TRAIN_SEASONS)]
        val_df = combined[combined["_season"].isin(VAL_SEASONS)]
        other_df = combined[~combined["_season"].isin(TRAIN_SEASONS | VAL_SEASONS)]

        # 未匹配赛季的归入训练集（避免数据浪费）
        if len(other_df) > 0:
            print(f"[特征工程] 未匹配赛季的样本 {len(other_df)} 行归入训练集")
            train_df = pd.concat([train_df, other_df], ignore_index=True)

        X_train = train_df[ALL_FEATURE_COLS].copy()
        y_train = train_df["label_wdl"].values
        X_val = val_df[ALL_FEATURE_COLS].copy()
        y_val = val_df["label_wdl"].values

        print(f"[特征工程] 训练集: {len(X_train)} 场 (赛季 {TRAIN_SEASONS})")
        print(f"  WDL 分布: H={np.sum(y_train==0)}, D={np.sum(y_train==1)}, A={np.sum(y_train==2)}")
        print(f"[特征工程] 验证集: {len(X_val)} 场 (赛季 {VAL_SEASONS})")
        print(f"  WDL 分布: H={np.sum(y_val==0)}, D={np.sum(y_val==1)}, A={np.sum(y_val==2)}")

        return X_train, y_train, X_val, y_val
    else:
        X = combined[ALL_FEATURE_COLS].copy()
        y = combined["label_wdl"].values
        print(f"[特征工程] 全部样本: {len(X)}")
        print(f"  WDL 分布: H={np.sum(y==0)}, D={np.sum(y==1)}, A={np.sum(y==2)}")
        return X, y


# ═══════════════════════════════════════════════════════════════
#  实时赔率 → 预测特征
# ═══════════════════════════════════════════════════════════════

def extract_features_from_odds(
    b365h: float,
    b365d: float,
    b365a: float,
    b365ch: float = None,
    b365cd: float = None,
    b365ca: float = None,
) -> pd.DataFrame:
    """
    从单场比赛的初盘 + 终盘赔率构建 13 维模型输入特征

    适用于: titan007 爬取的初盘+即时赔率 / 手动输入

    Args:
        b365h: Bet365 初盘主胜赔率
        b365d: Bet365 初盘平局赔率
        b365a: Bet365 初盘客胜赔率
        b365ch: Bet365 终盘(即时)主胜赔率，None 时 = 初盘
        b365cd: Bet365 终盘(即时)平局赔率，None 时 = 初盘
        b365ca: Bet365 终盘(即时)客胜赔率，None 时 = 初盘

    Returns: 包含 ALL_FEATURE_COLS 的单行 DataFrame
    """
    # 终盘缺失时用初盘填充（赔率变化 = 0）
    if b365ch is None:
        b365ch = b365h
    if b365cd is None:
        b365cd = b365d
    if b365ca is None:
        b365ca = b365a

    # 初盘概率
    prob_h, prob_d, prob_a, overround = odds_to_probs(b365h, b365d, b365a)

    # 赔率变化
    odds_move_h = b365ch - b365h
    odds_move_d = b365cd - b365d
    odds_move_a = b365ca - b365a

    # 终盘概率
    prob_h_c, prob_d_c, prob_a_c, _ = odds_to_probs(b365ch, b365cd, b365ca)

    # 赔率离散度特征（帮助识别平局场景）
    import numpy as np
    odds_arr = np.array([b365h, b365d, b365a])
    odds_spread = (odds_arr.max() - odds_arr.min()) / odds_arr.mean()
    odds_cv = odds_arr.std() / odds_arr.mean()
    sorted_odds = sorted(odds_arr)
    top2_gap = sorted_odds[1] - sorted_odds[0]

    # 交互特征
    move_x_prob_h = odds_move_h * prob_h
    move_x_prob_d = odds_move_d * prob_d
    move_x_prob_a = odds_move_a * prob_a

    row = {
        "B365H": b365h,
        "B365D": b365d,
        "B365A": b365a,
        "prob_h": prob_h,
        "prob_d": prob_d,
        "prob_a": prob_a,
        "overround": overround,
        "odds_move_h": odds_move_h,
        "odds_move_d": odds_move_d,
        "odds_move_a": odds_move_a,
        "prob_h_c": prob_h_c,
        "prob_d_c": prob_d_c,
        "prob_a_c": prob_a_c,
        "odds_spread": odds_spread,
        "odds_cv": odds_cv,
        "top2_gap": top2_gap,
        "move_x_prob_h": move_x_prob_h,
        "move_x_prob_d": move_x_prob_d,
        "move_x_prob_a": move_x_prob_a,
    }

    return pd.DataFrame([row], columns=ALL_FEATURE_COLS)


def extract_features_from_openclaw(match_data: dict) -> pd.DataFrame:
    """
    从 OpenClaw 推送的单场比赛原始字段中提取 13 维特征

    match_data 支持的字段:
      - 初盘: B365H, B365D, B365A（必需）
      - 终盘: B365CH, B365CD, B365CA（可选，缺失时 = 初盘）

    OpenClaw 推送的"当前赔率"若无初盘标注，会被视为终盘，
    此时初盘 = 终盘（赔率变化 = 0），模型仍可工作但损失赔率走势信号。
    """
    def _f(key):
        val = match_data.get(key, None)
        if val is None or val == "":
            return None
        return float(val)

    b365h = _f("B365H")
    b365d = _f("B365D")
    b365a = _f("B365A")

    if any(v is None for v in [b365h, b365d, b365a]):
        raise ValueError(
            f"初盘赔率数据不完整: B365H={b365h}, B365D={b365d}, B365A={b365a}"
        )

    # 终盘赔率（OpenClaw 可能不推送，此时用初盘）
    b365ch = _f("B365CH")
    b365cd = _f("B365CD")
    b365ca = _f("B365CA")

    return extract_features_from_odds(b365h, b365d, b365a, b365ch, b365cd, b365ca)
