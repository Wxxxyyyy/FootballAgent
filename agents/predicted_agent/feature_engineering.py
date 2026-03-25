# -*- coding: utf-8 -*-
"""
赔率特征工程模块

核心功能:
  1. 从 processed CSV / OpenClaw JSON 中提取 Bet365 初盘特征
  2. 赔率 → 隐含概率转换（含庄家利润去除）
  3. 构建 WDL（胜平负）和 OU（大小球）标签
  4. 统一的特征列定义，供训练和预测复用

特征列 (6 个原始 + 6 个衍生 = 12 维):
  原始: B365H, B365D, B365A, B365>2.5, B365<2.5, AHh
  衍生: prob_h, prob_d, prob_a, prob_over, prob_under, overround
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

# 训练 & 预测共用的原始特征列
RAW_FEATURE_COLS = ["B365H", "B365D", "B365A", "B365>2.5", "B365<2.5", "AHh"]

# 衍生特征列
DERIVED_FEATURE_COLS = ["prob_h", "prob_d", "prob_a", "prob_over", "prob_under", "overround"]

# 模型输入的完整特征列
ALL_FEATURE_COLS = RAW_FEATURE_COLS + DERIVED_FEATURE_COLS


# ═══════════════════════════════════════════════════════════════
#  赔率 → 隐含概率
# ═══════════════════════════════════════════════════════════════

def odds_to_probs(h_odds: float, d_odds: float, a_odds: float):
    """
    胜平负赔率 → 去除庄家利润后的真实隐含概率

    博彩公司赔率的倒数之和（overround）通常 > 1（如 1.05），
    将各赔率倒数除以 overround 即可得到归一化概率
    """
    inv_h = 1.0 / h_odds
    inv_d = 1.0 / d_odds
    inv_a = 1.0 / a_odds
    overround = inv_h + inv_d + inv_a
    return inv_h / overround, inv_d / overround, inv_a / overround, overround


def ou_odds_to_probs(over_odds: float, under_odds: float):
    """大小球赔率 → 去除庄家利润后的概率"""
    inv_o = 1.0 / over_odds
    inv_u = 1.0 / under_odds
    total = inv_o + inv_u
    return inv_o / total, inv_u / total


# ═══════════════════════════════════════════════════════════════
#  特征构建
# ═══════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从包含原始赔率列的 DataFrame 中构建完整特征

    输入 df 需包含 RAW_FEATURE_COLS 中的列
    输出 df 新增 DERIVED_FEATURE_COLS 中的列

    返回: 仅包含 ALL_FEATURE_COLS 的 DataFrame（已删除含 NaN 的行）
    """
    df = df.copy()

    # 确保原始特征列为 float
    for col in RAW_FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除原始特征列中有 NaN 的行
    df.dropna(subset=RAW_FEATURE_COLS, inplace=True)

    # 胜平负隐含概率
    wdl_probs = df.apply(
        lambda r: odds_to_probs(r["B365H"], r["B365D"], r["B365A"]),
        axis=1, result_type="expand",
    )
    df["prob_h"] = wdl_probs[0]
    df["prob_d"] = wdl_probs[1]
    df["prob_a"] = wdl_probs[2]
    df["overround"] = wdl_probs[3]

    # 大小球隐含概率
    ou_probs = df.apply(
        lambda r: ou_odds_to_probs(r["B365>2.5"], r["B365<2.5"]),
        axis=1, result_type="expand",
    )
    df["prob_over"] = ou_probs[0]
    df["prob_under"] = ou_probs[1]

    return df


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建训练标签

    WDL 标签: FTR 字段 → 数值编码 (H=0, D=1, A=2)
    OU 标签: FTHG + FTAG > 2.5 → 1(大球), 0(小球)
    """
    df = df.copy()

    # 确保比分列为数值
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")

    # 删除 FTR / FTHG / FTAG 有缺失的行
    df.dropna(subset=["FTR", "FTHG", "FTAG"], inplace=True)

    # WDL 标签: H=0, D=1, A=2
    wdl_map = {"H": 0, "D": 1, "A": 2}
    df["label_wdl"] = df["FTR"].map(wdl_map)

    # OU 标签: 大于 2.5 球 = 1
    df["label_ou"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)

    return df


# ═══════════════════════════════════════════════════════════════
#  加载训练数据
# ═══════════════════════════════════════════════════════════════

def load_training_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    读取 data/processed 下全部 CSV，构建特征 + 标签

    返回:
      X:       特征 DataFrame (ALL_FEATURE_COLS)
      y_wdl:   胜平负标签 ndarray (0/1/2)
      y_ou:    大小球标签 ndarray (0/1)
    """
    csv_pattern = os.path.join(PROCESSED_DIR, "*.csv")
    frames = []

    for filepath in sorted(glob.glob(csv_pattern)):
        filename = os.path.basename(filepath)
        if filename in ("ClubName.csv", ".gitkeep"):
            continue
        df = pd.read_csv(filepath)
        frames.append(df)

    if not frames:
        raise RuntimeError(f"未找到 CSV 文件: {PROCESSED_DIR}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"[特征工程] 读取 {len(frames)} 个文件，共 {len(combined)} 行")

    # 构建特征 + 标签
    combined = build_features(combined)
    combined = build_labels(combined)

    # 删除标签中还有 NaN 的行
    combined.dropna(subset=["label_wdl", "label_ou"], inplace=True)

    X = combined[ALL_FEATURE_COLS].copy()
    y_wdl = combined["label_wdl"].astype(int).values
    y_ou = combined["label_ou"].astype(int).values

    print(f"[特征工程] 清洗后有效样本: {len(X)}")
    print(f"  WDL 分布: H={np.sum(y_wdl==0)}, D={np.sum(y_wdl==1)}, A={np.sum(y_wdl==2)}")
    print(f"  OU  分布: Over={np.sum(y_ou==1)}, Under={np.sum(y_ou==0)}")

    return X, y_wdl, y_ou


# ═══════════════════════════════════════════════════════════════
#  OpenClaw 赔率 → 预测特征
# ═══════════════════════════════════════════════════════════════

def extract_features_from_odds(
    b365h: float,
    b365d: float,
    b365a: float,
    b365_over25: float,
    b365_under25: float,
    ahh: float = 0.0,
) -> pd.DataFrame:
    """
    从单场比赛的赔率构建模型输入特征

    适用于: OpenClaw 实时传来的赔率 / 手动输入的赔率
    返回: 包含 ALL_FEATURE_COLS 的单行 DataFrame
    """
    row = {
        "B365H": b365h,
        "B365D": b365d,
        "B365A": b365a,
        "B365>2.5": b365_over25,
        "B365<2.5": b365_under25,
        "AHh": ahh,
    }

    prob_h, prob_d, prob_a, overround = odds_to_probs(b365h, b365d, b365a)
    prob_over, prob_under = ou_odds_to_probs(b365_over25, b365_under25)

    row["prob_h"] = prob_h
    row["prob_d"] = prob_d
    row["prob_a"] = prob_a
    row["overround"] = overround
    row["prob_over"] = prob_over
    row["prob_under"] = prob_under

    return pd.DataFrame([row], columns=ALL_FEATURE_COLS)


def extract_features_from_openclaw(match_data: dict) -> pd.DataFrame:
    """
    从 OpenClaw 推送的单场比赛原始字段中提取特征

    match_data: OpenClaw 格式的比赛字典（含 B365H, B365D, B365A, B365>2.5, B365<2.5, AHh）
    """
    def _f(key):
        val = match_data.get(key, None)
        if val is None or val == "":
            return None
        return float(val)

    b365h = _f("B365H")
    b365d = _f("B365D")
    b365a = _f("B365A")
    over25 = _f("B365>2.5")
    under25 = _f("B365<2.5")
    ahh = _f("AHh") or 0.0

    if any(v is None for v in [b365h, b365d, b365a, over25, under25]):
        raise ValueError(f"赔率数据不完整: B365H={b365h}, B365D={b365d}, B365A={b365a}, "
                         f"Over25={over25}, Under25={under25}")

    return extract_features_from_odds(b365h, b365d, b365a, over25, under25, ahh)
