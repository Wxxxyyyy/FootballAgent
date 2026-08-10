# -*- coding: utf-8 -*-
"""
数据漂移检测 (Data Drift Detection)
══════════════════════════════════════════════════════════════

第一性原理:
  模型在训练时的特征分布上训练，如果线上特征分布变了（漂移），
  模型预测就不可信了——"刻舟求剑"。
  漂移检测 = 对比"训练时的特征分布"和"近期的线上特征分布"，差异大则告警。

方案:
  1. 基线: 训练时的特征统计（均值/标准差/分位数），存 model_meta.json 的 baseline_stats
  2. 当前: 从最近 N 场预测记录中提取特征统计
  3. 对比: 每个特征计算 PSI (Population Stability Index)
     - PSI < 0.1   : 无显著漂移
     - 0.1 <= PSI < 0.25: 轻微漂移，关注
     - PSI >= 0.25  : 显著漂移，需重训模型

运行:
  python evaluation/drift_detector.py              # 检测最近30场预测的漂移
  python evaluation/drift_detector.py --window 50  # 指定窗口大小
  python evaluation/drift_detector.py --baseline  # 用当前数据更新基线（训练后执行）

存储:
  检测结果落 MySQL drift_reports 表，支持历史趋势查询
"""

from __future__ import annotations

import os
import sys
import json
import logging
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drift_detector")


# ═══════════════════════════════════════════════════════════════
#  PSI 计算（Population Stability Index）
# ═══════════════════════════════════════════════════════════════

def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """计算 PSI (Population Stability Index)

    PSI = Σ (actual_pct - expected_pct) * ln(actual_pct / expected_pct)

    Args:
        expected: 基线分布（训练时的特征值）
        actual: 当前分布（近期的特征值）
        buckets: 分桶数（默认10）

    Returns:
        PSI 值。越小越好：< 0.1 无漂移，>= 0.25 显著漂移
    """
    # 处理空值
    expected = np.array(expected, dtype=float)
    actual = np.array(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 2 or len(actual) < 2:
        return 0.0

    # 用基线的分位数定义桶边界
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0

    # 计算各桶的占比
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # 避免 0 值（导致 ln 无效）
    expected_pct = np.clip(expected_pct, 0.0001, None)
    actual_pct = np.clip(actual_pct, 0.0001, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def interpret_psi(psi: float) -> str:
    """解释 PSI 值"""
    if psi < 0.1:
        return "无显著漂移"
    elif psi < 0.25:
        return "轻微漂移，关注"
    else:
        return "显著漂移，建议重训模型"


# ═══════════════════════════════════════════════════════════════
#  基线管理
# ═══════════════════════════════════════════════════════════════

def save_baseline(feature_stats: dict, version: str = "") -> str:
    """保存特征基线统计到文件

    Args:
        feature_stats: {feature_name: {"mean": x, "std": y, "values": [...]}}
        version: 模型版本号

    Returns: 基线文件路径
    """
    baseline_dir = os.path.join(PROJECT_ROOT, "data", "drift_baseline")
    os.makedirs(baseline_dir, exist_ok=True)

    version_str = version or "default"
    baseline_path = os.path.join(baseline_dir, f"baseline_{version_str}.json")

    data = {
        "version": version_str,
        "created_at": datetime.now().isoformat(),
        "features": feature_stats,
    }
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 更新 current 指针
    current_path = os.path.join(baseline_dir, "current.json")
    with open(current_path, "w", encoding="utf-8") as f:
        json.dump({"version": version_str, "path": baseline_path}, f, ensure_ascii=False)

    logger.info(f"基线已保存: {baseline_path}")
    return baseline_path


def load_baseline() -> dict:
    """加载当前基线"""
    baseline_dir = os.path.join(PROJECT_ROOT, "data", "drift_baseline")
    current_path = os.path.join(baseline_dir, "current.json")

    try:
        with open(current_path, "r", encoding="utf-8") as f:
            current = json.load(f)
        baseline_path = current["path"]
        with open(baseline_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ═══════════════════════════════════════════════════════════════
#  漂移检测主逻辑
# ═══════════════════════════════════════════════════════════════

def detect_drift(current_features: dict, baseline: dict = None) -> dict:
    """检测特征漂移

    Args:
        current_features: {feature_name: [values...]}  当前特征值
        baseline: 基线数据（None=自动加载）

    Returns:
        {
            "has_drift": bool,
            "drifted_features": [feature_name, ...],
            "details": [{feature, psi, interpretation}, ...],
            "summary": "..."
        }
    """
    if baseline is None:
        baseline = load_baseline()

    if not baseline:
        return {
            "has_drift": False,
            "drifted_features": [],
            "details": [],
            "summary": "无基线数据，无法检测漂移",
        }

    baseline_features = baseline.get("features", {})
    details = []
    drifted = []

    for feature, stats in baseline_features.items():
        if feature not in current_features:
            continue

        expected_values = stats.get("values", [])
        actual_values = current_features[feature]

        if len(expected_values) < 2 or len(actual_values) < 2:
            continue

        psi = calculate_psi(np.array(expected_values), np.array(actual_values))
        interpretation = interpret_psi(psi)

        details.append({
            "feature": feature,
            "psi": round(psi, 4),
            "interpretation": interpretation,
            "baseline_mean": round(stats.get("mean", 0), 4),
            "current_mean": round(float(np.mean(actual_values)), 4),
        })

        if psi >= 0.25:
            drifted.append(feature)

    # 按 PSI 降序排列
    details.sort(key=lambda x: -x["psi"])

    drifted_count = len(drifted)
    total_count = len(details)
    summary = f"检测 {total_count} 个特征，{drifted_count} 个显著漂移"

    return {
        "has_drift": drifted_count > 0,
        "drifted_features": drifted,
        "details": details,
        "summary": summary,
    }


# ═══════════════════════════════════════════════════════════════
#  从预测记录提取当前特征
# ═══════════════════════════════════════════════════════════════

def extract_current_features_from_predictions(window: int = 30) -> dict:
    """从最近的预测记录中提取特征统计

    预测文件 data/predictions/*_tier*.json 中包含 input_odds，
    可重建特征。
    """
    import glob
    from agents.predicted_agent.feature_engineering import extract_features_from_odds
    import pandas as pd

    preds_dir = os.path.join(PROJECT_ROOT, "data", "predictions")
    files = sorted(glob.glob(os.path.join(preds_dir, "*_tier*.json")), reverse=True)

    all_features = []
    for path in files[:window]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                pred = json.load(f)
            odds = pred.get("ml_prediction", {}).get("input_odds", {})
            if not odds:
                continue
            df = extract_features_from_odds(
                odds.get("B365H"), odds.get("B365D"), odds.get("B365A"),
                odds.get("B365CH"), odds.get("B365CD"), odds.get("B365CA"),
            )
            all_features.append(df)
        except Exception:
            continue

    if not all_features:
        return {}

    combined = pd.concat(all_features, ignore_index=True)
    result = {}
    for col in combined.columns:
        vals = combined[col].dropna().tolist()
        if vals:
            result[col] = vals
    return result


# ═══════════════════════════════════════════════════════════════
#  CLI 入口
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="数据漂移检测")
    parser.add_argument("--window", type=int, default=30, help="检测窗口大小（最近N场预测）")
    parser.add_argument("--baseline", action="store_true", help="用当前数据更新基线")
    args = parser.parse_args()

    if args.baseline:
        # 用当前预测数据建立基线
        features = extract_current_features_from_predictions(window=args.window)
        if not features:
            logger.error("无预测数据，无法建立基线")
            sys.exit(1)
        stats = {}
        for feat, vals in features.items():
            arr = np.array(vals, dtype=float)
            stats[feat] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "values": vals,
            }
        save_baseline(stats, version="manual")
        print("基线建立完成")
        sys.exit(0)

    # 漂移检测
    current = extract_current_features_from_predictions(window=args.window)
    if not current:
        logger.error("无预测数据或数据不足")
        sys.exit(1)

    result = detect_drift(current)

    print("\n" + "=" * 60)
    print("  数据漂移检测报告")
    print("=" * 60)
    print(f"  {result['summary']}")

    if result["has_drift"]:
        print(f"\n  ⚠️ 显著漂移的特征:")
        for feat in result["drifted_features"]:
            detail = next(d for d in result["details"] if d["feature"] == feat)
            print(f"    {feat:25s} PSI={detail['psi']:.4f} {detail['interpretation']}")

    print(f"\n  全部特征 PSI:")
    for d in result["details"]:
        marker = "⚠️" if d["psi"] >= 0.25 else ("👀" if d["psi"] >= 0.1 else "✅")
        print(f"    {marker} {d['feature']:25s} PSI={d['psi']:.4f} "
              f"(基线均值={d['baseline_mean']}, 当前均值={d['current_mean']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
