# -*- coding: utf-8 -*-
"""
赔率基座模型 - 基于 Bet365 初盘赔率的轻量级 ML 预测

包含两个 RandomForest 分类器:
  1. WDL 模型: 预测 主胜(H)/平(D)/客胜(A) 的概率
  2. OU  模型: 预测 大2.5球/小2.5球 的概率

使用方式:
  训练: OddsModel.train_and_save()
  预测: model = OddsModel.load(); result = model.predict_from_odds(...)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, log_loss

# 模型存储目录
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
WDL_MODEL_PATH = os.path.join(MODEL_DIR, "wdl_model.pkl")
OU_MODEL_PATH = os.path.join(MODEL_DIR, "ou_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

# WDL 标签名
WDL_LABELS = {0: "H", 1: "D", 2: "A"}
WDL_LABELS_CN = {0: "主胜", 1: "平局", 2: "客胜"}


class OddsModel:
    """赔率基座预测模型"""

    def __init__(self):
        self.wdl_model: RandomForestClassifier = None
        self.ou_model: RandomForestClassifier = None
        self.meta: dict = {}

    # ═══════════════════════════════════════════════════════════
    #  训练
    # ═══════════════════════════════════════════════════════════

    @classmethod
    def train_and_save(cls) -> "OddsModel":
        """
        读取全部历史数据 → 训练 WDL + OU 模型 → 保存到磁盘

        返回训练好的 OddsModel 实例
        """
        from agents.predicted_agent.feature_engineering import load_training_data, ALL_FEATURE_COLS

        print("=" * 60)
        print("  赔率基座模型 · 训练")
        print("=" * 60)

        # 加载数据
        X, y_wdl, y_ou = load_training_data()
        feature_names = ALL_FEATURE_COLS

        instance = cls()

        # ---------- WDL 模型 ----------
        print("\n[1/2] 训练 WDL 模型 (胜平负)...")
        instance.wdl_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        instance.wdl_model.fit(X, y_wdl)

        # 交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        wdl_cv = cross_val_score(instance.wdl_model, X, y_wdl, cv=cv, scoring="accuracy")
        print(f"  5折 CV 准确率: {wdl_cv.mean():.4f} (+/- {wdl_cv.std():.4f})")

        wdl_pred = instance.wdl_model.predict(X)
        print(f"  训练集准确率: {accuracy_score(y_wdl, wdl_pred):.4f}")
        wdl_proba = instance.wdl_model.predict_proba(X)
        print(f"  训练集 Log Loss: {log_loss(y_wdl, wdl_proba):.4f}")

        # 特征重要性
        print("  特征重要性:")
        for name, imp in sorted(zip(feature_names, instance.wdl_model.feature_importances_), key=lambda x: -x[1]):
            print(f"    {name:20s} {imp:.4f}")

        # ---------- OU 模型 ----------
        print("\n[2/2] 训练 OU 模型 (大小球)...")
        instance.ou_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        instance.ou_model.fit(X, y_ou)

        ou_cv = cross_val_score(instance.ou_model, X, y_ou, cv=cv, scoring="accuracy")
        print(f"  5折 CV 准确率: {ou_cv.mean():.4f} (+/- {ou_cv.std():.4f})")

        ou_pred = instance.ou_model.predict(X)
        print(f"  训练集准确率: {accuracy_score(y_ou, ou_pred):.4f}")
        ou_proba = instance.ou_model.predict_proba(X)
        print(f"  训练集 Log Loss: {log_loss(y_ou, ou_proba):.4f}")

        print("  特征重要性:")
        for name, imp in sorted(zip(feature_names, instance.ou_model.feature_importances_), key=lambda x: -x[1]):
            print(f"    {name:20s} {imp:.4f}")

        # ---------- 保存 ----------
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(instance.wdl_model, WDL_MODEL_PATH)
        joblib.dump(instance.ou_model, OU_MODEL_PATH)

        instance.meta = {
            "train_samples": int(len(X)),
            "feature_count": len(feature_names),
            "features": feature_names,
            "wdl_cv_accuracy": round(float(wdl_cv.mean()), 4),
            "ou_cv_accuracy": round(float(ou_cv.mean()), 4),
        }
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(instance.meta, f, ensure_ascii=False, indent=2)

        print(f"\n[保存] 模型已保存到 {MODEL_DIR}")
        print("=" * 60)

        return instance

    # ═══════════════════════════════════════════════════════════
    #  加载
    # ═══════════════════════════════════════════════════════════

    @classmethod
    def load(cls) -> "OddsModel":
        """从磁盘加载已训练的模型"""
        if not os.path.exists(WDL_MODEL_PATH) or not os.path.exists(OU_MODEL_PATH):
            raise FileNotFoundError(
                f"模型文件不存在，请先执行训练:\n"
                f"  python -m agents.predicted_agent.models.statistical_model\n"
                f"  预期路径: {MODEL_DIR}"
            )

        instance = cls()
        instance.wdl_model = joblib.load(WDL_MODEL_PATH)
        instance.ou_model = joblib.load(OU_MODEL_PATH)

        if os.path.exists(META_PATH):
            with open(META_PATH, "r", encoding="utf-8") as f:
                instance.meta = json.load(f)

        return instance

    # ═══════════════════════════════════════════════════════════
    #  预测
    # ═══════════════════════════════════════════════════════════

    def predict(self, X: pd.DataFrame) -> dict:
        """
        批量预测

        Args:
            X: 特征 DataFrame，列必须是 ALL_FEATURE_COLS
        Returns:
            {
                "wdl_proba": [[p_h, p_d, p_a], ...],
                "wdl_pred": ["H", ...],
                "ou_proba": [[p_over, p_under], ...],
                "ou_pred": ["Over", ...],
            }
        """
        wdl_proba = self.wdl_model.predict_proba(X)
        wdl_pred_idx = self.wdl_model.predict(X)
        wdl_pred = [WDL_LABELS[i] for i in wdl_pred_idx]

        ou_proba = self.ou_model.predict_proba(X)
        ou_pred_idx = self.ou_model.predict(X)
        ou_pred = ["Over" if i == 1 else "Under" for i in ou_pred_idx]

        return {
            "wdl_proba": wdl_proba.tolist(),
            "wdl_pred": wdl_pred,
            "ou_proba": ou_proba.tolist(),
            "ou_pred": ou_pred,
        }

    def predict_from_odds(
        self,
        b365h: float,
        b365d: float,
        b365a: float,
        b365_over25: float,
        b365_under25: float,
        ahh: float = 0.0,
    ) -> dict:
        """
        单场比赛赔率 → 预测概率（供 LLM Agent 调用）

        Args:
            b365h:       Bet365 主胜赔率
            b365d:       Bet365 平局赔率
            b365a:       Bet365 客胜赔率
            b365_over25: Bet365 大2.5球赔率
            b365_under25: Bet365 小2.5球赔率
            ahh:         亚盘让球数

        Returns:
            {
                "home_win_prob": 0.45,
                "draw_prob": 0.28,
                "away_win_prob": 0.27,
                "wdl_prediction": "H",
                "over25_prob": 0.55,
                "under25_prob": 0.45,
                "ou_prediction": "Over",
                "input_odds": {...},
            }
        """
        from agents.predicted_agent.feature_engineering import extract_features_from_odds

        X = extract_features_from_odds(b365h, b365d, b365a, b365_over25, b365_under25, ahh)
        result = self.predict(X)

        wdl_p = result["wdl_proba"][0]
        ou_p = result["ou_proba"][0]

        return {
            "home_win_prob": round(wdl_p[0], 4),
            "draw_prob": round(wdl_p[1], 4),
            "away_win_prob": round(wdl_p[2], 4),
            "wdl_prediction": result["wdl_pred"][0],
            "over25_prob": round(ou_p[1], 4) if len(ou_p) > 1 else round(ou_p[0], 4),
            "under25_prob": round(ou_p[0], 4),
            "ou_prediction": result["ou_pred"][0],
            "input_odds": {
                "B365H": b365h, "B365D": b365d, "B365A": b365a,
                "B365_Over25": b365_over25, "B365_Under25": b365_under25,
                "AHh": ahh,
            },
        }

    def predict_from_openclaw(self, match_data: dict) -> dict:
        """
        从 OpenClaw 原始比赛字典直接预测

        match_data: OpenClaw 推送的比赛数据（含 B365H, B365D, B365A 等字段）
        """
        from agents.predicted_agent.feature_engineering import extract_features_from_openclaw

        X = extract_features_from_openclaw(match_data)
        result = self.predict(X)

        wdl_p = result["wdl_proba"][0]
        ou_p = result["ou_proba"][0]

        home = match_data.get("HomeTeam", "主队")
        away = match_data.get("AwayTeam", "客队")

        return {
            "match": f"{home} vs {away}",
            "home_team": home,
            "away_team": away,
            "home_win_prob": round(wdl_p[0], 4),
            "draw_prob": round(wdl_p[1], 4),
            "away_win_prob": round(wdl_p[2], 4),
            "wdl_prediction": result["wdl_pred"][0],
            "over25_prob": round(ou_p[1], 4) if len(ou_p) > 1 else round(ou_p[0], 4),
            "under25_prob": round(ou_p[0], 4),
            "ou_prediction": result["ou_pred"][0],
        }


# ═══════════════════════════════════════════════════════════════
#  命令行入口: python -m agents.predicted_agent.models.statistical_model
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    OddsModel.train_and_save()
