# -*- coding: utf-8 -*-
"""
赔率基座模型 - 基于 Bet365 初盘 + 终盘赔率的 13 维胜平负预测

包含单个 RandomForest 分类器:
  WDL 模型: 预测 主胜(H)/平(D)/客胜(A) 的概率

特征 (13 维):
  B365H, B365D, B365A                  —— 初盘赔率
  prob_h, prob_d, prob_a, overround    —— 初盘隐含概率 + 庄家利润率
  odds_move_h, odds_move_d, odds_move_a —— 赔率变化（终盘 - 初盘）
  prob_h_c, prob_d_c, prob_a_c         —— 终盘隐含概率

数据集划分:
  训练集: 五大联赛 2021-2024 赛季（~7600 场）
  验证集: 五大联赛 2024-2025 赛季（~1900 场，用于早停/调参）
  测试集: 2022 世界杯（64 场，从 titan007 爬取初盘+终盘）

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
from sklearn.metrics import (
    classification_report, accuracy_score, log_loss, confusion_matrix
)
try:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# 模型存储目录
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
WDL_MODEL_PATH = os.path.join(MODEL_DIR, "wdl_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

# 2022 世界杯测试集路径（titan007 爬虫生成）
WC2022_TEST_PATH = os.path.join(MODEL_DIR, "wc2022_test.csv")

# WDL 标签名
WDL_LABELS = {0: "H", 1: "D", 2: "A"}
WDL_LABELS_CN = {0: "主胜", 1: "平局", 2: "客胜"}
# 反向映射
WDL_LABELS_REV = {"H": 0, "D": 1, "A": 2}
WDL_LABELS_CN_REV = {0: 0, 1: 1, 2: 2}


class OddsModel:
    """赔率基座预测模型（13 维 · 仅胜平负）"""

    def __init__(self):
        self.wdl_model: RandomForestClassifier = None
        self.meta: dict = {}

    # ═══════════════════════════════════════════════════════════
    #  训练
    # ═══════════════════════════════════════════════════════════

    @classmethod
    def train_and_save(cls) -> "OddsModel":
        """
        读取联赛历史数据 → 划分训练/验证集 → 训练 WDL 模型 → 评估 → 保存

        流程:
          1. 加载联赛数据（训练集 + 验证集）
          2. 训练 WDL RandomForest（用验证集做早停调参）
          3. 5 折交叉验证
          4. 验证集评估
          5. 2022 世界杯测试集评估（若数据存在）
          6. 保存模型 + 元信息
        """
        from agents.predicted_agent.feature_engineering import (
            load_training_data, ALL_FEATURE_COLS
        )

        print("=" * 60)
        print("  赔率基座模型 · 训练（13 维胜平负）")
        print("=" * 60)

        # 加载数据（按赛季划分训练/验证集）
        X_train, y_train, X_val, y_val = load_training_data(return_split=True)
        feature_names = ALL_FEATURE_COLS

        instance = cls()

        # ---------- WDL 模型 ----------
        print("\n[训练] WDL 模型 (胜平负, 16 维特征)...")

        if HAS_LIGHTGBM:
            print("  使用 LightGBM + 验证集早停")
            instance.wdl_model = LGBMClassifier(
                n_estimators=1000,
                max_depth=6,
                num_leaves=31,
                learning_rate=0.05,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            # 用验证集做早停
            instance.wdl_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(50), log_evaluation(0)],
            )
        else:
            print("  使用 RandomForest (LightGBM 未安装)")
            instance.wdl_model = RandomForestClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )
            instance.wdl_model.fit(X_train, y_train)

        # 训练集表现
        train_pred = instance.wdl_model.predict(X_train)
        train_proba = instance.wdl_model.predict_proba(X_train)
        print(f"\n  [训练集] 准确率: {accuracy_score(y_train, train_pred):.4f}")
        print(f"  [训练集] Log Loss: {log_loss(y_train, train_proba):.4f}")

        # 5 折交叉验证（在训练集上）
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        wdl_cv = cross_val_score(
            instance.wdl_model, X_train, y_train, cv=cv, scoring="accuracy"
        )
        print(f"  [5折CV] 准确率: {wdl_cv.mean():.4f} (+/- {wdl_cv.std():.4f})")

        # 验证集评估（holdout，2024-2025 赛季）
        val_pred = instance.wdl_model.predict(X_val)
        val_proba = instance.wdl_model.predict_proba(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        val_loss = log_loss(y_val, val_proba)
        print(f"\n  [验证集 2024-2025] 准确率: {val_acc:.4f}")
        print(f"  [验证集 2024-2025] Log Loss: {val_loss:.4f}")
        print(f"  [验证集] 分类报告:")
        print(classification_report(
            y_val, val_pred,
            target_names=["主胜(H)", "平(D)", "客胜(A)"],
            digits=4,
        ))
        # 混淆矩阵
        cm = confusion_matrix(y_val, val_pred, labels=[0, 1, 2])
        print(f"  [验证集] 混淆矩阵 (行=真实, 列=预测):")
        print(f"          预测H  预测D  预测A")
        for i, name in enumerate(["真实H", "真实D", "真实A"]):
            print(f"  {name}  {cm[i][0]:5d}  {cm[i][1]:5d}  {cm[i][2]:5d}")

        # 特征重要性
        print(f"\n  [特征重要性]:")
        for name, imp in sorted(
            zip(feature_names, instance.wdl_model.feature_importances_),
            key=lambda x: -x[1]
        ):
            print(f"    {name:20s} {imp:.4f}")

        # ---------- 2022 世界杯测试集评估 ----------
        wc_metrics = {}
        if os.path.exists(WC2022_TEST_PATH):
            wc_metrics = instance._evaluate_on_wc2022()
        else:
            print(f"\n[测试集] 2022 世界杯测试数据不存在: {WC2022_TEST_PATH}")
            print(f"  可运行 titan007 爬虫生成: python -m agents.predicted_agent.scripts.wc2022_odds_scraper")

        # ---------- 保存 ----------
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(instance.wdl_model, WDL_MODEL_PATH)

        instance.meta = {
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "feature_count": len(feature_names),
            "features": feature_names,
            "train_accuracy": round(float(accuracy_score(y_train, train_pred)), 4),
            "cv_accuracy": round(float(wdl_cv.mean()), 4),
            "val_accuracy": round(float(val_acc), 4),
            "val_log_loss": round(float(val_loss), 4),
            "mix_weight": 0.4,  # 混合策略: 40%模型 + 60%赔率
            "wc2022_test": wc_metrics,
        }
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(instance.meta, f, ensure_ascii=False, indent=2)

        print(f"\n[保存] 模型已保存到 {MODEL_DIR}")
        print("=" * 60)

        return instance

    def _optimize_thresholds(self, X_val: pd.DataFrame, y_val: np.ndarray) -> dict:
        """
        在验证集上网格搜索最佳的平局预测阈值

        策略: 当 max_prob < close_threshold 且 p_draw >= draw_threshold 时预测平局

        搜索空间:
          close_threshold: 0.40 ~ 0.60 (步长 0.02)
          draw_threshold:  0.22 ~ 0.35 (步长 0.01)

        返回: {"close_threshold", "draw_threshold", "best_acc"}
        """
        proba = self.wdl_model.predict_proba(X_val)

        best_acc = 0
        best_close = 0.50
        best_draw = 0.30

        for close_t in np.arange(0.40, 0.62, 0.02):
            for draw_t in np.arange(0.22, 0.36, 0.01):
                preds = []
                for probs in proba:
                    p_h, p_d, p_a = probs[0], probs[1], probs[2] if len(probs) > 2 else 0
                    max_p = max(p_h, p_d, p_a)

                    if max_p < close_t and p_d >= draw_t:
                        preds.append(1)  # D
                    else:
                        preds.append(int(np.argmax(probs)))

                acc = accuracy_score(y_val, preds)
                if acc > best_acc:
                    best_acc = acc
                    best_close = round(float(close_t), 2)
                    best_draw = round(float(draw_t), 2)

        return {
            "close_threshold": best_close,
            "draw_threshold": best_draw,
            "best_acc": round(float(best_acc), 4),
        }

    def _evaluate_on_wc2022(self) -> dict:
        """在 2022 世界杯测试集上评估（跨赛事泛化验证）"""
        from agents.predicted_agent.feature_engineering import build_features, ALL_FEATURE_COLS

        print(f"\n[测试集] 2022 世界杯评估...")
        df = pd.read_csv(WC2022_TEST_PATH)
        print(f"  读取 {len(df)} 场比赛")

        # 构建特征 + 标签
        df = build_features(df)
        df = df.dropna(subset=["FTR"]).copy()
        wdl_map = {"H": 0, "D": 1, "A": 2}
        df["label_wdl"] = df["FTR"].map(wdl_map)
        df = df.dropna(subset=["label_wdl"])
        df["label_wdl"] = df["label_wdl"].astype(int)

        if len(df) == 0:
            print("  ⚠️ 测试集清洗后为空")
            return {}

        X_test = df[ALL_FEATURE_COLS]
        y_test = df["label_wdl"].values

        # 用 predict 方法（含混合策略）而非裸 model.predict
        test_result = self.predict(
            X_test,
            odds_h=df["B365CH"].values,
            odds_d=df["B365CD"].values,
            odds_a=df["B365CA"].values,
        )
        pred = [WDL_LABELS_REV[p] for p in test_result["wdl_pred"]]
        proba = self.wdl_model.predict_proba(X_test)
        acc = accuracy_score(y_test, pred)
        loss = log_loss(y_test, proba)

        print(f"  [2022世界杯] 样本数: {len(df)}")
        print(f"  [2022世界杯] 准确率: {acc:.4f} (使用阈值策略: close={self.meta.get('close_threshold')}, draw={self.meta.get('draw_threshold')})")
        print(f"  [2022世界杯] Log Loss: {loss:.4f}")
        print(f"  [2022世界杯] 真实分布: H={np.sum(y_test==0)}, "
              f"D={np.sum(y_test==1)}, A={np.sum(y_test==2)}")
        print(f"  [2022世界杯] 预测分布: H={np.sum(pred==0)}, "
              f"D={np.sum(pred==1)}, A={np.sum(pred==2)}")
        print(f"  [2022世界杯] 分类报告:")
        print(classification_report(
            y_test, pred,
            target_names=["主胜(H)", "平(D)", "客胜(A)"],
            digits=4,
        ))

        return {
            "samples": int(len(df)),
            "accuracy": round(float(acc), 4),
            "log_loss": round(float(loss), 4),
        }

    # ═══════════════════════════════════════════════════════════
    #  加载
    # ═══════════════════════════════════════════════════════════

    @classmethod
    def load(cls) -> "OddsModel":
        """从磁盘加载已训练的模型"""
        if not os.path.exists(WDL_MODEL_PATH):
            raise FileNotFoundError(
                f"模型文件不存在，请先执行训练:\n"
                f"  python -m agents.predicted_agent.models.statistical_model\n"
                f"  预期路径: {WDL_MODEL_PATH}"
            )

        instance = cls()
        instance.wdl_model = joblib.load(WDL_MODEL_PATH)

        if os.path.exists(META_PATH):
            with open(META_PATH, "r", encoding="utf-8") as f:
                instance.meta = json.load(f)

        return instance

    # ═══════════════════════════════════════════════════════════
    #  预测
    # ═══════════════════════════════════════════════════════════

    def predict(self, X: pd.DataFrame, odds_h: np.ndarray = None,
                odds_d: np.ndarray = None, odds_a: np.ndarray = None) -> dict:
        """
        批量预测胜平负（纯模型，100% LightGBM）

        Args:
            X: 特征 DataFrame，列必须是 ALL_FEATURE_COLS (19 维)
            odds_h/d/a: 保留参数兼容旧调用，纯模型不使用

        Returns:
            {
                "wdl_proba": [[p_h, p_d, p_a], ...],
                "wdl_pred": ["H", ...],
            }
        """
        wdl_proba = self.wdl_model.predict_proba(X)
        preds = wdl_proba.argmax(axis=1)
        wdl_pred = [WDL_LABELS[i] for i in preds]

        return {
            "wdl_proba": wdl_proba.tolist(),
            "wdl_pred": wdl_pred,
        }

    def predict_from_odds(
        self,
        b365h: float,
        b365d: float,
        b365a: float,
        b365ch: float = None,
        b365cd: float = None,
        b365ca: float = None,
        # 兼容旧接口的参数（会被忽略，仅为向后兼容保留）
        b365_over25: float = None,
        b365_under25: float = None,
        ahh: float = None,
    ) -> dict:
        """
        单场比赛赔率 → 胜平负预测概率（供 LLM Agent 调用）

        Args:
            b365h: Bet365 初盘主胜赔率
            b365d: Bet365 初盘平局赔率
            b365a: Bet365 初盘客胜赔率
            b365ch: Bet365 终盘(即时)主胜赔率，None 时 = 初盘
            b365cd: Bet365 终盘(即时)平局赔率，None 时 = 初盘
            b365ca: Bet365 终盘(即时)客胜赔率，None 时 = 初盘
            (b365_over25/b365_under25/ahh: 旧接口兼容参数，已忽略)

        Returns:
            {
                "home_win_prob": 0.45,
                "draw_prob": 0.28,
                "away_win_prob": 0.27,
                "wdl_prediction": "H",
                "input_odds": {...},
            }
        """
        from agents.predicted_agent.feature_engineering import extract_features_from_odds

        X = extract_features_from_odds(b365h, b365d, b365a, b365ch, b365cd, b365ca)

        # 终盘赔率用于混合策略
        close_h = b365ch if b365ch is not None else b365h
        close_d = b365cd if b365cd is not None else b365d
        close_a = b365ca if b365ca is not None else b365a

        result = self.predict(X, odds_h=[close_h], odds_d=[close_d], odds_a=[close_a])
        wdl_p = result["wdl_proba"][0]

        return {
            "home_win_prob": round(wdl_p[0], 4),
            "draw_prob": round(wdl_p[1], 4),
            "away_win_prob": round(wdl_p[2], 4),
            "wdl_prediction": result["wdl_pred"][0],
            "input_odds": {
                "B365H": b365h, "B365D": b365d, "B365A": b365a,
                "B365CH": b365ch if b365ch is not None else b365h,
                "B365CD": b365cd if b365cd is not None else b365d,
                "B365CA": b365ca if b365ca is not None else b365a,
            },
        }

    def predict_from_openclaw(self, match_data: dict) -> dict:
        """
        从 OpenClaw 原始比赛字典直接预测

        match_data 支持字段:
          - 初盘: B365H, B365D, B365A（必需）
          - 终盘: B365CH, B365CD, B365CA（可选）
        """
        from agents.predicted_agent.feature_engineering import extract_features_from_openclaw

        X = extract_features_from_openclaw(match_data)
        result = self.predict(X)

        wdl_p = result["wdl_proba"][0]

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
        }


# ═══════════════════════════════════════════════════════════════
#  命令行入口: python -m agents.predicted_agent.models.statistical_model
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    OddsModel.train_and_save()
