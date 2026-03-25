# -*- coding: utf-8 -*-
"""
BERT 意图识别 · 推理预测 & 测试集评估
- 加载微调后的模型
- 对用户输入文本进行意图分类
- 供意图识别 Agent 调用
- 支持对测试集进行 Precision / Recall / F1 评估
"""

import os
import json
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 微调后的模型路径
BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "intent", "output", "best_model")

# 意图标签（与 train.py 保持一致）
INTENT_LABELS = ["predicted_agent", "information_agent", "otherchat_agent"]


class IntentClassifier:
    """意图分类器：加载微调后的 BERT 模型，预测用户输入的意图类别"""

    def __init__(self, model_path: str = BEST_MODEL_PATH):
        """
        Args:
            model_path: 微调后模型的本地路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"[✓] 意图分类器已加载, 设备: {self.device}")

    def predict(self, text: str) -> dict:
        """
        预测单条文本的意图。

        Args:
            text: 用户输入文本

        Returns:
            dict: {"intent": 意图标签, "confidence": 置信度, "all_scores": 各类别概率}
        """
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

        all_scores = {
            INTENT_LABELS[i]: round(probs[i].item(), 4)
            for i in range(len(INTENT_LABELS))
        }

        return {
            "intent": INTENT_LABELS[pred_idx],
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """批量预测多条文本"""
        return [self.predict(text) for text in texts]

    def evaluate(self, test_data_path: str | None = None) -> dict:
        """
        在测试集上进行评估，输出 Precision / Recall / F1 及混淆矩阵。

        Args:
            test_data_path: 测试集 JSON 文件路径，默认使用 intent/data/text.json

        Returns:
            dict: {
                "report": classification_report 字符串,
                "report_dict": 各类别及整体的 precision/recall/f1 字典,
                "confusion_matrix": 混淆矩阵 (list of list),
                "accuracy": 整体准确率,
                "error_samples": 预测错误的样本列表,
            }
        """
        # ── 默认测试集路径 ──
        if test_data_path is None:
            test_data_path = os.path.join(PROJECT_ROOT, "intent", "data", "text.json")

        # ── 加载测试数据 ──
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        print(f"\n[评估] 加载测试集: {test_data_path}")
        print(f"[评估] 测试样本数: {len(test_data)}")

        # ── 逐条预测 ──
        y_true = []          # 真实标签
        y_pred = []          # 预测标签
        error_samples = []   # 预测错误的样本

        for item in test_data:
            text = item["text"]
            true_label = item["label_name"]
            result = self.predict(text)
            pred_label = result["intent"]
            confidence = result["confidence"]

            y_true.append(true_label)
            y_pred.append(pred_label)

            if pred_label != true_label:
                error_samples.append({
                    "id": item.get("id", ""),
                    "text": text,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": confidence,
                })

        # ── 生成评估报告 ──
        report_str = classification_report(
            y_true, y_pred,
            labels=INTENT_LABELS,
            target_names=INTENT_LABELS,
            digits=4,
            zero_division=0,
        )

        report_dict = classification_report(
            y_true, y_pred,
            labels=INTENT_LABELS,
            target_names=INTENT_LABELS,
            digits=4,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_true, y_pred, labels=INTENT_LABELS)
        accuracy = report_dict.get("accuracy", 0.0)

        return {
            "report": report_str,
            "report_dict": report_dict,
            "confusion_matrix": cm.tolist(),
            "accuracy": accuracy,
            "error_samples": error_samples,
        }


# ─── 命令行测试 & 评估 ───────────────────────────────────────
if __name__ == "__main__":
    classifier = IntentClassifier()

    # ═══════════════════════════════════════════════════════════
    # 1. 单条预测示例
    # ═══════════════════════════════════════════════════════════
    test_texts = [
        "你好呀",
        "曼联上轮赢了吗",
        "介绍一下巴塞罗那的历史",
        "帮我预测明天利物浦对阿森纳的比分",
        "今天天气怎么样",
        "皇马和巴萨上赛季交手几次",
    ]

    print("\n[单条预测示例]")
    for text in test_texts:
        result = classifier.predict(text)
        print(f"  '{text}'")
        print(f"    → 意图: {result['intent']}  置信度: {result['confidence']}")
        print()

    # ═══════════════════════════════════════════════════════════
    # 2. 测试集评估（Precision / Recall / F1 / 混淆矩阵）
    # ═══════════════════════════════════════════════════════════
    print("=" * 70)
    print("  测试集评估（intent/data/text.json）")
    print("=" * 70)

    eval_result = classifier.evaluate()

    # 输出分类报告
    print("\n[分类报告]")
    print(eval_result["report"])

    # 输出混淆矩阵
    print("[混淆矩阵]")
    print(f"  标签顺序: {INTENT_LABELS}")
    cm = eval_result["confusion_matrix"]
    # 表头
    header = "  " + " " * 20 + "  ".join(f"{label[:8]:>10}" for label in INTENT_LABELS)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{val:>10}" for val in row)
        print(f"  {INTENT_LABELS[i]:<20}{row_str}")
    print()

    # 输出整体准确率
    print(f"[整体准确率] {eval_result['accuracy']:.4f}")

    # 输出预测错误的样本
    errors = eval_result["error_samples"]
    print(f"\n[预测错误样本] 共 {len(errors)} 条")
    if errors:
        for e in errors:
            print(f"  ID={e['id']}  真实={e['true_label']}  预测={e['pred_label']}  "
                  f"置信度={e['confidence']:.4f}")
            print(f"    文本: '{e['text']}'")
    else:
        print("  🎉 全部预测正确！")

    print("\n[评估完成]")

