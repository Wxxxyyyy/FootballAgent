# -*- coding: utf-8 -*-
"""
========================================
  BERT 意图识别 · 微调训练骨架脚本
========================================

功能说明：
  1. 加载本地 bert-base-chinese 预训练模型和 Tokenizer
  2. 定义意图分类标签体系
  3. 构建 Dataset & DataLoader
  4. 使用 HuggingFace Trainer 进行微调训练
  5. 保存微调后的模型，供后续意图识别 Agent 使用

使用方式：
  cd footballAgent
  python -m intent.train

目录结构：
  intent/
  ├── __init__.py           # 模块入口
  ├── train.py              # 本文件 - 微调训练骨架
  ├── predict.py            # 推理预测（后续开发）
  ├── bert-base-chinese/    # 预训练模型（本地下载）
  ├── data/                 # 训练数据
  │   ├── train.json        # 训练集（后续准备）
  │   └── val.json          # 验证集（后续准备）
  └── output/               # 微调输出目录
      └── best_model/       # 最优模型检查点
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ═══════════════════════════════════════════════════════════════
#  1. 路径 & 常量配置
# ═══════════════════════════════════════════════════════════════

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 本地 BERT 模型路径（下载后存放在此）
MODEL_PATH = os.path.join(PROJECT_ROOT, "intent", "bert-base-chinese")

# 训练数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "intent", "data")

# 微调后模型输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "intent", "output")

# ─── 意图标签定义 ─────────────────────────────────────────────
# 根据 Football Agent 业务场景，定义以下 3 类意图：
#   - predicted_agent:    比分/赛果预测类（赛前预测、盘口分析等）
#   - information_agent:  信息查询类（比赛结果、球队资料、赛程等）
#   - otherchat_agent:    闲聊/其他类（与足球业务无直接关系的对话）

INTENT_LABELS = [
    "predicted_agent",    # 预测类
    "information_agent",  # 信息查询类
    "otherchat_agent",    # 闲聊/其他
]

# 标签 → 数字索引 映射
LABEL2ID = {label: idx for idx, label in enumerate(INTENT_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(INTENT_LABELS)}
NUM_LABELS = len(INTENT_LABELS)

# ─── 训练超参数 ───────────────────────────────────────────────
MAX_SEQ_LENGTH = 128        # 输入序列最大长度（意图识别文本通常较短）
BATCH_SIZE = 16             # 批大小
LEARNING_RATE = 2e-5        # 学习率
NUM_EPOCHS = 10             # 最大训练轮数
WARMUP_RATIO = 0.1          # 学习率预热比例
WEIGHT_DECAY = 0.01         # 权重衰减
EARLY_STOPPING_PATIENCE = 3 # 早停耐心值


# ═══════════════════════════════════════════════════════════════
#  2. 自定义 Dataset
# ═══════════════════════════════════════════════════════════════

class IntentDataset(Dataset):
    """
    意图识别数据集。

    期望数据格式（JSON 文件）：
    [
        {"id": 1, "text": "这周末皇马打巴萨你觉得谁能赢？", "label_name": "predicted_agent"},
        {"id": 2, "text": "曼联的历史底蕴",              "label_name": "information_agent"},
        {"id": 3, "text": "你好呀",                       "label_name": "otherchat_agent"},
        ...
    ]
    """

    def __init__(self, filepath: str, tokenizer, max_length: int = MAX_SEQ_LENGTH):
        """
        Args:
            filepath:   JSON 数据文件路径
            tokenizer:  BERT Tokenizer
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 读取数据
        with open(filepath, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        print(f"  [数据集] {os.path.basename(filepath)}: {len(self.data)} 条样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = LABEL2ID[item["label_name"]]

        # 使用 Tokenizer 编码文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════
#  3. 评估指标
# ═══════════════════════════════════════════════════════════════

def compute_metrics(eval_pred):
    """
    计算分类评估指标，供 Trainer 在验证阶段调用。

    Returns:
        dict: 包含 accuracy 和 macro-f1
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")

    return {
        "accuracy": acc,
        "f1_macro": f1,
    }


# ═══════════════════════════════════════════════════════════════
#  4. 主训练流程
# ═══════════════════════════════════════════════════════════════

def load_model_and_tokenizer():
    """
    从本地路径加载 bert-base-chinese 模型和 Tokenizer。
    模型顶部添加一个分类头（num_labels 个输出）用于意图分类。
    """
    print(f"[模型路径] {MODEL_PATH}")

    # 加载 Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    print(f"[✓] Tokenizer 加载完成, 词表大小: {tokenizer.vocab_size}")

    # 加载 BERT + 序列分类头
    model = BertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=NUM_LABELS,      # 意图类别数
        id2label=ID2LABEL,          # 索引→标签
        label2id=LABEL2ID,          # 标签→索引
    )
    print(f"[✓] 模型加载完成, 分类头输出维度: {NUM_LABELS}")
    print(f"    标签体系: {INTENT_LABELS}")

    return model, tokenizer


def train():
    """完整的微调训练流程"""

    print("=" * 60)
    print("  Football Agent · BERT 意图识别微调训练")
    print("=" * 60)

    # ---- Step 1: 加载模型 ----
    print("\n[1/4] 加载预训练模型")
    model, tokenizer = load_model_and_tokenizer()

    # ---- Step 2: 加载数据集 ----
    print("\n[2/4] 加载训练数据")
    train_file = os.path.join(DATA_DIR, "train.json")
    val_file = os.path.join(DATA_DIR, "val.json")

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print(f"\n[!] 训练数据尚未准备！请在以下目录放置数据文件：")
        print(f"    训练集: {train_file}")
        print(f"    验证集: {val_file}")
        print(f"\n    数据格式示例：")
        print(f'    [')
        print(f'      {{"id": 1, "text": "周末皇马打巴萨谁能赢", "label_name": "predicted_agent"}},')
        print(f'      {{"id": 2, "text": "曼联上轮比分多少",     "label_name": "information_agent"}},')
        print(f'      {{"id": 3, "text": "你好",               "label_name": "otherchat_agent"}}')
        print(f'    ]')
        print(f"\n    标签可选值: {INTENT_LABELS}")
        print(f"\n[提示] 数据准备好后重新运行: python -m intent.train")
        return

    train_dataset = IntentDataset(train_file, tokenizer)
    val_dataset = IntentDataset(val_file, tokenizer)

    # ---- Step 3: 配置 Trainer ----
    print("\n[3/4] 配置 Trainer")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,                          # 输出目录
        num_train_epochs=NUM_EPOCHS,                    # 训练轮数
        per_device_train_batch_size=BATCH_SIZE,         # 训练批大小
        per_device_eval_batch_size=BATCH_SIZE * 2,      # 评估批大小（可以更大）
        learning_rate=LEARNING_RATE,                    # 学习率
        warmup_ratio=WARMUP_RATIO,                      # 预热比例
        weight_decay=WEIGHT_DECAY,                      # 权重衰减
        eval_strategy="epoch",                          # 每个 epoch 评估一次
        save_strategy="epoch",                          # 每个 epoch 保存一次
        load_best_model_at_end=True,                    # 训练结束时加载最优模型
        metric_for_best_model="f1_macro",               # 以 F1 为最优标准
        greater_is_better=True,                         # F1 越大越好
        save_total_limit=3,                             # 最多保留 3 个检查点
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),   # 日志目录
        logging_steps=10,                               # 每 10 步记录一次
        report_to="none",                               # 不上报到外部平台
        fp16=torch.cuda.is_available(),                 # 有 GPU 则用混合精度
        seed=42,                                        # 随机种子
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
        ],
    )

    print(f"  设备: {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"  训练轮数: {NUM_EPOCHS}, 批大小: {BATCH_SIZE}, 学习率: {LEARNING_RATE}")
    print(f"  早停耐心: {EARLY_STOPPING_PATIENCE} 轮")

    # ---- Step 4: 开始训练 ----
    print("\n[4/4] 开始训练")
    trainer.train()

    # ---- 保存最优模型 ----
    best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"\n[✓] 最优模型已保存到: {best_model_dir}")

    # ---- 最终评估 ----
    print("\n[评估] 在验证集上的最终表现：")
    eval_result = trainer.evaluate()
    for key, val in eval_result.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    # ---- 详细分类报告 ----
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids
    report = classification_report(
        true_labels, pred_labels,
        target_names=INTENT_LABELS,
        digits=4,
    )
    print(f"\n[分类报告]\n{report}")

    print("\n[✓] 训练完成！")
    print(f"    微调模型路径: {best_model_dir}")
    print(f"    后续在 intent/predict.py 中加载此模型即可进行意图预测")


# ═══════════════════════════════════════════════════════════════
#  5. 入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()

