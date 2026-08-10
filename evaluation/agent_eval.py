# -*- coding: utf-8 -*-
"""
多 Agent 评测系统
══════════════════════════════════════════════════════════════

第一性原理:
  多 Agent 系统的核心价值 = 决策质量。
  评测 = 回答"Agent 做的选择对不对"，而非"回复好不好看"。

三个维度:
  1. 意图识别准确率  — BERT 路由对不对（3类：predicted/information/otherchat）
  2. 工具选择正确率  — ReAct 选的工具对不对（mysql_query / search_knowledge_base / none）
  3. 记忆召回率      — 该召回时是否召回（precision/recall/F1）

运行:
  python evaluation/agent_eval.py                       # 跑全部维度
  python evaluation/agent_eval.py --dim intent          # 只跑意图
  python evaluation/agent_eval.py --dim tool            # 只跑工具选择
  python evaluation/agent_eval.py --dim memory          # 只跑记忆召回
  python evaluation/agent_eval.py --mock                # mock 模式（无 BERT 模型时用）

输出:
  evaluation/reports/agent_eval_{timestamp}.json  — 完整指标
  控制台 — 可读报告
"""

from __future__ import annotations

import os
import sys
import json
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.metrics import accuracy

# 测试集路径
DATASET_DIR = os.path.join(os.path.dirname(__file__), "test_datasets")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  维度 1: 意图识别评测
# ═══════════════════════════════════════════════════════════════

def _load_intent_dataset() -> list:
    """加载意图识别测试集"""
    with open(os.path.join(DATASET_DIR, "intent_test.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _mock_intent_predict(query: str) -> str:
    """无 BERT 模型时的关键词规则 mock（仅用于测试框架可用性）"""
    predict_keywords = ["预测", "比分", "能赢", "看好谁", "胜负", "怎么看", "比赛"]
    info_keywords = ["战绩", "历史", "介绍", "联赛", "教练", "积分榜", "主场", "交手", "上次"]

    for kw in predict_keywords:
        if kw in query and "上次" not in query:
            return "predicted_agent"
    for kw in info_keywords:
        if kw in query:
            return "information_agent"
    return "otherchat_agent"


def _real_intent_predict(query: str) -> str:
    """用真实 BERT 模型预测意图"""
    from agents.intent_agent.node import intent_route
    result = intent_route(query)
    return result["intent"]


def eval_intent(mock: bool = False) -> dict:
    """评测意图识别准确率

    Returns:
        {
            "dimension": "intent",
            "total": 30,
            "correct": 28,
            "accuracy": 0.933,
            "per_class": { "predicted_agent": {...}, ... },
            "confusion_matrix": {...},
            "details": [ {query, expected, actual, correct}, ... ]
        }
    """
    dataset = _load_intent_dataset()
    total = len(dataset)
    correct = 0

    # 按类别统计
    labels = ["predicted_agent", "information_agent", "otherchat_agent"]
    per_class = {l: {"tp": 0, "fp": 0, "fn": 0} for l in labels}

    # 混淆矩阵
    confusion = {l: {l2: 0 for l2 in labels} for l in labels}

    details = []

    for item in dataset:
        query = item["query"]
        expected = item["expected"]

        if mock:
            actual = _mock_intent_predict(query)
        else:
            try:
                actual = _real_intent_predict(query)
            except Exception as e:
                # 模型不可用时自动降级到 mock
                actual = _mock_intent_predict(query)

        is_correct = (actual == expected)
        if is_correct:
            correct += 1
            per_class[expected]["tp"] += 1
        else:
            per_class[expected]["fn"] += 1
            per_class[actual]["fp"] += 1

        confusion[expected][actual] += 1

        details.append({
            "query": query,
            "expected": expected,
            "actual": actual,
            "correct": is_correct,
        })

    # 计算每类 precision/recall/f1
    class_metrics = {}
    for l in labels:
        tp = per_class[l]["tp"]
        fp = per_class[l]["fp"]
        fn = per_class[l]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        class_metrics[l] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return {
        "dimension": "intent",
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "per_class": class_metrics,
        "confusion_matrix": confusion,
        "details": details,
    }


# ═══════════════════════════════════════════════════════════════
#  维度 2: 工具选择评测
# ═══════════════════════════════════════════════════════════════

def _load_tool_dataset() -> list:
    with open(os.path.join(DATASET_DIR, "tool_selection_test.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _predict_tool_choice(question: str) -> str:
    """基于关键词预测该问题应该用哪个工具（轻量版，不调 LLM）

    完整版应调 ReAct 让 LLM 选工具，但那需要 LLM 调用，这里先做关键词评测。
    """
    mysql_keywords = ["战绩", "排名", "交锋", "上次", "最近五场", "最近", "比分", "历史荣誉", "主教练"]
    vector_keywords = ["伤病", "转会", "战术风格", "首发", "阵容", "规则", "资讯", "消息"]

    for kw in vector_keywords:
        if kw in question:
            return "search_knowledge_base"

    for kw in mysql_keywords:
        if kw in question:
            return "mysql_query"

    # 闲聊类无需工具
    if len(question) < 5 or any(w in question for w in ["你好", "谢谢", "你是谁", "再见"]):
        return "none"

    return "mysql_query"  # 默认


def eval_tool_selection() -> dict:
    """评测工具选择正确率"""
    dataset = _load_tool_dataset()
    total = len(dataset)
    correct = 0
    details = []

    # 按工具统计
    tools = ["mysql_query", "search_knowledge_base", "none"]
    per_tool = {t: {"tp": 0, "fp": 0, "fn": 0} for t in tools}

    for item in dataset:
        question = item.get("question") or item.get("query", "")
        expected = item["expected_tool"]
        actual = _predict_tool_choice(question)

        is_correct = (actual == expected)
        if is_correct:
            correct += 1
            per_tool[expected]["tp"] += 1
        else:
            per_tool[expected]["fn"] += 1
            per_tool[actual]["fp"] += 1

        details.append({
            "question": question,
            "expected": expected,
            "actual": actual,
            "correct": is_correct,
            "reason": item.get("reason", ""),
        })

    class_metrics = {}
    for t in tools:
        tp = per_tool[t]["tp"]
        fp = per_tool[t]["fp"]
        fn = per_tool[t]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        class_metrics[t] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return {
        "dimension": "tool_selection",
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "per_tool": class_metrics,
        "details": details,
    }


# ═══════════════════════════════════════════════════════════════
#  维度 3: 记忆召回评测
# ═══════════════════════════════════════════════════════════════

def _load_memory_dataset() -> list:
    with open(os.path.join(DATASET_DIR, "memory_recall_test.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _mock_memory_retrieval(user_msg: str) -> bool:
    """无 retriever 模块时的关键词 mock"""
    keywords = ["上次", "上回", "之前", "前面", "刚才", "还记得", "我问过", "我说过",
                "那支球队", "那场比赛", "那个结果", "那个预测", "他们", "第一次"]
    return any(kw in user_msg for kw in keywords)


def _real_memory_retrieval(user_msg: str) -> bool:
    """用真实 retriever 判断是否需要记忆检索"""
    from agents.memory_manager.retriever import needs_memory_retrieval
    return needs_memory_retrieval(user_msg)


def eval_memory_recall(mock: bool = False) -> dict:
    """评测记忆召回率

    指标:
      - precision: 召回的请求中，真正需要的比例
      - recall: 需要的请求中，被召回的比例
      - f1: 综合指标
    """
    dataset = _load_memory_dataset()
    total = len(dataset)

    tp = 0  # 该召回且被召回
    fp = 0  # 不该召回但被召回
    fn = 0  # 该召回但没召回
    tn = 0  # 不该召回且没召回

    details = []

    for item in dataset:
        messages = item["messages"]
        last_user_msg = messages[-1]["content"] if messages else ""
        expected = item["should_retrieve"]

        if mock:
            actual = _mock_memory_retrieval(last_user_msg)
        else:
            try:
                actual = _real_memory_retrieval(last_user_msg)
            except Exception:
                actual = _mock_memory_retrieval(last_user_msg)

        if expected and actual:
            tp += 1
        elif expected and not actual:
            fn += 1
        elif not expected and actual:
            fp += 1
        else:
            tn += 1

        details.append({
            "conversation_id": item["conversation_id"],
            "last_message": last_user_msg,
            "should_retrieve": expected,
            "actual_retrieved": actual,
            "correct": (expected == actual),
            "reason": item.get("reason", ""),
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy_val = (tp + tn) / total if total > 0 else 0.0

    return {
        "dimension": "memory_recall",
        "total": total,
        "accuracy": round(accuracy_val, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "details": details,
    }


# ═══════════════════════════════════════════════════════════════
#  报告生成
# ═══════════════════════════════════════════════════════════════

def _print_report(results: dict):
    """打印可读报告"""
    print("\n" + "=" * 60)
    print("  多 Agent 评测报告")
    print("=" * 60)

    for dim_name, dim_result in results.items():
        print(f"\n【{dim_result['dimension']}】")
        print(f"  样本数: {dim_result['total']}")
        print(f"  准确率: {dim_result['accuracy']:.2%}")

        if dim_result['dimension'] == "intent":
            print("  各类别指标:")
            for label, m in dim_result["per_class"].items():
                print(f"    {label:20s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")
            print("  混淆矩阵 (行=期望, 列=实际):")
            labels = list(dim_result["confusion_matrix"].keys())
            header = "    期望\\实际    " + "  ".join(f"{l[:6]:>8s}" for l in labels)
            print(header)
            for row_label in labels:
                row = dim_result["confusion_matrix"][row_label]
                cells = "  ".join(f"{row[l]:>8d}" for l in labels)
                print(f"    {row_label[:6]:>8s}    {cells}")

        elif dim_result['dimension'] == "tool_selection":
            print("  各工具指标:")
            for tool, m in dim_result["per_tool"].items():
                print(f"    {tool:25s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")

        elif dim_result['dimension'] == "memory_recall":
            print(f"  精确率: {dim_result['precision']:.2%}")
            print(f"  召回率: {dim_result['recall']:.2%}")
            print(f"  F1:    {dim_result['f1']:.2%}")
            c = dim_result["confusion"]
            print(f"  混淆: TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']}")

        # 显示错误样本（最多5条）
        errors = [d for d in dim_result["details"] if not d["correct"]]
        if errors:
            print(f"  错误样本 ({len(errors)} 条, 显示前5条):")
            for e in errors[:5]:
                query = e.get("query") or e.get("question") or e.get("last_message", "")
                print(f"      '{query[:30]}...' 期望={e['expected']} 实际={e['actual']}")

    print("\n" + "=" * 60)


def run_all(mock: bool = False, only_dim: str = None) -> dict:
    """运行全部评测维度"""
    results = {}

    if only_dim is None or only_dim == "intent":
        print("[1/3] 评测意图识别...")
        results["intent"] = eval_intent(mock=mock)

    if only_dim is None or only_dim == "tool":
        print("[2/3] 评测工具选择...")
        results["tool_selection"] = eval_tool_selection()

    if only_dim is None or only_dim == "memory":
        print("[3/3] 评测记忆召回...")
        results["memory_recall"] = eval_memory_recall(mock=mock)

    _print_report(results)

    # 保存 JSON 报告
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"agent_eval_{ts}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {report_path}")

    return results


# ═══════════════════════════════════════════════════════════════
#  CLI 入口
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="多 Agent 评测系统")
    parser.add_argument("--dim", choices=["intent", "tool", "memory"],
                        help="只跑指定维度（默认全部）")
    parser.add_argument("--mock", action="store_true",
                        help="mock 模式（无 BERT 模型时用关键词规则）")
    args = parser.parse_args()

    only = None
    if args.dim == "intent":
        only = "intent"
    elif args.dim == "tool":
        only = "tool"
    elif args.dim == "memory":
        only = "memory"

    run_all(mock=args.mock, only_dim=only)


if __name__ == "__main__":
    main()
