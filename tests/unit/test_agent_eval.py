# -*- coding: utf-8 -*-
"""
多 Agent 评测系统契约测试。

核心契约:
  1. 三个维度的测试集格式正确（JSON 数组、必要字段）
  2. 评测函数能跑完不报错
  3. 评测结果含必要字段（accuracy/details/confusion_matrix）
  4. accuracy 在 [0,1] 合理范围
  5. mock 模式和真实模式接口一致
"""

import os
import json
import pytest

from evaluation import agent_eval


# ═══════════════════════════════════════════════════════════════
#  测试集格式契约
# ═══════════════════════════════════════════════════════════════

class TestDatasetFormat:
    def test_intent_dataset_valid(self):
        """意图测试集必须是有效 JSON 数组且每条含 query + expected"""
        data = agent_eval._load_intent_dataset()
        assert isinstance(data, list) and len(data) >= 20, "意图测试集至少20条"

        valid_labels = {"predicted_agent", "information_agent", "otherchat_agent"}
        for item in data:
            assert "query" in item and item["query"], "每条必须有 query"
            assert item["expected"] in valid_labels, f"expected 必须是合法标签: {item['expected']}"

    def test_tool_dataset_valid(self):
        """工具选择测试集必须含 question + expected_tool"""
        data = agent_eval._load_tool_dataset()
        assert isinstance(data, list) and len(data) >= 10

        valid_tools = {"mysql_query", "search_knowledge_base", "none"}
        for item in data:
            q = item.get("question") or item.get("query")
            assert q, "每条必须有 question 或 query"
            assert item["expected_tool"] in valid_tools

    def test_memory_dataset_valid(self):
        """记忆测试集必须含 messages + should_retrieve"""
        data = agent_eval._load_memory_dataset()
        assert isinstance(data, list) and len(data) >= 5

        for item in data:
            assert isinstance(item["messages"], list) and len(item["messages"]) >= 1
            assert isinstance(item["should_retrieve"], bool)


# ═══════════════════════════════════════════════════════════════
#  评测函数行为契约
# ═══════════════════════════════════════════════════════════════

class TestEvalFunctions:
    def test_eval_intent_returns_valid_result(self):
        """eval_intent 必须返回含 accuracy/details 的结果"""
        result = agent_eval.eval_intent(mock=True)

        assert result["dimension"] == "intent"
        assert result["total"] > 0
        assert 0 <= result["accuracy"] <= 1, "accuracy 必须在 [0,1]"
        assert "details" in result and len(result["details"]) == result["total"]
        assert "per_class" in result
        assert "confusion_matrix" in result

    def test_eval_tool_selection_returns_valid_result(self):
        """eval_tool_selection 必须返回完整结果"""
        result = agent_eval.eval_tool_selection()

        assert result["dimension"] == "tool_selection"
        assert result["total"] > 0
        assert 0 <= result["accuracy"] <= 1
        assert "per_tool" in result
        assert len(result["details"]) == result["total"]

    def test_eval_memory_recall_returns_valid_result(self):
        """eval_memory_recall 必须返回 precision/recall/f1"""
        result = agent_eval.eval_memory_recall(mock=True)

        assert result["dimension"] == "memory_recall"
        assert result["total"] > 0
        assert 0 <= result["accuracy"] <= 1
        assert 0 <= result["precision"] <= 1
        assert 0 <= result["recall"] <= 1
        assert 0 <= result["f1"] <= 1
        assert "confusion" in result

        c = result["confusion"]
        assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == result["total"], "混淆矩阵总和必须等于样本数"

    def test_run_all_returns_three_dimensions(self):
        """run_all 必须返回三个维度"""
        results = agent_eval.run_all(mock=True)

        assert len(results) == 3
        assert "intent" in results
        assert "tool_selection" in results
        assert "memory_recall" in results


# ═══════════════════════════════════════════════════════════════
#  降级契约
# ═══════════════════════════════════════════════════════════════

class TestDegradation:
    def test_mock_predict_returns_valid_label(self):
        """mock 意图预测必须返回合法标签"""
        labels = {"predicted_agent", "information_agent", "otherchat_agent"}

        test_cases = [
            "预测今晚的比赛",
            "曼联的战绩",
            "你好",
        ]
        for q in test_cases:
            result = agent_eval._mock_intent_predict(q)
            assert result in labels, f"mock 预测返回非法标签: {result}"

    def test_mock_memory_returns_bool(self):
        """mock 记忆检索必须返回布尔值"""
        assert isinstance(agent_eval._mock_memory_retrieval("上次说的"), bool)
        assert isinstance(agent_eval._mock_memory_retrieval("你好"), bool)

    def test_intent_eval_degrades_gracefully(self):
        """BERT 模型不可用时自动降级到 mock，不抛异常"""
        # 不传 mock=True，但 BERT 不存在时应自动降级
        result = agent_eval.eval_intent(mock=False)
        assert 0 <= result["accuracy"] <= 1
