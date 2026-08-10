# -*- coding: utf-8 -*-
"""
P2 三项工程实践的契约测试:
  1. 数据漂移检测 (drift_detector)
  2. A/B 测试框架 (ab_testing)
  3. Prometheus 指标采集 (metrics)
"""

import numpy as np
import pytest

from evaluation import drift_detector
from evaluation import ab_testing
from common import metrics


# ═══════════════════════════════════════════════════════════════
#  1. 数据漂移检测契约
# ═══════════════════════════════════════════════════════════════

class TestDriftDetector:
    def test_psi_no_drift_returns_low_value(self):
        """相同分布的 PSI 应接近 0"""
        data = np.random.RandomState(42).normal(0, 1, 1000)
        psi = drift_detector.calculate_psi(data, data)
        assert psi < 0.1, f"相同分布 PSI 应 < 0.1, 实际: {psi}"

    def test_psi_significant_drift_returns_high_value(self):
        """显著漂移时 PSI 应 >= 0.25"""
        baseline = np.random.RandomState(42).normal(0, 1, 1000)
        # 平移 + 扩大方差 → 显著漂移
        current = np.random.RandomState(42).normal(3, 2, 1000)
        psi = drift_detector.calculate_psi(baseline, current)
        assert psi >= 0.25, f"显著漂移 PSI 应 >= 0.25, 实际: {psi}"

    def test_interpret_psi_correct_interpretation(self):
        """PSI 解释应正确"""
        assert "无" in drift_detector.interpret_psi(0.05)
        assert "轻微" in drift_detector.interpret_psi(0.15)
        assert "显著" in drift_detector.interpret_psi(0.30)

    def test_detect_drift_returns_correct_structure(self):
        """detect_drift 应返回正确的结构"""
        baseline = {
            "features": {
                "feature1": {"mean": 0, "std": 1, "values": list(range(100))},
            }
        }
        current = {"feature1": list(range(100))}
        result = drift_detector.detect_drift(current, baseline)

        assert "has_drift" in result
        assert "drifted_features" in result
        assert "details" in result
        assert "summary" in result

    def test_detect_drift_no_baseline_returns_safe(self):
        """无基线时应安全返回，不抛异常"""
        result = drift_detector.detect_drift({}, baseline={})
        assert result["has_drift"] is False
        assert "无基线" in result["summary"]


# ═══════════════════════════════════════════════════════════════
#  2. A/B 测试框架契约
# ═══════════════════════════════════════════════════════════════

class TestABTesting:
    def test_assign_variant_is_deterministic(self):
        """同一 user_id + experiment 应始终返回相同桶"""
        v1 = ab_testing.assign_variant("user123", "test_exp")
        v2 = ab_testing.assign_variant("user123", "test_exp")
        assert v1 == v2, "分桶应确定性（同一用户始终在同一桶）"

    def test_assign_variant_returns_valid_variant(self):
        """分桶结果必须在 variants 列表中"""
        variants = ["control", "treatment"]
        result = ab_testing.assign_variant("user1", "exp1", variants)
        assert result in variants

    def test_assign_variant_distributes_users(self):
        """多个用户应分布到不同桶（不是全到一个桶）"""
        variants = ["control", "treatment"]
        assignments = [ab_testing.assign_variant(f"user{i}", "dist_exp", variants) for i in range(100)]
        control_count = assignments.count("control")
        treatment_count = assignments.count("treatment")
        # 100 个用户应大致均匀分布（允许 ±20 的偏差）
        assert 30 <= control_count <= 70, f"分桶不均匀: control={control_count}"
        assert 30 <= treatment_count <= 70, f"分桶不均匀: treatment={treatment_count}"

    def test_assign_variant_different_experiments_independent(self):
        """不同实验的分桶应独立（同一用户可能在不同实验的不同桶）"""
        # 不强制要求不同，但要验证不会报错
        v1 = ab_testing.assign_variant("user1", "exp_a")
        v2 = ab_testing.assign_variant("user1", "exp_b")
        assert isinstance(v1, str) and isinstance(v2, str)


# ═══════════════════════════════════════════════════════════════
#  3. Prometheus 指标采集契约
# ═══════════════════════════════════════════════════════════════

class TestMetrics:
    def test_record_http_request_does_not_raise(self):
        """记录 HTTP 请求指标不应抛异常"""
        metrics.record_http_request("GET", "/chat", 200, 0.5)

    def test_record_llm_call_does_not_raise(self):
        """记录 LLM 调用指标不应抛异常"""
        metrics.record_llm_call("kimi", True, 3.5)

    def test_record_prediction_does_not_raise(self):
        """记录预测触发指标不应抛异常"""
        metrics.record_prediction(tier=1)

    def test_set_component_health_does_not_raise(self):
        """设置组件健康状态不应抛异常"""
        metrics.set_component_health("mysql", True)
        metrics.set_component_health("redis", False)

    def test_get_metrics_returns_text(self):
        """get_metrics 应返回文本"""
        text, content_type = metrics.get_metrics()
        assert isinstance(text, (str, bytes))
        assert isinstance(content_type, str)
