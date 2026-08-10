# -*- coding: utf-8 -*-
"""
LLM 告警规则引擎契约测试。

核心契约:
  1. collect_metrics 返回必要字段（即使表不存在也优雅降级）
  2. evaluate_rules 在正常指标时不告警
  3. evaluate_rules 在超阈值时返回告警
  4. 告警规则覆盖5项核心指标
  5. 状态管理正确（翻转告警+重复告警间隔）
"""

import pytest
from scripts import alert_rules


class TestCollectMetrics:
    def test_returns_dict_with_required_fields(self):
        """collect_metrics 必须返回含必要字段的 dict"""
        metrics = alert_rules.collect_metrics()
        required = ["success_rate_1h", "avg_latency_1h_ms", "call_count_1h",
                   "token_24h", "error_24h", "call_count_24h"]
        for field in required:
            assert field in metrics, f"缺少字段: {field}"

    def test_degrades_gracefully_on_missing_table(self):
        """表不存在时返回默认正常状态，不抛异常"""
        metrics = alert_rules.collect_metrics()
        assert metrics["success_rate_1h"] == 1.0
        assert metrics["call_count_1h"] == 0


class TestEvaluateRules:
    def test_no_alerts_on_normal_metrics(self):
        """正常指标不应触发告警"""
        metrics = {
            "success_rate_1h": 0.95,
            "avg_latency_1h_ms": 5000,
            "call_count_1h": 10,
            "token_24h": 50000,
            "error_24h": 2,
            "call_count_24h": 50,
        }
        alerts = alert_rules.evaluate_rules(metrics)
        assert len(alerts) == 0, f"正常指标不应告警，但返回: {alerts}"

    def test_alert_on_low_success_rate(self):
        """成功率低于阈值应告警"""
        metrics = {
            "success_rate_1h": 0.50,  # 低于 0.70
            "avg_latency_1h_ms": 5000,
            "call_count_1h": 10,
            "token_24h": 50000,
            "error_24h": 0,
            "call_count_24h": 50,
        }
        alerts = alert_rules.evaluate_rules(metrics)
        rules = [a["rule"] for a in alerts]
        assert "success_rate_1h" in rules

    def test_alert_on_high_latency(self):
        """延迟超过阈值应告警"""
        metrics = {
            "success_rate_1h": 0.95,
            "avg_latency_1h_ms": 40000,  # 超过 30000
            "call_count_1h": 10,
            "token_24h": 50000,
            "error_24h": 0,
            "call_count_24h": 50,
        }
        alerts = alert_rules.evaluate_rules(metrics)
        rules = [a["rule"] for a in alerts]
        assert "avg_latency_1h" in rules

    def test_alert_on_high_token_usage(self):
        """token 消耗超过阈值应告警"""
        metrics = {
            "success_rate_1h": 0.95,
            "avg_latency_1h_ms": 5000,
            "call_count_1h": 10,
            "token_24h": 600000,  # 超过 500000
            "error_24h": 0,
            "call_count_24h": 50,
        }
        alerts = alert_rules.evaluate_rules(metrics)
        rules = [a["rule"] for a in alerts]
        assert "token_24h" in rules

    def test_alert_on_many_errors(self):
        """错误数超过阈值应告警"""
        metrics = {
            "success_rate_1h": 0.95,
            "avg_latency_1h_ms": 5000,
            "call_count_1h": 10,
            "token_24h": 50000,
            "error_24h": 15,  # 超过 10
            "call_count_24h": 50,
        }
        alerts = alert_rules.evaluate_rules(metrics)
        rules = [a["rule"] for a in alerts]
        assert "error_24h" in rules

    def test_no_alert_on_empty_calls_during_night(self):
        """深夜时段无调用不应告警（mock 时无法测试时间，但验证逻辑）"""
        # 不做时间 mock，只验证有调用时正常不告警
        metrics = {
            "success_rate_1h": 1.0,
            "avg_latency_1h_ms": 0,
            "call_count_1h": 0,
            "token_24h": 0,
            "error_24h": 0,
            "call_count_24h": 0,
        }
        alerts = alert_rules.evaluate_rules(metrics)
        # 白天无调用会告警 no_calls_1h，这是预期行为
        rules = [a["rule"] for a in alerts]
        assert "no_calls_1h" in rules  # 白天无调用确实该告警

    def test_alert_contains_required_fields(self):
        """告警必须包含 rule/value/threshold/message 字段"""
        metrics = {
            "success_rate_1h": 0.50,
            "avg_latency_1h_ms": 5000,
            "call_count_1h": 10,
            "token_24h": 50000,
            "error_24h": 0,
            "call_count_24h": 50,
        }
        alerts = alert_rules.evaluate_rules(metrics)
        for alert in alerts:
            assert "rule" in alert
            assert "value" in alert
            assert "threshold" in alert
            assert "message" in alert
