# -*- coding: utf-8 -*-
"""
notifier 推送渲染契约测试。

锁死: llm_analysis（llm_predictor 的输出契约）→ 推送标题/正文的渲染，
以及推送层去重行为。真实发送被 monkeypatch 掉，不产生真实推送。
"""

import notifier


PREDICTION_RESULT = {
    "llm_analysis": {
        "wdl_prediction": {"primary": "H", "secondary": "D", "confidence": "高"},
        "score_predictions": [
            {"score": "2:0", "prob": 0.14, "reason": "主队攻强守稳"},
            {"score": "1:0", "prob": 0.12, "reason": "小胜格局"},
        ],
        "upset_prediction": {"result": "A", "score": "0:1", "reason": "客队反击犀利"},
        "key_points": ["核心前锋伤缺", "主队战意强烈"],
    },
    "trigger_time": "2026-07-02T19:00:00",
}


def _capture_send(monkeypatch):
    sent = {}

    def fake_send(payload, desc="", max_attempts=4):
        sent.update(payload)
        return True

    monkeypatch.setattr(notifier, "_send_with_retry", fake_send)
    return sent


class TestPushPredictionRender:
    def test_full_render(self, monkeypatch):
        sent = _capture_send(monkeypatch)
        ok = notifier.push_prediction("法国", "瑞典", PREDICTION_RESULT,
                                      tier=1, hours_to_kickoff=3.5)
        assert ok is True
        assert "3.5h" in sent["title"] and "法国 vs 瑞典" in sent["title"]
        body = sent["body"]
        assert "胜负: 主胜" in body
        assert "次选: 平局" in body
        assert "置信度: 高" in body
        assert "2:0" in body and "1:0" in body
        assert "爆冷可能: 0:1" in body
        assert "核心前锋伤缺" in body
        assert "N/A" not in body, "出现 N/A 说明字段契约破裂（7/2 事故回归）"

    def test_tier2_title_is_final(self, monkeypatch):
        sent = _capture_send(monkeypatch)
        notifier.push_prediction("英格兰", "加纳", PREDICTION_RESULT,
                                 tier=2, hours_to_kickoff=0.5)
        assert "最终预测" in sent["title"]

    def test_dedup_second_push_skipped(self, monkeypatch):
        _capture_send(monkeypatch)
        assert notifier.push_prediction("巴西", "日本", PREDICTION_RESULT,
                                        tier=1, hours_to_kickoff=2.5) is True
        # 同场同触发点第二次 → 去重拦截
        assert notifier.push_prediction("巴西", "日本", PREDICTION_RESULT,
                                        tier=1, hours_to_kickoff=2.5) is False

    def test_missing_key_points_uses_reasons(self, monkeypatch):
        """key_points 缺失时从 score_predictions/upset 的 reason 提取（最后防线）"""
        sent = _capture_send(monkeypatch)
        result = {
            "llm_analysis": {
                "wdl_prediction": {"primary": "A", "secondary": "D", "confidence": "中"},
                "score_predictions": [{"score": "0:1", "prob": 0.1, "reason": "客队状态更佳"}],
                "key_points": [],
            },
        }
        notifier.push_prediction("埃及", "伊朗", result, tier=1, hours_to_kickoff=1.5)
        assert "客队状态更佳" in sent["body"]


class TestPushAlert:
    def test_alert_payload(self, monkeypatch):
        sent = _capture_send(monkeypatch)
        assert notifier.push_alert("预测服务异常", "HTTP 500") is True
        assert sent["group"] == "系统告警"
        assert sent["level"] == "timeSensitive"
        assert "预测服务异常" in sent["title"]
