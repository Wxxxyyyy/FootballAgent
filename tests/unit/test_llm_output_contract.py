# -*- coding: utf-8 -*-
"""
llm_predictor 输出契约测试。

背景: 2026-07-02 事故 — llm_analysis 缺 wdl_prediction/key_points 字段，
notifier 渲染出"置信度 N/A"。此文件锁死: 无论 LLM 输出什么，
归一化/兜底后必须产出 notifier 可渲染的完整结构。
"""

from agents.predicted_agent.models.llm_predictor import (
    _fallback_from_models,
    _normalize_output,
    _parse_response,
    _to_float,
    _wdl_from_probs,
)

ML_RESULT = {"home_win_prob": 0.60, "draw_prob": 0.25, "away_win_prob": 0.15}
MC_RESULT = {
    "home_win_prob": 0.62, "draw_prob": 0.22, "away_win_prob": 0.16,
    "most_likely_score": "1:0",
    "score_distribution": {"1:0": 0.14, "2:0": 0.13, "1:1": 0.10},
}


def _assert_pushable(out: dict):
    """notifier 渲染所需的最小契约"""
    wdl = out.get("wdl_prediction")
    assert isinstance(wdl, dict), "缺 wdl_prediction"
    assert wdl.get("primary") in ("H", "D", "A")
    assert wdl.get("secondary") in ("H", "D", "A")
    assert wdl["secondary"] != wdl["primary"], "首选次选不能相同"
    assert wdl.get("confidence"), "缺 confidence"
    kps = out.get("key_points")
    assert isinstance(kps, list) and len(kps) >= 1, "缺 key_points"


class TestNormalizeOutput:
    def test_complete_output_untouched(self):
        parsed = {
            "wdl_prediction": {"primary": "H", "secondary": "D", "confidence": "高"},
            "key_points": ["情报1", "情报2"],
        }
        out = _normalize_output(parsed, ML_RESULT, MC_RESULT)
        assert out["wdl_prediction"]["primary"] == "H"
        assert out["key_points"] == ["情报1", "情报2"]
        _assert_pushable(out)

    def test_missing_wdl_synthesized_from_llm_submodels(self):
        """LLM 漏填 wdl_prediction 但给了 ml/mc 判读 → 用其合成（7/2 事故场景）"""
        parsed = {
            "ml_prediction": {"result": "H", "confidence": "中", "reason": "模型倾向主胜"},
            "monte_carlo_prediction": {"result": "H", "most_likely_score": "1:0", "reason": "模拟主胜"},
            "score_predictions": [{"score": "1:0", "prob": 0.14, "reason": "高频比分"}],
        }
        out = _normalize_output(parsed, ML_RESULT, MC_RESULT)
        _assert_pushable(out)
        assert out["wdl_prediction"]["primary"] == "H"

    def test_empty_llm_output_falls_back_to_ml_probs(self):
        out = _normalize_output({}, ML_RESULT, MC_RESULT)
        _assert_pushable(out)
        assert out["wdl_prediction"]["primary"] == "H"  # ML 概率 argmax

    def test_no_data_at_all_does_not_crash(self):
        out = _normalize_output({}, None, None)
        assert isinstance(out, dict)  # 允许无 wdl，但绝不能抛异常


class TestFallbackFromModels:
    def test_fallback_is_pushable(self):
        out = _fallback_from_models(ML_RESULT, MC_RESULT, "connection timeout")
        _assert_pushable(out)
        assert out["degraded"] is True
        assert out["score_predictions"], "兜底应含蒙特卡洛比分"

    def test_fallback_without_any_models(self):
        out = _fallback_from_models(None, None, "boom")
        _assert_pushable(out)  # 即使全无数据也要产出可推送结构


class TestParseResponse:
    def test_plain_json(self):
        assert _parse_response('{"a": 1}') == {"a": 1}

    def test_markdown_fenced_json(self):
        text = '好的，以下是预测:\n```json\n{"wdl_prediction": {"primary": "H"}}\n```'
        assert _parse_response(text)["wdl_prediction"]["primary"] == "H"

    def test_json_embedded_in_text(self):
        text = '分析如下 {"a": {"b": 2}} 完毕'
        assert _parse_response(text) == {"a": {"b": 2}}

    def test_garbage_returns_parse_error(self):
        out = _parse_response("完全不是JSON")
        assert "parse_error" in out
        assert out["overall_analysis"] == "完全不是JSON"


class TestHelpers:
    def test_to_float_percent_string(self):
        assert _to_float("52.1%") == 0.521
        assert _to_float("0.52") == 0.52
        assert _to_float(0.3) == 0.3
        assert _to_float("垃圾", 0.0) == 0.0

    def test_wdl_from_probs_confidence_bands(self):
        p, s, c = _wdl_from_probs({"home_win_prob": 0.70, "draw_prob": 0.20, "away_win_prob": 0.10})
        assert (p, s, c) == ("H", "D", "高")  # gap 0.5 ≥ 0.20
        p, s, c = _wdl_from_probs({"home_win_prob": 0.40, "draw_prob": 0.33, "away_win_prob": 0.27})
        assert c == "低"  # gap 0.07 < 0.08

    def test_wdl_from_probs_no_data(self):
        assert _wdl_from_probs(None) == (None, None, None)
        assert _wdl_from_probs({"error": "x"}) == (None, None, None)
