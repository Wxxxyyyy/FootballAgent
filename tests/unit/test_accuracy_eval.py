# -*- coding: utf-8 -*-
"""
accuracy_tracker 评估逻辑单测（纯函数，不连库不联网）。
锁死: 预测文件 → 评估行 的字段映射与命中/Brier 计算。
"""

from accuracy_tracker import _argmax_wdl, _brier, _norm_score, evaluate

OUTCOME_H = {"score": "2-0", "hg": 2, "ag": 0, "result": "H"}
OUTCOME_D = {"score": "1-1", "hg": 1, "ag": 1, "result": "D"}


def _pred(llm_analysis: dict) -> dict:
    return {
        "match_id": "2907000",
        "home_team": "法国",
        "away_team": "瑞典",
        "tier": 2,
        "prediction_time": "2026-07-01T04:40:00",
        "_file": "test_pred.json",
        "ml_prediction": {"home_win_prob": 0.6, "draw_prob": 0.25, "away_win_prob": 0.15},
        "monte_carlo": {
            "home_win_prob": 0.62, "draw_prob": 0.22, "away_win_prob": 0.16,
            "most_likely_score": "2:0",
        },
        "llm_analysis": llm_analysis,
    }


class TestEvaluate:
    def test_hit_all(self):
        row = evaluate(_pred({
            "wdl_prediction": {"primary": "H", "secondary": "D", "confidence": "高"},
            "score_predictions": [{"score": "2:0", "prob": 0.14}, {"score": "1:0", "prob": 0.1}],
        }), OUTCOME_H)
        assert row["llm_hit"] == 1
        assert row["secondary_hit"] == 1
        assert row["ml_hit"] == 1 and row["mc_hit"] == 1
        assert row["mc_score_hit"] == 1      # 2:0 归一化为 2-0 后精确命中
        assert row["llm_score_hit"] == 1
        assert row["actual_result"] == "H"

    def test_miss_primary_hit_secondary(self):
        row = evaluate(_pred({
            "wdl_prediction": {"primary": "H", "secondary": "D", "confidence": "中"},
            "score_predictions": [{"score": "2:0", "prob": 0.14}],
        }), OUTCOME_D)
        assert row["llm_hit"] == 0
        assert row["secondary_hit"] == 1
        assert row["mc_score_hit"] == 0

    def test_legacy_format_uses_submodel_result(self):
        """7/2 之前的旧文件没有 wdl_prediction → 取 llm 的 ml_prediction.result"""
        row = evaluate(_pred({
            "ml_prediction": {"result": "H", "confidence": "中", "reason": "模型倾向主胜"},
            "monte_carlo_prediction": {"result": "H", "most_likely_score": "1:0"},
        }), OUTCOME_H)
        assert row["pred_primary"] == "H"
        assert row["llm_hit"] == 1
        assert row["confidence"] == "中"

    def test_brier_computed(self):
        row = evaluate(_pred({
            "wdl_prediction": {"primary": "H", "secondary": "D", "confidence": "高"},
        }), OUTCOME_H)
        # ML: (0.6-1)^2 + 0.25^2 + 0.15^2 = 0.245
        assert abs(row["brier_ml"] - 0.245) < 1e-6
        assert row["brier_mc"] is not None


class TestHelpers:
    def test_argmax(self):
        assert _argmax_wdl({"home_win_prob": 0.2, "draw_prob": 0.5, "away_win_prob": 0.3}) == "D"

    def test_brier_perfect_and_worst(self):
        assert _brier({"home_win_prob": 1.0, "draw_prob": 0, "away_win_prob": 0}, "H") == 0.0
        assert _brier({"home_win_prob": 0, "draw_prob": 0, "away_win_prob": 1.0}, "H") == 2.0
        assert _brier({}, "H") is None

    def test_norm_score(self):
        assert _norm_score("1:0") == "1-0"
        assert _norm_score("1 - 0") == "1-0"
        assert _norm_score(None) == ""
