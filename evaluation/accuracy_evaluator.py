# -*- coding: utf-8 -*-
"""
预测准确率评估：胜平负、精确比分、大小球；支持按联赛/赛季分组。
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _hit_rate(mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return 0.0
    return float(np.mean(m))


class AccuracyEvaluator:
    """
    使用长度 n 的平行数组描述比赛；分组键 league / season 可为 None（不分组）。
    """

    def __init__(
        self,
        y_true_wdl: Sequence[str],
        y_pred_wdl: Sequence[str],
        y_true_score: Sequence[str] | None = None,
        y_pred_score: Sequence[str] | None = None,
        y_true_ou: Sequence[str] | None = None,
        y_pred_ou: Sequence[str] | None = None,
        *,
        league: Sequence[Any] | None = None,
        season: Sequence[Any] | None = None,
    ) -> None:
        self._wdl_t = np.asarray(y_true_wdl)
        self._wdl_p = np.asarray(y_pred_wdl)
        self._sc_t = None if y_true_score is None else np.asarray(y_true_score)
        self._sc_p = None if y_pred_score is None else np.asarray(y_pred_score)
        self._ou_t = None if y_true_ou is None else np.asarray(y_true_ou)
        self._ou_p = None if y_pred_ou is None else np.asarray(y_pred_ou)
        self._league = np.asarray(league) if league is not None else None
        self._season = np.asarray(season) if season is not None else None

    def _rows(self, idx: slice | np.ndarray) -> AccuracyEvaluator:
        return AccuracyEvaluator(
            self._wdl_t[idx],
            self._wdl_p[idx],
            self._sc_t[idx] if self._sc_t is not None else None,
            self._sc_p[idx] if self._sc_p is not None else None,
            self._ou_t[idx] if self._ou_t is not None else None,
            self._ou_p[idx] if self._ou_p is not None else None,
            league=self._league[idx] if self._league is not None else None,
            season=self._season[idx] if self._season is not None else None,
        )

    def evaluate(self) -> dict[str, Any]:
        """返回总体指标 dict。"""
        n = len(self._wdl_t)
        out: dict[str, Any] = {
            "n_matches": int(n),
            "wdl_hit_rate": _hit_rate(self._wdl_t == self._wdl_p),
        }
        if self._sc_t is not None and self._sc_p is not None:
            out["score_hit_rate"] = _hit_rate(self._sc_t == self._sc_p)
        if self._ou_t is not None and self._ou_p is not None:
            out["over_under_hit_rate"] = _hit_rate(self._ou_t == self._ou_p)
        return out

    def evaluate_by_groups(self) -> dict[str, dict[str, dict[str, float]]]:
        """按联赛、赛季分组；键为分组值字符串，值为 evaluate() 中的标量指标。"""
        result: dict[str, dict[str, dict[str, float]]] = {
            "by_league": {},
            "by_season": {},
        }
        if self._league is not None:
            for g in np.unique(self._league):
                mask = self._league == g
                sub = self._rows(mask)
                result["by_league"][str(g)] = {k: v for k, v in sub.evaluate().items() if isinstance(v, (int, float))}
        if self._season is not None:
            for g in np.unique(self._season):
                mask = self._season == g
                sub = self._rows(mask)
                result["by_season"][str(g)] = {k: v for k, v in sub.evaluate().items() if isinstance(v, (int, float))}
        return result
