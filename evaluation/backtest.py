# -*- coding: utf-8 -*-
"""历史回测：按时间排序滚动切窗；模型需实现 predict_batch(DataFrame) 返回预测列。"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

from .metrics import accuracy, brier_score, log_loss


class _PredictModel(Protocol):
    def predict_batch(self, chunk: pd.DataFrame) -> pd.DataFrame | pd.Series | np.ndarray:
        ...


def _wdl_to_idx(series: pd.Series) -> np.ndarray:
    m = {"H": 0, "D": 1, "A": 2}
    return np.array([m.get(str(x), -1) for x in series], dtype=int)


class Backtester:
    """历史表 + 模型；按日期升序每 window_size 场一窗，输出各窗 accuracy 等。"""

    def __init__(
        self,
        history: pd.DataFrame,
        model: _PredictModel,
        *,
        date_col: str = "Date",
        league_col: str = "Div",
        season_col: str | None = None,
        result_col: str = "FTR",
    ) -> None:
        self.df = history.copy()
        self.model = model
        self.date_col = date_col
        self.league_col = league_col
        self.season_col = season_col
        self.result_col = result_col
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df = self.df.sort_values(date_col).reset_index(drop=True)

    def run(
        self,
        *,
        season_range: tuple[str | None, str | None] = (None, None),
        leagues: list[str] | None = None,
        window_size: int = 50,
        prob_cols: tuple[str, str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """season_range 需 season_col；prob_cols 为 (pH,pD,pA) 时追加 Brier / log_loss。"""
        df = self.df
        if leagues is not None:
            df = df[df[self.league_col].isin(leagues)]
        lo, hi = season_range
        if self.season_col and (lo is not None or hi is not None):
            s = df[self.season_col].astype(str)
            if lo is not None:
                df = df.loc[s >= lo]
            if hi is not None:
                s = df[self.season_col].astype(str)
                df = df.loc[s <= hi]
        df = df.reset_index(drop=True)
        rows_out: list[dict[str, Any]] = []
        n = len(df)
        if n == 0 or window_size <= 0:
            return rows_out
        for start in range(0, n, window_size):
            chunk = df.iloc[start : start + window_size]
            y_true = chunk[self.result_col].astype(str)
            pred = self.model.predict_batch(chunk)
            if isinstance(pred, pd.DataFrame):
                y_pred = pred.iloc[:, 0] if pred.shape[1] else pred.squeeze()
            else:
                y_pred = pred
            y_pred = pd.Series(y_pred).astype(str).values
            acc = accuracy(y_true.values, y_pred)
            rec: dict[str, Any] = {
                "window_start": int(start),
                "window_end": int(min(start + window_size, n) - 1),
                "n": int(len(chunk)),
                "date_start": str(chunk[self.date_col].iloc[0]),
                "date_end": str(chunk[self.date_col].iloc[-1]),
                "accuracy": float(acc),
            }
            if prob_cols is not None and all(c in chunk.columns for c in prob_cols):
                ph, pd_, pa = prob_cols
                P = chunk[[ph, pd_, pa]].values.astype(float)
                P = P / P.sum(axis=1, keepdims=True)
                yt = _wdl_to_idx(y_true)
                if (yt >= 0).all():
                    rec["brier_score"] = brier_score(yt, P)
                    rec["log_loss"] = log_loss(yt, P)
            rows_out.append(rec)
        return rows_out
