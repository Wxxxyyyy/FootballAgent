# -*- coding: utf-8 -*-
"""
盈利率评估：固定注额与凯利注额两种路径，输出 ROI、最大回撤、夏普比率。
"""
from __future__ import annotations

import numpy as np

from .metrics import kelly_criterion, roi as roi_metric


def _max_drawdown(equity: np.ndarray) -> float:
    """权益曲线相对历史峰值的 maximum drawdown（负数或 0）。"""
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    return float(np.min(dd))


def _sharpe(period_returns: np.ndarray) -> float:
    """简单夏普：均值/标准差（未年化）；样本过少时返回 0.0。"""
    r = np.asarray(period_returns, dtype=float).ravel()
    if r.size < 2:
        return 0.0
    mu, sig = float(np.mean(r)), float(np.std(r, ddof=1))
    if sig < 1e-12:
        return 0.0
    return mu / sig


class ProfitEvaluator:
    """根据每场是否猜中、赔率与小样本胜率，模拟资金曲线并汇总风险收益指标。"""

    def __init__(
        self,
        won: np.ndarray,
        odds_decimal: np.ndarray,
        *,
        p_win: np.ndarray | None = None,
        initial_bankroll: float = 1000.0,
        fixed_stake: float = 10.0,
        kelly_cap: float = 0.25,
        kelly_outer_fraction: float = 0.5,
    ) -> None:
        self.won = np.asarray(won, dtype=bool).ravel()
        self.odds = np.asarray(odds_decimal, dtype=float).ravel()
        self.p_win = None if p_win is None else np.asarray(p_win, dtype=float).ravel()
        self.initial_bankroll = float(initial_bankroll)
        self.fixed_stake = float(fixed_stake)
        self.kelly_cap = float(kelly_cap)
        self.kelly_outer_fraction = float(kelly_outer_fraction)

    def _run_fixed(self) -> dict[str, float]:
        stake = np.full(self.won.shape[0], self.fixed_stake)
        profit = np.where(self.won, stake * (self.odds - 1.0), -stake)
        equity = self.initial_bankroll + np.cumsum(profit)
        ret = profit / np.maximum(stake, 1e-9)
        return {
            "roi": roi_metric(profit, stake),
            "max_drawdown": _max_drawdown(equity),
            "sharpe": _sharpe(ret),
            "final_bankroll": float(equity[-1]) if equity.size else self.initial_bankroll,
        }

    def _run_kelly(self) -> dict[str, float]:
        if self.p_win is None:
            return {"roi": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "final_bankroll": self.initial_bankroll}
        bank = self.initial_bankroll
        equity_list = [bank]
        profits: list[float] = []
        stakes: list[float] = []
        for i in range(self.won.shape[0]):
            k = kelly_criterion(float(self.p_win[i]), float(self.odds[i]))
            k = min(k, self.kelly_cap) * self.kelly_outer_fraction
            stake = min(bank * k, bank * 0.99)
            stakes.append(stake)
            pnl = stake * (self.odds[i] - 1.0) if self.won[i] else -stake
            profits.append(pnl)
            bank += pnl
            bank = max(bank, 1e-6)
            equity_list.append(bank)
        equity = np.array(equity_list)
        profit_arr = np.array(profits)
        stake_arr = np.array(stakes)
        ret = profit_arr / np.maximum(np.array(equity_list[:-1]), 1e-9)
        return {
            "roi": roi_metric(profit_arr, stake_arr),
            "max_drawdown": _max_drawdown(equity),
            "sharpe": _sharpe(ret),
            "final_bankroll": float(equity[-1]),
        }

    def evaluate(self) -> dict[str, dict[str, float]]:
        """返回 {"fixed_stake": {...}, "kelly": {...}}。"""
        return {"fixed_stake": self._run_fixed(), "kelly": self._run_kelly()}
