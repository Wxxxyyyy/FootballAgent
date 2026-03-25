# -*- coding: utf-8 -*-
"""
评估指标：命中率、Brier、对数损失、ROI、凯利比例。
均基于 numpy，输入为数组；空样本时返回 0.0 或 nan（见各函数说明）。
"""
from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """分类命中率：y_true / y_pred 形状相同，元素为类别标签或 0/1。"""
    # 逐元素比较后取平均
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    多类 Brier：y_true 为 one-hot (n, C)，y_prob 为预测概率 (n, C)，每行和约为 1。
    二分类时可用 y_true 为 (n,) 的 0/1，y_prob 为正类概率 (n,) 或 (n, 2)。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.ndim == 1 and y_prob.ndim == 2 and y_prob.shape[1] >= 2:
        n = y_true.shape[0]
        oh = np.zeros_like(y_prob)
        oh[np.arange(n), y_true.astype(int)] = 1.0
        return float(np.mean(np.sum((y_prob - oh) ** 2, axis=1)))
    if y_true.ndim == 1 and y_prob.ndim == 1:
        p = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
        return float(np.mean((p - y_true) ** 2))
    if y_true.shape != y_prob.shape:
        raise ValueError("brier_score: y_true 与 y_prob 形状不匹配")
    return float(np.mean(np.sum((y_prob - y_true) ** 2, axis=1)))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """多类对数损失；y_true 为 (n,) 类别索引，y_prob 为 (n, C) 概率。"""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return 0.0
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # 行归一化
    idx = (np.arange(y_true.shape[0]), y_true)
    return float(-np.mean(np.log(y_prob[idx])))


def roi(net_returns: np.ndarray, stakes: np.ndarray) -> float:
    """
    投资回报率：总净收益 / 总投注额。
    net_returns[i] 为第 i 笔在结算后的盈亏（可负），stakes[i] 为该笔下注金额。
    """
    net_returns = np.asarray(net_returns, dtype=float).ravel()
    stakes = np.asarray(stakes, dtype=float).ravel()
    s = np.sum(stakes)
    if s <= 0:
        return 0.0
    return float(np.sum(net_returns) / s)


def kelly_criterion(p_win: float, odds_decimal: float) -> float:
    """
    凯利公式最优投注比例（占资金比例）。
    odds_decimal 为欧洲盘小数赔率；b = odds - 1，f* = (p*b - (1-p)) / b。
    若分母<=0 或最优比例<0，返回 0.0。
    """
    if odds_decimal <= 1.0:
        return 0.0
    b = odds_decimal - 1.0
    q = 1.0 - p_win
    num = p_win * b - q
    if b <= 0 or num <= 0:
        return 0.0
    return float(num / b)
