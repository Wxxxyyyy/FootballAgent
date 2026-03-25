# -*- coding: utf-8 -*-
"""
LLM 调用量 / Token 消耗统计
记录每次 LLM 调用的模型、Token 用量、耗时、费用等（进程内内存列表，单例）
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# 简易默认单价：美元 / 百万 token（输入+输出加权时可在外部传入 cost 覆盖）
_DEFAULT_COST_PER_1M: Dict[str, tuple[float, float]] = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4-turbo": (10.0, 30.0),
    "deepseek-chat": (0.14, 0.28),
    "default": (1.0, 2.0),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """按模型名估算费用（美元）；未知模型走 default。"""
    key = next((k for k in _DEFAULT_COST_PER_1M if k != "default" and k in model.lower()), None)
    inp_rate, out_rate = _DEFAULT_COST_PER_1M.get(key or "default", _DEFAULT_COST_PER_1M["default"])
    return (input_tokens / 1_000_000.0) * inp_rate + (output_tokens / 1_000_000.0) * out_rate


@dataclass
class LLMCallRecord:
    """单次 LLM 调用记录"""

    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost: float
    created_at: float = field(default_factory=time.time)


class LLMUsageTracker:
    """LLM 调用追踪器（单例），数据保存在内存 list 中"""

    _instance: Optional["LLMUsageTracker"] = None
    _singleton_lock = threading.Lock()

    def __new__(cls) -> "LLMUsageTracker":
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._records: List[LLMCallRecord] = []
                    inst._list_lock = threading.Lock()
                    cls._instance = inst
        return cls._instance

    def track_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost: Optional[float] = None,
    ) -> None:
        """记录一次调用；未传 cost 时按模型粗略估算。"""
        c = cost if cost is not None else _estimate_cost(model, input_tokens, output_tokens)
        rec = LLMCallRecord(
            model=model,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            latency_ms=float(latency_ms),
            cost=float(c),
        )
        with self._list_lock:
            self._records.append(rec)

    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """按模型汇总：调用次数、总 token、总费用。"""
        with self._list_lock:
            snapshot = list(self._records)
        by_model: Dict[str, Dict[str, Any]] = {}
        for r in snapshot:
            agg = by_model.setdefault(
                r.model,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_cost": 0.0},
            )
            agg["calls"] += 1
            agg["input_tokens"] += r.input_tokens
            agg["output_tokens"] += r.output_tokens
            agg["total_cost"] += r.cost
        return by_model

    def clear(self) -> None:
        """清空记录（测试或进程内重置用）"""
        with self._list_lock:
            self._records.clear()
