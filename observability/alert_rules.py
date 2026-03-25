# -*- coding: utf-8 -*-
"""
告警规则引擎
预置阈值规则；check_alerts 根据当前指标返回触发列表
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """单条告警规则"""

    name: str
    metric: str  # 与 metrics 字典中的键对应
    threshold: float
    window_seconds: int  # 语义上为统计窗口（具体聚合由上游指标服务完成）
    severity: str  # info / warning / critical


@dataclass
class AlertEvent:
    """已触发的告警"""

    rule: AlertRule
    actual: float
    message: str


# 预置：预测失败率、LLM P95、入库队列长度、Redis 内存
PRESET_RULES: List[AlertRule] = [
    AlertRule(
        name="prediction_failure_rate_high",
        metric="prediction_failure_rate",
        threshold=0.20,
        window_seconds=300,
        severity="warning",
    ),
    AlertRule(
        name="llm_latency_p95_high",
        metric="llm_latency_p95_seconds",
        threshold=20.0,
        window_seconds=300,
        severity="warning",
    ),
    AlertRule(
        name="ingest_queue_backlog",
        metric="ingest_queue_length",
        threshold=20.0,
        window_seconds=60,
        severity="critical",
    ),
    AlertRule(
        name="redis_memory_high",
        metric="redis_used_memory_mb",
        threshold=200.0,
        window_seconds=60,
        severity="warning",
    ),
]


def _collect_redis_metrics() -> Dict[str, float]:
    """尝试从 Redis 读取队列长度与内存用量（失败则返回空 dict）"""
    out: Dict[str, float] = {}
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        import redis

        r = redis.from_url(url, decode_responses=True)
        qlen = r.llen("openclaw:ingest_queue")
        out["ingest_queue_length"] = float(qlen)
        info = r.info("memory")
        used = info.get("used_memory", 0)
        out["redis_used_memory_mb"] = used / (1024.0 * 1024.0)
    except Exception as e:
        logger.debug("Redis 指标采集跳过: %s", e)
    return out


def check_alerts(metrics: Optional[Dict[str, float]] = None, rules: Optional[List[AlertRule]] = None) -> List[AlertEvent]:
    """
    检查所有规则。metrics 可为 None，此时会尝试合并 Redis 中的部分指标；
    预测失败率、LLM P95 需由业务方注入 metrics。
    """
    merged: Dict[str, float] = {}
    if metrics:
        merged.update(metrics)
    merged.update(_collect_redis_metrics())

    active_rules = rules if rules is not None else PRESET_RULES
    fired: List[AlertEvent] = []
    for rule in active_rules:
        if rule.metric not in merged:
            continue
        actual = float(merged[rule.metric])
        if actual > rule.threshold:
            fired.append(
                AlertEvent(
                    rule=rule,
                    actual=actual,
                    message=f"[{rule.severity}] {rule.name}: {rule.metric}={actual:.4f} > {rule.threshold}",
                )
            )
    return fired
