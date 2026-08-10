# -*- coding: utf-8 -*-
"""
Prometheus 指标采集
══════════════════════════════════════════════════════════════

第一性原理:
  日志是"事后查"，仪表盘是"实时看"。
  Prometheus 采集 = 把系统状态变成可查询的时序指标，
  配合 Grafana 实现实时可视化（QPS/延迟/错误率/成功率）。

方案:
  1. 定义核心指标: Counter（累计请求数/错误数）/ Gauge（当前值）/ Histogram（延迟分布）
  2. 在 API 中间件中自动记录 HTTP 请求指标
  3. /metrics 端点暴露给 Prometheus 抓取

核心指标:
  - http_requests_total{method, path, status}  — HTTP 请求总数
  - http_request_duration_seconds{path}          — HTTP 延迟分布
  - llm_calls_total{model, status}              — LLM 调用总数
  - llm_duration_seconds{model}                 — LLM 延迟分布
  - prediction_total{tier}                      — 预测触发总数
  - system_health                               — 组件健康状态 (0/1)
"""

import os
import time
import threading
from collections import defaultdict

# 使用 prometheus_client 库
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


# ═══════════════════════════════════════════════════════════════
#  指标定义
# ═══════════════════════════════════════════════════════════════

if HAS_PROMETHEUS:
    # HTTP 请求指标
    http_requests_total = Counter(
        "football_http_requests_total",
        "Total HTTP requests",
        ["method", "path", "status"],
    )
    http_request_duration = Histogram(
        "football_http_request_duration_seconds",
        "HTTP request duration in seconds",
        ["method", "path"],
    )

    # LLM 调用指标
    llm_calls_total = Counter(
        "football_llm_calls_total",
        "Total LLM API calls",
        ["model", "status"],
    )
    llm_duration = Histogram(
        "football_llm_duration_seconds",
        "LLM API call duration in seconds",
        ["model"],
    )

    # 预测指标
    prediction_total = Counter(
        "football_prediction_total",
        "Total predictions triggered",
        ["tier"],
    )

    # 系统健康
    system_health = Gauge(
        "football_system_health",
        "System component health (1=ok, 0=down)",
        ["component"],
    )

else:
    # 无 prometheus_client 时的 mock（不阻断运行）
    _mock_store = defaultdict(lambda: defaultdict(int))
    http_requests_total = None
    http_request_duration = None
    llm_calls_total = None
    llm_duration = None
    prediction_total = None
    system_health = None


# ═══════════════════════════════════════════════════════════════
#  便捷记录函数
# ═══════════════════════════════════════════════════════════════

def record_http_request(method: str, path: str, status: int, duration: float):
    """记录一次 HTTP 请求"""
    if not HAS_PROMETHEUS:
        return
    http_requests_total.labels(method=method, path=path, status=str(status)).inc()
    http_request_duration.labels(method=method, path=path).observe(duration)


def record_llm_call(model: str, success: bool, duration: float):
    """记录一次 LLM 调用"""
    if not HAS_PROMETHEUS:
        return
    llm_calls_total.labels(model=model, status="success" if success else "error").inc()
    llm_duration.labels(model=model).observe(duration)


def record_prediction(tier: int = 1):
    """记录一次预测触发"""
    if not HAS_PROMETHEUS:
        return
    prediction_total.labels(tier=str(tier)).inc()


def set_component_health(component: str, ok: bool):
    """设置组件健康状态"""
    if not HAS_PROMETHEUS:
        return
    system_health.labels(component=component).set(1 if ok else 0)


def get_metrics() -> tuple:
    """获取 Prometheus 格式的指标文本

    Returns:
        (text, content_type) 供 FastAPI 返回
    """
    if not HAS_PROMETHEUS:
        return "# prometheus_client not installed\n", "text/plain"
    return generate_latest(), CONTENT_TYPE_LATEST


# ═══════════════════════════════════════════════════════════════
#  FastAPI 中间件
# ═══════════════════════════════════════════════════════════════

def create_middleware():
    """创建 FastAPI 中间件，自动记录 HTTP 请求指标

    用法:
        from common.metrics import create_middleware
        app.middleware("http")(create_middleware())
    """
    async def metrics_middleware(request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start

        # 归一化路径（把 /chat/123 变成 /chat/:id）
        path = request.url.path
        # 简单归一化：数字段替换为 :id
        import re
        normalized = re.sub(r"/\d+", "/:id", path)

        record_http_request(
            method=request.method,
            path=normalized,
            status=response.status_code,
            duration=duration,
        )
        return response

    return metrics_middleware


# ═══════════════════════════════════════════════════════════════
#  FastAPI 路由
# ═══════════════════════════════════════════════════════════════

def mount_metrics_endpoint(app):
    """在 FastAPI app 上挂载 /metrics 端点

    用法:
        from common.metrics import mount_metrics_endpoint
        mount_metrics_endpoint(app)
    """
    from fastapi import Response

    @app.get("/metrics")
    async def prometheus_metrics():
        text, content_type = get_metrics()
        return Response(content=text, media_type=content_type)
