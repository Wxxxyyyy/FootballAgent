# -*- coding: utf-8 -*-
"""
FastAPI / Starlette 中间件装配
- 跨域 CORS
- 请求唯一 ID（可与日志 trace 对齐）
- 基础访问日志（方法、路径、耗时、状态码）
"""
import time
import uuid
from typing import Iterable, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .logger import REQUEST_ID_CTX, get_logger

_log = get_logger()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    为每个请求生成或透传 X-Request-ID，并写入上下文供日志过滤器使用。
    客户端可传入 X-Request-ID 以保持链路一致。
    """

    header_name = "X-Request-ID"

    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get(self.header_name)
        rid = incoming.strip() if incoming else str(uuid.uuid4())
        token = REQUEST_ID_CTX.set(rid)
        try:
            response: Response = await call_next(request)
            response.headers[self.header_name] = rid
            return response
        finally:
            REQUEST_ID_CTX.reset(token)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """记录 HTTP 方法、路径、状态码与耗时（毫秒），不记录敏感 query 全文。"""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        cost_ms = round((time.perf_counter() - start) * 1000, 2)
        _log.bind(
            http_method=request.method,
            http_path=request.url.path,
            status_code=response.status_code,
            duration_ms=cost_ms,
        ).info("http_access")
        return response


def setup_middlewares(
    app: FastAPI,
    *,
    cors_origins: Optional[Iterable[str]] = None,
    cors_allow_credentials: bool = False,
) -> None:
    """
    注册中间件（Starlette 规则：最后 add 的中间件最先处理入站请求）。

    实际入站顺序：CORS → RequestID（写入 request_id 上下文）→ RequestLogging → 路由。
    这样在 RequestLogging 记录耗时与状态码时，日志过滤器仍能读到 request_id。
    """
    origins = list(cors_origins) if cors_origins is not None else ["*"]

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
