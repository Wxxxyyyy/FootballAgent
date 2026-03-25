# -*- coding: utf-8 -*-
"""
Redis 异步客户端封装（redis.asyncio + 连接池）
用于缓存、会话与分布式限流等场景。
"""
from typing import Optional

import redis.asyncio as redis

from .config import get_settings

_redis: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    """
    获取全局 Redis 客户端（单例）。

    - ``decode_responses=True``：返回值统一为 ``str``，避免业务层手动 decode。
    - ``max_connections=20``：与常见 API 并发量级匹配，可按压测调大。
    - URL 由 ``Settings.redis_url`` 拼装，支持密码中的特殊字符（URL 编码）。
    """
    global _redis
    if _redis is None:
        settings = get_settings()
        _redis = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
    return _redis


async def close_redis() -> None:
    """
    关闭底层连接池并释放资源。

    建议在 FastAPI ``lifespan`` 的 shutdown 阶段与 ``close_db``、``close_driver`` 一并调用，
    避免进程退出时残留 TCP 连接。
    """
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


async def health_check() -> bool:
    """
    对 Redis 执行 ``PING``，成功返回 ``True``。

    部署在 K8s / Docker 时可作为 readiness 探针；失败时返回 ``False`` 不打栈，
    由上层决定日志级别。
    """
    try:
        client = get_redis()
        pong = await client.ping()
        return bool(pong)
    except Exception:
        return False
