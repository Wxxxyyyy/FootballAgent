# -*- coding: utf-8 -*-
"""
轻量同步 Redis 助手（缓存 + 去重标记）。

设计原则（第一性原理：Redis 是"提效层"，不是"正确性依赖"）:
  1. 任何 Redis 异常都静默降级，绝不向上抛，绝不阻断预测/推送主链路。
  2. 熔断: 一旦失败，冷却 60s 内直接跳过 Redis，避免每次调用都吃 connect 超时拖慢链路。
  3. 无第三方依赖（仅 stdlib + redis），可同时被宿主机预测服务与 OpenClaw 容器复用。

连接地址由环境变量决定:
  - 宿主机预测服务:   REDIS_HOST=127.0.0.1（redis 端口发布在 127.0.0.1:6379）
  - OpenClaw 容器:    REDIS_HOST=football_redis（走 docker football_net 服务名）

注意: 本文件在 common/ 与 docker/openclaw/ 各存一份且内容一致，
      因 OpenClaw 镜像构建上下文限定在 docker/openclaw/，无法 COPY 仓库根文件。
      修改时请同步两处。
"""

import os
import json
import time
import logging
import threading

try:
    import redis as _redis_lib
except ImportError:  # 容器/环境未装 redis 包时，整体降级为"无缓存"
    _redis_lib = None

logger = logging.getLogger(__name__)

_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
_PORT = int(os.getenv("REDIS_PORT", "6379"))
_DB = int(os.getenv("REDIS_DB", "0"))
_PASSWORD = os.getenv("REDIS_PASSWORD") or None
_TIMEOUT = float(os.getenv("REDIS_TIMEOUT", "1.0"))  # connect/read 超时，短以便快速降级
_BREAKER_COOLDOWN = float(os.getenv("REDIS_BREAKER_COOLDOWN", "60"))

_client = None
_client_lock = threading.Lock()
_breaker_until = 0.0  # 熔断截止时间戳；now < 此值时跳过 Redis


def _get_client():
    global _client
    if _redis_lib is None:
        return None
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = _redis_lib.Redis(
                    host=_HOST,
                    port=_PORT,
                    db=_DB,
                    password=_PASSWORD,
                    decode_responses=True,
                    socket_connect_timeout=_TIMEOUT,
                    socket_timeout=_TIMEOUT,
                    health_check_interval=30,
                )
    return _client


def _trip_breaker():
    global _breaker_until
    _breaker_until = time.time() + _BREAKER_COOLDOWN


def _safe(op, default):
    """执行一次 Redis 操作；任何异常/熔断中 → 返回 default，并触发熔断。"""
    if _redis_lib is None or time.time() < _breaker_until:
        return default
    client = _get_client()
    if client is None:
        return default
    try:
        return op(client)
    except Exception as e:  # 连接失败/超时/协议错误 一律降级
        logger.warning(f"[redis] 操作失败，降级 {_BREAKER_COOLDOWN:.0f}s: {e}")
        _trip_breaker()
        return default


def available() -> bool:
    """PING 探活（受熔断保护）。仅用于诊断/日志，不作为主链路依赖。"""
    return bool(_safe(lambda c: c.ping(), False))


# ─────────────────────────── JSON 缓存 ───────────────────────────

def cache_get_json(key: str):
    """读取 JSON 缓存；未命中/降级返回 None。"""
    raw = _safe(lambda c: c.get(key), None)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def cache_set_json(key: str, value, ttl: int) -> bool:
    """写入 JSON 缓存（带 TTL 秒）。成功 True，降级 False。"""
    try:
        payload = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return False
    return _safe(lambda c: c.set(key, payload, ex=ttl), None) is not None


# ─────────────────────────── 去重标记 ───────────────────────────

def mark(key: str, ttl: int) -> bool:
    """打标记（带 TTL，到期自动清理）。成功 True，降级 False。"""
    return _safe(lambda c: c.set(key, "1", ex=ttl), None) is not None


def is_marked(key: str) -> bool:
    """是否存在标记。降级时返回 False（交由上层文件兜底判断）。"""
    return _safe(lambda c: c.exists(key), 0) == 1


def unmark(key: str) -> bool:
    """删除标记。成功 True，降级 False。"""
    return _safe(lambda c: c.delete(key), None) is not None
