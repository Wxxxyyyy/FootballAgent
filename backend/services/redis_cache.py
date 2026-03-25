# -*- coding: utf-8 -*-
"""
Redis 异步缓存封装
- JSON 序列化 / 反序列化
- TTL 过期
- 足球预测等业务键构造
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import redis.asyncio as redis


class RedisCacheService:
    """基于 redis.asyncio 的通用缓存服务。"""

    def __init__(self, client: redis.Redis) -> None:
        self._r = client

    @classmethod
    def from_url(cls, url: str | None = None) -> "RedisCacheService":
        """从 REDIS_URL 或默认本机地址创建客户端。"""
        u = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        return cls(redis.from_url(u, decode_responses=True))

    async def get(self, key: str) -> Optional[Any]:
        raw = await self._r.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        payload = json.dumps(value, ensure_ascii=False, default=str)
        if ttl_seconds is not None:
            await self._r.setex(key, ttl_seconds, payload)
        else:
            await self._r.set(key, payload)

    async def delete(self, key: str) -> int:
        return int(await self._r.delete(key))

    async def exists(self, key: str) -> bool:
        return bool(await self._r.exists(key))

    def _prediction_key(self, home: str, away: str, date: str) -> str:
        """预测缓存键：主客队 + 比赛日。"""
        return f"pred:{home.strip()}:{away.strip()}:{date.strip()}"

    async def get_prediction_cache(
        self, home: str, away: str, date: str
    ) -> Optional[dict[str, Any]]:
        return await self.get(self._prediction_key(home, away, date))

    async def set_prediction_cache(
        self,
        home: str,
        away: str,
        date: str,
        data: dict[str, Any],
        ttl_seconds: int = 3600,
    ) -> None:
        await self.set(self._prediction_key(home, away, date), data, ttl_seconds)
