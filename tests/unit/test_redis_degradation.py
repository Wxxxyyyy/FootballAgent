# -*- coding: utf-8 -*-
"""
redis_cache 降级契约测试。

第一性原理: Redis 是提效层不是正确性依赖。
conftest 已把 REDIS_PORT 指向不存在的端口，这里锁死:
任何操作在 Redis 不可用时 ① 不抛异常 ② 返回安全默认值 ③ 熔断后快速返回。
"""

import time

import redis_cache as rc


class TestDegradation:
    def test_all_ops_safe_defaults(self):
        assert rc.cache_get_json("k") is None
        assert rc.cache_set_json("k", {"a": 1}, 60) is False
        assert rc.mark("k", 60) is False
        assert rc.is_marked("k") is False
        assert rc.unmark("k") is False
        assert rc.available() is False

    def test_breaker_makes_subsequent_calls_fast(self):
        rc.is_marked("warmup")  # 触发熔断
        start = time.time()
        for _ in range(50):
            rc.is_marked("x")
            rc.cache_get_json("y")
        elapsed = time.time() - start
        assert elapsed < 0.5, f"熔断后 100 次调用耗时 {elapsed:.2f}s，疑似仍在连接重试"

    def test_unserializable_value_returns_false(self):
        assert rc.cache_set_json("k", {"bad": object()}, 60) is False
