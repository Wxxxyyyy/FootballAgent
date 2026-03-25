# -*- coding: utf-8 -*-
"""
backend.core 公共导出

典型启动顺序（``lifespan``）：

1. ``setup_logger()`` 初始化日志；
2. ``init_db()``、``get_redis()`` / ``get_driver()`` 懒连或按需预热；
3. ``setup_middlewares(app)`` 注册 CORS 与访问日志；
4. shutdown 时 ``close_db``、``close_redis``、``close_neo4j_driver``。
"""
from .config import Settings, get_settings
from .database import close_db, get_engine, get_session, init_db
from .logger import REQUEST_ID_CTX, get_logger, setup_logger
from .middleware import setup_middlewares
from .neo4j_conn import close_driver as close_neo4j_driver
from .neo4j_conn import get_driver as get_neo4j_driver
from .neo4j_conn import health_check as neo4j_health_check
from .redis_client import close_redis, get_redis, health_check as redis_health_check
from .security import (
    TokenPayload,
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
    verify_token,
)

__all__ = [
    "Settings",
    "REQUEST_ID_CTX",
    "TokenPayload",
    "close_db",
    "close_neo4j_driver",
    "close_redis",
    "create_access_token",
    "get_current_user",
    "get_engine",
    "get_logger",
    "get_neo4j_driver",
    "get_redis",
    "get_session",
    "get_settings",
    "hash_password",
    "init_db",
    "neo4j_health_check",
    "redis_health_check",
    "setup_logger",
    "setup_middlewares",
    "verify_password",
    "verify_token",
]
