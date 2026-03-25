# -*- coding: utf-8 -*-
"""
MySQL 异步连接池（SQLAlchemy 2 + aiomysql）
提供引擎、会话上下文与生命周期管理。
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .config import get_settings

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine() -> AsyncEngine:
    """懒加载创建全局 AsyncEngine（单例）。"""
    global _engine, _session_factory
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.sqlalchemy_database_uri,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=settings.app_debug,
        )
        _session_factory = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
    return _engine


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    异步会话上下文管理器。
    用法::
        async with get_session() as session:
            await session.execute(...)
    """
    get_engine()
    assert _session_factory is not None
    async with _session_factory() as session:
        yield session


async def init_db() -> None:
    """
    初始化数据库：校验连接可用。
    若已定义 SQLAlchemy Base，可在此调用 create_all（需避免循环导入）。
    """
    engine = get_engine()
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


async def close_db() -> None:
    """应用关闭时释放连接池。"""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
