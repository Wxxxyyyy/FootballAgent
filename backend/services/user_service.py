# -*- coding: utf-8 -*-
"""用户业务：bcrypt、异步会话、users 表 Core 定义；get_db 供路由注入。"""
from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import Any, Optional

import bcrypt
from sqlalchemy import Column, Integer, MetaData, String, Table, insert, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

metadata = MetaData()
users_table = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(64), unique=True, nullable=False, index=True),
    Column("hashed_password", String(255), nullable=False),
    Column("email", String(128), nullable=True),
)
# 优先使用 DATABASE_URL，否则从独立环境变量拼接
def _build_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    from urllib.parse import quote_plus
    _host = os.getenv("MYSQL_HOST", "127.0.0.1")
    _port = os.getenv("MYSQL_PORT", "3306")
    _user = quote_plus(os.getenv("MYSQL_USER", "root"))
    _pwd = quote_plus(os.getenv("MYSQL_PASSWORD", ""))
    _db = os.getenv("MYSQL_DATABASE", "football_agent")
    return f"mysql+aiomysql://{_user}:{_pwd}@{_host}:{_port}/{_db}"

_database_url = _build_database_url()
_engine = create_async_engine(_database_url, echo=False, pool_pre_ping=True)
async_session_maker = async_sessionmaker(
    _engine, expire_on_commit=False, class_=AsyncSession
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session


class UserService:
    @staticmethod
    def _hash_password(plain: str) -> str:
        return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    @staticmethod
    def _verify_password(plain: str, hashed: str) -> bool:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))

    async def create_user(
        self,
        session: AsyncSession,
        username: str,
        password: str,
        email: str | None = None,
    ) -> dict[str, Any]:
        dup = await session.scalar(
            select(users_table.c.id).where(users_table.c.username == username)
        )
        if dup is not None:
            raise ValueError("用户名已存在")
        hp = self._hash_password(password)
        r = await session.execute(
            insert(users_table).values(
                username=username, hashed_password=hp, email=email
            )
        )
        await session.commit()
        pk = r.inserted_primary_key
        uid = int(pk[0]) if pk else 0
        return {"id": uid, "username": username, "email": email}

    async def authenticate(
        self, session: AsyncSession, username: str, password: str
    ) -> Optional[dict[str, Any]]:
        res = await session.execute(
            select(users_table).where(users_table.c.username == username)
        )
        row = res.mappings().first()
        if row is None or not self._verify_password(password, row["hashed_password"]):
            return None
        return {"id": row["id"], "username": row["username"], "email": row["email"]}

    async def get_user_by_id(
        self, session: AsyncSession, user_id: int
    ) -> Optional[dict[str, Any]]:
        res = await session.execute(
            select(users_table).where(users_table.c.id == user_id)
        )
        row = res.mappings().first()
        if row is None:
            return None
        return {"id": row["id"], "username": row["username"], "email": row["email"]}
