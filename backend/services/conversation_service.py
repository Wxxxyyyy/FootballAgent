# -*- coding: utf-8 -*-
"""会话与消息表（Core），管理用户与 Agent 的对话历史。"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    insert,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession

from services.user_service import metadata

conversations_table = Table(
    "conversations",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, nullable=False, index=True),
    Column("title", String(256), nullable=True),
    Column("created_at", DateTime, default=datetime.utcnow),
)
messages_table = Table(
    "messages",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "conversation_id",
        Integer,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("role", String(32), nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", DateTime, default=datetime.utcnow),
)


class ConversationService:
    async def create_conversation(
        self, session: AsyncSession, user_id: int, title: str | None = None
    ) -> dict[str, Any]:
        r = await session.execute(
            insert(conversations_table).values(user_id=user_id, title=title)
        )
        await session.commit()
        cid = r.inserted_primary_key[0] if r.inserted_primary_key else 0
        return {"id": int(cid), "user_id": user_id, "title": title}

    async def get_conversation(
        self, session: AsyncSession, conversation_id: int
    ) -> Optional[dict[str, Any]]:
        res = await session.execute(
            select(conversations_table).where(
                conversations_table.c.id == conversation_id
            )
        )
        row = res.mappings().first()
        return dict(row) if row else None

    async def add_message(
        self,
        session: AsyncSession,
        conversation_id: int,
        role: str,
        content: str,
    ) -> dict[str, Any]:
        r = await session.execute(
            insert(messages_table).values(
                conversation_id=conversation_id, role=role, content=content
            )
        )
        await session.commit()
        mid = r.inserted_primary_key[0] if r.inserted_primary_key else 0
        return {"id": int(mid), "conversation_id": conversation_id, "role": role}

    async def get_messages(
        self, session: AsyncSession, conversation_id: int, limit: int = 200
    ) -> list[dict[str, Any]]:
        q = (
            select(messages_table)
            .where(messages_table.c.conversation_id == conversation_id)
            .order_by(messages_table.c.id.asc())
            .limit(limit)
        )
        res = await session.execute(q)
        return [dict(x) for x in res.mappings().all()]
