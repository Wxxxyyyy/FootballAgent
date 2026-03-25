# -*- coding: utf-8 -*-
"""
预测任务 API：提交、查询单条、历史列表（内存存储，生产可换 DB/Redis）。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(tags=["预测"])

# 进程内结果缓存；多实例部署请改为共享存储
_store: dict[str, dict[str, Any]] = {}


class PredictionRequest(BaseModel):
    home_team: str = Field(min_length=1)
    away_team: str = Field(min_length=1)
    match_date: str = Field(description="YYYY-MM-DD")
    extra: dict[str, Any] | None = None


@router.post("/prediction")
async def submit_prediction(req: PredictionRequest):
    """创建预测任务，返回任务 id（后续可异步填充结果）。"""
    pid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    _store[pid] = {
        "id": pid,
        "status": "pending",
        "created_at": now,
        "request": req.model_dump(),
        "result": None,
    }
    return {"id": pid, "status": "pending"}


@router.get("/prediction/history")
async def prediction_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """按创建时间倒序返回历史（内存实现）。"""
    items = sorted(
        _store.values(),
        key=lambda x: x.get("created_at", ""),
        reverse=True,
    )
    total = len(items)
    page = items[offset : offset + limit]
    return {"total": total, "items": page}


@router.get("/prediction/{prediction_id}")
async def get_prediction(prediction_id: str):
    """按 id 查询预测详情。"""
    row = _store.get(prediction_id)
    if row is None:
        raise HTTPException(404, "预测记录不存在")
    return row
