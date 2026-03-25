# -*- coding: utf-8 -*-
"""
模型评估：触发离线评估任务、读取最近一次报告 JSON。
"""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["评估"])

_report: dict[str, Any] | None = None
_last_run_id: str | None = None


class EvaluationRunBody(BaseModel):
    dataset_path: str | None = Field(None, description="可选评估数据集路径")
    metrics: list[str] = Field(default_factory=lambda: ["accuracy", "mae"])


@router.post("/evaluation/run")
async def run_evaluation(body: EvaluationRunBody):
    """异步触发评估（此处用短 sleep 模拟耗时任务），生成报告摘要。"""
    global _report, _last_run_id
    rid = str(uuid.uuid4())
    _last_run_id = rid
    # 模拟评估流水线
    await asyncio.sleep(0.05)
    _report = {
        "run_id": rid,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "dataset": body.dataset_path or os.getenv("EVAL_DATASET", "default"),
        "metrics": body.metrics,
        "summary": {"note": "请将此处替换为真实离线评估脚本输出"},
    }
    return {"status": "ok", "run_id": rid}


@router.get("/evaluation/report")
async def get_report(run_id: Optional[str] = None):
    """获取评估报告；可指定 run_id，默认最近一次。"""
    if _report is None:
        raise HTTPException(404, "尚未运行评估")
    if run_id and _report.get("run_id") != run_id:
        raise HTTPException(404, "找不到该次评估")
    return _report
