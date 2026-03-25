# -*- coding: utf-8 -*-
"""
OpenClaw 每日数据同步
经中继 relay_to_openclaw 拉取昨日比赛 JSON，再调用 openclaw_ingestion 入库
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 保证以脚本直接运行时能 import pipeline
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_RELAY_ENV = "RELAY_URL"
_DEFAULT_RELAY = "http://localhost:15000"
_MAX_RETRIES = 3
_RETRY_BASE_SEC = 1.5


def _relay_url() -> str:
    load_dotenv()
    return __import__("os").getenv(_RELAY_ENV, _DEFAULT_RELAY).rstrip("/")


def _normalize_daily_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """将中继/OpenClaw HTTP 响应整理为与 /receive_data daily_matches 一致的入库结构"""
    oc = raw.get("openclaw_response", raw)
    if isinstance(oc, dict) and "result" in oc:
        oc = oc["result"]
    if not isinstance(oc, dict):
        return {"data_type": "daily_matches", "content": {}}
    if oc.get("data_type") == "daily_matches" and "content" in oc:
        return oc
    content = oc.get("content")
    if content is None:
        content = oc
    return {"data_type": "daily_matches", "content": content, "task_id": raw.get("task_id")}


def sync_daily_matches() -> Dict[str, Any]:
    """
    请求昨日比赛数据并写入 MySQL / Neo4j。
    失败时按指数退避重试，错误写入日志。
    """
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    relay = _relay_url()
    url = f"{relay}/relay_to_openclaw"
    payload = {
        "task_id": str(uuid.uuid4()),
        "task_type": "fetch_daily_matches",
        "params": {"date": yesterday},
        "async_mode": False,
        "timestamp": datetime.now().isoformat(),
    }
    last_err: Exception | None = None
    raw: Dict[str, Any] = {}
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                raw = resp.json()
            last_err = None
            break
        except Exception as e:
            last_err = e
            logger.warning(
                "sync_daily_matches 请求失败 (%s/%s): %s",
                attempt,
                _MAX_RETRIES,
                e,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BASE_SEC**attempt)

    if last_err is not None:
        logger.error("sync_daily_matches 放弃: %s", last_err)
        return {"status": "error", "message": str(last_err)}

    data = _normalize_daily_payload(raw if isinstance(raw, dict) else {"openclaw_response": raw})
    try:
        from pipeline.openclaw_ingestion import ingest_openclaw_data

        ingested = ingest_openclaw_data(data)
        logger.info("sync_daily_matches 入库完成: %s", ingested)
        return {"status": "ok", "date": yesterday, "ingestion": ingested}
    except Exception as e:
        logger.exception("sync_daily_matches 入库异常: %s", e)
        return {"status": "error", "message": str(e), "date": yesterday}
