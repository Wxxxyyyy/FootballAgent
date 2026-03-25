# -*- coding: utf-8 -*-
"""
比赛数据 API：列表筛选、单场详情、积分榜（示例数据，可对接 MySQL/CSV）。
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(tags=["比赛"])

# 示例数据，便于联调；接入真实数据源时替换查询层
_SAMPLE: list[dict[str, Any]] = [
    {
        "id": 1,
        "league": "英超",
        "home": "Arsenal",
        "away": "Chelsea",
        "match_date": "2025-03-23",
        "home_score": None,
        "away_score": None,
    },
    {
        "id": 2,
        "league": "英超",
        "home": "Liverpool",
        "away": "Man City",
        "match_date": "2025-03-24",
        "home_score": 2,
        "away_score": 1,
    },
]

_STANDINGS: dict[str, list[dict[str, Any]]] = {
    "英超": [
        {"team": "Liverpool", "played": 28, "points": 67},
        {"team": "Arsenal", "played": 28, "points": 65},
    ]
}


def _match_filters(
    league: Optional[str],
    match_date: Optional[str],
    team: Optional[str],
) -> list[dict[str, Any]]:
    out = []
    for m in _SAMPLE:
        if league and m.get("league") != league:
            continue
        if match_date and m.get("match_date") != match_date:
            continue
        if team and team.lower() not in (
            str(m.get("home", "")).lower(),
            str(m.get("away", "")).lower(),
        ):
            continue
        out.append(m)
    return out


@router.get("/matches")
async def list_matches(
    league: str | None = Query(None, description="联赛名称"),
    match_date: str | None = Query(None, description="YYYY-MM-DD"),
    team: str | None = Query(None, description="主客队模糊匹配"),
):
    """比赛列表，支持联赛/日期/球队筛选。"""
    return {"items": _match_filters(league, match_date, team)}


@router.get("/matches/standings")
async def standings(league: str = Query("英超", description="联赛名称")):
    """积分榜。"""
    rows = _STANDINGS.get(league)
    if rows is None:
        raise HTTPException(404, "暂无该联赛积分榜")
    return {"league": league, "table": rows}


@router.get("/matches/{match_id}")
async def match_detail(match_id: int):
    """单场比赛详情。"""
    for m in _SAMPLE:
        if m["id"] == match_id:
            return m
    raise HTTPException(404, "比赛不存在")
