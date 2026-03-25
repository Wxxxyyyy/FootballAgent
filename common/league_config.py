# -*- coding: utf-8 -*-
"""
五大联赛静态配置：代码、中英文名称、国家、典型球队数量、赛季字符串格式等。

用于校验 CSV ``Div`` 列、展示层文案与赛季选择器默认值。
"""

from __future__ import annotations

from typing import TypedDict


class LeagueInfo(TypedDict):
    """单联赛配置结构。"""

    code: str
    name_en: str
    name_zh: str
    country: str
    teams_count: int
    season_format: str  # 例如 "YYYY-YYYY" 表示跨年赛季展示方式


# 与 football-data 风格 Div 代码一致
LEAGUE_CONFIG: dict[str, LeagueInfo] = {
    "E0": {
        "code": "E0",
        "name_en": "Premier League",
        "name_zh": "英超",
        "country": "England",
        "teams_count": 20,
        "season_format": "YYYY-YYYY",
    },
    "D1": {
        "code": "D1",
        "name_en": "Bundesliga",
        "name_zh": "德甲",
        "country": "Germany",
        "teams_count": 18,
        "season_format": "YYYY-YYYY",
    },
    "I1": {
        "code": "I1",
        "name_en": "Serie A",
        "name_zh": "意甲",
        "country": "Italy",
        "teams_count": 20,
        "season_format": "YYYY-YYYY",
    },
    "SP1": {
        "code": "SP1",
        "name_en": "La Liga",
        "name_zh": "西甲",
        "country": "Spain",
        "teams_count": 20,
        "season_format": "YYYY-YYYY",
    },
    "F1": {
        "code": "F1",
        "name_en": "Ligue 1",
        "name_zh": "法甲",
        "country": "France",
        "teams_count": 18,
        "season_format": "YYYY-YYYY",
    },
}


def get_league_by_code(code: str) -> LeagueInfo | None:
    """按 Div 代码查询；大小写不敏感，未找到返回 ``None``。"""
    key = code.strip().upper()
    return LEAGUE_CONFIG.get(key)


def get_all_leagues() -> list[LeagueInfo]:
    """返回全部联赛配置列表（按 code 排序）。"""
    return [LEAGUE_CONFIG[k] for k in sorted(LEAGUE_CONFIG.keys())]


def league_exists(code: str) -> bool:
    """判断联赛代码是否受支持。"""
    return get_league_by_code(code) is not None
