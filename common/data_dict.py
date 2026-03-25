# -*- coding: utf-8 -*-
"""
CSV 列字段语义与类型说明。

与 ``football-data.co.uk`` 风格导出一致；类型字段为粗粒度提示（分析/校验用），
并非运行时强制 Schema。
"""

from __future__ import annotations

# 每条: (中文说明, 类型提示)
FieldDesc = tuple[str, str]

MATCH_FIELDS: dict[str, FieldDesc] = {
    "Div": ("联赛代码（如 E0=英超）", "str"),
    "Date": ("比赛日期", "str"),
    "Time": ("开球时间（若存在）", "str"),
    "HomeTeam": ("主队名称", "str"),
    "AwayTeam": ("客队名称", "str"),
    "FTHG": ("全场主队进球 Full Time Home Goals", "int"),
    "FTAG": ("全场客队进球 Full Time Away Goals", "int"),
    "FTR": ("全场赛果 H/D/A（主胜/平/客胜）", "category"),
    "HTHG": ("半场主队进球", "int"),
    "HTAG": ("半场客队进球", "int"),
    "HTR": ("半场赛果 H/D/A", "category"),
    "Referee": ("主裁判（若存在）", "str"),
}

ODDS_FIELDS: dict[str, FieldDesc] = {
    "B365H": ("Bet365 主胜欧赔", "float"),
    "B365D": ("Bet365 平局欧赔", "float"),
    "B365A": ("Bet365 客胜欧赔", "float"),
    "BWH": ("BetWin 主胜欧赔", "float"),
    "BWD": ("BetWin 平局欧赔", "float"),
    "BWA": ("BetWin 客胜欧赔", "float"),
    "PSH": ("Pinnacle 主胜欧赔", "float"),
    "PSD": ("Pinnacle 平局欧赔", "float"),
    "PSA": ("Pinnacle 客胜欧赔", "float"),
    "WHH": ("William Hill 主胜欧赔", "float"),
    "WHD": ("William Hill 平局欧赔", "float"),
    "WHA": ("William Hill 客胜欧赔", "float"),
    "MaxH": ("各公司主胜最高欧赔", "float"),
    "MaxD": ("各公司平局最高欧赔", "float"),
    "MaxA": ("各公司客胜最高欧赔", "float"),
    "AvgH": ("主胜欧赔平均值", "float"),
    "AvgD": ("平局欧赔平均值", "float"),
    "AvgA": ("客胜欧赔平均值", "float"),
    "B365>2.5": ("Bet365 大于 2.5 球赔率", "float"),
    "B365<2.5": ("Bet365 小于 2.5 球赔率", "float"),
    "P>2.5": ("Pinnacle 大 2.5 球赔率", "float"),
    "P<2.5": ("Pinnacle 小 2.5 球赔率", "float"),
    "AHh": ("亚洲盘口让球数（主队视角，负表示受让）", "float"),
    "B365AHH": ("Bet365 亚盘主队水位", "float"),
    "B365AHA": ("Bet365 亚盘客队水位", "float"),
    "PAHH": ("Pinnacle 亚盘主队水位", "float"),
    "PAHA": ("Pinnacle 亚盘客队水位", "float"),
}

STATS_FIELDS: dict[str, FieldDesc] = {
    "HS": ("主队射门次数 Shots", "int"),
    "AS": ("客队射门次数 Shots", "int"),
    "HST": ("主队射正 Shots on Target", "int"),
    "AST": ("客队射正 Shots on Target", "int"),
    "HF": ("主队犯规 Fouls", "int"),
    "AF": ("客队犯规 Fouls", "int"),
    "HC": ("主队角球 Corners", "int"),
    "AC": ("客队角球 Corners", "int"),
    "HY": ("主队黄牌 Yellow", "int"),
    "AY": ("客队黄牌 Yellow", "int"),
    "HR": ("主队红牌 Red", "int"),
    "AR": ("客队红牌 Red", "int"),
    "HO": ("主队越位 Offsides（若存在）", "int"),
    "AO": ("客队越位 Offsides（若存在）", "int"),
}


def describe_field(group: str, column: str) -> FieldDesc | None:
    """按分组与列名返回 ``(中文说明, 类型)``；未知列返回 ``None``。"""
    tables = {
        "match": MATCH_FIELDS,
        "odds": ODDS_FIELDS,
        "stats": STATS_FIELDS,
    }
    return tables.get(group, {}).get(column)
