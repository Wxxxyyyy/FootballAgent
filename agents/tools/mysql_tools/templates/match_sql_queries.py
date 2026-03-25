# -*- coding: utf-8 -*-
"""
比赛相关 SQL 静态查询模板
─────────────────────────
数据源: football_agent.match_master 宽表（football-data.co.uk）

每个函数返回 (sql: str, params: tuple)，
使用 %s 占位符 + 元组参数，由 pymysql 参数化执行以防注入。

由 tool_entry.py 统一提交给 MySQL 执行。
"""


def recent_matches_sql(team: str, limit: int = 10) -> tuple[str, tuple]:
    """
    查询某支球队最近的比赛（主场或客场均包含），按日期降序。

    返回字段: 日期、联赛、赛季、主队、客队、全场比分、半场比分、B365初盘赔率

    Args:
        team:  球队英文名（如 "Arsenal"）
        limit: 返回记录数上限（默认 10）
    """
    sql = """
    SELECT
        Date            AS match_date,
        league,
        season,
        HomeTeam,
        AwayTeam,
        FTHG, FTAG, FTR,
        HTHG, HTAG, HTR,
        B365H, B365D, B365A
    FROM match_master
    WHERE HomeTeam = %s OR AwayTeam = %s
    ORDER BY Date DESC
    LIMIT %s
    """
    return sql, (team, team, limit)


def head_to_head_sql(team_a: str, team_b: str, limit: int = 10) -> tuple[str, tuple]:
    """
    两队历史交锋记录，包含全场比分和 B365 初盘赔率。

    支持双方主客场互换，按日期降序。

    Args:
        team_a: 第一支球队英文名
        team_b: 第二支球队英文名
        limit:  返回记录数上限（默认 10）
    """
    sql = """
    SELECT
        Date            AS match_date,
        league,
        season,
        HomeTeam,
        AwayTeam,
        FTHG, FTAG, FTR,
        B365H, B365D, B365A,
        B365_Over25, B365_Under25,
        AHh, B365AHH, B365AHA
    FROM match_master
    WHERE (HomeTeam = %s AND AwayTeam = %s)
       OR (HomeTeam = %s AND AwayTeam = %s)
    ORDER BY Date DESC
    LIMIT %s
    """
    return sql, (team_a, team_b, team_b, team_a, limit)


def team_season_stats_sql(team: str, season: str) -> tuple[str, tuple]:
    """
    查询某队某赛季所有的比赛记录。

    返回字段: 日期、主客队、比分、结果、赔率、大小球、亚盘

    Args:
        team:   球队英文名
        season: 赛季标识（如 "2024-2025"）
    """
    sql = """
    SELECT
        Date            AS match_date,
        league,
        HomeTeam,
        AwayTeam,
        FTHG, FTAG, FTR,
        HTHG, HTAG, HTR,
        B365H, B365D, B365A,
        B365_Over25, B365_Under25,
        AHh, B365AHH, B365AHA
    FROM match_master
    WHERE (HomeTeam = %s OR AwayTeam = %s)
      AND season = %s
    ORDER BY Date ASC
    """
    return sql, (team, team, season)

