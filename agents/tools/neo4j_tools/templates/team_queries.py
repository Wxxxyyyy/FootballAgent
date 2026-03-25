# -*- coding: utf-8 -*-
"""
球队相关 Cypher 静态查询模板
───────────────────────────
每个函数返回 (cypher: str, params: dict)，
由 tool_entry 统一提交给 Neo4j 执行。
"""


def team_info(team: str) -> tuple[str, dict]:
    """
    查询球队基本信息（名称 + 所属联赛）。
    """
    cypher = """
    MATCH (t:Team {name: $team})
    RETURN t.name AS name, t.league AS league
    """
    return cypher, {"team": team}


def league_teams(league: str) -> tuple[str, dict]:
    """
    列出某联赛所有球队。
    """
    cypher = """
    MATCH (t:Team {league: $league})
    RETURN t.name AS name
    ORDER BY t.name
    """
    return cypher, {"league": league}


def team_season_all(team: str, season: str) -> tuple[str, dict]:
    """
    球队某赛季全部比赛汇总（含主客场区分）。
    通过 startNode(r) 判断谁是主队来标记 venue。
    """
    cypher = """
    MATCH (t:Team {name: $team})-[r:PLAYED_AGAINST]-(opp:Team)
    WHERE r.season = $season
    RETURN r.match_date   AS date,
           r.match_result AS result,
           r.total_goals  AS total_goals,
           opp.name       AS opponent,
           CASE WHEN startNode(r) = t THEN 'Home' ELSE 'Away' END AS venue
    ORDER BY r.match_date
    """
    return cypher, {"team": team, "season": season}


def team_opponents(team: str, limit: int = 10) -> tuple[str, dict]:
    """
    与某球队交手最多的对手 TOP N。
    """
    cypher = """
    MATCH (t:Team {name: $team})-[r:PLAYED_AGAINST]-(opp:Team)
    RETURN opp.name AS opponent, count(r) AS match_count
    ORDER BY match_count DESC
    LIMIT $limit
    """
    return cypher, {"team": team, "limit": limit}


def team_goal_stats(team: str, season: str) -> tuple[str, dict]:
    """
    球队某赛季进球 / 失球统计。
    利用 match_result 格式 "{主队} {主队进球}:{客队进球} {客队}" 进行解析。
    """
    cypher = """
    MATCH (t:Team {name: $team})-[r:PLAYED_AGAINST]-(opp:Team)
    WHERE r.season = $season
    RETURN r.match_date   AS date,
           r.match_result AS result,
           r.total_goals  AS total_goals,
           opp.name       AS opponent,
           CASE WHEN startNode(r) = t THEN 'Home' ELSE 'Away' END AS venue
    ORDER BY r.match_date
    """
    return cypher, {"team": team, "season": season}

