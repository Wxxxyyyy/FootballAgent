# -*- coding: utf-8 -*-
"""
比赛相关 Cypher 静态查询模板
───────────────────────────
图谱方向约定（neo4j_loader 写入时确定）：
  (HomeTeam)-[:PLAYED_AGAINST]->(AwayTeam)

每个函数返回 (cypher: str, params: dict)，
由 tool_entry 统一提交给 Neo4j 执行。
"""


def head_to_head(team_a: str, team_b: str, limit: int = 20, years_back: int | None = None, score_only: bool = False) -> tuple[str, dict]:
    """
    历史交锋记录（不限主客场）。
    使用无向匹配 `-[r:PLAYED_AGAINST]-` 同时捕获双方主客场记录。
    
    Args:
        team_a: 第一支球队英文名
        team_b: 第二支球队英文名
        limit: 返回记录数上限
        years_back: 最近N年（可选），如果提供则只返回最近N年的记录
        score_only: 是否只返回比分（日期、赛季、比分），不返回赔率等详细信息
    """
    params = {"team_a": team_a, "team_b": team_b, "limit": limit}
    
    # 确定返回字段
    if score_only:
        return_fields = """
               r.match_date   AS date,
               r.season       AS season,
               r.match_result AS result"""
    else:
        return_fields = """
               r.match_date   AS date,
               r.season       AS season,
               r.match_result AS result,
               r.total_goals  AS total_goals,
               r.odds_info    AS odds,
               r.asian_handicap_info AS asian_handicap,
               r.over_under_odds     AS over_under"""
    
    # 如果有年份限制，计算起始日期
    if years_back is not None:
        from datetime import datetime
        # 计算起始年份：当前年份往前推 N 年（例如：2026年往前推2年 = 2024年）
        current_year = datetime.now().year
        start_year = current_year - years_back
        # 使用年份的第一天作为起始日期，确保包含完整年份
        start_date_str = f"{start_year}-01-01"
        
        cypher = f"""
        MATCH (a:Team {{name: $team_a}})-[r:PLAYED_AGAINST]-(b:Team {{name: $team_b}})
        WHERE r.match_date >= $start_date
        RETURN{return_fields}
        ORDER BY r.match_date DESC
        LIMIT $limit
        """
        params["start_date"] = start_date_str
    else:
        cypher = f"""
        MATCH (a:Team {{name: $team_a}})-[r:PLAYED_AGAINST]-(b:Team {{name: $team_b}})
        RETURN{return_fields}
        ORDER BY r.match_date DESC
        LIMIT $limit
        """
    
    return cypher, params


def recent_matches(team: str, n: int = 10) -> tuple[str, dict]:
    """
    球队最近 N 场比赛（不限主客场）。
    """
    cypher = """
    MATCH (t:Team {name: $team})-[r:PLAYED_AGAINST]-(opp:Team)
    RETURN r.match_date   AS date,
           r.season       AS season,
           r.match_result AS result,
           opp.name       AS opponent,
           r.odds_info    AS odds,
           r.over_under_odds AS over_under
    ORDER BY r.match_date DESC
    LIMIT $n
    """
    return cypher, {"team": team, "n": n}


def season_matches(team: str, season: str) -> tuple[str, dict]:
    """
    球队某赛季全部比赛。
    """
    cypher = """
    MATCH (t:Team {name: $team})-[r:PLAYED_AGAINST {season: $season}]-(opp:Team)
    RETURN r.match_date   AS date,
           r.match_result AS result,
           opp.name       AS opponent,
           r.total_goals  AS total_goals,
           r.odds_info    AS odds,
           r.over_under_odds AS over_under
    ORDER BY r.match_date
    """
    return cypher, {"team": team, "season": season}


def home_record(team: str, limit: int = 10) -> tuple[str, dict]:
    """
    球队主场战绩。
    方向：(home)-[:PLAYED_AGAINST]->(away)，所以主场 = 正向。
    """
    cypher = """
    MATCH (t:Team {name: $team})-[r:PLAYED_AGAINST]->(away:Team)
    RETURN r.match_date   AS date,
           r.season       AS season,
           r.match_result AS result,
           away.name      AS opponent,
           r.odds_info    AS odds
    ORDER BY r.match_date DESC
    LIMIT $limit
    """
    return cypher, {"team": team, "limit": limit}


def away_record(team: str, limit: int = 10) -> tuple[str, dict]:
    """
    球队客场战绩。
    方向：(home)-[:PLAYED_AGAINST]->(away)，所以客场 = 反向（被指向）。
    """
    cypher = """
    MATCH (home:Team)-[r:PLAYED_AGAINST]->(t:Team {name: $team})
    RETURN r.match_date   AS date,
           r.season       AS season,
           r.match_result AS result,
           home.name      AS opponent,
           r.odds_info    AS odds
    ORDER BY r.match_date DESC
    LIMIT $limit
    """
    return cypher, {"team": team, "limit": limit}


def match_with_odds(team_a: str, team_b: str, season: str = "", years_back: int | None = None) -> tuple[str, dict]:
    """
    两队交锋（含完整赔率数据），可选按赛季过滤或按年份范围过滤。
    
    Args:
        team_a: 第一支球队英文名
        team_b: 第二支球队英文名
        season: 赛季标识（如 "2024-2025"），如果提供则按赛季过滤
        years_back: 最近N年（可选），如果提供则只返回最近N年的记录
    """
    params = {"team_a": team_a, "team_b": team_b}
    where_clauses = []
    
    # 赛季过滤
    if season:
        where_clauses.append("r.season = $season")
        params["season"] = season
    
    # 年份范围过滤（优先级低于赛季过滤）
    if years_back is not None and not season:
        from datetime import datetime
        # 计算起始年份：当前年份往前推 N 年（例如：2026年往前推2年 = 2024年）
        current_year = datetime.now().year
        start_year = current_year - years_back
        # 使用年份的第一天作为起始日期，确保包含完整年份
        start_date_str = f"{start_year}-01-01"
        where_clauses.append("r.match_date >= $start_date")
        params["start_date"] = start_date_str
    
    where_clause = ""
    if where_clauses:
        where_clause = "WHERE " + " AND ".join(where_clauses)
    
    cypher = f"""
    MATCH (a:Team {{name: $team_a}})-[r:PLAYED_AGAINST]-(b:Team {{name: $team_b}})
    {where_clause}
    RETURN r.match_date        AS date,
           r.season            AS season,
           r.match_result      AS result,
           r.odds_info         AS odds,
           r.closing_odds_info AS closing_odds,
           r.asian_handicap_info AS asian_handicap,
           r.over_under_odds     AS over_under,
           r.closing_over_under  AS closing_over_under
    ORDER BY r.match_date DESC
    """
    
    return cypher, params

