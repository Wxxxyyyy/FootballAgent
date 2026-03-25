# -*- coding: utf-8 -*-
"""
Neo4j 查询安全校验 · 四道防线
══════════════════════════════
防线1 (读写隔离)  : 正则扫描，拦截一切写操作关键字
防线2 (方向纠正)  : 离线 Schema 比对，纠正/拦截错误的关系类型与箭头
防线3 (语法验证)  : EXPLAIN 前缀发送给 Neo4j，只做编译不跑数据
防线4 (值映射校验) : 探测查询，核实 WHERE 子句中实体是否真实存在

任何一道防线失败都会抛出对应的自定义异常，
供 text2cypher.py 的重试回环捕获并组装上下文重新生成。
"""

import re
from typing import Optional

# ═══════════════════════════════════════════════════════════════
#  自定义异常
# ═══════════════════════════════════════════════════════════════

class CypherSecurityError(Exception):
    """防线1: 检测到写操作（CREATE / DELETE / SET / MERGE 等）"""
    pass


class CypherDirectionError(Exception):
    """防线2: 关系类型不存在或箭头方向与 Schema 不符"""
    pass


class CypherSyntaxError(Exception):
    """防线3: Cypher 语法错误（EXPLAIN 编译失败）"""
    pass


class CypherMappingError(Exception):
    """防线4: WHERE 子句中的实体值在库中不存在"""
    pass


# ═══════════════════════════════════════════════════════════════
#  图谱 Schema（离线常量，与 neo4j_loader.py 保持一致）
# ═══════════════════════════════════════════════════════════════

# 合法的关系类型 → (起点标签, 终点标签)
VALID_RELATIONSHIPS = {
    "PLAYED_AGAINST": ("Team", "Team"),
}

# 合法的节点标签
VALID_LABELS = {"Team"}

# 合法的关系属性
VALID_REL_PROPERTIES = {
    "match_date", "season", "match_result", "total_goals",
    "odds_info", "over_under_odds", "closing_odds_info",
    "closing_over_under", "asian_handicap_info", "league",
}

# 合法的节点属性
VALID_NODE_PROPERTIES = {"name", "league"}


# ═══════════════════════════════════════════════════════════════
#  防线 1 — 读写隔离
# ═══════════════════════════════════════════════════════════════

# 写操作关键字正则（忽略大小写，匹配完整单词）
_WRITE_PATTERN = re.compile(
    r'\b(CREATE|DELETE|DETACH\s+DELETE|SET|REMOVE|MERGE|DROP|'
    r'CALL\s*\{|FOREACH)\b',
    re.IGNORECASE,
)


def check_read_only(cypher: str) -> None:
    """
    防线1: 扫描 Cypher 是否包含写操作关键字。
    只允许 MATCH / RETURN / WHERE / WITH / ORDER BY / LIMIT / EXPLAIN / UNION 等读操作。

    Raises:
        CypherSecurityError: 检测到写操作
    """
    match = _WRITE_PATTERN.search(cypher)
    if match:
        raise CypherSecurityError(
            f"[防线1·读写隔离] 检测到写操作关键字: '{match.group().strip()}'。"
            f"只允许执行只读查询。请移除写操作后重试。"
        )


# ═══════════════════════════════════════════════════════════════
#  防线 2 — 方向纠正（离线 Schema 校验，不查库）
# ═══════════════════════════════════════════════════════════════

# 匹配关系模式: -[xxx:REL_TYPE]-> 或 <-[xxx:REL_TYPE]-
_REL_PATTERN = re.compile(
    r'(<-\s*\[)'               # 左箭头开始 <-[
    r'([^:\]]*)'               # 变量名（可为空）
    r':(\w+)'                  # :关系类型
    r'([^\]]*)\]'              # 属性过滤等
    r'\s*-'                    # 右端 -
    r'|'                       # ---- 或 ----
    r'-\s*\['                  # 右箭头开始 -[
    r'([^:\]]*)'               # 变量名
    r':(\w+)'                  # :关系类型
    r'([^\]]*)\]'              # 属性过滤等
    r'\s*->'                   # 右端 ->
    r'|'                       # ---- 或 ----
    r'-\s*\['                  # 无向 -[
    r'([^:\]]*)'               # 变量名
    r':(\w+)'                  # :关系类型
    r'([^\]]*)\]'              # 属性过滤等
    r'\s*-(?!>)',              # 右端 -（不跟 >）
    re.IGNORECASE,
)

# 更简洁的关系类型提取
_REL_TYPE_SIMPLE = re.compile(r'\[:?(\w+)', re.IGNORECASE)


def check_direction(cypher: str) -> str:
    """
    防线2: 离线校验关系类型是否存在于 Schema 中。
    - 如果出现未知关系类型 → 抛出 CypherDirectionError
    - 如果关系类型正确，返回原始 Cypher（方向在查询场景下通常双向均合法）

    Args:
        cypher: 待检查的 Cypher 语句

    Returns:
        str: 校验通过后的 Cypher（可能经过修正）

    Raises:
        CypherDirectionError: 关系类型不在 Schema 中
    """
    # 提取所有 [:XXX] 中的关系类型
    rel_types_in_cypher = set()
    for match in re.finditer(r'\[\s*\w*\s*:\s*(\w+)', cypher):
        rel_types_in_cypher.add(match.group(1))

    if not rel_types_in_cypher:
        # 没有显式关系类型，可能是 ()-[]-() 之类的，不做校验
        return cypher

    unknown = rel_types_in_cypher - set(VALID_RELATIONSHIPS.keys())
    if unknown:
        known_list = ", ".join(VALID_RELATIONSHIPS.keys())
        raise CypherDirectionError(
            f"[防线2·方向纠正] 检测到未知关系类型: {unknown}。"
            f"Schema 中合法的关系类型为: [{known_list}]。"
            f"请仅使用已有的关系类型重新生成 Cypher。"
        )

    # 检查节点标签是否合法
    label_matches = re.findall(r'\(\s*\w*\s*:\s*(\w+)', cypher)
    if label_matches:
        labels_in_cypher = set(label_matches)
        unknown_labels = labels_in_cypher - VALID_LABELS
        if unknown_labels:
            raise CypherDirectionError(
                f"[防线2·方向纠正] 检测到未知节点标签: {unknown_labels}。"
                f"Schema 中合法的节点标签为: {VALID_LABELS}。"
                f"请仅使用已有的节点标签重新生成 Cypher。"
            )

    return cypher


# ═══════════════════════════════════════════════════════════════
#  防线 3 — 语法验证（EXPLAIN，只编译不跑数据）
# ═══════════════════════════════════════════════════════════════

def validate_syntax(cypher: str, driver, database: str = "neo4j") -> None:
    """
    防线3: 为 Cypher 加上 EXPLAIN 前缀发送给 Neo4j。
    只做查询规划（编译），不会实际扫描数据，从而零成本验证语法。

    Args:
        cypher:   待验证的 Cypher 语句
        driver:   Neo4j Driver 实例
        database: 数据库名

    Raises:
        CypherSyntaxError: Cypher 语法错误
    """
    explain_cypher = f"EXPLAIN {cypher}"
    try:
        with driver.session(database=database) as session:
            session.run(explain_cypher).consume()
    except Exception as e:
        error_msg = str(e)
        raise CypherSyntaxError(
            f"[防线3·语法验证] Cypher 编译失败:\n"
            f"  错误信息: {error_msg}\n"
            f"  问题语句: {cypher}"
        )


# ═══════════════════════════════════════════════════════════════
#  防线 4 — 值映射校验（探测查询 Probe Query）
# ═══════════════════════════════════════════════════════════════

# 匹配 Cypher 中内联的字符串值（用于 {name: "xxx"} 或 WHERE t.name = "xxx"）
_INLINE_VALUE_PATTERN = re.compile(
    r"""(?:name|league)\s*[:=]\s*["']([^"']+)["']""",
    re.IGNORECASE,
)


def validate_values(
    cypher: str,
    params: Optional[dict],
    driver,
    database: str = "neo4j",
) -> None:
    """
    防线4: 提取 Cypher 中的实体过滤值，发送探测查询核实库中是否存在。
    - 提取内联字符串值（如 {name: "Arsenal"}）
    - 提取参数化的值（如 $team → params["team"]）
    - 对每个值发送极简 Probe Query 验证存在性

    Args:
        cypher:   待验证的 Cypher 语句
        params:   参数化查询的参数字典（可选）
        driver:   Neo4j Driver 实例
        database: 数据库名

    Raises:
        CypherMappingError: 实体值在库中不存在
    """
    values_to_check = []

    # 1. 提取内联值
    for match in _INLINE_VALUE_PATTERN.finditer(cypher):
        values_to_check.append(match.group(1))

    # 2. 提取参数化值（只关注与 name / league 相关的参数）
    if params:
        for key, val in params.items():
            if isinstance(val, str) and key in (
                "team", "team_a", "team_b", "home_team", "away_team",
                "name", "league",
            ):
                values_to_check.append(val)

    if not values_to_check:
        return  # 无需校验

    # 3. 对每个值发送 Probe Query
    missing = []
    with driver.session(database=database) as session:
        for val in set(values_to_check):
            # 先探测是否为球队名
            probe_team = "MATCH (t:Team {name: $val}) RETURN t LIMIT 1"
            result = session.run(probe_team, val=val).single()
            if result:
                continue

            # 再探测是否为联赛名
            probe_league = "MATCH (t:Team {league: $val}) RETURN t LIMIT 1"
            result = session.run(probe_league, val=val).single()
            if result:
                continue

            missing.append(val)

    if missing:
        raise CypherMappingError(
            f"[防线4·值映射校验] 以下实体在图谱中不存在: {missing}。"
            f"请检查球队名称是否为英文原名（如 'Arsenal' 而非 '阿森纳'），"
            f"联赛名是否为 'England'/'Spain'/'Italy'/'Germany'/'France'。"
        )


# ═══════════════════════════════════════════════════════════════
#  统一执行：四道防线一次性跑完
# ═══════════════════════════════════════════════════════════════

def run_all_defenses(
    cypher: str,
    params: Optional[dict],
    driver,
    database: str = "neo4j",
) -> str:
    """
    按顺序执行四道防线。
    - 防线1 和 防线2 纯离线，不需要 driver
    - 防线3 和 防线4 需要连接 Neo4j

    任何一道防线失败都会直接抛出对应异常，
    由 text2cypher.py 的重试回环统一捕获。

    Args:
        cypher:   LLM 生成的 Cypher 语句
        params:   参数字典（可选）
        driver:   Neo4j Driver 实例
        database: 数据库名

    Returns:
        str: 通过全部防线后的 Cypher（可能经防线2修正）

    Raises:
        CypherSecurityError  : 防线1 失败
        CypherDirectionError : 防线2 失败
        CypherSyntaxError    : 防线3 失败
        CypherMappingError   : 防线4 失败
    """
    # 防线1: 读写隔离
    check_read_only(cypher)

    # 防线2: 方向/关系类型纠正
    cypher = check_direction(cypher)

    # 防线3: 语法验证
    validate_syntax(cypher, driver, database)

    # 防线4: 值映射校验
    validate_values(cypher, params, driver, database)

    return cypher

