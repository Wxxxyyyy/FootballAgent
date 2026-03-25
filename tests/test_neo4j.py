# -*- coding: utf-8 -*-
"""
Neo4j 图谱查询工具 · 全模块测试
═══════════════════════════════════
测试覆盖:
  [离线测试 - 无需 Neo4j / LLM]
    Test 1  : 防线1 — 读写隔离（各种写操作关键字拦截）
    Test 2  : 防线2 — 关系类型/节点标签校验
    Test 3  : 中英文球队名映射加载与解析
    Test 4  : 从自然语言中提取球队实体
    Test 5  : 赛季标识提取（多种格式）
    Test 6  : 数字提取（近N场）
    Test 7  : 联赛名映射
    Test 8  : Cypher 模板函数返回格式校验
    Test 9  : LLM 输出中 Cypher 提取（去 markdown / 注释）
    Test 10 : Fast Path 模板匹配（意图路由覆盖）

  [在线测试 - 需要 Neo4j 连接]
    Test 11 : 防线3 — EXPLAIN 语法验证
    Test 12 : 防线4 — 值映射探测查询
    Test 13 : run_all_defenses 四道防线联合执行
    Test 14 : Cypher 执行与结果格式化
    Test 15 : @tool 完整查询流程（Fast Path 端到端）

运行方式:
  conda activate football
  python tests/test_neo4j.py
"""

import sys
import os
import traceback
import argparse

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════
#  测试统计
# ═══════════════════════════════════════════════════════════════

_pass_count = 0
_fail_count = 0
_skip_count = 0


def _report(test_id: str, title: str, passed: bool, detail: str = ""):
    """统一输出测试结果"""
    global _pass_count, _fail_count
    if passed:
        _pass_count += 1
        tag = "✅ PASS"
    else:
        _fail_count += 1
        tag = "❌ FAIL"
    print(f"  {tag}  [{test_id}] {title}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")


def _skip(test_id: str, title: str, reason: str):
    """跳过的测试"""
    global _skip_count
    _skip_count += 1
    print(f"  ⏭️ SKIP [{test_id}] {title} — {reason}")


# ═══════════════════════════════════════════════════════════════
#  离线测试（无需 Neo4j / LLM）
# ═══════════════════════════════════════════════════════════════

def test_01_read_only_check():
    """防线1: 读写隔离 — 各种写操作关键字拦截"""
    from agents.tools.neo4j_tools.security import check_read_only, CypherSecurityError

    # 应该通过的只读语句
    safe_queries = [
        "MATCH (t:Team) RETURN t",
        "MATCH (a)-[r:PLAYED_AGAINST]-(b) WHERE a.name = 'Arsenal' RETURN r",
        "MATCH (t:Team) WITH t ORDER BY t.name LIMIT 10 RETURN t",
        "MATCH (a:Team) RETURN a UNION MATCH (b:Team) RETURN b",
    ]
    for q in safe_queries:
        try:
            check_read_only(q)
        except CypherSecurityError:
            _report("1a", f"只读语句应通过: {q[:50]}", False)
            return
    _report("1a", "只读语句全部通过 (4条)", True)

    # 应该被拦截的写操作
    write_queries = {
        "CREATE": "CREATE (t:Team {name: 'Fake'})",
        "DELETE": "MATCH (t:Team) DELETE t",
        "DETACH DELETE": "MATCH (t:Team) DETACH DELETE t",
        "SET": "MATCH (t:Team {name: 'Arsenal'}) SET t.league = 'X'",
        "REMOVE": "MATCH (t:Team) REMOVE t.league",
        "MERGE": "MERGE (t:Team {name: 'Fake'})",
        "DROP": "DROP CONSTRAINT my_constraint",
        "FOREACH": "FOREACH (x IN [1,2] | CREATE (t:Team))",
    }
    blocked = 0
    for keyword, q in write_queries.items():
        try:
            check_read_only(q)
        except CypherSecurityError:
            blocked += 1
    total = len(write_queries)
    _report("1b", f"写操作拦截 ({blocked}/{total})", blocked == total,
            f"未拦截数: {total - blocked}" if blocked < total else "")


def test_02_direction_check():
    """防线2: 关系类型/节点标签校验"""
    from agents.tools.neo4j_tools.security import check_direction, CypherDirectionError

    # 合法关系类型
    try:
        result = check_direction(
            "MATCH (a:Team)-[r:PLAYED_AGAINST]-(b:Team) RETURN r"
        )
        _report("2a", "合法关系类型 PLAYED_AGAINST 通过", True)
    except CypherDirectionError as e:
        _report("2a", "合法关系类型应通过", False, str(e))

    # 合法但有方向箭头
    try:
        check_direction(
            "MATCH (a:Team)-[r:PLAYED_AGAINST]->(b:Team) RETURN r"
        )
        _report("2b", "PLAYED_AGAINST 正向箭头通过", True)
    except CypherDirectionError as e:
        _report("2b", "正向箭头应通过", False, str(e))

    # 反向箭头也应通过
    try:
        check_direction(
            "MATCH (a:Team)<-[r:PLAYED_AGAINST]-(b:Team) RETURN r"
        )
        _report("2c", "PLAYED_AGAINST 反向箭头通过", True)
    except CypherDirectionError as e:
        _report("2c", "反向箭头应通过", False, str(e))

    # 未知关系类型应被拦截
    try:
        check_direction("MATCH (a:Team)-[r:VS]-(b:Team) RETURN r")
        _report("2d", "未知关系类型 VS 应被拦截", False)
    except CypherDirectionError:
        _report("2d", "未知关系类型 VS 被正确拦截", True)

    # LLM 常见错误: PLAYS_AGAINST
    try:
        check_direction("MATCH (a:Team)-[r:PLAYS_AGAINST]-(b:Team) RETURN r")
        _report("2e", "未知关系类型 PLAYS_AGAINST 应被拦截", False)
    except CypherDirectionError:
        _report("2e", "未知关系类型 PLAYS_AGAINST 被正确拦截", True)

    # 未知节点标签
    try:
        check_direction("MATCH (p:Player)-[r:PLAYED_AGAINST]-(t:Team) RETURN r")
        _report("2f", "未知节点标签 Player 应被拦截", False)
    except CypherDirectionError:
        _report("2f", "未知节点标签 Player 被正确拦截", True)

    # 无显式关系类型的查询应直接通过
    try:
        check_direction("MATCH (a:Team)--(b:Team) RETURN a, b")
        _report("2g", "无显式关系类型的查询通过", True)
    except CypherDirectionError as e:
        _report("2g", "无显式关系类型应通过", False, str(e))


def test_03_name_map_loading():
    """中英文球队名映射加载与解析"""
    from agents.tools.neo4j_tools.tool_entry import _load_name_map, _resolve_team_name

    name_map = _load_name_map()

    # 检查映射总数（应 > 100 条）
    _report("3a", f"映射总数: {len(name_map)} 条", len(name_map) > 100)

    # 核心映射验证
    test_cases = {
        "阿森纳": "Arsenal",
        "枪手": "Arsenal",
        "Arsenal": "Arsenal",
        "arsenal": "Arsenal",       # 小写英文
        "切尔西": "Chelsea",
        "蓝军": "Chelsea",
        "巴萨": "Barcelona",
        "拜仁": "Bayern Munich",
        "曼联": "Man United",
        "红魔": "Man United",
        "皇马": "Real Madrid",
        "尤文": "Juventus",
        "大巴黎": "Paris SG",
    }
    passed = 0
    failed_details = []
    for zh, expected_en in test_cases.items():
        resolved = _resolve_team_name(zh)
        if resolved == expected_en:
            passed += 1
        else:
            failed_details.append(f"'{zh}' → '{resolved}'（期望 '{expected_en}'）")

    total = len(test_cases)
    _report("3b", f"核心映射验证 ({passed}/{total})", passed == total,
            "\n".join(failed_details) if failed_details else "")


def test_04_extract_teams():
    """从自然语言中提取球队实体"""
    from agents.tools.neo4j_tools.tool_entry import _extract_teams

    test_cases = [
        ("阿森纳和切尔西的历史交锋记录", ["Arsenal", "Chelsea"]),
        ("拜仁近5场比赛", ["Bayern Munich"]),
        ("巴萨本赛季客场战绩", ["Barcelona"]),
        ("皇马vs巴萨谁更强", ["Real Madrid", "Barcelona"]),
        ("利物浦和曼联的比赛", ["Liverpool", "Man United"]),
    ]

    passed = 0
    failed_details = []
    for question, expected in test_cases:
        result = _extract_teams(question)
        # 只检查预期球队是否全部被提取到（不严格检查顺序和多余匹配）
        if all(t in result for t in expected):
            passed += 1
        else:
            failed_details.append(
                f"'{question}' → {result}（期望包含 {expected}）"
            )

    total = len(test_cases)
    _report("4", f"球队实体提取 ({passed}/{total})", passed == total,
            "\n".join(failed_details) if failed_details else "")


def test_05_extract_season():
    """赛季标识提取（多种格式）"""
    from agents.tools.neo4j_tools.tool_entry import _extract_season

    test_cases = [
        ("2024-2025赛季", "2024-2025"),
        ("24-25赛季", "2024-2025"),
        ("2024/2025", "2024-2025"),
        ("本赛季", "2025-2026"),
        ("这个赛季", "2025-2026"),
        ("上赛季", "2024-2025"),
        ("上个赛季", "2024-2025"),
        ("今天天气真好", None),  # 无赛季信息
    ]

    passed = 0
    failed_details = []
    for question, expected in test_cases:
        result = _extract_season(question)
        if result == expected:
            passed += 1
        else:
            failed_details.append(
                f"'{question}' → '{result}'（期望 '{expected}'）"
            )

    total = len(test_cases)
    _report("5", f"赛季提取 ({passed}/{total})", passed == total,
            "\n".join(failed_details) if failed_details else "")


def test_06_extract_number():
    """数字提取（近N场）"""
    from agents.tools.neo4j_tools.tool_entry import _extract_number

    test_cases = [
        ("近5场比赛", 5),
        ("最近3", 3),
        ("近10场", 10),
        ("最近20场表现", 20),
        ("历史交锋", 10),  # 无数字时返回默认值
    ]

    passed = 0
    failed_details = []
    for question, expected in test_cases:
        result = _extract_number(question)
        if result == expected:
            passed += 1
        else:
            failed_details.append(
                f"'{question}' → {result}（期望 {expected}）"
            )

    total = len(test_cases)
    _report("6", f"数字提取 ({passed}/{total})", passed == total,
            "\n".join(failed_details) if failed_details else "")


def test_07_resolve_league():
    """联赛名映射"""
    from agents.tools.neo4j_tools.tool_entry import _resolve_league

    test_cases = [
        ("英超有哪些球队", "England"),
        ("西甲球队列表", "Spain"),
        ("意甲球队", "Italy"),
        ("德甲球队名单", "Germany"),
        ("法甲有哪些队伍", "France"),
        ("Premier League teams", "England"),
        ("没有联赛关键词的问题", None),
    ]

    passed = 0
    failed_details = []
    for question, expected in test_cases:
        result = _resolve_league(question)
        if result == expected:
            passed += 1
        else:
            failed_details.append(
                f"'{question}' → '{result}'（期望 '{expected}'）"
            )

    total = len(test_cases)
    _report("7", f"联赛映射 ({passed}/{total})", passed == total,
            "\n".join(failed_details) if failed_details else "")


def test_08_template_format():
    """Cypher 模板函数返回格式校验"""
    from agents.tools.neo4j_tools.templates import match_queries, team_queries

    templates_to_test = [
        ("match.head_to_head", match_queries.head_to_head("Arsenal", "Chelsea")),
        ("match.recent_matches", match_queries.recent_matches("Arsenal", 5)),
        ("match.season_matches", match_queries.season_matches("Arsenal", "2024-2025")),
        ("match.home_record", match_queries.home_record("Arsenal")),
        ("match.away_record", match_queries.away_record("Arsenal")),
        ("match.match_with_odds", match_queries.match_with_odds("Arsenal", "Chelsea")),
        ("match.match_with_odds(season)", match_queries.match_with_odds("Arsenal", "Chelsea", "2024-2025")),
        ("team.team_info", team_queries.team_info("Arsenal")),
        ("team.league_teams", team_queries.league_teams("England")),
        ("team.team_season_all", team_queries.team_season_all("Arsenal", "2024-2025")),
        ("team.team_opponents", team_queries.team_opponents("Arsenal")),
        ("team.team_goal_stats", team_queries.team_goal_stats("Arsenal", "2024-2025")),
    ]

    passed = 0
    failed_details = []
    for name, result in templates_to_test:
        ok = True
        reasons = []
        # 检查返回值是 tuple(str, dict)
        if not isinstance(result, tuple) or len(result) != 2:
            ok = False
            reasons.append("返回值不是 (str, dict) 元组")
        else:
            cypher, params = result
            if not isinstance(cypher, str) or not cypher.strip():
                ok = False
                reasons.append("Cypher 为空")
            if not isinstance(params, dict):
                ok = False
                reasons.append("params 不是 dict")
            # Cypher 应包含 MATCH 和 RETURN
            if "MATCH" not in cypher.upper():
                ok = False
                reasons.append("缺少 MATCH 关键字")
            if "RETURN" not in cypher.upper():
                ok = False
                reasons.append("缺少 RETURN 关键字")

        if ok:
            passed += 1
        else:
            failed_details.append(f"{name}: {', '.join(reasons)}")

    total = len(templates_to_test)
    _report("8", f"模板格式校验 ({passed}/{total})", passed == total,
            "\n".join(failed_details) if failed_details else "")


def test_09_extract_cypher():
    """LLM 输出中 Cypher 提取（去 markdown / 注释）"""
    from agents.tools.neo4j_tools.text2cypher import _extract_cypher

    # 纯 Cypher
    assert_eq = []

    raw1 = "MATCH (t:Team) RETURN t"
    r1 = _extract_cypher(raw1)
    assert_eq.append(("纯 Cypher", r1 == raw1, f"'{r1}'"))

    # markdown 包裹
    raw2 = "```cypher\nMATCH (t:Team) RETURN t\n```"
    r2 = _extract_cypher(raw2)
    assert_eq.append(("markdown cypher 块", "MATCH (t:Team) RETURN t" in r2, f"'{r2}'"))

    # neo4j 标签的 markdown
    raw3 = "```neo4j\nMATCH (t:Team)\nRETURN t\n```"
    r3 = _extract_cypher(raw3)
    assert_eq.append(("markdown neo4j 块", "MATCH" in r3 and "RETURN" in r3, f"'{r3}'"))

    # 带行首注释
    raw4 = "// 查询所有球队\nMATCH (t:Team)\nRETURN t"
    r4 = _extract_cypher(raw4)
    assert_eq.append(("去行首注释", "MATCH" in r4 and "//" not in r4, f"'{r4}'"))

    # 混合: markdown + 注释 + 多余文字
    raw5 = "以下是查询:\n```cypher\n// 历史交锋\nMATCH (a:Team)-[r:PLAYED_AGAINST]-(b:Team)\nRETURN r\n```\n希望对你有帮助"
    r5 = _extract_cypher(raw5)
    assert_eq.append(("混合格式", "MATCH" in r5 and "//" not in r5, f"'{r5}'"))

    passed = sum(1 for _, ok, _ in assert_eq if ok)
    total = len(assert_eq)
    details = [f"{name}: {detail}" for name, ok, detail in assert_eq if not ok]
    _report("9", f"Cypher 提取 ({passed}/{total})", passed == total,
            "\n".join(details) if details else "")


def test_10_template_matching():
    """Fast Path 模板匹配（意图路由覆盖）"""
    from agents.tools.neo4j_tools.tool_entry import _try_template_match

    test_cases = [
        # (问题, 是否应命中模板, 预期参数中应包含的 key)
        ("阿森纳和切尔西的历史交锋记录", True, ["team_a", "team_b"]),
        ("Arsenal vs Chelsea 对阵记录", True, ["team_a", "team_b"]),
        ("拜仁近5场比赛", True, ["team", "n"]),
        ("巴萨最近表现如何", True, ["team", "n"]),
        ("利物浦主场战绩", True, ["team"]),
        ("曼联客场战绩", True, ["team"]),
        ("阿森纳2024-2025赛季", True, ["team", "season"]),
        ("英超有哪些球队", True, ["league"]),
        ("阿森纳属于什么联赛", True, ["team"]),
        ("切尔西经常跟谁交手", True, ["team"]),
        ("阿森纳和切尔西的赔率数据", True, ["team_a", "team_b"]),
        ("今天天气真好", False, []),     # 无法匹配模板
        ("你好啊", False, []),            # 纯闲聊
    ]

    passed = 0
    failed_details = []
    for question, should_hit, expected_keys in test_cases:
        result = _try_template_match(question)
        hit = result is not None

        if hit == should_hit:
            if hit:
                _, params = result
                # 检查预期参数 key 是否存在
                missing_keys = [k for k in expected_keys if k not in params]
                if missing_keys:
                    failed_details.append(
                        f"'{question}' → 命中但缺少参数 {missing_keys}"
                    )
                else:
                    passed += 1
            else:
                passed += 1
        else:
            if should_hit:
                failed_details.append(f"'{question}' → 未命中（期望命中）")
            else:
                failed_details.append(f"'{question}' → 意外命中（期望未命中）")

    total = len(test_cases)
    _report("10", f"模板匹配覆盖 ({passed}/{total})", passed == total,
            "\n".join(failed_details) if failed_details else "")


# ═══════════════════════════════════════════════════════════════
#  在线测试（需要 Neo4j 连接）
# ═══════════════════════════════════════════════════════════════

def _get_neo4j_driver():
    """尝试获取 Neo4j 连接，失败返回 None"""
    try:
        from agents.tools.neo4j_tools.tool_entry import _get_driver, NEO4J_DATABASE
        driver = _get_driver()
        # 快速验证连接
        with driver.session(database=NEO4J_DATABASE) as session:
            session.run("RETURN 1").consume()
        return driver, NEO4J_DATABASE
    except Exception as e:
        print(f"  ⚠️ Neo4j 连接失败: {e}")
        return None, None


def test_11_syntax_validation(driver, database):
    """防线3: EXPLAIN 语法验证"""
    from agents.tools.neo4j_tools.security import validate_syntax, CypherSyntaxError

    # 正确语法
    try:
        validate_syntax("MATCH (t:Team) RETURN t LIMIT 5", driver, database)
        _report("11a", "正确语法 EXPLAIN 通过", True)
    except CypherSyntaxError as e:
        _report("11a", "正确语法应通过", False, str(e))

    # 更复杂的正确语法
    try:
        validate_syntax(
            "MATCH (a:Team {name: 'Arsenal'})-[r:PLAYED_AGAINST]-(b:Team) "
            "RETURN r.match_result ORDER BY r.match_date DESC LIMIT 10",
            driver, database,
        )
        _report("11b", "复杂正确语法通过", True)
    except CypherSyntaxError as e:
        _report("11b", "复杂正确语法应通过", False, str(e))

    # 拼写错误的语法
    try:
        validate_syntax("MATC (t:Team) RETRUN t", driver, database)
        _report("11c", "拼写错误应被拦截", False)
    except CypherSyntaxError:
        _report("11c", "拼写错误被正确拦截", True)

    # 不完整的语法
    try:
        validate_syntax("MATCH (t:Team) WHERE", driver, database)
        _report("11d", "不完整语法应被拦截", False)
    except CypherSyntaxError:
        _report("11d", "不完整语法被正确拦截", True)


def test_12_value_validation(driver, database):
    """防线4: 值映射探测查询"""
    from agents.tools.neo4j_tools.security import validate_values, CypherMappingError

    # 存在的球队 — 内联值
    try:
        validate_values(
            'MATCH (t:Team {name: "Arsenal"}) RETURN t', None, driver, database
        )
        _report("12a", "Arsenal（内联值）存在", True)
    except CypherMappingError as e:
        _report("12a", "Arsenal 应存在于图谱中", False, str(e))

    # 存在的球队 — 参数化值
    try:
        validate_values(
            "MATCH (t:Team {name: $team}) RETURN t",
            {"team": "Chelsea"},
            driver, database,
        )
        _report("12b", "Chelsea（参数化值）存在", True)
    except CypherMappingError as e:
        _report("12b", "Chelsea 应存在于图谱中", False, str(e))

    # 不存在的球队 — 应拦截
    try:
        validate_values(
            'MATCH (t:Team {name: "FakeTeamXYZ"}) RETURN t', None, driver, database
        )
        _report("12c", "不存在球队应被拦截", False)
    except CypherMappingError:
        _report("12c", "不存在球队 FakeTeamXYZ 被正确拦截", True)

    # 存在的联赛
    try:
        validate_values(
            'MATCH (t:Team {league: "England"}) RETURN t', None, driver, database
        )
        _report("12d", "联赛 England 存在", True)
    except CypherMappingError as e:
        _report("12d", "England 应存在于图谱中", False, str(e))

    # 不存在的联赛 — 应拦截
    try:
        validate_values(
            'MATCH (t:Team {league: "中超"}) RETURN t', None, driver, database
        )
        _report("12e", "不存在联赛应被拦截", False)
    except CypherMappingError:
        _report("12e", "不存在联赛 '中超' 被正确拦截", True)

    # 两个队都存在
    try:
        validate_values(
            "MATCH (a:Team)-[r:PLAYED_AGAINST]-(b:Team) RETURN r",
            {"team_a": "Arsenal", "team_b": "Chelsea"},
            driver, database,
        )
        _report("12f", "两队参数化值均存在", True)
    except CypherMappingError as e:
        _report("12f", "Arsenal + Chelsea 应均存在", False, str(e))


def test_13_run_all_defenses(driver, database):
    """run_all_defenses 四道防线联合执行"""
    from agents.tools.neo4j_tools.security import (
        run_all_defenses,
        CypherSecurityError,
        CypherDirectionError,
        CypherSyntaxError,
        CypherMappingError,
    )

    # 完全合法的查询 → 应全部通过
    try:
        result = run_all_defenses(
            'MATCH (a:Team {name: "Arsenal"})-[r:PLAYED_AGAINST]-(b:Team {name: "Chelsea"}) '
            'RETURN r.match_result LIMIT 5',
            None, driver, database,
        )
        _report("13a", "合法查询通过全部四道防线", True)
    except Exception as e:
        _report("13a", "合法查询应通过", False, f"{type(e).__name__}: {e}")

    # 防线1 应先于其他防线触发
    try:
        run_all_defenses(
            'MATCH (t:Team) DELETE t', None, driver, database,
        )
        _report("13b", "写操作应在防线1被拦截", False)
    except CypherSecurityError:
        _report("13b", "写操作在防线1被正确拦截", True)
    except Exception as e:
        _report("13b", "应是 CypherSecurityError", False, f"实际: {type(e).__name__}")

    # 防线2: 未知关系类型
    try:
        run_all_defenses(
            'MATCH (a:Team)-[r:UNKNOWN_REL]-(b:Team) RETURN r',
            None, driver, database,
        )
        _report("13c", "未知关系应在防线2被拦截", False)
    except CypherDirectionError:
        _report("13c", "未知关系在防线2被正确拦截", True)
    except Exception as e:
        _report("13c", "应是 CypherDirectionError", False, f"实际: {type(e).__name__}")

    # 防线4: 不存在的实体
    try:
        run_all_defenses(
            'MATCH (t:Team {name: "NoSuchTeam123"}) RETURN t',
            None, driver, database,
        )
        _report("13d", "不存在实体应在防线4被拦截", False)
    except CypherMappingError:
        _report("13d", "不存在实体在防线4被正确拦截", True)
    except Exception as e:
        _report("13d", "应是 CypherMappingError", False, f"实际: {type(e).__name__}")


def test_14_execute_cypher(driver, database):
    """Cypher 执行与结果格式化"""
    from agents.tools.neo4j_tools.tool_entry import _execute_cypher

    # 有结果的查询
    result = _execute_cypher(
        "MATCH (t:Team {league: 'England'}) RETURN t.name AS name LIMIT 5",
        None, driver,
    )
    has_data = "[查询结果]" in result and "name:" in result
    _report("14a", "有结果查询格式化正确", has_data,
            result[:200] if not has_data else "")

    # 无结果的查询
    result_empty = _execute_cypher(
        "MATCH (t:Team {name: 'NoTeam999'}) RETURN t.name AS name",
        None, driver,
    )
    is_empty = "未找到" in result_empty
    _report("14b", "无结果查询返回提示信息", is_empty,
            result_empty if not is_empty else "")

    # 参数化查询
    result_param = _execute_cypher(
        "MATCH (a:Team {name: $team_a})-[r:PLAYED_AGAINST]-(b:Team {name: $team_b}) "
        "RETURN r.match_result AS result LIMIT 3",
        {"team_a": "Arsenal", "team_b": "Chelsea"},
        driver,
    )
    has_match = "[查询结果]" in result_param
    _report("14c", "参数化查询执行成功", has_match,
            result_param[:200] if not has_match else "")


def test_15_full_query_fast_path(driver, database):
    """@tool 完整查询流程（Fast Path 端到端）"""
    from agents.tools.neo4j_tools.tool_entry import neo4j_query

    test_cases = [
        (
            "阿森纳和切尔西的历史交锋",
            lambda r: "[查询结果]" in r and "result:" in r,
            "应返回交锋记录",
        ),
        (
            "拜仁近3场比赛",
            lambda r: "[查询结果]" in r,
            "应返回近期比赛",
        ),
        (
            "巴萨主场战绩",
            lambda r: "[查询结果]" in r,
            "应返回主场记录",
        ),
        (
            "英超有哪些球队",
            lambda r: "[查询结果]" in r and "name:" in r,
            "应返回英超球队列表",
        ),
        (
            "利物浦上赛季的比赛",
            lambda r: "[查询结果]" in r or "未找到" in r,
            "应正常返回（有数据或空提示）",
        ),
    ]

    passed = 0
    failed_details = []
    for question, checker, desc in test_cases:
        try:
            result = neo4j_query.invoke(question)
            if checker(result):
                passed += 1
            else:
                failed_details.append(f"'{question}': {desc}\n  返回: {result[:150]}")
        except Exception as e:
            failed_details.append(f"'{question}': 异常 {type(e).__name__}: {e}")

    total = len(test_cases)
    _report("15", f"@tool 端到端查询 ({passed}/{total})", passed == total,
            "\n".join(failed_details) if failed_details else "")


# ═══════════════════════════════════════════════════════════════
#  交互式查询模式
# ═══════════════════════════════════════════════════════════════

def interactive_query():
    """
    交互式查询模式：允许用户输入自然语言查询，查看 Text2Cypher 转换和查询结果。
    """
    from agents.tools.neo4j_tools.tool_entry import neo4j_query

    print("=" * 65)
    print("  Neo4j 图谱查询工具 · 交互式查询模式")
    print("=" * 65)
    print("  输入自然语言查询，系统将自动转换为 Cypher 并返回结果")
    print("  支持 Fast Path（模板匹配）和 Deep Path（Text2Cypher）")
    print("  输入 'q' 或 'quit' 退出，输入 'help' 查看示例")
    print("=" * 65)

    # 检查 Neo4j 连接
    driver, database = _get_neo4j_driver()
    if driver is None:
        print("\n❌ Neo4j 未连接，无法进行查询。")
        print("   请确保 Neo4j 服务正在运行，并检查 .env 中的连接配置。")
        return 1

    print("\n✅ Neo4j 连接成功！\n")

    example_queries = [
        "阿森纳和切尔西的历史交锋",
        "拜仁近5场比赛",
        "巴萨主场战绩",
        "英超有哪些球队",
        "利物浦上赛季的比赛",
        "皇马和巴萨的赔率数据",
    ]

    while True:
        try:
            print("\n>> 请输入查询: ", end="", flush=True)
            query = sys.stdin.readline()
            if not query:          # EOF (Ctrl+D)
                print("\n再见！")
                break
            query = query.strip()
        except KeyboardInterrupt:
            print("\n再见！")
            break

        if not query:
            continue

        if query.lower() in ("q", "quit", "exit"):
            print("\n👋 再见！")
            break

        if query.lower() in ("help", "h", "?"):
            print("\n📝 示例查询:")
            for i, ex in enumerate(example_queries, 1):
                print(f"  {i}. {ex}")
            continue

        # 执行查询
        print()
        print("─" * 65)
        print(f"📥 查询: {query}")
        print("─" * 65)
        print()

        try:
            # 调用 neo4j_query，它会自动打印 Fast Path / Deep Path 日志
            result = neo4j_query.invoke(query)
            
            print()
            print("─" * 65)
            print("📤 查询结果:")
            print("─" * 65)
            print(result)
            print("─" * 65)
        except Exception as e:
            print()
            print("─" * 65)
            print(f"❌ 查询失败: {type(e).__name__}: {e}")
            print("─" * 65)
            if "--debug" in sys.argv:
                traceback.print_exc()

    return 0


# ═══════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Neo4j 图谱查询工具 · 全模块测试")
    print("=" * 65)

    # ── 离线测试 ──
    print()
    print("─── 离线测试（无需 Neo4j 连接）───")
    offline_tests = [
        test_01_read_only_check,
        test_02_direction_check,
        test_03_name_map_loading,
        test_04_extract_teams,
        test_05_extract_season,
        test_06_extract_number,
        test_07_resolve_league,
        test_08_template_format,
        test_09_extract_cypher,
        test_10_template_matching,
    ]
    for test_fn in offline_tests:
        try:
            test_fn()
        except Exception as e:
            print(f"  💥 {test_fn.__name__} 意外崩溃:")
            traceback.print_exc()

    # ── 在线测试 ──
    print()
    print("─── 在线测试（需要 Neo4j 连接）───")
    driver, database = _get_neo4j_driver()

    if driver is None:
        online_tests = [
            ("11", "防线3 — EXPLAIN 语法验证"),
            ("12", "防线4 — 值映射探测查询"),
            ("13", "run_all_defenses 四道防线联合"),
            ("14", "Cypher 执行与结果格式化"),
            ("15", "@tool 完整端到端查询"),
        ]
        for tid, title in online_tests:
            _skip(tid, title, "Neo4j 未连接")
    else:
        online_fns = [
            test_11_syntax_validation,
            test_12_value_validation,
            test_13_run_all_defenses,
            test_14_execute_cypher,
            test_15_full_query_fast_path,
        ]
        for test_fn in online_fns:
            try:
                test_fn(driver, database)
            except Exception as e:
                print(f"  💥 {test_fn.__name__} 意外崩溃:")
                traceback.print_exc()

    # ── 汇总 ──
    print()
    print("=" * 65)
    total = _pass_count + _fail_count + _skip_count
    print(f"  测试汇总: {total} 项 "
          f"| ✅ 通过: {_pass_count} "
          f"| ❌ 失败: {_fail_count} "
          f"| ⏭️ 跳过: {_skip_count}")
    if _fail_count == 0:
        print("  🎉 全部测试通过！")
    else:
        print(f"  ⚠️ 有 {_fail_count} 项测试失败，请检查上方日志")
    print("=" * 65)

    return 0 if _fail_count == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Neo4j 图谱查询工具测试与交互式查询",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python tests/test_neo4j.py              # 运行全部测试
  python tests/test_neo4j.py --interactive # 进入交互式查询模式
  python tests/test_neo4j.py -i            # 简写形式
        """.strip(),
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="进入交互式查询模式（跳过测试，直接查询）",
    )
    args = parser.parse_args()

    if args.interactive:
        exit(interactive_query())
    else:
        exit(main())

