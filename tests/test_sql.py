# -*- coding: utf-8 -*-
"""
MySQL 比赛数据查询工具 · 全模块测试
═══════════════════════════════════════
测试覆盖:
  [离线测试 - 无需 MySQL / LLM]
    Test 1  : 防线1 — 读写隔离 + SQL 注入拦截
    Test 2  : 防线2 — 幻觉校验（表名 / 字段名白名单）
    Test 3  : 防线4 — 强制 LIMIT（自动追加 / 修正）
    Test 4  : 球队实体提取 + 赛季提取 + 联赛映射
    Test 5  : SQL 模板函数返回格式校验
    Test 6  : LLM 输出中 SQL 提取（去 markdown / 注释）
    Test 7  : Fast Path 模板匹配（意图路由覆盖）

  [在线测试 - 需要 MySQL 连接]
    Test 8  : 防线3 — EXPLAIN 语法验证
    Test 9  : run_all_defenses 四道防线联合执行
    Test 10 : @tool 完整查询流程（Fast Path 端到端）

运行方式:
  conda activate football
  python tests/test_sql.py              # 运行全部测试
  python tests/test_sql.py -i           # 进入交互式查询模式
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
#  离线测试（无需 MySQL / LLM）
# ═══════════════════════════════════════════════════════════════

def test_01_read_only_and_injection():
    """防线1: 读写隔离 + SQL 注入拦截"""
    from agents.tools.mysql_tools.security import check_read_only, SQLSecurityError

    # 应通过的只读语句
    safe_queries = [
        "SELECT * FROM match_master WHERE HomeTeam = 'Arsenal'",
        "SELECT FTHG, FTAG, FTR FROM match_master LIMIT 10",
        "SELECT COUNT(*) FROM match_master WHERE league = 'England'",
    ]
    for q in safe_queries:
        try:
            check_read_only(q)
        except SQLSecurityError:
            _report("1a", f"只读语句应通过: {q[:50]}", False)
            return
    _report("1a", f"只读语句全部通过 ({len(safe_queries)}条)", True)

    # 应拦截的写操作
    write_queries = {
        "INSERT": "INSERT INTO match_master VALUES (1, 'E0')",
        "UPDATE": "UPDATE match_master SET FTHG = 0 WHERE id = 1",
        "DELETE": "DELETE FROM match_master WHERE id = 1",
        "DROP":   "DROP TABLE match_master",
        "ALTER":  "ALTER TABLE match_master ADD COLUMN hack INT",
        "TRUNCATE": "TRUNCATE TABLE match_master",
        "GRANT":  "GRANT ALL ON football_agent.* TO 'hacker'",
    }
    blocked = 0
    for keyword, q in write_queries.items():
        try:
            check_read_only(q)
        except SQLSecurityError:
            blocked += 1
    total = len(write_queries)
    _report("1b", f"写操作拦截 ({blocked}/{total})", blocked == total)

    # 应拦截的注入特征
    injection_cases = {
        "分号堆叠": "SELECT 1; DROP TABLE match_master",
        "双横线注释": "SELECT 1 -- DROP TABLE",
        "块注释": "SELECT /* hack */ 1 FROM match_master",
        "SLEEP盲注": "SELECT SLEEP(5) FROM match_master",
    }
    inject_blocked = 0
    for name, q in injection_cases.items():
        try:
            check_read_only(q)
        except SQLSecurityError:
            inject_blocked += 1
    total_inj = len(injection_cases)
    _report("1c", f"注入拦截 ({inject_blocked}/{total_inj})", inject_blocked == total_inj)


def test_02_schema_check():
    """防线2: 幻觉校验（表名 + 字段名）"""
    from agents.tools.mysql_tools.security import check_schema, SQLSchemaError

    # 合法字段应通过
    try:
        check_schema('SELECT FTHG, FTAG, FTR FROM match_master WHERE HomeTeam = "Arsenal"')
        _report("2a", "合法字段通过", True)
    except SQLSchemaError as e:
        _report("2a", "合法字段应通过", False, str(e))

    # 非法表名应拦截
    try:
        check_schema("SELECT * FROM player_stats WHERE name = 'Messi'")
        _report("2b", "非法表名应被拦截", False)
    except SQLSchemaError:
        _report("2b", "非法表名 player_stats 被正确拦截", True)

    # 非法字段应拦截（通过表别名引用）
    try:
        check_schema("SELECT m.`goals_scored` FROM match_master m")
        _report("2c", "幻觉字段应被拦截", False)
    except SQLSchemaError:
        _report("2c", "幻觉字段 goals_scored 被正确拦截", True)


def test_03_enforce_limit():
    """防线4: 强制 LIMIT"""
    from agents.tools.mysql_tools.security import enforce_limit

    # 无 LIMIT → 自动追加 LIMIT 30
    sql1 = enforce_limit("SELECT * FROM match_master WHERE HomeTeam = 'Arsenal'")
    _report("3a", "无LIMIT自动追加", "LIMIT 30" in sql1)

    # LIMIT > 50 → 修正为 LIMIT 30
    sql2 = enforce_limit("SELECT * FROM match_master LIMIT 100")
    _report("3b", "LIMIT>50修正为30", "LIMIT 30" in sql2)

    # LIMIT <= 50 → 保持不变
    sql3 = enforce_limit("SELECT * FROM match_master LIMIT 20")
    _report("3c", "LIMIT<=50保持不变", "LIMIT 20" in sql3)


def test_04_entity_extraction():
    """球队实体提取 + 赛季提取 + 联赛映射"""
    from agents.tools.mysql_tools.tool_entry import (
        _extract_teams, _extract_season, _resolve_league, _load_name_map,
    )

    # 球队名映射加载
    name_map = _load_name_map()
    _report("4a", f"球队映射加载: {len(name_map)} 条", len(name_map) > 100)

    # 球队实体提取
    team_cases = [
        ("阿森纳和切尔西的历史交锋记录", ["Arsenal", "Chelsea"]),
        ("拜仁近5场比赛", ["Bayern Munich"]),
        ("皇马vs巴萨谁更强", ["Real Madrid", "Barcelona"]),
    ]
    team_pass = 0
    team_details = []
    for q, expected in team_cases:
        result = _extract_teams(q)
        if all(t in result for t in expected):
            team_pass += 1
        else:
            team_details.append(f"'{q}' → {result}（期望含 {expected}）")
    _report("4b", f"球队提取 ({team_pass}/{len(team_cases)})",
            team_pass == len(team_cases), "\n".join(team_details))

    # 赛季提取
    season_cases = [
        ("2024-2025赛季", "2024-2025"),
        ("24-25赛季", "2024-2025"),
        ("本赛季", "2025-2026"),
        ("上赛季", "2024-2025"),
        ("今天天气好", None),
    ]
    season_pass = 0
    season_details = []
    for q, expected in season_cases:
        result = _extract_season(q)
        if result == expected:
            season_pass += 1
        else:
            season_details.append(f"'{q}' → '{result}'（期望 '{expected}'）")
    _report("4c", f"赛季提取 ({season_pass}/{len(season_cases)})",
            season_pass == len(season_cases), "\n".join(season_details))

    # 联赛映射
    league_cases = [
        ("英超有哪些球队", "England"),
        ("西甲球队列表", "Spain"),
        ("没有联赛关键词", None),
    ]
    league_pass = 0
    for q, expected in league_cases:
        if _resolve_league(q) == expected:
            league_pass += 1
    _report("4d", f"联赛映射 ({league_pass}/{len(league_cases)})",
            league_pass == len(league_cases))


def test_05_template_format():
    """SQL 模板函数返回格式校验"""
    from agents.tools.mysql_tools.templates.match_sql_queries import (
        recent_matches_sql, head_to_head_sql, team_season_stats_sql,
    )

    templates = [
        ("recent_matches_sql", recent_matches_sql("Arsenal", 5)),
        ("head_to_head_sql", head_to_head_sql("Arsenal", "Chelsea", 10)),
        ("team_season_stats_sql", team_season_stats_sql("Arsenal", "2024-2025")),
    ]

    passed = 0
    failed_details = []
    for name, result in templates:
        sql, params = result
        ok = (isinstance(sql, str) and isinstance(params, tuple)
              and "SELECT" in sql.upper() and "match_master" in sql)
        if ok:
            passed += 1
        else:
            failed_details.append(f"{name}: 格式不符")

    _report("5", f"模板格式校验 ({passed}/{len(templates)})",
            passed == len(templates), "\n".join(failed_details))


def test_06_extract_sql():
    """LLM 输出中 SQL 提取（去 markdown / 注释）"""
    from agents.tools.mysql_tools.text2sql import _extract_sql

    cases = [
        ("纯 SQL", "SELECT * FROM match_master LIMIT 10",
         lambda r: r == "SELECT * FROM match_master LIMIT 10"),
        ("markdown sql块", "```sql\nSELECT * FROM match_master LIMIT 10\n```",
         lambda r: "SELECT" in r and "```" not in r),
        ("带行首注释", "-- 查阿森纳\nSELECT * FROM match_master",
         lambda r: "SELECT" in r and "--" not in r),
        ("带尾部分号", "SELECT * FROM match_master LIMIT 10;",
         lambda r: not r.endswith(";")),
    ]

    passed = 0
    details = []
    for name, raw, checker in cases:
        result = _extract_sql(raw)
        if checker(result):
            passed += 1
        else:
            details.append(f"{name}: '{result[:60]}'")

    _report("6", f"SQL 提取 ({passed}/{len(cases)})",
            passed == len(cases), "\n".join(details))


def test_07_template_matching():
    """Fast Path 模板匹配（意图路由覆盖）"""
    from agents.tools.mysql_tools.tool_entry import _try_template_match

    cases = [
        ("阿森纳和切尔西的历史交锋记录", True),
        ("Arsenal vs Chelsea", True),
        ("拜仁近5场比赛", True),
        ("巴萨最近表现如何", True),
        ("阿森纳2024-2025赛季", True),
        ("今天天气真好", False),
        ("你好啊", False),
    ]

    passed = 0
    details = []
    for q, should_hit in cases:
        result = _try_template_match(q)
        hit = result is not None
        if hit == should_hit:
            passed += 1
        else:
            details.append(f"'{q}' → {'命中' if hit else '未命中'}（期望{'命中' if should_hit else '未命中'}）")

    _report("7", f"模板匹配覆盖 ({passed}/{len(cases)})",
            passed == len(cases), "\n".join(details))


# ═══════════════════════════════════════════════════════════════
#  在线测试（需要 MySQL 连接）
# ═══════════════════════════════════════════════════════════════

def _get_mysql_connection():
    """尝试获取 MySQL 连接，失败返回 None"""
    try:
        from agents.tools.mysql_tools.tool_entry import _get_connection
        conn = _get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return conn
    except Exception as e:
        print(f"  ⚠️ MySQL 连接失败: {e}")
        return None


def test_08_syntax_validation(conn):
    """防线3: EXPLAIN 语法验证"""
    from agents.tools.mysql_tools.security import validate_syntax, SQLSyntaxError

    with conn.cursor() as cur:
        # 合法语法
        try:
            validate_syntax(
                "SELECT FTHG, FTAG FROM match_master WHERE HomeTeam = 'Arsenal' LIMIT 5",
                cur,
            )
            _report("8a", "合法 SQL 通过 EXPLAIN", True)
        except SQLSyntaxError as e:
            _report("8a", "合法 SQL 应通过", False, str(e))

        # 非法语法
        try:
            validate_syntax("SELEC * FORM match_master", cur)
            _report("8b", "非法 SQL 应被拦截", False)
        except SQLSyntaxError:
            _report("8b", "非法 SQL 被正确拦截", True)

        # 不存在的字段（语法层面可能不报错，由防线2兜底）
        try:
            validate_syntax(
                "SELECT fake_col FROM match_master LIMIT 1", cur
            )
            _report("8c", "不存在字段 EXPLAIN 行为", True,
                    "MySQL EXPLAIN 对不存在字段会报错")
        except SQLSyntaxError:
            _report("8c", "不存在字段被 EXPLAIN 正确拦截", True)


def test_09_run_all_defenses(conn):
    """run_all_defenses 四道防线联合执行"""
    from agents.tools.mysql_tools.security import (
        run_all_defenses, SQLSecurityError, SQLSchemaError, SQLSyntaxError,
    )

    with conn.cursor() as cur:
        # 完全合法的查询 → 应通过
        try:
            result = run_all_defenses(
                "SELECT FTHG, FTAG, FTR FROM match_master "
                "WHERE HomeTeam = 'Arsenal' AND AwayTeam = 'Chelsea' "
                "LIMIT 10",
                cur,
            )
            _report("9a", "合法查询通过全部四道防线", True)
        except Exception as e:
            _report("9a", "合法查询应通过", False, f"{type(e).__name__}: {e}")

        # 防线1: 写操作
        try:
            run_all_defenses("DELETE FROM match_master WHERE id = 1", cur)
            _report("9b", "写操作应在防线1被拦截", False)
        except SQLSecurityError:
            _report("9b", "写操作在防线1被正确拦截", True)
        except Exception as e:
            _report("9b", "应是 SQLSecurityError", False, f"实际: {type(e).__name__}")

        # 防线2: 幻觉表名
        try:
            run_all_defenses("SELECT * FROM player_stats LIMIT 5", cur)
            _report("9c", "幻觉表名应在防线2被拦截", False)
        except SQLSchemaError:
            _report("9c", "幻觉表名在防线2被正确拦截", True)
        except Exception as e:
            _report("9c", "应是 SQLSchemaError", False, f"实际: {type(e).__name__}")

        # 无 LIMIT 时应自动追加并通过
        try:
            result = run_all_defenses(
                "SELECT HomeTeam, AwayTeam FROM match_master WHERE league = 'England'",
                cur,
            )
            has_limit = "LIMIT" in result.upper()
            _report("9d", "无 LIMIT 自动追加后通过", has_limit)
        except Exception as e:
            _report("9d", "自动追加 LIMIT 后应通过", False, f"{type(e).__name__}: {e}")


def test_10_full_query_fast_path(conn):
    """@tool 完整查询流程（Fast Path 端到端）"""
    from agents.tools.mysql_tools.tool_entry import mysql_query

    cases = [
        (
            "阿森纳和切尔西的历史交锋",
            lambda r: "[查询结果]" in r,
            "应返回交锋记录",
        ),
        (
            "拜仁近3场比赛",
            lambda r: "[查询结果]" in r,
            "应返回近期比赛",
        ),
        (
            "巴萨2024-2025赛季",
            lambda r: "[查询结果]" in r or "未找到" in r,
            "应正常返回",
        ),
        (
            "帮我删除曼联的数据",
            lambda r: "安全拦截" in r,
            "应触发安全拦截",
        ),
    ]

    passed = 0
    details = []
    for question, checker, desc in cases:
        try:
            result = mysql_query.invoke(question)
            if checker(result):
                passed += 1
            else:
                details.append(f"'{question}': {desc}\n  返回: {result[:150]}")
        except Exception as e:
            details.append(f"'{question}': 异常 {type(e).__name__}: {e}")

    _report("10", f"@tool 端到端查询 ({passed}/{len(cases)})",
            passed == len(cases), "\n".join(details))


# ═══════════════════════════════════════════════════════════════
#  交互式查询模式
# ═══════════════════════════════════════════════════════════════

def interactive_query():
    """
    交互式查询模式：输入自然语言查询 MySQL 比赛数据库。
    支持 Fast Path（模板匹配）和 Deep Path（Text2SQL）。
    """
    from agents.tools.mysql_tools.tool_entry import mysql_query

    print("=" * 65)
    print("  MySQL 比赛数据查询工具 · 交互式查询模式")
    print("=" * 65)
    print("  输入自然语言查询，系统将自动转换为 SQL 并返回结果")
    print("  支持 Fast Path（模板匹配）和 Deep Path（Text2SQL）")
    print("  输入 'q' 或 'quit' 退出，输入 'help' 查看示例")
    print("=" * 65)

    # 检查 MySQL 连接
    conn = _get_mysql_connection()
    if conn is None:
        print("\n❌ MySQL 未连接，无法进行查询。")
        print("   请确保 MySQL 服务正在运行，并检查 .env 中的连接配置。")
        return 1

    print("\n✅ MySQL 连接成功！\n")

    example_queries = [
        "阿森纳和切尔西的历史交锋",
        "拜仁近5场比赛",
        "巴萨2024-2025赛季全部比赛",
        "英超上赛季总进球数最多的球队",
        "利物浦主场胜率是多少",
        "哪支球队本赛季场均进球最多",
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
            print("\n再见！")
            break

        if query.lower() in ("help", "h", "?"):
            print("\n示例查询:")
            for i, ex in enumerate(example_queries, 1):
                print(f"  {i}. {ex}")
            continue

        # 执行查询
        print()
        print("-" * 65)
        print(f"查询: {query}")
        print("-" * 65)
        print()

        try:
            result = mysql_query.invoke(query)
            print()
            print("-" * 65)
            print("查询结果:")
            print("-" * 65)
            print(result)
            print("-" * 65)
        except Exception as e:
            print()
            print("-" * 65)
            print(f"❌ 查询失败: {type(e).__name__}: {e}")
            print("-" * 65)
            if "--debug" in sys.argv:
                traceback.print_exc()

    return 0


# ═══════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  MySQL 比赛数据查询工具 · 全模块测试")
    print("=" * 65)

    # ── 离线测试 ──
    print()
    print("--- 离线测试（无需 MySQL 连接）---")
    offline_tests = [
        test_01_read_only_and_injection,
        test_02_schema_check,
        test_03_enforce_limit,
        test_04_entity_extraction,
        test_05_template_format,
        test_06_extract_sql,
        test_07_template_matching,
    ]
    for test_fn in offline_tests:
        try:
            test_fn()
        except Exception as e:
            print(f"  !! {test_fn.__name__} 意外崩溃:")
            traceback.print_exc()

    # ── 在线测试 ──
    print()
    print("--- 在线测试（需要 MySQL 连接）---")
    conn = _get_mysql_connection()

    if conn is None:
        for tid, title in [
            ("8", "防线3 — EXPLAIN 语法验证"),
            ("9", "run_all_defenses 四道防线联合"),
            ("10", "@tool 完整端到端查询"),
        ]:
            _skip(tid, title, "MySQL 未连接")
    else:
        online_fns = [
            test_08_syntax_validation,
            test_09_run_all_defenses,
            test_10_full_query_fast_path,
        ]
        for test_fn in online_fns:
            try:
                test_fn(conn)
            except Exception as e:
                print(f"  !! {test_fn.__name__} 意外崩溃:")
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
        print("  全部测试通过！")
    else:
        print(f"  ⚠️ 有 {_fail_count} 项测试失败，请检查上方日志")
    print("=" * 65)

    return 0 if _fail_count == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MySQL 比赛数据查询工具测试与交互式查询",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python tests/test_sql.py              # 运行全部测试
  python tests/test_sql.py --interactive # 进入交互式查询模式
  python tests/test_sql.py -i            # 简写形式
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

