# -*- coding: utf-8 -*-
"""
向量知识检索工具 · 全模块测试
═══════════════════════════════
测试覆盖:
  [离线测试 - 无需向量库]
    Test 1 : 配置文件路径 & 阈值合法性
    Test 2 : 模块导入完整性

  [在线测试 - 需要 ChromaDB + bge-m3]
    Test 3 : ChromaDB 连接 & Collection 文档数
    Test 4 : 防线1 — 距离阈值拦截（无关问题全拦截）
    Test 5 : 防线2 — 强制 Top-K 上限（返回条数 ≤ MAX_RESULTS）
    Test 6 : 精确球队检索（含球队名 → 命中对应球队）
    Test 7 : 语义检索（不含球队名 → 语义召回）
    Test 8 : @tool 端到端（返回格式校验）

运行方式:
  conda activate football
  python tests/test_vector.py              # 运行全部测试
  python tests/test_vector.py -i           # 进入交互式查询模式
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
    global _skip_count
    _skip_count += 1
    print(f"  ⏭️ SKIP [{test_id}] {title} — {reason}")


# ═══════════════════════════════════════════════════════════════
#  离线测试
# ═══════════════════════════════════════════════════════════════

def test_01_config():
    """配置文件路径 & 阈值合法性"""
    from agents.tools.vector_tools.config import (
        CHROMA_DB_PATH, BGE_M3_MODEL_PATH, COLLECTION_NAME,
        DISTANCE_THRESHOLD, MAX_RESULTS,
    )

    checks = [
        ("CHROMA_DB_PATH 存在", os.path.isdir(CHROMA_DB_PATH)),
        ("BGE_M3_MODEL_PATH 存在", os.path.isdir(BGE_M3_MODEL_PATH)),
        ("COLLECTION_NAME 非空", bool(COLLECTION_NAME)),
        ("DISTANCE_THRESHOLD > 0", DISTANCE_THRESHOLD > 0),
        ("MAX_RESULTS ∈ [1, 20]", 1 <= MAX_RESULTS <= 20),
    ]
    passed = sum(1 for _, ok in checks if ok)
    details = [f"{name}: ✗" for name, ok in checks if not ok]
    _report("1", f"配置校验 ({passed}/{len(checks)})",
            passed == len(checks), "\n".join(details))


def test_02_imports():
    """模块导入完整性"""
    try:
        from agents.tools.vector_tools.config import DISTANCE_THRESHOLD, MAX_RESULTS  # noqa: F401
        from agents.tools.vector_tools.retriever import search_team_profiles  # noqa: F401
        from agents.tools.vector_tools.tool_entry import search_knowledge_base  # noqa: F401
        from agents.tools.vector_tools import search_knowledge_base as skb  # noqa: F401
        _report("2", "全部模块导入成功", True)
    except Exception as e:
        _report("2", "模块导入失败", False, str(e))


# ═══════════════════════════════════════════════════════════════
#  在线测试（需要 ChromaDB + bge-m3）
# ═══════════════════════════════════════════════════════════════

def _check_vector_ready() -> bool:
    """检查向量库和模型是否可用，返回 True/False"""
    try:
        from agents.tools.vector_tools.retriever import _get_collection
        col = _get_collection()
        return col.count() > 0
    except Exception as e:
        print(f"  ⚠️ 向量库连接失败: {e}")
        return False


def test_03_collection_info():
    """ChromaDB 连接 & Collection 文档数"""
    from agents.tools.vector_tools.retriever import _get_collection
    col = _get_collection()
    count = col.count()
    _report("3", f"Collection 文档数: {count}", count > 0,
            f"预期 ~130 条 (五大联赛球队简介)")


def test_04_threshold_filter():
    """防线1: 距离阈值拦截 — 完全无关问题应全部被拦截"""
    from agents.tools.vector_tools.retriever import search_team_profiles

    irrelevant_queries = [
        "量子力学的基本原理是什么",
        "Python装饰器怎么写",
        "今天中午吃什么好",
    ]
    all_blocked = True
    details = []
    for q in irrelevant_queries:
        results = search_team_profiles(q)
        if results:
            all_blocked = False
            names = ", ".join(r["club_name_zh"] for r in results)
            details.append(f"'{q}' 未被完全拦截 → {names}")

    _report("4", f"无关问题阈值拦截 ({len(irrelevant_queries)} 条)",
            all_blocked, "\n".join(details))


def test_05_topk_limit():
    """防线2: 强制 Top-K — 返回条数 ≤ MAX_RESULTS"""
    from agents.tools.vector_tools.retriever import search_team_profiles
    from agents.tools.vector_tools.config import MAX_RESULTS

    # 用一个宽泛的足球问题触发多条召回
    results = search_team_profiles("英超球队有哪些")
    within_limit = len(results) <= MAX_RESULTS
    _report("5", f"Top-K 限制: 返回 {len(results)} 条 (上限 {MAX_RESULTS})",
            within_limit)


def test_06_exact_team_retrieval():
    """精确球队检索: 含球队名 → 命中对应球队"""
    from agents.tools.vector_tools.retriever import search_team_profiles

    cases = [
        ("阿森纳的历史底蕴", "Arsenal"),
        ("皇马的背景介绍", "Real Madrid"),
        ("拜仁慕尼黑的辉煌战绩", "Bayern Munich"),
    ]
    passed = 0
    details = []
    for query, expected_club in cases:
        results = search_team_profiles(query)
        if results and any(expected_club in r["club_name"] for r in results):
            passed += 1
        else:
            got = [r["club_name"] for r in results] if results else "空"
            details.append(f"'{query}' → {got}（期望含 {expected_club}）")

    _report("6", f"精确球队检索 ({passed}/{len(cases)})",
            passed == len(cases), "\n".join(details))


def test_07_semantic_retrieval():
    """语义检索: 不含球队名 → 语义召回相关球队"""
    from agents.tools.vector_tools.retriever import search_team_profiles

    cases = [
        ("哪支球队联赛不败夺冠", True),      # 应能召回阿森纳等
        ("欧冠最多冠军的球队", True),          # 应能召回皇马等
    ]
    passed = 0
    details = []
    for query, should_have_result in cases:
        results = search_team_profiles(query)
        has_result = len(results) > 0
        if has_result == should_have_result:
            passed += 1
            if results:
                names = ", ".join(f"{r['club_name_zh']}({r['distance']})" for r in results)
                details.append(f"'{query}' → {names}")
        else:
            details.append(f"'{query}' → {'有' if has_result else '无'}结果（期望{'有' if should_have_result else '无'}）")

    _report("7", f"语义检索 ({passed}/{len(cases)})",
            passed == len(cases), "\n".join(details))


def test_08_tool_e2e():
    """@tool 端到端: 返回格式校验"""
    from agents.tools.vector_tools.tool_entry import search_knowledge_base

    # 正常检索 → 应包含格式化信息
    result_ok = search_knowledge_base.invoke("皇马的背景介绍")
    has_header = "向量检索" in result_ok
    has_club = "Real Madrid" in result_ok or "皇马" in result_ok or "皇家马德里" in result_ok
    _report("8a", "@tool 正常检索返回格式", has_header and has_club,
            f"前100字: {result_ok[:100]}")

    # 无关问题 → 应返回未找到
    result_empty = search_knowledge_base.invoke("量子力学的基本原理是什么")
    has_not_found = "未找到" in result_empty
    _report("8b", "@tool 无关问题返回空提示", has_not_found,
            f"返回: {result_empty[:80]}")


# ═══════════════════════════════════════════════════════════════
#  交互式查询模式
# ═══════════════════════════════════════════════════════════════

def interactive_query():
    """交互式查询：输入自然语言检索球队向量知识库。"""
    from agents.tools.vector_tools.tool_entry import search_knowledge_base

    print("=" * 65)
    print("  向量知识检索工具 · 交互式查询模式")
    print("=" * 65)
    print("  输入自然语言问题检索球队知识库")
    print("  输入 'q' 或 'quit' 退出，输入 'help' 查看示例")
    print("=" * 65)

    # 检查向量库
    if not _check_vector_ready():
        print("\n❌ 向量库未就绪，无法进行查询。")
        print("   请确保 ChromaDB 和 bge-m3 模型路径正确。")
        return 1

    print("\n✅ 向量库连接成功！\n")

    example_queries = [
        "阿森纳的历史底蕴",
        "皇马的背景介绍",
        "哪支球队联赛不败夺冠",
        "欧冠最多冠军的球队",
        "巴黎圣日耳曼的老板是谁",
        "米兰双雄是哪两支球队",
    ]

    while True:
        try:
            print("\n>> 请输入查询: ", end="", flush=True)
            query = sys.stdin.readline()
            if not query:
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

        print()
        print("-" * 65)
        print(f"查询: {query}")
        print("-" * 65)

        try:
            result = search_knowledge_base.invoke(query)
            print()
            print("-" * 65)
            print("检索结果:")
            print("-" * 65)
            print(result)
            print("-" * 65)
        except Exception as e:
            print()
            print("-" * 65)
            print(f"❌ 检索失败: {type(e).__name__}: {e}")
            print("-" * 65)
            if "--debug" in sys.argv:
                traceback.print_exc()

    return 0


# ═══════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  向量知识检索工具 · 全模块测试")
    print("=" * 65)

    # ── 离线测试 ──
    print()
    print("--- 离线测试（无需向量库）---")
    for fn in [test_01_config, test_02_imports]:
        try:
            fn()
        except Exception:
            print(f"  !! {fn.__name__} 意外崩溃:")
            traceback.print_exc()

    # ── 在线测试 ──
    print()
    print("--- 在线测试（需要 ChromaDB + bge-m3）---")
    ready = _check_vector_ready()

    if not ready:
        for tid, title in [
            ("3", "ChromaDB 连接"),
            ("4", "距离阈值拦截"),
            ("5", "Top-K 限制"),
            ("6", "精确球队检索"),
            ("7", "语义检索"),
            ("8", "@tool 端到端"),
        ]:
            _skip(tid, title, "向量库未就绪")
    else:
        for fn in [
            test_03_collection_info,
            test_04_threshold_filter,
            test_05_topk_limit,
            test_06_exact_team_retrieval,
            test_07_semantic_retrieval,
            test_08_tool_e2e,
        ]:
            try:
                fn()
            except Exception:
                print(f"  !! {fn.__name__} 意外崩溃:")
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
        description="向量知识检索工具测试与交互式查询",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python tests/test_vector.py              # 运行全部测试
  python tests/test_vector.py --interactive # 进入交互式查询模式
  python tests/test_vector.py -i           # 简写形式
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

