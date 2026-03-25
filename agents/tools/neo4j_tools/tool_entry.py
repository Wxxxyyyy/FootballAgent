# -*- coding: utf-8 -*-
"""
Neo4j 图谱查询工具 · @tool 统一入口
══════════════════════════════════════
核心执行流程:
  阶段一 (Fast Path) : 意图拦截 + 模板匹配
    → 判断用户问题是否命中高频模板（历史交锋、近N场、主客场等）
    → 命中则提取实体，填充静态 Cypher 模板，直接执行
  阶段二 (Deep Path) : Text2Cypher 终极回环
    → 模板未命中，进入 LLM 生成 Cypher + 四道防线 + 最多3次重试

依赖:
  - data/team_profiles/*.json 提供中英文球队名映射
  - .env 提供 Neo4j 连接配置
"""

import os
import re
import csv
from typing import Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_core.tools import tool

from agents.tools.neo4j_tools.templates import match_queries, team_queries
from agents.tools.neo4j_tools.text2cypher import generate_cypher

# ═══════════════════════════════════════════════════════════════
#  环境 & 常量
# ═══════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

NEO4J_URI = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

CLUB_NAME_CSV = os.path.join(PROJECT_ROOT, "data", "English2Chinese", "中英文对照.csv")

# 结果最大行数（防止返回过长的文本卡住 Agent）
MAX_RESULT_ROWS = 30


# ═══════════════════════════════════════════════════════════════
#  Neo4j Driver 管理（模块级单例）
# ═══════════════════════════════════════════════════════════════

_driver = None


def _get_driver():
    """获取 Neo4j Driver 单例"""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        print(f"[Neo4j] 已连接: {NEO4J_URI}")
    return _driver


# ═══════════════════════════════════════════════════════════════
#  中英文球队名映射（从 中英文对照.csv 加载）
# ═══════════════════════════════════════════════════════════════

_name_map: dict[str, str] | None = None  # {中文名/别名/英文名 → 英文名}


def _load_name_map() -> dict[str, str]:
    """
    加载 data/English2Chinese/中英文对照.csv，构建 中文/别名/英文 → 英文名 的映射。

    CSV 格式: ClubName,League,ClubNameZh,AliasZh
    注意: AliasZh 可能包含多个别名，用中文逗号 `，` 分隔。
    示例: "Man United,England,曼彻斯特联,曼联，红魔"
    """
    global _name_map
    if _name_map is not None:
        return _name_map

    _name_map = {}
    try:
        with open(CLUB_NAME_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                en_name = row.get("ClubName", "").strip()
                if not en_name:
                    continue

                # 英文名 → 自身（原始 + 小写）
                _name_map[en_name] = en_name
                _name_map[en_name.lower()] = en_name

                # 中文全名
                zh = row.get("ClubNameZh", "").strip()
                if zh:
                    _name_map[zh] = en_name

                # 中文别名（可能有多个，用中文逗号 `，` 或英文逗号 `,` 分隔）
                alias_raw = row.get("AliasZh", "").strip()
                if alias_raw:
                    # 统一分隔符后拆分
                    for alias in alias_raw.replace("，", ",").split(","):
                        alias = alias.strip()
                        if alias and alias != zh:
                            _name_map[alias] = en_name
    except Exception as e:
        print(f"[Neo4j] ⚠️ 加载球队映射失败 ({CLUB_NAME_CSV}): {e}")

    print(f"[Neo4j] 已加载 {len(_name_map)} 条球队名映射")
    return _name_map


def _resolve_team_name(raw: str) -> Optional[str]:
    """
    将用户输入中的球队名（中文/英文/别名）解析为 Neo4j 中存储的英文名。
    """
    name_map = _load_name_map()
    raw = raw.strip()
    # 精确匹配
    if raw in name_map:
        return name_map[raw]
    # 小写匹配
    if raw.lower() in name_map:
        return name_map[raw.lower()]
    return None


def _extract_teams(question: str) -> list[str]:
    """
    从用户问题中提取球队名（返回英文名列表）。
    策略: 将名称映射表中的所有 key 按长度降序逐一匹配。
    """
    name_map = _load_name_map()

    # 按 key 长度降序排列，优先匹配更长的名字（如"曼联"优先于"联"）
    sorted_keys = sorted(name_map.keys(), key=len, reverse=True)

    found = []
    remaining = question
    for key in sorted_keys:
        if key in remaining:
            en_name = name_map[key]
            if en_name not in found:
                found.append(en_name)
            # 去掉已匹配的部分，避免子串重复匹配
            remaining = remaining.replace(key, "", 1)

    return found


def _extract_season(question: str) -> Optional[str]:
    """
    从问题中提取赛季标识。
    支持格式: "2024-2025", "24-25赛季", "这个赛季", "上赛季"
    """
    # 精确格式 "2024-2025"
    m = re.search(r'(20\d{2})\s*[-/]\s*(20\d{2})', question)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    # 简写格式 "24-25"
    m = re.search(r'(\d{2})\s*[-/]\s*(\d{2})(?:\s*赛季)?', question)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        return f"20{y1:02d}-20{y2:02d}"

    # "本赛季" / "这个赛季" → 默认当前赛季 2025-2026
    if re.search(r'本赛季|这个赛季|这赛季|当前赛季', question):
        return "2025-2026"

    # "上赛季" / "上个赛季"
    if re.search(r'上赛季|上个赛季', question):
        return "2024-2025"

    return None


def _extract_number(question: str, default: int = 10) -> int:
    """提取问题中的数字（如"近5场"中的5）"""
    m = re.search(r'近\s*(\d+)\s*场', question)
    if m:
        return int(m.group(1))
    m = re.search(r'最近\s*(\d+)', question)
    if m:
        return int(m.group(1))
    return default


def _extract_years_back(question: str) -> int | None:
    """
    提取"最近N年"中的年份数。
    支持: "最近两年"、"近3年"、"最近5年"、"近两年" 等。
    
    Returns:
        int | None: 年份数，如果未找到则返回 None
    """
    # 中文数字映射
    chinese_numbers = {
        "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    }
    
    # 匹配"最近N年"或"近N年"（支持中文数字和阿拉伯数字）
    patterns = [
        r'最近\s*(\d+)\s*年',
        r'近\s*(\d+)\s*年',
        r'最近\s*([一二两三五六七八九十])\s*年',
        r'近\s*([一二两三五六七八九十])\s*年',
    ]
    
    for pattern in patterns:
        m = re.search(pattern, question)
        if m:
            num_str = m.group(1)
            # 尝试转换为数字
            if num_str.isdigit():
                return int(num_str)
            # 中文数字转换
            if num_str in chinese_numbers:
                return chinese_numbers[num_str]
    
    return None


# ═══════════════════════════════════════════════════════════════
#  联赛名映射
# ═══════════════════════════════════════════════════════════════

_LEAGUE_MAP = {
    "英超": "England", "英格兰": "England", "epl": "England", "premier league": "England",
    "西甲": "Spain", "西班牙": "Spain", "la liga": "Spain",
    "意甲": "Italy", "意大利": "Italy", "serie a": "Italy",
    "德甲": "Germany", "德国": "Germany", "bundesliga": "Germany",
    "法甲": "France", "法国": "France", "ligue 1": "France",
}


def _resolve_league(question: str) -> Optional[str]:
    """从问题中提取联赛名并映射为英文"""
    q_lower = question.lower()
    for key, val in _LEAGUE_MAP.items():
        if key in q_lower:
            return val
    return None


# ═══════════════════════════════════════════════════════════════
#  阶段一: Fast Path — 意图拦截 + 模板匹配
# ═══════════════════════════════════════════════════════════════

def _try_template_match(question: str) -> Optional[tuple[str, dict]]:
    """
    尝试将用户问题匹配到预定义的 Cypher 模板。
    匹配成功返回 (cypher, params)，未命中返回 None。

    支持的高频意图:
      - 两队历史交锋 / 对战记录
      - 某队近N场比赛
      - 某队主场/客场战绩
      - 某队某赛季比赛
      - 两队赔率数据
      - 联赛球队列表
      - 球队交手最多对手
    """
    teams = _extract_teams(question)
    season = _extract_season(question)
    years_back = _extract_years_back(question)

    # ── 1. 两队历史交锋（优先级最高，即使有"最近"关键词，两支球队时优先匹配） ──
    if len(teams) >= 2:
        # 检测是否问的是两队之间的比赛（有明确的交锋关键词，或者只是问"比分"、"结果"等）
        # 注意：只要有两支球队，即使没有明确的"交锋"关键词，也认为是问两队之间的比赛
        has_match_keywords = re.search(
            r'交锋|对战|对阵|对决|vs|VS|比赛记录|交手|head.?to.?head|比分|结果', question
        )
        
        # 有两支球队时，默认就是问两队之间的比赛
        if has_match_keywords or len(teams) >= 2:
            # 有赔率关键词 → 返回完整赔率数据
            if re.search(r'赔率|盘口|亚盘|大小球|让球|odds', question, re.IGNORECASE):
                return match_queries.match_with_odds(teams[0], teams[1], season or "", years_back=years_back)
            # 只问比分/结果 → 只返回比分相关字段
            if re.search(r'比分|结果', question) and not re.search(r'赔率|盘口|数据|详细|完整', question):
                return match_queries.head_to_head(teams[0], teams[1], years_back=years_back, score_only=True)
            return match_queries.head_to_head(teams[0], teams[1], years_back=years_back)

    # ── 2. 某队近N场（只有一支球队时才匹配） ──
    if len(teams) == 1 and re.search(r'近\d+场|最近|近期|recent', question, re.IGNORECASE):
        n = _extract_number(question)
        return match_queries.recent_matches(teams[0], n)

    # ── 3. 主场战绩 ──
    if len(teams) >= 1 and re.search(r'主场', question):
        n = _extract_number(question, default=10)
        return match_queries.home_record(teams[0], n)

    # ── 4. 客场战绩 ──
    if len(teams) >= 1 and re.search(r'客场', question):
        n = _extract_number(question, default=10)
        return match_queries.away_record(teams[0], n)

    # ── 5. 某队某赛季比赛 ──
    if len(teams) >= 1 and season:
        return match_queries.season_matches(teams[0], season)

    # ── 6. 两队赔率数据（无交锋关键词但提到赔率） ──
    if len(teams) >= 2 and re.search(r'赔率|盘口|亚盘|大小球|让球|odds', question, re.IGNORECASE):
        return match_queries.match_with_odds(teams[0], teams[1], season or "")

    # ── 7. 联赛球队列表 ──
    league = _resolve_league(question)
    if league and re.search(r'球队|队伍|名单|有哪些|列表|all.?teams', question, re.IGNORECASE):
        return team_queries.league_teams(league)

    # ── 8. 球队信息 ──
    if len(teams) >= 1 and re.search(r'什么联赛|哪个联赛|属于|league', question, re.IGNORECASE):
        return team_queries.team_info(teams[0])

    # ── 9. 交手最多对手 ──
    if len(teams) >= 1 and re.search(r'最多|频繁|经常|对手|rivals|opponents', question, re.IGNORECASE):
        return team_queries.team_opponents(teams[0])

    # ── 10. 只提到一支球队 + 赛季 → 赛季汇总 ──
    if len(teams) == 1 and season:
        return team_queries.team_season_all(teams[0], season)

    return None


# ═══════════════════════════════════════════════════════════════
#  Cypher 执行 & 结果格式化
# ═══════════════════════════════════════════════════════════════

def _execute_cypher(cypher: str, params: Optional[dict], driver) -> str:
    """
    执行 Cypher 并将结果格式化为可读字符串。

    Args:
        cypher: Cypher 查询语句
        params: 参数字典（可选）
        driver: Neo4j Driver 实例

    Returns:
        str: 格式化的查询结果文本
    """
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher, **(params or {}))
            records = result.data()
    except Exception as e:
        return f"[Neo4j 执行错误] {type(e).__name__}: {e}"

    if not records:
        return "[查询结果] 未找到匹配数据。请检查球队名称或赛季是否正确。"

    # 截断过长结果
    total = len(records)
    if total > MAX_RESULT_ROWS:
        records = records[:MAX_RESULT_ROWS]
        truncated = True
    else:
        truncated = False

    # 格式化为表格文本
    lines = []
    for i, rec in enumerate(records, 1):
        parts = []
        for k, v in rec.items():
            parts.append(f"{k}: {v}")
        lines.append(f"  [{i}] {' | '.join(parts)}")

    header = f"[查询结果] 共 {total} 条记录"
    if truncated:
        header += f"（仅展示前 {MAX_RESULT_ROWS} 条）"

    return header + "\n" + "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  @tool 统一入口
# ═══════════════════════════════════════════════════════════════

@tool
def neo4j_query(question: str) -> str:
    """
    足球图谱查询工具 —— 在 Neo4j 知识图谱中查询球队比赛数据、历史交锋、赔率等信息。

    支持查询类型:
      - 两队历史交锋记录（含赔率）
      - 某队近N场比赛
      - 某队主场/客场战绩
      - 某队某赛季全部比赛
      - 联赛球队列表
      - 球队对手统计
      - 以及更多通过自然语言描述的复杂查询

    Args:
        question: 用户的自然语言查询问题（中文或英文均可）

    Returns:
        str: 格式化的查询结果文本
    """
    print(f"\n{'='*60}")
    print(f"[Neo4j Tool] 收到查询: {question}")
    print(f"{'='*60}")

    # ━━━ 阶段零: 用户意图预拦截 — 拒绝明显的写操作/危险请求 ━━━
    _DANGEROUS_INTENT_RE = re.compile(
        r'删除|删掉|移除|清空|修改|更新|更改|添加|新增|创建|插入|'
        r'DROP|DELETE|REMOVE|CREATE|SET|MERGE|DETACH|TRUNCATE',
        re.IGNORECASE,
    )
    if _DANGEROUS_INTENT_RE.search(question):
        print("[Neo4j Tool] ⛔ 用户输入包含写操作意图，已拦截")
        return "[安全拦截] 本工具仅支持只读查询，不允许对图谱数据进行删除、修改或新增操作。请改用查询类问题。"

    driver = _get_driver()

    # ━━━ 阶段一: Fast Path — 模板匹配 ━━━
    print("[Neo4j Tool] 阶段一: 尝试模板匹配...")
    template_result = _try_template_match(question)

    if template_result is not None:
        cypher, params = template_result
        print(f"[Neo4j Tool] ✅ Fast Path 命中模板")
        print(f"  Cypher: {cypher.strip()[:150]}...")
        print(f"  Params: {params}")
        return _execute_cypher(cypher, params, driver)

    # ━━━ 阶段二: Deep Path — Text2Cypher ━━━
    print("[Neo4j Tool] 模板未命中，进入 Deep Path (Text2Cypher)...")

    # 预处理: 将问题中的中文球队名替换为英文，帮助 LLM 生成更准确的 Cypher
    enriched_question = question
    teams = _extract_teams(question)
    if teams:
        enriched_question += f"\n（提示: 涉及的球队英文名为 {', '.join(teams)}）"

    season = _extract_season(question)
    if season:
        enriched_question += f"\n（提示: 赛季标识为 {season}）"

    cypher, params = generate_cypher(
        question=enriched_question,
        driver=driver,
        database=NEO4J_DATABASE,
    )

    # 检查是否是错误说明（generate_cypher 失败时返回错误文本 + None）
    if params is None and cypher.startswith("[Text2Cypher"):
        print(f"[Neo4j Tool] ❌ Text2Cypher 彻底失败")
        return cypher

    print(f"[Neo4j Tool] ✅ Deep Path 生成 Cypher 成功")
    return _execute_cypher(cypher, params, driver)


# ═══════════════════════════════════════════════════════════════
#  命令行测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Neo4j 图谱查询工具 · 交互式测试")
    print("=" * 60)
    print("  输入自然语言问题查询图谱，输入 q 退出")
    print("=" * 60)

    while True:
        try:
            q = input("\n🔍 请输入查询: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if q.lower() in ("q", "quit", "exit"):
            break

        if not q:
            continue

        result = neo4j_query.invoke(q)
        print(f"\n{result}")

    # 关闭连接
    if _driver:
        _driver.close()
        print("\n[Neo4j] 连接已关闭")

