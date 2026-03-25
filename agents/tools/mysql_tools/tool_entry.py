# -*- coding: utf-8 -*-
"""
MySQL 比赛数据查询工具 · @tool 统一入口
══════════════════════════════════════════
核心执行流程:
  阶段零 (Guard)     : 用户意图预拦截 — 拒绝写操作 / 危险请求
  阶段一 (Fast Path) : 意图拦截 + 模板匹配
    → 判断用户问题是否命中高频模板（历史交锋、近N场、某赛季）
    → 命中则提取实体，填充静态 SQL 模板，直接执行
  阶段二 (Deep Path) : Text2SQL 终极回环
    → 模板未命中，进入 LLM 生成 SQL + 四道防线 + 最多3次重试

依赖:
  - data/English2Chinese/中英文对照.csv 提供中英文球队名映射
  - .env 提供 MySQL 连接配置
"""

import os
import re
import csv
import json
from typing import Optional
from dotenv import load_dotenv
import pymysql
from langchain_core.tools import tool

from agents.tools.mysql_tools.templates import match_sql_queries
from agents.tools.mysql_tools.text2sql import generate_sql

# ═══════════════════════════════════════════════════════════════
#  环境 & 常量
# ═══════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "football_agent")

CLUB_NAME_CSV = os.path.join(PROJECT_ROOT, "data", "English2Chinese", "中英文对照.csv")

# 结果最大行数（防止返回过长的文本卡住 Agent）
MAX_RESULT_ROWS = 30


# ═══════════════════════════════════════════════════════════════
#  MySQL 连接管理（模块级单例）
# ═══════════════════════════════════════════════════════════════

_connection: pymysql.Connection | None = None


def _get_connection() -> pymysql.Connection:
    """获取 MySQL 连接单例（带自动重连）"""
    global _connection
    if _connection is None or not _connection.open:
        _connection = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=10,
            read_timeout=30,
        )
        print(f"[MySQL] 已连接: {MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")
    # ping 检测连接是否存活，断了则自动重连
    _connection.ping(reconnect=True)
    return _connection


# ═══════════════════════════════════════════════════════════════
#  中英文球队名映射（从 中英文对照.csv 加载，复用 neo4j_tools 同一份数据）
# ═══════════════════════════════════════════════════════════════

_name_map: dict[str, str] | None = None


def _load_name_map() -> dict[str, str]:
    """
    加载 data/English2Chinese/中英文对照.csv，构建 中文/别名/英文 → 英文名 的映射。

    CSV 格式: ClubName,League,ClubNameZh,AliasZh
    注意: AliasZh 可能包含多个别名，用中文逗号 `，` 分隔。
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
                    for alias in alias_raw.replace("，", ",").split(","):
                        alias = alias.strip()
                        if alias and alias != zh:
                            _name_map[alias] = en_name
    except Exception as e:
        print(f"[MySQL] ⚠️ 加载球队映射失败 ({CLUB_NAME_CSV}): {e}")

    print(f"[MySQL] 已加载 {len(_name_map)} 条球队名映射")
    return _name_map


def _resolve_team_name(raw: str) -> Optional[str]:
    """将用户输入中的球队名（中文/英文/别名）解析为 MySQL 中存储的英文名。"""
    name_map = _load_name_map()
    raw = raw.strip()
    if raw in name_map:
        return name_map[raw]
    if raw.lower() in name_map:
        return name_map[raw.lower()]
    return None


def _extract_teams(question: str) -> list[str]:
    """
    从用户问题中提取球队名（返回英文名列表）。
    策略: 将名称映射表中的所有 key 按长度降序逐一匹配。
    """
    name_map = _load_name_map()
    sorted_keys = sorted(name_map.keys(), key=len, reverse=True)

    found = []
    remaining = question
    for key in sorted_keys:
        if key in remaining:
            en_name = name_map[key]
            if en_name not in found:
                found.append(en_name)
            remaining = remaining.replace(key, "", 1)

    return found


def _extract_season(question: str) -> Optional[str]:
    """
    从问题中提取赛季标识。
    支持格式: "2024-2025", "24-25赛季", "这个赛季", "上赛季"
    """
    m = re.search(r'(20\d{2})\s*[-/]\s*(20\d{2})', question)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    m = re.search(r'(\d{2})\s*[-/]\s*(\d{2})(?:\s*赛季)?', question)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        return f"20{y1:02d}-20{y2:02d}"

    if re.search(r'本赛季|这个赛季|这赛季|当前赛季', question):
        return "2025-2026"

    if re.search(r'上赛季|上个赛季', question):
        return "2024-2025"

    return None


def _extract_number(question: str, default: int = 10) -> int:
    """
    提取问题中的数字（如"近5场"或"近五场"中的数字）。
    同时支持阿拉伯数字和中文数字。
    """
    # 中文数字映射
    _CN_NUM = {
        "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
        "二十": 20, "三十": 30,
    }

    # ── 1. 阿拉伯数字: "近5场", "最近10场" ──
    m = re.search(r'近\s*(\d+)\s*场', question)
    if m:
        return int(m.group(1))
    m = re.search(r'最近\s*(\d+)', question)
    if m:
        return int(m.group(1))

    # ── 2. 中文数字: "近五场", "最近三场" ──
    # 按长度降序匹配，确保"十五"优先于"十"和"五"
    cn_keys_sorted = sorted(_CN_NUM.keys(), key=len, reverse=True)
    cn_pattern = '|'.join(cn_keys_sorted)
    m = re.search(rf'近\s*({cn_pattern})\s*场', question)
    if m:
        return _CN_NUM[m.group(1)]
    m = re.search(rf'最近\s*({cn_pattern})\s*(?:场|轮)', question)
    if m:
        return _CN_NUM[m.group(1)]

    return default


def _extract_years_back(question: str) -> int | None:
    """
    提取"最近N年"中的年份数。
    支持: "最近两年"、"近3年"、"最近5年" 等。
    """
    chinese_numbers = {
        "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    }

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
            if num_str.isdigit():
                return int(num_str)
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

def _try_template_match(question: str) -> Optional[tuple[str, tuple]]:
    """
    尝试将用户问题匹配到预定义的 SQL 模板。
    匹配成功返回 (sql, params)，未命中返回 None。

    支持的高频意图:
      - 两队历史交锋记录
      - 某队近N场比赛
      - 某队某赛季全部比赛
    """
    teams = _extract_teams(question)
    season = _extract_season(question)

    # ── 1. 两队历史交锋（优先级最高，两支球队时优先匹配） ──
    if len(teams) >= 2:
        has_match_keywords = re.search(
            r'交锋|对战|对阵|对决|vs|VS|比赛记录|交手|head.?to.?head|比分|结果|赔率|盘口',
            question
        )
        if has_match_keywords or len(teams) >= 2:
            n = _extract_number(question, default=10)
            return match_sql_queries.head_to_head_sql(teams[0], teams[1], limit=n)

    # ── 2. 某队近N场比赛 ──
    # 触发条件: "近5场" / "近五场" / "最近" / "近期" / "战绩" / "表现" / "recent"
    if len(teams) == 1 and re.search(
        r'近[\d一二两三四五六七八九十]+场|最近|近期|战绩|表现|recent',
        question, re.IGNORECASE,
    ):
        n = _extract_number(question)
        return match_sql_queries.recent_matches_sql(teams[0], limit=n)

    # ── 3. 某队某赛季全部比赛 ──
    if len(teams) >= 1 and season:
        return match_sql_queries.team_season_stats_sql(teams[0], season)

    return None


# ═══════════════════════════════════════════════════════════════
#  SQL 执行 & 结果格式化
# ═══════════════════════════════════════════════════════════════

def _execute_sql(sql: str, params: Optional[tuple], connection: pymysql.Connection) -> str:
    """
    执行 SQL 并将结果格式化为易于 Summary 节点阅读的文本。

    Args:
        sql:        SQL 查询语句
        params:     参数元组（可选，用于参数化查询）
        connection: pymysql Connection 实例

    Returns:
        str: 格式化的查询结果文本
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            records = cursor.fetchall()
    except Exception as e:
        return f"[MySQL 执行错误] {type(e).__name__}: {e}"

    if not records:
        return "[查询结果] 未找到匹配数据。请检查球队名称或赛季是否正确。"

    # 截断过长结果
    total = len(records)
    if total > MAX_RESULT_ROWS:
        records = records[:MAX_RESULT_ROWS]
        truncated = True
    else:
        truncated = False

    # 格式化为结构化文本（每条记录一行，key: value 格式）
    lines = []
    for i, rec in enumerate(records, 1):
        parts = []
        for k, v in rec.items():
            # 跳过 None 值和 id 字段，精简输出
            if v is None or k == "id":
                continue
            parts.append(f"{k}: {v}")
        lines.append(f"  [{i}] {' | '.join(parts)}")

    header = f"[查询结果] 共 {total} 条记录"
    if truncated:
        header += f"（仅展示前 {MAX_RESULT_ROWS} 条）"

    return header + "\n" + "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  @tool 统一入口
# ═══════════════════════════════════════════════════════════════

# 用户意图预拦截正则 — 拒绝写操作/危险请求
_DANGEROUS_INTENT_RE = re.compile(
    r'删除|删掉|移除|清空|修改|更新|更改|添加|新增|创建|插入|'
    r'DROP|DELETE|REMOVE|CREATE|SET|INSERT|UPDATE|ALTER|TRUNCATE',
    re.IGNORECASE,
)


@tool
def mysql_query(question: str) -> str:
    """
    足球比赛数据查询工具 —— 在 MySQL 数据库中查询五大联赛比赛数据、赔率、盘口等详细信息。

    适合查询需要精确数据的场景，如:
      - 两队历史交锋记录（含赔率和亚盘）
      - 某队近N场比赛详情
      - 某队某赛季全部比赛数据
      - 赔率走势、大小球数据、亚盘让球数据
      - 以及更多通过自然语言描述的复杂统计查询

    Args:
        question: 用户的自然语言查询问题（中文或英文均可）

    Returns:
        str: 格式化的查询结果文本
    """
    print(f"\n{'='*60}")
    print(f"[MySQL Tool] 收到查询: {question}")
    print(f"{'='*60}")

    # ━━━ 阶段零: 用户意图预拦截 ━━━
    if _DANGEROUS_INTENT_RE.search(question):
        print("[MySQL Tool] ⛔ 用户输入包含写操作意图，已拦截")
        return "[安全拦截] 本工具仅支持只读查询，不允许对数据库进行删除、修改或新增操作。请改用查询类问题。"

    connection = _get_connection()

    # ━━━ 阶段一: Fast Path — 模板匹配 ━━━
    print("[MySQL Tool] 阶段一: 尝试模板匹配...")
    template_result = _try_template_match(question)

    if template_result is not None:
        sql, params = template_result
        print(f"[MySQL Tool] ✅ Fast Path 命中模板")
        print(f"  SQL: {sql.strip()[:150]}...")
        print(f"  Params: {params}")
        return _execute_sql(sql, params, connection)

    # ━━━ 阶段二: Deep Path — Text2SQL ━━━
    print("[MySQL Tool] 模板未命中，进入 Deep Path (Text2SQL)...")

    # 预处理: 将问题中的中文球队名替换为英文，帮助 LLM 生成更准确的 SQL
    enriched_question = question
    teams = _extract_teams(question)
    if teams:
        enriched_question += f"\n（提示: 涉及的球队英文名为 {', '.join(teams)}）"

    season = _extract_season(question)
    if season:
        enriched_question += f"\n（提示: 赛季标识为 {season}）"

    league = _resolve_league(question)
    if league:
        enriched_question += f"\n（提示: 联赛为 {league}）"

    # 获取 cursor 供 text2sql 的 EXPLAIN 防线使用
    with connection.cursor() as cursor:
        sql, _ = generate_sql(
            question=enriched_question,
            cursor=cursor,
        )

    # 检查是否是错误说明（generate_sql 失败时返回错误文本 + None）
    if sql.startswith("[Text2SQL"):
        print(f"[MySQL Tool] ❌ Text2SQL 彻底失败")
        return sql

    print(f"[MySQL Tool] ✅ Deep Path 生成 SQL 成功")
    return _execute_sql(sql, None, connection)


# ═══════════════════════════════════════════════════════════════
#  命令行测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  MySQL 比赛数据查询工具 · 交互式测试")
    print("=" * 60)
    print("  输入自然语言问题查询数据库，输入 q 退出")
    print("=" * 60)

    while True:
        try:
            print("\n>> 请输入查询: ", end="", flush=True)
            q = sys.stdin.readline()
            if not q:
                break
            q = q.strip()
        except KeyboardInterrupt:
            print("\n再见！")
            break

        if q.lower() in ("q", "quit", "exit"):
            break

        if not q:
            continue

        result = mysql_query.invoke(q)
        print(f"\n{result}")

    # 关闭连接
    if _connection and _connection.open:
        _connection.close()
        print("\n[MySQL] 连接已关闭")

