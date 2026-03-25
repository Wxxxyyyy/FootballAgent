# -*- coding: utf-8 -*-
"""
MySQL 查询安全校验 · 四道防线
══════════════════════════════
防线1 (SQLSecurityError)  : 读写隔离 + SQL 注入拦截
    正则扫描 INSERT / UPDATE / DELETE / DROP / ALTER / TRUNCATE / GRANT / REVOKE
    严格拦截分号 `;`、注释符 `--` 和 `/*`，防止 SQL 堆叠注入

防线2 (SQLSchemaError)    : 幻觉校验
    校验表名必须为 match_master，字段名必须在合法列表中
    防止 LLM 编造不存在的表或字段

防线3 (SQLSyntaxError)    : 语法编译
    利用 EXPLAIN {sql} 让 MySQL 引擎做零成本语法检查
    捕获 pymysql.Error 并抛出携带具体报错原因的异常

防线4 (防 Token 爆炸)     : 强制 LIMIT
    检查 SQL 是否包含 LIMIT，若无或 LIMIT > 50 则自动添加/修正为 LIMIT 30

任何一道防线失败都会抛出对应的自定义异常，
供 text2sql.py 的重试回环捕获并组装上下文重新生成。
"""

import re
from typing import Optional


# ═══════════════════════════════════════════════════════════════
#  自定义异常
# ═══════════════════════════════════════════════════════════════

class SQLSecurityError(Exception):
    """防线1: 检测到写操作或 SQL 注入特征"""
    pass


class SQLSchemaError(Exception):
    """防线2: 表名或字段名不在合法 Schema 中（LLM 幻觉）"""
    pass


class SQLSyntaxError(Exception):
    """防线3: SQL 语法错误（EXPLAIN 编译失败）"""
    pass


class SQLLimitError(Exception):
    """防线4: LIMIT 缺失或过大（防 Token 爆炸）"""
    pass


# ═══════════════════════════════════════════════════════════════
#  数据库 Schema（合法表名和字段名白名单）
# ═══════════════════════════════════════════════════════════════

VALID_TABLE = "match_master"

# match_master 表的全部合法字段（与 mysql_loader.py 写入时保持一致）
VALID_COLUMNS = {
    # 基础信息
    "id", "Div", "Date", "HomeTeam", "AwayTeam", "league", "season",
    # 全场比分
    "FTHG", "FTAG", "FTR",
    # 半场比分
    "HTHG", "HTAG", "HTR",
    # Bet365 初盘胜平负
    "B365H", "B365D", "B365A",
    # Pinnacle 初盘胜平负
    "PSH", "PSD", "PSA",
    # 最大值 初盘胜平负
    "MaxH", "MaxD", "MaxA",
    # 平均值 初盘胜平负
    "AvgH", "AvgD", "AvgA",
    # 大小球 初盘
    "B365_Over25", "B365_Under25",
    "P_Over25", "P_Under25",
    "Max_Over25", "Max_Under25",
    "Avg_Over25", "Avg_Under25",
    # 亚盘 初盘
    "AHh", "B365AHH", "B365AHA",
    "PAHH", "PAHA",
    "MaxAHH", "MaxAHA",
    "AvgAHH", "AvgAHA",
    # Bet365 终盘胜平负
    "B365CH", "B365CD", "B365CA",
    # Pinnacle 终盘胜平负
    "PSCH", "PSCD", "PSCA",
    # 最大值 终盘胜平负
    "MaxCH", "MaxCD", "MaxCA",
    # 平均值 终盘胜平负
    "AvgCH", "AvgCD", "AvgCA",
    # 大小球 终盘
    "B365C_Over25", "B365C_Under25",
    "PC_Over25", "PC_Under25",
    "MaxC_Over25", "MaxC_Under25",
    "AvgC_Over25", "AvgC_Under25",
    # 亚盘 终盘
    "AHCh", "B365CAHH", "B365CAHA",
    "PCAHH", "PCAHA",
    "MaxCAHH", "MaxCAHA",
    "AvgCAHH", "AvgCAHA",
}

# 转为小写集合，用于大小写不敏感匹配
_VALID_COLUMNS_LOWER = {c.lower() for c in VALID_COLUMNS}

# 允许出现在 SQL 中但不属于表字段的"伪列/表达式"（如聚合别名、函数名等）
_ALLOWED_NON_COLUMNS = {
    "count", "sum", "avg", "max", "min", "round", "if", "ifnull",
    "case", "when", "then", "else", "end", "as", "distinct",
    "concat", "group_concat", "cast", "coalesce",
    "year", "month", "day", "date_format", "str_to_date",
    "null", "true", "false",
}


# ═══════════════════════════════════════════════════════════════
#  防线 1 — 读写隔离 + SQL 注入拦截
# ═══════════════════════════════════════════════════════════════

# 写操作关键字正则（忽略大小写，匹配完整单词）
_WRITE_PATTERN = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|'
    r'REPLACE|LOAD\s+DATA|INTO\s+OUTFILE|RENAME|CALL|EXECUTE|'
    r'CREATE|LOCK|UNLOCK)\b',
    re.IGNORECASE,
)

# SQL 注入特征正则
_INJECTION_PATTERN = re.compile(
    r';'                     # 分号 → 堆叠注入
    r'|--'                   # 双横线注释
    r"|/\*"                  # 块注释开始
    r'|\bSLEEP\s*\('        # SLEEP 时间盲注
    r'|\bBENCHMARK\s*\(',   # BENCHMARK 时间盲注
)


def check_read_only(sql: str) -> None:
    """
    防线1: 扫描 SQL 是否包含写操作关键字或注入特征。
    只允许 SELECT / EXPLAIN 等只读操作。

    Args:
        sql: 待检查的 SQL 语句

    Raises:
        SQLSecurityError: 检测到写操作或注入特征
    """
    # 检查写操作关键字
    match = _WRITE_PATTERN.search(sql)
    if match:
        raise SQLSecurityError(
            f"[防线1·读写隔离] 检测到写操作关键字: '{match.group().strip()}'。"
            f"只允许执行 SELECT 只读查询。请移除写操作后重试。"
        )

    # 检查注入特征
    match = _INJECTION_PATTERN.search(sql)
    if match:
        raise SQLSecurityError(
            f"[防线1·注入拦截] 检测到潜在 SQL 注入特征: '{match.group().strip()}'。"
            f"不允许使用分号、注释符或危险函数。请清理后重试。"
        )


# ═══════════════════════════════════════════════════════════════
#  防线 2 — 幻觉校验（表名 + 字段名白名单）
# ═══════════════════════════════════════════════════════════════

# 提取 FROM / JOIN 后的表名
_TABLE_PATTERN = re.compile(
    r'\b(?:FROM|JOIN)\s+`?(\w+)`?',
    re.IGNORECASE,
)

# 提取可能的字段引用（点号后 或 SELECT/WHERE/ORDER BY 中的标识符）
# 仅做"可能的字段"提取，后续再过滤
_IDENTIFIER_PATTERN = re.compile(
    r'(?:'
    r'(?:match_master|mm|m|t)\s*\.\s*`?(\w+)`?'   # 表别名.字段
    r'|'
    r'`(\w+)`'                                      # 反引号包裹的字段
    r')',
    re.IGNORECASE,
)


def check_schema(sql: str) -> None:
    """
    防线2: 校验 SQL 中引用的表名和字段名是否在合法白名单中。
    防止 LLM 幻觉编造如 goals_scored、player_name 等不存在的字段。

    Args:
        sql: 待检查的 SQL 语句

    Raises:
        SQLSchemaError: 表名或字段名不在 Schema 中
    """
    # ── 校验表名 ──
    tables = _TABLE_PATTERN.findall(sql)
    for tbl in tables:
        if tbl.lower() != VALID_TABLE.lower():
            raise SQLSchemaError(
                f"[防线2·幻觉校验] 检测到未知表名: '{tbl}'。"
                f"数据库中只有一张表: '{VALID_TABLE}'。"
                f"请修正 SQL 中的表名。"
            )

    # ── 校验字段名 ──
    # 提取通过表别名引用的字段
    aliased_fields = set()
    for m in _IDENTIFIER_PATTERN.finditer(sql):
        field = m.group(1) or m.group(2)
        if field:
            aliased_fields.add(field)

    # 过滤掉 SQL 关键字 / 函数名 / 数字 / 已知的非字段标识符
    unknown_fields = []
    for field in aliased_fields:
        fl = field.lower()
        # 跳过已知的非字段标识符
        if fl in _ALLOWED_NON_COLUMNS:
            continue
        # 跳过纯数字
        if field.isdigit():
            continue
        # 校验是否在合法字段列表中
        if fl not in _VALID_COLUMNS_LOWER:
            unknown_fields.append(field)

    if unknown_fields:
        raise SQLSchemaError(
            f"[防线2·幻觉校验] 检测到未知字段: {unknown_fields}。"
            f"这些字段不存在于 `{VALID_TABLE}` 表中。"
            f"请仅使用以下合法字段: {sorted(VALID_COLUMNS)}"
        )


# ═══════════════════════════════════════════════════════════════
#  防线 3 — 语法编译（EXPLAIN，零成本语法检查）
# ═══════════════════════════════════════════════════════════════

def validate_syntax(sql: str, cursor) -> None:
    """
    防线3: 为 SQL 加上 EXPLAIN 前缀发送给 MySQL 引擎。
    只做查询计划编译，不实际扫描数据，从而零成本验证语法。

    Args:
        sql:    待验证的 SQL 语句
        cursor: pymysql Cursor 实例

    Raises:
        SQLSyntaxError: SQL 语法错误
    """
    explain_sql = f"EXPLAIN {sql}"
    try:
        cursor.execute(explain_sql)
        # 消费结果集，防止后续查询报 "Commands out of sync"
        cursor.fetchall()
    except Exception as e:
        error_msg = str(e)
        raise SQLSyntaxError(
            f"[防线3·语法验证] SQL 编译失败:\n"
            f"  错误信息: {error_msg}\n"
            f"  问题语句: {sql}"
        )


# ═══════════════════════════════════════════════════════════════
#  防线 4 — 强制 LIMIT（防 Token 爆炸）
# ═══════════════════════════════════════════════════════════════

# 匹配 SQL 末尾的 LIMIT 子句
_LIMIT_PATTERN = re.compile(
    r'\bLIMIT\s+(\d+)\s*$',
    re.IGNORECASE,
)

MAX_LIMIT = 50       # 允许的最大 LIMIT 值
DEFAULT_LIMIT = 30   # 自动追加的默认 LIMIT 值


def enforce_limit(sql: str) -> str:
    """
    防线4: 强制 SQL 包含合理的 LIMIT 子句，防止返回海量数据导致 Token 爆炸。
    - 若 SQL 末尾无 LIMIT → 自动追加 LIMIT 30
    - 若 LIMIT 值 > 50 → 修改为 LIMIT 30

    Args:
        sql: 待检查的 SQL 语句

    Returns:
        str: 带有合理 LIMIT 的 SQL
    """
    sql_stripped = sql.strip().rstrip(";")

    match = _LIMIT_PATTERN.search(sql_stripped)
    if match:
        limit_val = int(match.group(1))
        if limit_val > MAX_LIMIT:
            print(f"[防线4·强制LIMIT] LIMIT {limit_val} 超过上限 {MAX_LIMIT}，"
                  f"已修正为 LIMIT {DEFAULT_LIMIT}")
            sql_stripped = _LIMIT_PATTERN.sub(f"LIMIT {DEFAULT_LIMIT}", sql_stripped)
        return sql_stripped
    else:
        # 无 LIMIT 子句，自动追加
        print(f"[防线4·强制LIMIT] SQL 缺少 LIMIT，已自动追加 LIMIT {DEFAULT_LIMIT}")
        return f"{sql_stripped}\nLIMIT {DEFAULT_LIMIT}"


# ═══════════════════════════════════════════════════════════════
#  统一执行：四道防线一次性跑完
# ═══════════════════════════════════════════════════════════════

def run_all_defenses(sql: str, cursor) -> str:
    """
    按顺序执行四道防线。
    - 防线1 和 防线2 纯离线，不需要数据库连接
    - 防线3 需要 cursor 做 EXPLAIN 语法检查
    - 防线4 纯离线，自动修正 LIMIT

    任何一道防线失败都会直接抛出对应异常，
    由 text2sql.py 的重试回环统一捕获。

    Args:
        sql:    LLM 生成的 SQL 语句
        cursor: pymysql Cursor 实例

    Returns:
        str: 通过全部防线后的 SQL（可能经防线4修正 LIMIT）

    Raises:
        SQLSecurityError : 防线1 失败
        SQLSchemaError   : 防线2 失败
        SQLSyntaxError   : 防线3 失败
    """
    # 防线1: 读写隔离 + 注入拦截
    check_read_only(sql)

    # 防线2: 幻觉校验（表名 + 字段名）
    check_schema(sql)

    # 防线4: 强制 LIMIT（在 EXPLAIN 之前修正，避免无 LIMIT 的 EXPLAIN 也报错）
    sql = enforce_limit(sql)

    # 防线3: 语法编译（用修正后的 SQL 做 EXPLAIN）
    validate_syntax(sql, cursor)

    return sql

