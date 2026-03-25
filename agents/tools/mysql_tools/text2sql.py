# -*- coding: utf-8 -*-
"""
Text2SQL · LLM 生成 SQL + 纠错重试回环
═══════════════════════════════════════════
流程:
  1. 将用户问题 + 数据库 Schema 组装为 Prompt，让 LLM 生成 SQL
  2. 生成结果送入 security.py 四道防线逐一校验
  3. 任何防线报错 → 组装「task + statement + errors + schema」重试上下文
  4. while 循环最多重试 MAX_RETRIES 次，全部失败则优雅退出

默认使用 LLM_MODEL_QWEN_SIMPLE_NAME（Coder 模型），擅长写代码。
"""

import re
from langchain_core.messages import SystemMessage, HumanMessage

from common.llm_select import llm_call, LLM_MODEL_QWEN_SIMPLE_NAME
from agents.tools.mysql_tools.security import (
    run_all_defenses,
    SQLSecurityError,
    SQLSchemaError,
    SQLSyntaxError,
    SQLLimitError,
)

# ═══════════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════════

MAX_RETRIES = 3

# 数据库 Schema 描述（喂给 LLM，防幻觉）
DB_SCHEMA = """
## MySQL 数据库 Schema

### 表名: match_master
数据来源: football-data.co.uk（五大联赛 2021-2026 赛季）

### 字段说明（中英文对照）

#### 基础信息
| 字段        | 类型    | 说明 |
|-------------|---------|------|
| id          | BIGINT  | 自增主键 |
| Div         | TEXT    | 联赛代码（E0=英超, SP1=西甲, I1=意甲, D1=德甲, F1=法甲）|
| Date        | TEXT    | 比赛日期（格式: "YYYY-MM-DD"，如 "2024-08-16"）|
| HomeTeam    | TEXT    | 主队英文名（如 "Arsenal", "Man United", "Barcelona"）|
| AwayTeam    | TEXT    | 客队英文名 |
| league      | TEXT    | 联赛名（取值: "England", "Spain", "Italy", "Germany", "France"）|
| season      | TEXT    | 赛季标识（如 "2024-2025"）|

#### 比赛结果
| 字段 | 类型    | 说明 |
|------|---------|------|
| FTHG | BIGINT  | 全场主队进球 (Full Time Home Goals) |
| FTAG | BIGINT  | 全场客队进球 (Full Time Away Goals) |
| FTR  | TEXT    | 全场结果 (H=主胜, D=平, A=客胜) |
| HTHG | DOUBLE  | 半场主队进球 (Half Time Home Goals) |
| HTAG | DOUBLE  | 半场客队进球 (Half Time Away Goals) |
| HTR  | TEXT    | 半场结果 |

#### 胜平负赔率（初盘 / 终盘）
| 字段         | 说明 |
|--------------|------|
| B365H/B365D/B365A | Bet365 初盘：主胜/平局/客胜赔率 |
| PSH/PSD/PSA       | Pinnacle 初盘 |
| AvgH/AvgD/AvgA    | 市场平均初盘 |
| MaxH/MaxD/MaxA     | 市场最大初盘 |
| B365CH/B365CD/B365CA | Bet365 终盘 |
| PSCH/PSCD/PSCA       | Pinnacle 终盘 |
| AvgCH/AvgCD/AvgCA    | 市场平均终盘 |
| MaxCH/MaxCD/MaxCA     | 市场最大终盘 |

#### 大小球赔率
| 字段               | 说明 |
|--------------------|------|
| B365_Over25 / B365_Under25 | Bet365 初盘 大2.5球 / 小2.5球 |
| P_Over25 / P_Under25       | Pinnacle 初盘 |
| Avg_Over25 / Avg_Under25   | 市场平均初盘 |
| Max_Over25 / Max_Under25   | 市场最大初盘 |
| B365C_Over25 / B365C_Under25 | Bet365 终盘 |
| PC_Over25 / PC_Under25       | Pinnacle 终盘 |
| AvgC_Over25 / AvgC_Under25   | 市场平均终盘 |
| MaxC_Over25 / MaxC_Under25   | 市场最大终盘 |

#### 亚盘让球
| 字段       | 说明 |
|------------|------|
| AHh        | 主队让球盘口大小（初盘）|
| B365AHH / B365AHA | Bet365 初盘：主队亚盘赔率 / 客队亚盘赔率 |
| PAHH / PAHA       | Pinnacle 初盘亚盘 |
| AvgAHH / AvgAHA   | 市场平均初盘亚盘 |
| MaxAHH / MaxAHA   | 市场最大初盘亚盘 |
| AHCh               | 主队让球盘口（终盘）|
| B365CAHH / B365CAHA | Bet365 终盘亚盘 |
| PCAHH / PCAHA       | Pinnacle 终盘亚盘 |
| AvgCAHH / AvgCAHA   | 市场平均终盘亚盘 |
| MaxCAHH / MaxCAHA   | 市场最大终盘亚盘 |

### 注意事项
1. 数据库中 **只有** `match_master` 一张表，不要编造其他表名。
2. 球队名必须使用**英文原名**（如 "Arsenal" 而非 "阿森纳"，"Man United" 而非 "曼联"）。
3. Date 字段存储为 TEXT 格式 "YYYY-MM-DD"，可直接用字符串比较做日期范围过滤。
4. 赛季格式固定为 "YYYY-YYYY"（如 "2024-2025"）。
5. 你只能生成 **SELECT** 只读查询，绝不能使用 INSERT/UPDATE/DELETE/DROP 等写操作。
6. 查两队交锋时，需同时考虑主客场互换:
   WHERE (HomeTeam='A' AND AwayTeam='B') OR (HomeTeam='B' AND AwayTeam='A')
7. 请直接返回纯 SQL 代码，不要加 markdown 代码块标记。
8. 查询结果默认按 Date DESC 排序，LIMIT 30。
""".strip()


# ═══════════════════════════════════════════════════════════════
#  Prompt 组装
# ═══════════════════════════════════════════════════════════════

def _build_initial_prompt(question: str) -> list:
    """
    首次生成 SQL 的 Prompt 组装。

    Args:
        question: 用户的自然语言问题

    Returns:
        list: LangChain Message 列表
    """
    system_msg = SystemMessage(content=f"""你是一个专业的 MySQL SQL 查询生成器。
你的唯一任务是：根据用户问题和数据库 Schema，生成精确的 SELECT 查询语句。

{DB_SCHEMA}

## 输出要求
- 只输出纯 SQL 代码，不要任何解释、注释或 markdown 包裹
- 不要使用分号结尾
- 确保查询结果包含足够的上下文字段（日期、赛季、比分等）
- 结果默认按 Date DESC 排序，LIMIT 30
""")

    human_msg = HumanMessage(content=f"请根据以下问题生成 SQL 查询:\n\n{question}")

    return [system_msg, human_msg]


def _build_retry_prompt(
    task: str,
    statement: str,
    errors: str,
) -> list:
    """
    重试时的 Prompt 组装。
    包含四要素: task（原始问题）、statement（错误SQL）、errors（错误堆栈）、schema。

    Args:
        task:      用户原始问题（保持意图不偏离）
        statement: 上一次生成的错误 SQL
        errors:    四道防线抛出的具体错误信息

    Returns:
        list: LangChain Message 列表
    """
    system_msg = SystemMessage(content=f"""你是一个专业的 MySQL SQL 查询生成器。
上一次生成的 SQL 未能通过安全校验，请根据错误信息修正后重新生成。

{DB_SCHEMA}

## 输出要求
- 只输出修正后的纯 SQL 代码，不要任何解释
- 不要使用分号结尾
- 仔细阅读下方错误信息，针对性修复
""")

    human_msg = HumanMessage(content=f"""## 用户原始问题（task）
{task}

## 上一次生成的错误 SQL（statement）
{statement}

## 错误信息（errors）
{errors}

请修正上述错误，重新生成正确的 SQL 查询。只输出纯 SQL 代码。""")

    return [system_msg, human_msg]


# ═══════════════════════════════════════════════════════════════
#  SQL 提取（从 LLM 回复中剥离多余内容）
# ═══════════════════════════════════════════════════════════════

def _extract_sql(llm_output: str) -> str:
    """
    从 LLM 的输出中提取纯 SQL 代码。
    处理可能的 markdown 代码块包裹、多余解释等。

    Args:
        llm_output: LLM 的原始输出文本

    Returns:
        str: 提取出的纯 SQL 代码
    """
    text = llm_output.strip()

    # 去掉 markdown 代码块 ```sql ... ``` 或 ```mysql ... ```
    code_block = re.search(r'```(?:sql|mysql)?\s*\n?(.*?)```', text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()

    # 去掉可能的行首 SQL 注释 --
    lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('--'):
            continue
        lines.append(line)

    # 去掉尾部分号
    result = '\n'.join(lines).strip()
    result = result.rstrip(';').strip()

    return result


# ═══════════════════════════════════════════════════════════════
#  核心函数: Text2SQL 带纠错重试回环
# ═══════════════════════════════════════════════════════════════

def generate_sql(
    question: str,
    cursor,
) -> tuple[str, None]:
    """
    Text2SQL 终极回环：LLM 生成 SQL → 四道防线校验 → 失败重试。

    流程:
      1. 首次调用 LLM（轻量 Coder 模型）生成 SQL
      2. 送入四道防线校验
      3. 失败 → 组装 (task, statement, errors, schema) 重试上下文
      4. 最多重试 MAX_RETRIES 次
      5. 全部失败 → 优雅退出，返回错误说明

    Args:
        question: 用户的自然语言问题
        cursor:   pymysql Cursor 实例

    Returns:
        tuple: (sql: str, None)
            - 成功: 通过全部防线的 SQL 和 None
            - 失败: (错误说明字符串, None)
    """
    last_sql = ""
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[Text2SQL] 第 {attempt}/{MAX_RETRIES} 次尝试...")

        # ── 组装 Prompt ──
        if attempt == 1:
            messages = _build_initial_prompt(question)
        else:
            messages = _build_retry_prompt(
                task=question,
                statement=last_sql,
                errors=last_error,
            )

        # ── 调用 LLM 生成 SQL ──
        try:
            response = llm_call(messages, model=LLM_MODEL_QWEN_SIMPLE_NAME)
            raw_sql = _extract_sql(response.content)
        except Exception as e:
            last_error = f"LLM 调用失败: {type(e).__name__}: {e}"
            print(f"[Text2SQL] ❌ {last_error}")
            continue

        if not raw_sql:
            last_error = "LLM 返回了空内容，未能生成有效的 SQL"
            print(f"[Text2SQL] ❌ {last_error}")
            continue

        last_sql = raw_sql
        print(f"[Text2SQL] LLM 生成 SQL:\n  {raw_sql[:200]}...")

        # ── 送入四道防线 ──
        try:
            verified_sql = run_all_defenses(
                sql=raw_sql,
                cursor=cursor,
            )
            print(f"[Text2SQL] ✅ 通过全部四道防线")
            return verified_sql, None

        except (SQLSecurityError, SQLSchemaError,
                SQLSyntaxError, SQLLimitError) as e:
            last_error = str(e)
            print(f"[Text2SQL] ⚠️ 防线拦截 ({type(e).__name__}): {last_error}")
            # 继续下一轮重试

    # ── 全部重试耗尽 ──
    print(f"[Text2SQL] ❌ {MAX_RETRIES} 次重试全部失败，优雅退出")
    return (
        f"[Text2SQL 查询失败] 经过 {MAX_RETRIES} 次尝试仍无法生成合法 SQL。\n"
        f"用户问题: {question}\n"
        f"最后一次 SQL: {last_sql}\n"
        f"最后一次错误: {last_error}\n"
        f"建议: 请尝试更具体的问题描述，或直接指定球队英文名称。"
    ), None

