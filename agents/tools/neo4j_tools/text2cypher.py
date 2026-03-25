# -*- coding: utf-8 -*-
"""
Text2Cypher · LLM 生成 Cypher + 纠错重试回环
═══════════════════════════════════════════════
流程:
  1. 将用户问题 + 图谱 Schema 组装为 Prompt，让 LLM 生成 Cypher
  2. 生成结果送入 security.py 四道防线逐一校验
  3. 任何防线报错 → 组装「task + statement + errors + schema」重试上下文
  4. while 循环最多重试 MAX_RETRIES 次，全部失败则优雅退出
"""

import re
from langchain_core.messages import SystemMessage, HumanMessage

from common.llm_select import llm_call, LLM_MODEL_QWEN_SIMPLE_NAME
from agents.tools.neo4j_tools.security import (
    run_all_defenses,
    CypherSecurityError,
    CypherDirectionError,
    CypherSyntaxError,
    CypherMappingError,
)

# ═══════════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════════

MAX_RETRIES = 3

# 图谱 Schema 描述（喂给 LLM，让它理解数据结构）
GRAPH_SCHEMA = """
## Neo4j 图谱 Schema

### 节点
(:Team)
  属性:
    - name   : STRING  球队英文名称（如 "Arsenal", "Man United", "Barcelona", "Bayern Munich"）
    - league : STRING  所属联赛，取值范围: "England", "Spain", "Italy", "Germany", "France"

### 关系
(:Team)-[:PLAYED_AGAINST]->(:Team)
  方向语义: 起点 = 主队(HomeTeam), 终点 = 客队(AwayTeam)
  属性:
    - match_date        : STRING   比赛日期（如 "2024-08-16"）
    - season            : STRING   赛季标识（如 "2024-2025"）
    - match_result      : STRING   比分文本（格式: "{主队名} {主队进球}:{客队进球} {客队名}"，如 "Arsenal 3:1 Chelsea"）
    - total_goals       : INTEGER  总进球数
    - odds_info         : STRING   Bet365 初盘胜平赔率（如 "Arsenal 胜赔率: 1.50 | 平局赔率: 4.20 | Chelsea 胜赔率: 5.25"）
    - over_under_odds   : STRING   初盘大小球赔率
    - closing_odds_info : STRING   Bet365 终盘赔率
    - closing_over_under: STRING   终盘大小球赔率
    - asian_handicap_info: STRING  亚盘信息
    - league            : STRING   联赛
    - season            : STRING   赛季

### 注意事项
1. 图中 **只有** Team 节点和 PLAYED_AGAINST 关系，没有其他类型。
2. 查两队历史交锋时请使用**无向匹配** `-[:PLAYED_AGAINST]-`，以同时获取双方主客场记录。
3. 查某队主场记录: (t:Team {name:$team})-[:PLAYED_AGAINST]->(opp)
4. 查某队客场记录: (opp)-[:PLAYED_AGAINST]->(t:Team {name:$team})
5. 球队名称必须使用**英文原名**（如 "Arsenal" 而非 "阿森纳"）。
6. 赛季格式为 "YYYY-YYYY"（如 "2024-2025"）。
7. 你只能生成 **只读查询**（MATCH/RETURN/WHERE/WITH/ORDER BY/LIMIT/UNION），绝不能使用 CREATE/DELETE/SET/MERGE 等写操作。
8. 请直接返回纯 Cypher 代码，不要加 markdown 代码块标记。
""".strip()


# ═══════════════════════════════════════════════════════════════
#  Prompt 组装
# ═══════════════════════════════════════════════════════════════

def _build_initial_prompt(question: str) -> list:
    """
    首次生成 Cypher 的 Prompt 组装。

    Args:
        question: 用户的自然语言问题

    Returns:
        list: LangChain Message 列表
    """
    system_msg = SystemMessage(content=f"""你是一个专业的 Neo4j Cypher 查询生成器。
你的唯一任务是：根据用户问题和图谱 Schema，生成精确的 Cypher 查询语句。

{GRAPH_SCHEMA}

## 输出要求
- 只输出纯 Cypher 代码，不要任何解释、注释或 markdown 包裹
- 使用参数化查询时，参数名用 $xxx 格式
- 确保查询结果包含足够的上下文字段（日期、赛季、比分等）
- 结果默认按日期降序排列，LIMIT 20
""")

    human_msg = HumanMessage(content=f"请根据以下问题生成 Cypher 查询:\n\n{question}")

    return [system_msg, human_msg]


def _build_retry_prompt(
    task: str,
    statement: str,
    errors: str,
) -> list:
    """
    重试时的 Prompt 组装。
    包含四要素: task（原始问题）、statement（错误Cypher）、errors（错误堆栈）、schema。

    Args:
        task:      用户原始问题（保持意图不偏离）
        statement: 上一次生成的错误 Cypher
        errors:    四道防线抛出的具体错误信息

    Returns:
        list: LangChain Message 列表
    """
    system_msg = SystemMessage(content=f"""你是一个专业的 Neo4j Cypher 查询生成器。
上一次生成的 Cypher 未能通过安全校验，请根据错误信息修正后重新生成。

{GRAPH_SCHEMA}

## 输出要求
- 只输出修正后的纯 Cypher 代码，不要任何解释
- 仔细阅读下方错误信息，针对性修复
""")

    human_msg = HumanMessage(content=f"""## 用户原始问题（task）
{task}

## 上一次生成的错误 Cypher（statement）
{statement}

## 错误信息（errors）
{errors}

请修正上述错误，重新生成正确的 Cypher 查询。只输出纯 Cypher 代码。""")

    return [system_msg, human_msg]


# ═══════════════════════════════════════════════════════════════
#  Cypher 提取（从 LLM 回复中剥离多余内容）
# ═══════════════════════════════════════════════════════════════

def _extract_cypher(llm_output: str) -> str:
    """
    从 LLM 的输出中提取纯 Cypher 代码。
    处理可能的 markdown 代码块包裹、多余解释等。
    """
    text = llm_output.strip()

    # 去掉 markdown 代码块
    code_block = re.search(r'```(?:cypher|neo4j)?\s*\n?(.*?)```', text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()

    # 去掉可能的行首注释 //
    lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('//'):
            continue
        lines.append(line)

    return '\n'.join(lines).strip()


# ═══════════════════════════════════════════════════════════════
#  核心函数: Text2Cypher 带纠错重试回环
# ═══════════════════════════════════════════════════════════════

def generate_cypher(
    question: str,
    driver,
    database: str = "neo4j",
) -> tuple[str, dict | None]:
    """
    Text2Cypher 终极回环：LLM 生成 Cypher → 四道防线校验 → 失败重试。

    流程:
      1. 首次调用 LLM 生成 Cypher
      2. 送入四道防线校验
      3. 失败 → 组装 (task, statement, errors, schema) 重试上下文
      4. 最多重试 MAX_RETRIES 次
      5. 全部失败 → 优雅退出，返回错误说明

    Args:
        question: 用户的自然语言问题
        driver:   Neo4j Driver 实例
        database: 数据库名

    Returns:
        tuple: (cypher: str, params: dict | None)
            - 成功: 通过全部防线的 Cypher 和空参数
            - 失败: (错误说明字符串, None)
    """
    last_cypher = ""
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[Text2Cypher] 第 {attempt}/{MAX_RETRIES} 次尝试...")

        # ── 组装 Prompt ──
        if attempt == 1:
            messages = _build_initial_prompt(question)
        else:
            messages = _build_retry_prompt(
                task=question,
                statement=last_cypher,
                errors=last_error,
            )

        # ── 调用 LLM 生成 Cypher ──
        try:
            response = llm_call(messages, model=LLM_MODEL_QWEN_SIMPLE_NAME)
            raw_cypher = _extract_cypher(response.content)
        except Exception as e:
            last_error = f"LLM 调用失败: {type(e).__name__}: {e}"
            print(f"[Text2Cypher] ❌ {last_error}")
            continue

        if not raw_cypher:
            last_error = "LLM 返回了空内容，未能生成有效的 Cypher"
            print(f"[Text2Cypher] ❌ {last_error}")
            continue

        last_cypher = raw_cypher
        print(f"[Text2Cypher] LLM 生成 Cypher:\n  {raw_cypher[:200]}...")

        # ── 送入四道防线 ──
        try:
            verified_cypher = run_all_defenses(
                cypher=raw_cypher,
                params=None,  # LLM 生成的是内联值，无参数
                driver=driver,
                database=database,
            )
            print(f"[Text2Cypher] ✅ 通过全部四道防线")
            return verified_cypher, None

        except (CypherSecurityError, CypherDirectionError,
                CypherSyntaxError, CypherMappingError) as e:
            last_error = str(e)
            print(f"[Text2Cypher] ⚠️ 防线拦截 ({type(e).__name__}): {last_error}")
            # 继续下一轮重试

    # ── 全部重试耗尽 ──
    print(f"[Text2Cypher] ❌ {MAX_RETRIES} 次重试全部失败，优雅退出")
    return (
        f"[Text2Cypher 查询失败] 经过 {MAX_RETRIES} 次尝试仍无法生成合法 Cypher。\n"
        f"用户问题: {question}\n"
        f"最后一次 Cypher: {last_cypher}\n"
        f"最后一次错误: {last_error}\n"
        f"建议: 请尝试更具体的问题描述，或直接指定球队英文名称。"
    ), None

