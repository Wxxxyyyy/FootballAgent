# -*- coding: utf-8 -*-
"""
信息查询 Agent · 技能模块（Skill）
════════════════════════════════════
职责：
  1. 调用 Planner 获取子问题列表（含代词消解 + 工具分类）
  2. 按 tool 字段串行分发到 mysql_query / search_knowledge_base
  3. 聚合所有子问题结果，拼出完整的 raw_agent_response

对外暴露 query() 作为唯一入口，node.py 只需调用它即可。

使用方式：
    from agents.information_agent.skill import query
    result = query(user_msg="阿森纳近5场比赛和它的历史底蕴",
                   messages=state["messages"])
    # result["response"]: 聚合后的完整查询结果文本
"""

from typing import Sequence
from langchain_core.messages import BaseMessage

from agents.information_agent.planner import plan
from agents.tools.mysql_tools.tool_entry import mysql_query
from agents.tools.vector_tools.tool_entry import search_knowledge_base


# ═══════════════════════════════════════════════════════════════
#  工具名 → 展示名（聚合输出用）
# ═══════════════════════════════════════════════════════════════

_TOOL_DISPLAY = {
    "mysql":  "MySQL 比赛数据",
    "vector": "向量知识库",
}

# LLM / 工具全部不可用时的兜底回复
_FALLBACK_REPLY = (
    "抱歉，信息查询过程中遇到了技术问题，暂时无法返回数据。"
    "请稍后再试，或者换一种方式描述您的问题。"
)


# ═══════════════════════════════════════════════════════════════
#  单子问题执行
# ═══════════════════════════════════════════════════════════════

def _dispatch_one(question: str, tool: str) -> dict:
    """
    执行单个子问题的工具调用。

    Args:
        question: 消解后的子问题文本
        tool:     "mysql" 或 "vector"

    Returns:
        dict: {
            "question": 原始子问题,
            "tool":     工具名,
            "success":  bool,
            "result":   工具返回的文本（成功时）或错误信息（失败时）,
        }
    """
    tool_display = _TOOL_DISPLAY.get(tool, tool)
    print(f"[Skill] 调用 {tool_display}: '{question[:60]}'")

    try:
        if tool == "vector":
            result_text = search_knowledge_base.invoke(question)
        else:
            # 默认走 mysql（包括 tool 字段异常的情况）
            result_text = mysql_query.invoke(question)

        # 判断是否为空结果 / 安全拦截
        is_empty = (
            "未找到" in result_text
            or "安全拦截" in result_text
            or "Text2SQL" in result_text
        )

        if is_empty:
            print(f"[Skill] ⚠️ {tool_display} 返回空/拦截结果")

        return {
            "question": question,
            "tool": tool,
            "success": True,
            "result": result_text,
        }

    except Exception as e:
        error_msg = f"[工具调用失败] {tool_display} 出现异常: {type(e).__name__}: {e}"
        print(f"[Skill] ❌ {error_msg}")
        return {
            "question": question,
            "tool": tool,
            "success": False,
            "result": error_msg,
        }


# ═══════════════════════════════════════════════════════════════
#  结果聚合
# ═══════════════════════════════════════════════════════════════

def _aggregate_results(sub_results: list[dict]) -> str:
    """
    将多个子问题的执行结果聚合为一个完整的 raw_agent_response。

    格式：
      - 头部汇总：成功/失败数量
      - 每个子问题带序号、问题文本、来源工具、查询结果
      - 尾部列出失败的子问题（如果有）

    Args:
        sub_results: _dispatch_one 返回的 dict 列表

    Returns:
        str: 聚合后的完整文本
    """
    total = len(sub_results)
    success_list = [r for r in sub_results if r["success"]]
    fail_list = [r for r in sub_results if not r["success"]]

    lines = []

    # ── 头部汇总 ──
    if total == 1:
        lines.append(f"[信息查询] 共 1 个问题")
    else:
        lines.append(
            f"[信息查询] 共 {total} 个子问题 "
            f"| 查询成功: {len(success_list)} "
            f"| 查询失败: {len(fail_list)}"
        )
    lines.append("")

    # ── 逐个子问题输出 ──
    for i, r in enumerate(sub_results, 1):
        tool_display = _TOOL_DISPLAY.get(r["tool"], r["tool"])
        status_tag = "✅" if r["success"] else "❌"

        if total == 1:
            lines.append(f"用户问题: {r['question']}")
            lines.append(f"数据来源: {tool_display}")
        else:
            lines.append(f"【子问题 {i}/{total}】{r['question']}")
            lines.append(f"数据来源: {tool_display} {status_tag}")
        lines.append("─" * 40)
        lines.append(r["result"])
        lines.append("")

    # ── 尾部失败汇总（仅多问题 + 有失败时） ──
    if fail_list and total > 1:
        lines.append("─" * 40)
        lines.append(f"⚠️ 以下 {len(fail_list)} 个子问题查询失败:")
        for r in fail_list:
            lines.append(f"  · {r['question']}（{_TOOL_DISPLAY.get(r['tool'], r['tool'])}）")
        lines.append("")

    return "\n".join(lines).strip()


# ═══════════════════════════════════════════════════════════════
#  对外接口
# ═══════════════════════════════════════════════════════════════

def query(user_msg: str, messages: Sequence[BaseMessage]) -> dict:
    """
    信息查询技能主入口 —— 规划 + 调度 + 聚合，一站式完成。

    流程：
      1. 调 planner.plan() 获取子问题列表（含代词消解 + 工具分类）
      2. 串行遍历子问题，调对应的 @tool 获取结果
      3. 聚合所有结果，生成 raw_agent_response

    Args:
        user_msg:  用户最新输入文本
        messages:  state 中的完整对话历史（含当前消息）

    Returns:
        dict: {
            "response": 聚合后的完整查询结果文本（供 summary_agent 消费）
        }
    """
    # ── 步骤1: 调 Planner ──
    print("[Skill] 步骤1: 调用 Planner 进行问题规划...")
    try:
        sub_questions = plan(user_msg=user_msg, messages=messages)
    except Exception as e:
        print(f"[Skill] ❌ Planner 异常: {e}，使用原始输入作为单问题")
        sub_questions = [{"question": user_msg, "tool": "mysql"}]

    if not sub_questions:
        print("[Skill] ⚠️ Planner 返回空列表，使用原始输入作为单问题")
        sub_questions = [{"question": user_msg, "tool": "mysql"}]

    print(f"[Skill] 步骤2: 串行调度 {len(sub_questions)} 个子问题...")

    # ── 步骤2: 串行调度工具 ──
    sub_results = []
    for i, item in enumerate(sub_questions, 1):
        q = item["question"]
        t = item["tool"]
        print(f"\n[Skill] ── 执行子问题 {i}/{len(sub_questions)} ──")
        result = _dispatch_one(question=q, tool=t)
        sub_results.append(result)

    # ── 步骤3: 聚合结果 ──
    print(f"\n[Skill] 步骤3: 聚合 {len(sub_results)} 个结果...")
    aggregated = _aggregate_results(sub_results)

    if not aggregated.strip():
        aggregated = _FALLBACK_REPLY

    return {
        "response": aggregated,
    }

