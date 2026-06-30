# -*- coding: utf-8 -*-
"""
信息查询 · ReAct 执行器（节点内局部 messages，不污染全局 state.messages）

流程：
  System + Human → LLM(bind_tools) → 若有 tool_calls 则 invoke 工具并追加 ToolMessage → 循环
  直到无 tool_calls 或达到最大轮次。最终聚合所有工具返回 + 可选模型收尾句。
"""

from __future__ import annotations

import json
from typing import Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agents.information_agent.prompts import get_react_system_prompt
from agents.information_agent.planner import _extract_history_text
from agents.tools.mysql_tools.tool_entry import mysql_query
from agents.tools.vector_tools.tool_entry import search_knowledge_base
from common.llm_select import LLM_MODEL_KIMI_NAME, get_llm

# 与 planner 解耦：仅复用历史抽取
TOOLS = [mysql_query, search_knowledge_base]

_TOOL_MAP = {
    "mysql_query": mysql_query,
    "search_knowledge_base": search_knowledge_base,
}

# 单轮用户任务内，LLM 推理-行动最大轮次（每轮 = 1 次 invoke，可含多 tool_call）
MAX_REACT_ITERATIONS = 6


def _make_bound_llm(force_fallback: bool = False):
    llm = get_llm(LLM_MODEL_KIMI_NAME, force_fallback=force_fallback)
    return llm.bind(temperature=0.2).bind_tools(TOOLS)


def _normalize_tool_args(args) -> dict:
    if args is None:
        return {}
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {}
    if isinstance(args, dict):
        return args
    return {}


def _execute_tool_call(name: str, args: dict) -> str:
    q = str(args.get("question", "")).strip()
    if not q:
        return "[工具参数错误] question 为空"
    fn = _TOOL_MAP.get(name)
    if fn is None:
        return f"[未知工具] {name}"
    try:
        return fn.invoke(q)
    except Exception as e:
        return f"[工具执行异常] {type(e).__name__}: {e}"


def _aggregate_react_output(
    tool_traces: list[tuple[str, str]],
    final_ai: AIMessage | None,
) -> str:
    lines: list[str] = []
    n = len(tool_traces)
    if n == 0:
        text = (final_ai.content or "").strip() if final_ai else ""
        return text or "[信息查询] 未获得工具结果。"

    lines.append(f"[信息查询 · ReAct] 共 {n} 次工具调用")
    lines.append("")
    for i, (tool_name, result_text) in enumerate(tool_traces, 1):
        display = "MySQL/Text2SQL" if tool_name == "mysql_query" else "向量知识库/RAG"
        lines.append(f"【第 {i} 步 · {display}】({tool_name})")
        lines.append("─" * 40)
        lines.append(result_text)
        lines.append("")

    if final_ai and (final_ai.content or "").strip():
        lines.append("【模型备注】")
        lines.append(final_ai.content.strip())
    return "\n".join(lines).strip()


def run_react(user_msg: str, messages: Sequence[BaseMessage]) -> str:
    """
    在局部消息列表上运行 ReAct，返回供 Summary 使用的聚合文本。

    不向全局 state.messages 写入任何中间 AIMessage/ToolMessage。
    """
    history_text = _extract_history_text(messages, user_msg=user_msg)
    system_text = get_react_system_prompt(history_text)

    local: list[BaseMessage] = [
        SystemMessage(content=system_text),
        HumanMessage(content=user_msg),
    ]

    tool_traces: list[tuple[str, str]] = []
    final_ai: AIMessage | None = None

    used_fallback = False
    bound = _make_bound_llm(force_fallback=False)

    for iteration in range(MAX_REACT_ITERATIONS):
        try:
            ai = bound.invoke(local)
        except Exception as e:
            print(f"[ReAct] LLM invoke 失败: {e}，尝试备用模型…")
            if not used_fallback:
                used_fallback = True
                bound = _make_bound_llm(force_fallback=True)
                ai = bound.invoke(local)
            else:
                raise

        if not isinstance(ai, AIMessage):
            ai = AIMessage(content=str(ai))

        local.append(ai)
        final_ai = ai

        if not ai.tool_calls:
            break

        for tc in ai.tool_calls:
            if isinstance(tc, dict):
                name = (tc.get("name") or "").strip()
                tid = tc.get("id") or ""
                args = _normalize_tool_args(tc.get("args"))
            else:
                name = (getattr(tc, "name", None) or "").strip()
                tid = getattr(tc, "id", "") or ""
                args = _normalize_tool_args(getattr(tc, "args", None))

            result_text = _execute_tool_call(name, args)
            tool_traces.append((name, result_text))
            print(f"[ReAct] 工具 {name} | question={args.get('question', '')[:80]!r}…")

            if tid:
                local.append(ToolMessage(content=result_text, tool_call_id=tid))
            else:
                local.append(ToolMessage(content=result_text, tool_call_id="call_missing"))

    else:
        print(f"[ReAct] ⚠️ 已达到最大迭代次数 {MAX_REACT_ITERATIONS}（最后一轮可能仍有未满足的工具意图）")

    # 从未调用过工具：可能模型直接文字回答，或空
    if not tool_traces:
        content = (final_ai.content or "").strip() if final_ai else ""
        if content:
            return f"[信息查询]\n{content}"
        print("[ReAct] 无工具且无文本，兜底单次 mysql_query")
        try:
            fallback_text = mysql_query.invoke(user_msg)
        except Exception as e:
            fallback_text = f"[兜底查询失败] {type(e).__name__}: {e}"
        return _aggregate_react_output([("mysql_query", fallback_text)], None)

    return _aggregate_react_output(tool_traces, final_ai)
