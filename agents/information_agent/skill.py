# -*- coding: utf-8 -*-
"""
信息查询 Agent · 技能模块（Skill）
════════════════════════════════════
职责：
  在 **局部消息列表** 上运行 **ReAct**（bind_tools + 多轮 ToolMessage），
  在 mysql_query（Text2SQL）与 search_knowledge_base（RAG）间自主选择与补查；
  将聚合结果写入 raw_agent_response，**不**把中间轨迹写入全局 state.messages。

对外暴露 query() 作为唯一入口，node.py 只需调用它即可。
"""

from typing import Sequence

from langchain_core.messages import BaseMessage

from agents.information_agent.react_runner import run_react

# LLM / 工具全部不可用时的兜底回复
_FALLBACK_REPLY = (
    "抱歉，信息查询过程中遇到了技术问题，暂时无法返回数据。"
    "请稍后再试，或者换一种方式描述您的问题。"
)


def query(user_msg: str, messages: Sequence[BaseMessage]) -> dict:
    """
    信息查询技能主入口 —— ReAct 多轮工具循环 + 聚合。

    Args:
        user_msg:  用户最新输入文本
        messages:  state 中的完整对话历史（含当前消息），仅用于构造历史上下文

    Returns:
        dict: {"response": 供 summary_agent 使用的聚合文本}
    """
    print("[Skill] 使用 ReAct（局部 messages）查询…")

    try:
        aggregated = run_react(user_msg=user_msg, messages=messages)
    except Exception as e:
        print(f"[Skill] ❌ ReAct 异常: {e}")
        return {"response": _FALLBACK_REPLY}

    if not aggregated.strip():
        return {"response": _FALLBACK_REPLY}

    return {"response": aggregated}
