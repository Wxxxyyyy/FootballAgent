# -*- coding: utf-8 -*-
"""
信息查询 Agent · LangGraph 节点
- 职责单一：读 state → 调 skill → 写 state
- 查询逻辑在 skill.py：节点内 **ReAct**（局部 messages + mysql / RAG 工具循环），
  不污染全局 messages
"""

from agents.states import AgentState
from agents.information_agent.skill import query


def information_agent_node(state: AgentState) -> dict:
    """
    信息查询 Agent 节点（LangGraph Node）。

    职责：
      1. 从 state 中读取 messages（对话历史 + 当前用户输入）
      2. 调用 skill.query()：ReAct 多轮工具调用 → 聚合 → raw_agent_response
      3. 将聚合后的查询结果写入 raw_agent_response，供 summary_agent 消费
    """
    messages = state.get("messages", [])
    user_msg = messages[-1].content if messages else ""

    print("=" * 50)
    print("[information_agent] 进入信息查询 Agent 节点")
    print(f"  用户输入: '{user_msg}'")
    print("=" * 50)

    # 调用 skill 执行完整操作链
    result = query(user_msg=user_msg, messages=messages)

    # 写入 state
    return {
        "raw_agent_response": result["response"],
    }
