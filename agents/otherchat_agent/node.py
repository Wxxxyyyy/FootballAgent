# -*- coding: utf-8 -*-
"""
闲聊 Agent · LangGraph 节点
- 职责单一：读 state → 调 skill → 写 state
- 所有操作逻辑（组装对话历史 + 调 LLM）都在 skill.py 中
"""

from agents.states import AgentState
from agents.otherchat_agent.skill import chat


def otherchat_agent_node(state: AgentState) -> dict:
    """
    闲聊 Agent 节点（LangGraph Node）。

    职责：
      1. 从 state 中读取 messages 和 is_fallback
      2. 调用 skill.chat() 执行完整的闲聊操作链
      3. 将回复写入 raw_agent_response，供后续 summary_agent 消费
    """
    is_fallback = state.get("is_fallback", False)
    messages = state.get("messages", [])
    user_msg = messages[-1].content if messages else ""

    status_tag = "兜底模式" if is_fallback else "正常闲聊"
    print("=" * 50)
    print(f"[otherchat_node] 进入闲聊 Agent 节点 ({status_tag})")
    print(f"  用户输入: '{user_msg}'")
    print("=" * 50)

    # 调用 skill 执行完整操作链
    result = chat(messages=messages, is_fallback=is_fallback)

    # 写入 state
    return {
        "raw_agent_response": result["reply"],
    }
