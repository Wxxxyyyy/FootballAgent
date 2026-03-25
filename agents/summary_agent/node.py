# -*- coding: utf-8 -*-
"""
总结 Agent · LangGraph 节点
- 职责单一：读 state → 调 skill → 写 state
- 所有操作逻辑（LLM 润色 + 安全检查）都在 skill.py 中
- 对话锁状态下（等待用户补充输入），跳过 LLM 润色，仅做安全检查
"""

from agents.states import AgentState
from agents.summary_agent.skill import summarize
from agents.summary_agent.safety_check import safety_check


# 处于这些对话锁状态时，raw_agent_response 是简单的提示信息，
# 不需要消耗远程 LLM 进行润色，直接走安全检查即可
_PASSTHROUGH_DIALOG_STATES = {
    "waiting_prediction_input",
    "waiting_realtime_confirm",
}


def summary_agent_node(state: AgentState) -> dict:
    """
    总结 Agent 节点（LangGraph Node）。

    职责：
      1. 从 state 中读取 raw_agent_response 和 current_intent
      2. 若处于对话锁状态 → 跳过 LLM 润色，仅做安全检查（节省开销）
      3. 否则调用 skill.summarize() 执行完整的润色 + 安全检查流水线
      4. 将最终安全文本作为 assistant 消息写入 state
    """
    raw = state.get("raw_agent_response", "")
    intent = state.get("current_intent", "")
    dialog_state = state.get("dialog_state", "normal")

    print("=" * 50)
    print(f"[summary_node] 进入总结 Agent 节点")
    print(f"  当前意图: {intent}")
    print(f"  对话状态: {dialog_state}")
    print(f"  原始数据长度: {len(raw)} 字符")
    print("=" * 50)

    # ── 对话锁状态：轻量处理，跳过 LLM 润色 ──
    if dialog_state in _PASSTHROUGH_DIALOG_STATES:
        print(f"[summary_node] 对话锁 '{dialog_state}' → 跳过 LLM 润色，仅做安全检查")
        check_result = safety_check(text=raw, intent=intent)
        return {
            "messages": [("assistant", check_result["text"])],
        }

    # ── 正常流程：完整的 LLM 润色 + 安全检查 ──
    result = summarize(raw_agent_response=raw, intent=intent)

    return {
        "messages": [("assistant", result["final_text"])],
    }
