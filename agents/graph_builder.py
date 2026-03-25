# -*- coding: utf-8 -*-
"""
LangGraph 主图构建入口
- 定义 Agent 节点和边
- 意图识别 → 子 Agent 路由 → 总结输出
- 编译生成可执行的 graph 对象

流转示意：
  [START]
     │
     ▼
  intent_node  ──(BERT 意图识别 + 置信度阈值)──┐
     │                                          │
     ├─ predicted_agent ────────┐                │
     ├─ information_agent ──────┤  (条件路由)     │
     └─ otherchat_agent ────────┤                │
                                │                │
                                ▼                │
                          summary_agent          │
                                │                │
                                ▼                │
                             [END]               │
"""

import os
import sys
import uuid

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ─── 全局状态 ───────────────────────────────────────────────────
from agents.states import AgentState

# ─── 意图识别 ───────────────────────────────────────────────────
from agents.intent_agent.node import intent_route

# ─── 4 个子 Agent 节点 ──────────────────────────────────────────
from agents.predicted_agent.node import predicted_agent_node
from agents.information_agent.node import information_agent_node
from agents.otherchat_agent.node import otherchat_agent_node
from agents.summary_agent.node import summary_agent_node


# ═══════════════════════════════════════════════════════════════
#  1. 意图识别节点包装（适配 AgentState）
# ═══════════════════════════════════════════════════════════════

def intent_node(state: AgentState) -> dict:
    """
    意图识别节点 —— 包装 intent_route() 使其适配 LangGraph AgentState。

    流程：
      1. 检查 dialog_state 对话锁
         - 如果是 "waiting_realtime_confirm"，跳过 BERT，直接路由回 predicted_agent
      2. 否则从 messages 末尾取出用户输入，调用 BERT 进行意图识别
      3. 将识别结果写入 state 的意图识别区字段
    """
    # ── 对话锁检查 ──
    dialog_state = state.get("dialog_state", "normal")
    if dialog_state in ("waiting_realtime_confirm", "waiting_prediction_input"):
        print(f"[intent_node] 检测到对话锁: {dialog_state} → 跳过 BERT，直接路由到 predicted_agent")
        return {
            "current_intent": "predicted_agent",
            "intent_confidence": 1.0,
            "is_fallback": False,
        }

    # ── 正常流程：从 messages 中提取用户最新输入 ──
    user_input = ""
    if state.get("messages"):
        user_input = state["messages"][-1].content

    if not user_input.strip():
        print("[intent_node] 用户输入为空，兜底到 otherchat_agent")
        return {
            "current_intent": "otherchat_agent",
            "intent_confidence": 0.0,
            "is_fallback": True,
            "dialog_state": "normal",
        }

    # ── 调用 BERT 意图识别 ──
    result = intent_route(user_input)

    return {
        "current_intent": result["intent"],
        "intent_confidence": result["confidence"],
        "is_fallback": result["is_fallback"],
        "dialog_state": "normal",
    }


# ═══════════════════════════════════════════════════════════════
#  2. 路由函数（条件边）
# ═══════════════════════════════════════════════════════════════

def route_by_intent(state: AgentState) -> str:
    """
    根据 state 中的 current_intent 值，决定下一步流向哪个子 Agent。

    Returns:
        str: 节点名称，对应 add_node 时注册的名字
    """
    intent = state.get("current_intent", "otherchat_agent")
    print(f"\n[路由] current_intent = '{intent}' → 转发到 {intent} 节点\n")

    # 映射意图标签到节点名
    route_map = {
        "predicted_agent": "predicted_agent",
        "information_agent": "information_agent",
        "otherchat_agent": "otherchat_agent",
    }

    return route_map.get(intent, "otherchat_agent")


# ═══════════════════════════════════════════════════════════════
#  3. 构建主图
# ═══════════════════════════════════════════════════════════════

def build_graph():
    """
    构建 LangGraph 主图，编译并返回可执行的 graph 对象。

    图结构：
      intent_node → (条件路由) → predicted_agent / information_agent / otherchat_agent
                                         ↓                    ↓                    ↓
                                     summary_agent ← ─ ─ ─ ─ ┘
                                         ↓
                                        END
    """
    # 创建状态图
    workflow = StateGraph(AgentState)

    # ── 添加节点 ──
    workflow.add_node("intent_node", intent_node)
    workflow.add_node("predicted_agent", predicted_agent_node)
    workflow.add_node("information_agent", information_agent_node)
    workflow.add_node("otherchat_agent", otherchat_agent_node)
    workflow.add_node("summary_agent", summary_agent_node)

    # ── 设置入口节点 ──
    workflow.set_entry_point("intent_node")

    # ── 条件边：意图识别 → 三个子 Agent ──
    workflow.add_conditional_edges(
        "intent_node",
        route_by_intent,
        {
            "predicted_agent": "predicted_agent",
            "information_agent": "information_agent",
            "otherchat_agent": "otherchat_agent",
        },
    )

    # ── 三个子 Agent → summary_agent ──
    workflow.add_edge("predicted_agent", "summary_agent")
    workflow.add_edge("information_agent", "summary_agent")
    workflow.add_edge("otherchat_agent", "summary_agent")

    # ── summary_agent → END ──
    workflow.add_edge("summary_agent", END)

    # ── 使用 MemorySaver 做本地记忆，编译图 ──
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    print("[✓] LangGraph 主图构建完成！")
    return graph


# ═══════════════════════════════════════════════════════════════
#  4. 终端交互测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    print("=" * 60)
    print("  Football Agent · LangGraph 终端交互测试")
    print("=" * 60)
    print("  输入问题即可测试意图识别 → 子Agent路由 → 总结输出")
    print("  输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    # 构建图
    graph = build_graph()

    # 使用固定 thread_id 保持对话记忆
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n[会话ID] {thread_id}\n")

    while True:
        try:
            user_input = input("👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[退出] 感谢使用，再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\n[退出] 感谢使用，再见！")
            break

        print(f"\n{'─' * 60}")
        print(f"[用户输入] {user_input}")
        print(f"{'─' * 60}\n")

        # 调用图
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        # 提取最终回复（messages 列表最后一条 assistant 消息）
        final_msg = result["messages"][-1]
        if hasattr(final_msg, "content"):
            reply = final_msg.content
        else:
            reply = str(final_msg)

        print(f"\n{'─' * 60}")
        print(f"🤖 助手: {reply}")
        print(f"{'─' * 60}\n")
