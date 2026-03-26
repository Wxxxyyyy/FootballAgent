# -*- coding: utf-8 -*-
"""
主图状态定义（AgentState）
- 对话记忆区：LangGraph 自动管理的消息历史
- 意图识别区：由 BERT intent_node 填充
- 流程控制区：对话锁，用于多轮交互跳过重复意图识别
- 数据交接区：子 Agent 输出 → 总结 Agent 消费
"""

from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

# add_messages 是 LangGraph 的核心机制：新消息自动"追加"到列表末尾，而非覆盖
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    全局图状态定义 (The State of the Graph)
    在整个 Multi-Agent 路由和执行过程中传递的数据结构。
    """

    # ═══════════════════════════════════════════════════════════
    # 1. 对话记忆区（LangGraph 自动管理历史记录）
    # ═══════════════════════════════════════════════════════════
    # Annotated[..., add_messages] 意味着当有新消息时，会自动 append 到列表末尾
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # ═══════════════════════════════════════════════════════════
    # 2. 意图识别区（由 BERT intent_node 填充）
    # ═══════════════════════════════════════════════════════════
    current_intent: str        # 例如: "predicted_agent"
    intent_confidence: float   # 例如: 0.95
    is_fallback: bool          # 是否因为置信度过低触发了闲聊兜底

    # ═══════════════════════════════════════════════════════════
    # 3. 流程控制区 / 对话锁（极其关键！）
    # ═══════════════════════════════════════════════════════════
    # 默认值为 "normal"。
    # 当预测节点问出"是否需要实时预测？"时，改为 "waiting_realtime_confirm"。
    # 下一次用户说话时，入口路由查到这个锁，直接扔回给预测节点，跳过 BERT。
    dialog_state: str

    # ═══════════════════════════════════════════════════════════
    # 4. 数据交接区（子 Agent 给总结 Agent 的传球）
    # ═══════════════════════════════════════════════════════════
    raw_agent_response: str

    # ═══════════════════════════════════════════════════════════
    # 5. 记忆管理区（Memory Manager 节点读写）
    # ═══════════════════════════════════════════════════════════
    # Compaction 时 Flush 提取的关键信息（实体、事实、偏好等）
    memory_metadata: dict
