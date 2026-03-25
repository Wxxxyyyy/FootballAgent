# -*- coding: utf-8 -*-
"""
闲聊 Agent · 技能模块（Skill）
- 真正的操作逻辑：组装 prompt + 对话历史 → 调用轻量 LLM 生成回复
- 对外暴露 chat() 作为唯一入口，node.py 只需调用它即可

使用方式：
    from agents.otherchat_agent.skill import chat
    result = chat(messages=state["messages"], is_fallback=False)
    # result["reply"]: LLM 生成的闲聊回复文本
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from typing import Sequence

from agents.otherchat_agent.prompts import get_chat_system_prompt
from common.llm_select import llm_call, LLM_MODEL_QWEN_SIMPLE_NAME

# 对话历史最大保留条数（避免 token 过多）
HISTORY_LIMIT = 6

# LLM 全部不可用时的兜底回复
_FALLBACK_REPLY = (
    "抱歉，我现在遇到了一些技术问题，暂时无法回复。"
    "请稍后再试，或者直接告诉我你想查询哪场比赛的信息吧！"
)


def _build_chat_messages(
    messages: Sequence[BaseMessage],
    is_fallback: bool,
) -> list:
    """
    组装发送给 LLM 的消息列表。

    操作步骤：
      1. 通过 prompts.py 获取 System Prompt（根据 is_fallback 选择模板）
      2. 从 state messages 中提取最近几轮对话历史
      3. 拼接为 [SystemMessage, ...历史消息] 的标准格式
    
    Args:
        messages:    state 中的完整对话历史
        is_fallback: 是否为低置信度兜底
        
    Returns:
        list: 可直接传入 LLM 的消息列表
    """
    # 获取系统提示词
    system_prompt = get_chat_system_prompt(is_fallback)
    chat_messages = [SystemMessage(content=system_prompt)]

    # 提取最近的对话历史
    recent = messages[-HISTORY_LIMIT:] if len(messages) > HISTORY_LIMIT else messages

    for msg in recent:
        if hasattr(msg, "type"):
            if msg.type == "human":
                chat_messages.append(HumanMessage(content=msg.content))
            elif msg.type == "ai":
                chat_messages.append(AIMessage(content=msg.content))
        # 跳过 system 类型等其他消息

    return chat_messages


def chat(messages: Sequence[BaseMessage], is_fallback: bool) -> dict:
    """
    闲聊技能：组装对话上下文 + 调用轻量 LLM 生成闲聊回复。

    操作流程：
      1. 组装对话消息列表（System Prompt + 历史对话）
      2. 调用轻量模型生成回复（失败自动降级到本地 Ollama）
      3. 返回回复文本

    Args:
        messages:    state 中的完整对话历史
        is_fallback: 是否为低置信度兜底

    Returns:
        dict: {
            "reply": LLM 生成的闲聊回复文本,
        }
    """
    # ── 步骤1: 组装消息列表 ──
    chat_messages = _build_chat_messages(messages, is_fallback)

    # ── 步骤2: 调用轻量模型 ──
    print("[otherchat_skill] 正在调用轻量模型生成闲聊回复...")
    try:
        response = llm_call(chat_messages, model=LLM_MODEL_QWEN_SIMPLE_NAME)
        reply = response.content
        print(f"[otherchat_skill] 回复成功，长度: {len(reply)} 字符")
    except Exception as e:
        # 极端情况：所有模型全部不可用
        print(f"[otherchat_skill] ❌ LLM 调用失败: {e}")
        reply = _FALLBACK_REPLY

    return {
        "reply": reply,
    }
