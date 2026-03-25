# -*- coding: utf-8 -*-
"""
总结 Agent · 技能模块（Skill）
- 真正的操作逻辑：组装 prompt → 调用 LLM 润色 → 执行安全检查
- 对外暴露 summarize() 作为唯一入口，node.py 只需调用它即可

使用方式：
    from agents.summary_agent.skill import summarize
    result = summarize(raw_agent_response="...", intent="otherchat_agent")
    # result["final_text"]     : 最终安全文本
    # result["safety_status"]  : "pass" | "modified" | "blocked"
    # result["safety_warnings"]: 安全检查触发的警告列表
"""

from langchain_core.messages import SystemMessage, HumanMessage

from agents.summary_agent.prompts import get_summary_prompt
from agents.summary_agent.safety_check import safety_check
from common.llm_select import llm_call, LLM_MODEL_KIMI_NAME


def summarize(raw_agent_response: str, intent: str) -> dict:
    """
    总结技能：对子 Agent 的原始输出进行润色 + 安全审核，返回最终安全文本。

    操作流程：
      1. 通过 prompts.py 组装 System Prompt（根据意图动态选择模板）
      2. 根据意图选择合适的模型进行话术润色
         - information_agent / predicted_agent：数据量大、需要整理排版
           → 使用远程模型（LLM_MODEL_QWEN_SIMPLE_NAME），处理能力更强
         - otherchat_agent 等简单任务
           → 使用本地 Ollama 7b，省时省钱
      3. 执行 safety_check 安全检查（敏感词过滤 / 赌博免责 / 预测免责）
      4. 返回最终结果

    Args:
        raw_agent_response: 子 Agent 的原始输出文本
        intent:             当前意图标签（如 "otherchat_agent"）

    Returns:
        dict: {
            "final_text":      最终安全文本（可直接返回给用户）,
            "polished_text":   LLM 润色后的中间文本（安全检查前）,
            "safety_status":   "pass" | "modified" | "blocked",
            "safety_warnings": 安全检查触发的警告列表,
        }
    """
    # ── 步骤1: 组装 System Prompt ──
    system_prompt = get_summary_prompt(current_intent=intent, raw_agent_response=raw_agent_response)

    # ── 步骤2: 根据意图选择模型进行话术润色 ──
    llm_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="请根据上述原始数据，生成最终回复。"),
    ]

    # 数据密集型意图（信息查询、预测）用远程模型，处理能力更强
    # 闲聊等简单任务用本地 Ollama，省资源
    _HEAVY_INTENTS = {"information_agent", "predicted_agent"}
    use_remote = intent in _HEAVY_INTENTS

    if use_remote:
        print(f"[summary_skill] 意图={intent}，数据量较大，使用远程模型 {LLM_MODEL_KIMI_NAME} 润色...")
    else:
        print(f"[summary_skill] 意图={intent}，使用本地 Ollama 进行话术润色...")

    try:
        if use_remote:
            response = llm_call(llm_messages, model=LLM_MODEL_KIMI_NAME)
        else:
            response = llm_call(llm_messages, force_fallback=True)
        polished_text = response.content
        print(f"[summary_skill] 润色完成，长度: {len(polished_text)} 字符")
    except Exception as e:
        # 模型不可用时，直接使用原始数据兜底
        print(f"[summary_skill] ❌ LLM 润色失败: {e}，使用原始数据兜底")
        polished_text = raw_agent_response if raw_agent_response else "抱歉，系统暂时无法生成回复，请稍后再试~"

    # ── 步骤3: 执行安全检查 ──
    print("[summary_skill] 正在执行安全检查...")
    check_result = safety_check(text=polished_text, intent=intent)

    final_text = check_result["text"]
    safety_status = check_result["status"]
    safety_warnings = check_result["warnings"]

    if safety_warnings:
        print(f"[summary_skill] 安全检查警告: {safety_warnings}")
    print(f"[summary_skill] 安全检查状态: {safety_status}")
    print(f"[summary_skill] 最终输出长度: {len(final_text)} 字符")

    return {
        "final_text": final_text,
        "polished_text": polished_text,
        "safety_status": safety_status,
        "safety_warnings": safety_warnings,
    }
