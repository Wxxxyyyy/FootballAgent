# -*- coding: utf-8 -*-
"""
信息查询 Agent · Planner（问题规划器）
══════════════════════════════════════
职责：
  ① 代词消解 —— "它/他们/这支球队" → 从对话历史中找到具体实体
  ② 多问题拆分 —— 复合问题 → 独立子问题列表
  ③ 工具分类   —— 每个子问题标注 "mysql" 或 "vector"

快速路径：
  - 若用户输入 **无代词 + 无多问题信号**，则跳过 LLM Planner，
    直接使用关键词规则分类工具，节省一次 LLM 调用。

兜底机制：
  - LLM 返回的 JSON 解析失败时，降级为单问题（原始输入），
    工具默认走 mysql。
"""

import re
import json
from typing import Sequence

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from agents.information_agent.prompts import get_planner_prompt
from common.llm_select import llm_call, LLM_MODEL_QWEN_SIMPLE_NAME


# ═══════════════════════════════════════════════════════════════
#  常量 & 正则
# ═══════════════════════════════════════════════════════════════

# 对话历史提取条数上限（不含当前消息）
HISTORY_LIMIT = 6

# ── 代词检测 ──
_PRONOUN_RE = re.compile(
    r'它的?|他们的?|她们的?|'
    r'这支球队|那支球队|该队|这个队|那个队|'
    r'这支队伍|那支队伍|上述球队|前面那个|前面那支'
)

# ── 多问题信号检测 ──
_MULTI_Q_RE = re.compile(
    r'还有|另外|以及|同时|顺便|再帮我|再查一?下|再问|并且|除此之外'
)

# ── 工具分类关键词（用于快速路径） ──
_VECTOR_KEYWORDS_RE = re.compile(
    r'历史底蕴|背景介绍|背景|简介|介绍一下|别名|绰号|'
    r'战术风格|球队文化|成立|创始|球场信息|球衣|队徽|'
    r'传奇球星|荣誉殿堂|故事|起源|昵称|什么来头|什么背景|'
    r'是一支怎样的球队|球队历史'
)


# ═══════════════════════════════════════════════════════════════
#  内部工具函数
# ═══════════════════════════════════════════════════════════════

def _needs_planner(user_msg: str) -> bool:
    """
    判断用户输入是否需要调用 LLM Planner。

    触发条件（满足任一即需要 Planner）：
      1. 包含代词（需要消解）
      2. 包含多问题信号词
      3. 包含 2 个及以上问号（可能有多个独立问题）

    Returns:
        True  → 需要调 LLM Planner
        False → 跳过 Planner，走关键词快速路径
    """
    if _PRONOUN_RE.search(user_msg):
        return True
    if _MULTI_Q_RE.search(user_msg):
        return True
    if user_msg.count('？') >= 2 or user_msg.count('?') >= 2:
        return True
    return False


def _classify_tool_by_keywords(question: str) -> str:
    """
    基于关键词的工具分类（无需 LLM，用于快速路径）。

    策略：
      - 若命中 vector 关键词 → "vector"
      - 其余一律 → "mysql"（信息查询的主力数据源）

    Returns:
        "mysql" 或 "vector"
    """
    if _VECTOR_KEYWORDS_RE.search(question):
        return "vector"
    return "mysql"


def _extract_history_text(messages: Sequence[BaseMessage], user_msg: str = "") -> str:
    """
    从 state.messages 中提取近期对话历史 + 条件触发长期记忆检索。

    策略：
      1. 取最近 HISTORY_LIMIT 条作为近期上下文
      2. 如果用户输入包含回指/代词且窗口内消解失败，检索 ChromaDB 长期记忆
      3. 返回: [历史记忆]\n[近期对话]

    Returns:
        str: 格式化的历史上下文文本
    """
    # 排除最后一条（当前用户消息）
    history = messages[:-1] if len(messages) > 1 else []
    if not history:
        return ""

    recent = history[-HISTORY_LIMIT:]

    parts = []

    # ── 条件触发长期记忆检索 ──
    if user_msg:
        try:
            from agents.memory_manager.retriever import maybe_retrieve_memory
            memory_context = maybe_retrieve_memory(
                user_msg=user_msg,
                recent_messages=recent,
            )
            if memory_context:
                parts.append(memory_context)
        except Exception as e:
            print(f"[Planner] 长期记忆检索异常（不影响主流程）: {e}")

    # ── 近期对话历史 ──
    lines = []
    for msg in recent:
        msg_type = getattr(msg, 'type', '')
        if msg_type == 'human':
            lines.append(f"用户: {msg.content}")
        elif msg_type == 'ai':
            lines.append(f"助手: {msg.content}")

    if lines:
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def _parse_planner_response(content: str) -> list[dict] | None:
    """
    尝试从 LLM 的响应文本中解析出 JSON 数组。

    处理策略：
      1. 去掉 <think>...</think> 思考标签
      2. 去掉 markdown ```json ``` 代码块
      3. 尝试直接 json.loads
      4. 若失败，尝试正则提取 [...] 部分再解析
      5. 验证每个元素包含 question + tool 字段

    Returns:
        list[dict] | None: 解析成功返回列表，失败返回 None（触发兜底）
    """
    text = content.strip()

    # 去掉 <think> 标签（部分模型会输出思考过程）
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # 去掉 markdown 代码块
    if '```' in text:
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    def _validate_list(raw) -> list[dict] | None:
        """验证并过滤合法的子问题列表"""
        if not isinstance(raw, list) or len(raw) == 0:
            return None
        validated = []
        for item in raw:
            if isinstance(item, dict) and "question" in item and "tool" in item:
                tool = item["tool"].lower().strip()
                if tool not in ("mysql", "vector"):
                    tool = "mysql"  # 非法工具名兜底
                validated.append({
                    "question": str(item["question"]).strip(),
                    "tool": tool,
                })
        return validated if validated else None

    # 尝试直接解析
    try:
        result = json.loads(text)
        validated = _validate_list(result)
        if validated:
            return validated
    except (json.JSONDecodeError, TypeError):
        pass

    # 尝试正则提取 JSON 数组部分
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            validated = _validate_list(result)
            if validated:
                return validated
        except (json.JSONDecodeError, TypeError):
            pass

    return None  # 解析失败 → 调用方触发兜底


# ═══════════════════════════════════════════════════════════════
#  对外接口
# ═══════════════════════════════════════════════════════════════

def plan(user_msg: str, messages: Sequence[BaseMessage]) -> list[dict]:
    """
    规划用户问题 —— 信息查询 Agent 的 Planner 主入口。

    流程：
      1. 判断是否需要 LLM Planner（代词 / 多问题）
      2a. 不需要 → 关键词快速路径分类，返回单问题列表
      2b. 需要   → 调 LLM Planner，解析 JSON 响应
      3. LLM 解析失败 → 兜底为单问题 + 关键词分类

    Args:
        user_msg:  用户最新输入（已从 state.messages[-1] 提取）
        messages:  state 中的完整对话历史（含当前消息）

    Returns:
        list[dict]: 子问题列表，每个 dict 包含:
            - "question": 消解后的完整子问题文本
            - "tool":     "mysql" 或 "vector"
    """
    # ── 快速路径：单问题 + 无代词 → 跳过 LLM ──
    if not _needs_planner(user_msg):
        tool = _classify_tool_by_keywords(user_msg)
        print(f"[Planner] 快速路径: 单问题，工具={tool}")
        return [{"question": user_msg, "tool": tool}]

    # ── 需要 LLM Planner ──
    print("[Planner] 检测到代词/多问题信号，启动 LLM Planner...")

    # 提取对话历史（含条件触发的长期记忆检索）
    history_text = _extract_history_text(messages, user_msg=user_msg)

    # 组装消息
    system_prompt = get_planner_prompt(history_text)
    llm_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg),
    ]

    # 调用轻量模型
    try:
        response = llm_call(llm_messages, model=LLM_MODEL_QWEN_SIMPLE_NAME)
        raw_content = response.content
        print(f"[Planner] LLM 响应: {raw_content[:200]}...")
    except Exception as e:
        print(f"[Planner] ❌ LLM 调用失败: {e}，启用兜底")
        tool = _classify_tool_by_keywords(user_msg)
        return [{"question": user_msg, "tool": tool}]

    # 解析 JSON 响应
    parsed = _parse_planner_response(raw_content)
    if parsed is None:
        print("[Planner] ⚠️ JSON 解析失败，启用兜底: 原始输入作为单问题")
        tool = _classify_tool_by_keywords(user_msg)
        return [{"question": user_msg, "tool": tool}]

    print(f"[Planner] 规划完成: {len(parsed)} 个子问题")
    for i, item in enumerate(parsed, 1):
        print(f"  [{i}] {item['tool']}: {item['question']}")

    return parsed

