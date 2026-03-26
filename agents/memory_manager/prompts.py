# -*- coding: utf-8 -*-
"""
记忆管理 · Prompt 模板
- Memory Flush: 压缩前提取关键信息，防止摘要丢失细节
- Compaction:    将旧消息压缩为结构化摘要
"""


def get_flush_prompt(messages_text: str) -> str:
    """
    Memory Flush Prompt —— 压缩前先提取关键标识符和事实。

    Args:
        messages_text: 即将被压缩的消息文本（格式化后的多轮对话）

    Returns:
        System Prompt 字符串
    """
    return f"""你是一个对话信息提取器。请从以下对话片段中提取关键信息，输出严格的 JSON 格式。

## 对话内容
{messages_text}

## 输出要求（严格 JSON，不要任何解释）
{{
  "entities": ["提到的球队名、球员名、联赛名等实体"],
  "key_facts": ["重要的数据结论，如'阿森纳近5场3胜1平1负'、'利物浦主场胜率72%'"],
  "user_preferences": ["用户表露的偏好，如'关注英超'、'喜欢阿森纳'、'偏好预测类问题'"],
  "decisions": ["用户做出的确认或决策，如'用户选择查询阿森纳'"]
}}

只输出 JSON，不要其他内容。"""


def get_compaction_prompt(messages_text: str, flush_result: str) -> str:
    """
    Compaction Prompt —— 将旧消息压缩为摘要，严格保留 Flush 提取的关键信息。

    Args:
        messages_text: 即将被压缩的消息文本
        flush_result:  Memory Flush 提取的结构化关键信息（JSON 字符串）

    Returns:
        System Prompt 字符串
    """
    return f"""你是一个对话摘要生成器。请将以下对话片段压缩为一段简洁的摘要。

## 对话内容
{messages_text}

## 必须保留的关键信息（来自预提取，不可遗漏）
{flush_result}

## 输出要求
- 200 字以内的中文摘要
- 必须包含上述关键信息中的所有实体名称、数字和结论
- 按时间顺序组织，突出用户关注点
- 不要添加对话中没有的信息
- 直接输出摘要文本，不要标题或格式标记"""
