# -*- coding: utf-8 -*-
"""
Langfuse LLM 可观测性集成（v3 SDK，自托管/云均可）

职责:
  - langfuse_enabled()      —— 是否已配置密钥
  - get_client()            —— Langfuse 单例客户端（未配置时返回 None）
  - get_langchain_handler() —— LangChain/LangGraph 回调处理器，
                               挂到 graph.invoke 的 config.callbacks 即可自动捕获
                               全图 trace（每个节点/LLM 调用的输入输出、Token、延迟）

两条接入路径:
  1. LangGraph 主链路（backend/api/chat.py、graph_builder.py CLI）
     → LangChain CallbackHandler，图内所有 LLM 调用自动嵌套进一条 trace
  2. 预测链路（llm_predictor.py，OpenAI SDK 直连，不经 LangChain）
     → langfuse.openai 的 drop-in 包装，自动上报 token/延迟/模型

设计原则: 纯增强。未配置密钥或未安装 SDK 时全部返回 None/False，业务零感知。

环境变量:
  LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST（默认 http://localhost:3000）
  自托管启动: docker compose --profile langfuse up -d
"""

import os

_client = None
_checked = False


def langfuse_enabled() -> bool:
    """是否配置了 Langfuse 密钥（有即视为启用）"""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_client():
    """获取 Langfuse 单例客户端；未配置/未安装 SDK 时返回 None"""
    global _client, _checked
    if _client is not None:
        return _client
    if _checked:
        return None
    _checked = True
    if not langfuse_enabled():
        return None
    try:
        from langfuse import get_client as _get
        _client = _get()  # 从环境变量读取密钥与 host
        return _client
    except Exception:  # noqa: BLE001 — 观测组件绝不阻断业务
        return None


def get_langchain_handler():
    """获取 LangChain 回调处理器（未配置时返回 None）

    用法:
        handler = get_langchain_handler()
        config = {"configurable": {"thread_id": tid}}
        if handler:
            config["callbacks"] = [handler]
        graph.invoke(input, config=config)
    """
    if not langfuse_enabled():
        return None
    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except Exception:  # noqa: BLE001
        return None
