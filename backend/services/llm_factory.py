# -*- coding: utf-8 -*-
"""
LLM 客户端工厂
- OpenAI 兼容 API（含 DeepSeek 等）
- 本地 Ollama
- 按模型名缓存已创建的客户端实例
"""
from __future__ import annotations

import os
from threading import Lock
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

Provider = Literal["openai_compatible", "ollama"]


class LLMFactory:
    """
    根据环境变量与模型名创建聊天模型。
    - LLM_PROVIDER: openai_compatible | ollama
    - OPENAI_API_KEY / OPENAI_BASE_URL（兼容端）
    - OLLAMA_BASE_URL（默认 http://localhost:11434）
    """

    def __init__(self) -> None:
        self._cache: dict[str, BaseChatModel] = {}
        self._lock = Lock()

    def _provider(self) -> Provider:
        p = os.getenv("LLM_PROVIDER", "openai_compatible").lower()
        if p == "ollama":
            return "ollama"
        return "openai_compatible"

    def create_client(self, model_name: str) -> BaseChatModel:
        """返回缓存的模型实例；同一 model_name 复用同一对象。"""
        with self._lock:
            if model_name in self._cache:
                return self._cache[model_name]
            client = self._build(model_name)
            self._cache[model_name] = client
            return client

    def _build(self, model_name: str) -> BaseChatModel:
        if self._provider() == "ollama":
            base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return ChatOllama(model=model_name, base_url=base)

        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return ChatOpenAI(
            model=model_name,
            api_key=api_key or None,
            base_url=base_url,
        )

    def clear_cache(self) -> None:
        """测试或热切换配置时可清空缓存。"""
        with self._lock:
            self._cache.clear()
