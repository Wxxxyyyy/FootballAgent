# -*- coding: utf-8 -*-
"""
向量知识检索工具包（RAG）
- config.py     : 本地路径配置及防线阈值
- retriever.py  : 连接 ChromaDB，向量检索 + 两道防线（距离阈值 / Top-K）
- tool_entry.py : 暴露给 LangChain/LangGraph 的 @tool 统一入口
"""

from agents.tools.vector_tools.tool_entry import search_knowledge_base   # noqa: F401

