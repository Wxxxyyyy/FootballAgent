# -*- coding: utf-8 -*-
"""
Neo4j 图谱查询工具包
- templates/  : 高频查询的静态 Cypher 模板
- security.py : 四道防线（读写隔离 / 方向纠正 / 语法验证 / 值映射校验）
- text2cypher.py : LLM 生成 Cypher + 纠错重试回环
- tool_entry.py  : 暴露给 LangChain/LangGraph 的 @tool 统一入口
"""

from agents.tools.neo4j_tools.tool_entry import neo4j_query   # noqa: F401

