# -*- coding: utf-8 -*-
"""
信息查询 Agent
- prompts.py  : Planner 的 Prompt 模板（代词消解 + 拆分 + 工具分类）
- planner.py  : 问题规划器（快速路径 / LLM Planner / 兜底）
- skill.py    : 工具调度 + 结果聚合（mysql_query / search_knowledge_base）
- node.py     : LangGraph 节点（薄包装）
"""

from agents.information_agent.node import information_agent_node   # noqa: F401

