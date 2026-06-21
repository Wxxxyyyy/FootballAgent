# -*- coding: utf-8 -*-
"""
信息查询 Agent
- prompts.py     : Planner / ReAct 的 Prompt 模板
- planner.py     : 历史抽取与旧版 plan()（供复用，主路径已改为 ReAct）
- react_runner.py: 节点内 ReAct（局部 messages + bind_tools 多轮）
- skill.py       : query() → run_react → 聚合 raw 文本
- node.py        : LangGraph 节点（薄包装）
"""

from agents.information_agent.node import information_agent_node   # noqa: F401
