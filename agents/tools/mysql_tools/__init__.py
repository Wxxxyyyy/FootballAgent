# -*- coding: utf-8 -*-
"""
MySQL 比赛数据查询工具包
- templates/  : 高频查询的静态 SQL 模板（参数化防注入）
- security.py : 四道防线（读写隔离 / 幻觉校验 / 语法验证 / 强制 LIMIT）
- text2sql.py : LLM 生成 SQL + 纠错重试回环
- tool_entry.py  : 暴露给 LangChain/LangGraph 的 @tool 统一入口
"""

from agents.tools.mysql_tools.tool_entry import mysql_query   # noqa: F401

