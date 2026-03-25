# -*- coding: utf-8 -*-
"""
全局常量定义。

涵盖五大联赛代码、各 Agent 标识、Redis 键前缀、网络与 LLM 默认参数等，
供各模块统一引用，避免魔法字符串分散在代码中。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 五大联赛：数据源常用 Div 代码 → 中文简称（与 football-data 等 CSV 约定一致）
# ---------------------------------------------------------------------------
LEAGUE_CODE_TO_ZH: dict[str, str] = {
    "E0": "英超",
    "D1": "德甲",
    "I1": "意甲",
    "SP1": "西甲",
    "F1": "法甲",
}

# 反向映射（展示或检索用）
ZH_TO_LEAGUE_CODE: dict[str, str] = {v: k for k, v in LEAGUE_CODE_TO_ZH.items()}

BIG_FIVE_LEAGUE_CODES: frozenset[str] = frozenset(LEAGUE_CODE_TO_ZH.keys())

# ---------------------------------------------------------------------------
# Agent / 服务名称（与路由、日志、队列消费者标识对齐）
# ---------------------------------------------------------------------------
AGENT_PREDICTED = "predicted_agent"
AGENT_INFORMATION = "information_agent"
AGENT_SUMMARY = "summary_agent"
AGENT_OTHERCHAT = "otherchat_agent"
AGENT_OPENCLAW_BRIDGE = "openclaw_bridge"

DEFAULT_AGENT_NAME = AGENT_INFORMATION

# ---------------------------------------------------------------------------
# Redis 键前缀（环境前缀建议由部署层拼接，此处仅业务命名空间）
# ---------------------------------------------------------------------------
REDIS_PREFIX_SESSION = "football:session:"
REDIS_PREFIX_RATE_LIMIT = "football:ratelimit:"
REDIS_PREFIX_CACHE_MATCH = "football:cache:match:"
REDIS_PREFIX_CACHE_ODDS = "football:cache:odds:"
REDIS_PREFIX_TASK_QUEUE = "football:task:"
REDIS_PREFIX_LLM_CACHE = "football:llm:cache:"

# ---------------------------------------------------------------------------
# 超时与重试（秒；具体业务可覆盖）
# ---------------------------------------------------------------------------
DEFAULT_HTTP_TIMEOUT_SEC = 30.0
DEFAULT_DB_QUERY_TIMEOUT_SEC = 15.0
DEFAULT_REDIS_TIMEOUT_SEC = 5.0
DEFAULT_LLM_REQUEST_TIMEOUT_SEC = 120.0
DEFAULT_OPENCLAW_CALLBACK_TIMEOUT_SEC = 60.0

# ---------------------------------------------------------------------------
# LLM 通用常量（模型名由 llm_select 等模块再细化）
# ---------------------------------------------------------------------------
LLM_DEFAULT_MAX_TOKENS = 4096
LLM_DEFAULT_TEMPERATURE = 0.2
LLM_JSON_RESPONSE_HINT = "请严格输出可解析的 JSON，不要包含 Markdown 代码围栏。"

# 预留：与外部 API 版本或功能开关相关的占位（便于配置化扩展）
API_VERSION_V1 = "v1"
