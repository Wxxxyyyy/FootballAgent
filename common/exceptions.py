# -*- coding: utf-8 -*-
"""
自定义异常层级。

所有业务异常均携带 ``message``（人类可读）与 ``code``（机器可读/监控聚合），
便于统一日志格式与 HTTP/API 错误码映射。
"""

from __future__ import annotations


class FootballAgentError(Exception):
    """项目内异常的基类。"""

    def __init__(self, message: str, code: str = "FOOTBALL_AGENT_ERROR") -> None:
        self.message = message
        self.code = code
        super().__init__(message)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.code!r}, message={self.message!r})"


class LLMError(FootballAgentError):
    """大模型调用、解析或配额相关错误。"""

    def __init__(self, message: str, code: str = "LLM_ERROR") -> None:
        super().__init__(message, code)


class DatabaseError(FootballAgentError):
    """关系型数据库 / 连接池 / 查询执行错误。"""

    def __init__(self, message: str, code: str = "DATABASE_ERROR") -> None:
        super().__init__(message, code)


class OpenClawError(FootballAgentError):
    """与 OpenClaw 或外部编排回调相关的错误。"""

    def __init__(self, message: str, code: str = "OPENCLAW_ERROR") -> None:
        super().__init__(message, code)


class SecurityError(FootballAgentError):
    """鉴权、签名、敏感配置或越权访问。"""

    def __init__(self, message: str, code: str = "SECURITY_ERROR") -> None:
        super().__init__(message, code)


class ParseError(FootballAgentError):
    """结构化数据解析失败（JSON/XML/比分字符串等）。"""

    def __init__(self, message: str, code: str = "PARSE_ERROR") -> None:
        super().__init__(message, code)


class ValidationError(FootballAgentError):
    """输入校验失败（联赛代码、日期范围等）。"""

    def __init__(self, message: str, code: str = "VALIDATION_ERROR") -> None:
        super().__init__(message, code)
