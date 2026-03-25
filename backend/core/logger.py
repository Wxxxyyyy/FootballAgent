# -*- coding: utf-8 -*-
"""
结构化日志（loguru）
- 控制台与文件双通道
- JSON 序列化便于 ELK / Loki 采集
- 请求级 request_id 贯穿访问日志
"""
import contextvars
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# 供中间件注入当前请求的追踪 ID（异步安全）
REQUEST_ID_CTX: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)

_logger_configured = False


def _inject_request_id(record: dict) -> bool:
    """将 request_id 写入 extra，便于 JSON 与文本格式共用。"""
    rid = REQUEST_ID_CTX.get()
    record["extra"]["request_id"] = rid if rid else "-"
    return True


def setup_logger(
    *,
    level: str = "INFO",
    log_dir: Optional[Path] = None,
) -> None:
    """
    初始化全局日志：控制台 + 滚动文件。
    重复调用不会重复挂载 handler（幂等）。
    """
    global _logger_configured
    if _logger_configured:
        return

    base = log_dir or (Path(__file__).resolve().parent.parent / "logs")
    base.mkdir(parents=True, exist_ok=True)
    log_file = base / "app.log"

    logger.remove()

    # 控制台：JSON 一行一条，便于开发机与容器采集
    logger.add(
        sys.stdout,
        level=level,
        serialize=True,
        filter=_inject_request_id,
        enqueue=True,
    )

    # 文件：按大小切割、按天保留
    logger.add(
        str(log_file),
        level=level,
        encoding="utf-8",
        serialize=True,
        filter=_inject_request_id,
        rotation="10 MB",
        retention="7 days",
        enqueue=True,
    )

    _logger_configured = True


def get_logger():
    """获取全局 loguru logger（业务代码统一从此处取）。"""
    return logger
