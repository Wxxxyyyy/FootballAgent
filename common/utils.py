# -*- coding: utf-8 -*-
"""
通用工具函数：重试、字符串与比分解析、JSON 安全加载、项目根路径等。

仅使用标准库，便于在任意子模块中无额外依赖引用。
"""

from __future__ import annotations

import json
import re
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def get_project_root() -> Path:
    """返回仓库根目录（``common`` 的上一级）。"""
    return Path(__file__).resolve().parent.parent


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay_sec: float = 0.5,
    max_delay_sec: float = 30.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    指数退避重试装饰器；第 n 次等待 ``min(base * 2**(n-1), max_delay)`` 秒。
    仅在捕获到 ``exceptions`` 元组中的类型时重试。
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if max_attempts < 1:
            raise ValueError("max_attempts 必须 >= 1")

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    if attempt >= max_attempts:
                        raise
                    delay = min(base_delay_sec * (2 ** (attempt - 1)), max_delay_sec)
                    time.sleep(delay)

        return wrapper

    return decorator


def normalize_team_name(name: str) -> str:
    """球队名标准化：去首尾空白、合并连续空白、统一常见缩写前后空格。"""
    s = name.strip()
    s = re.sub(r"\s+", " ", s)
    return s


_SCORE_RE = re.compile(r"^\s*(\d+)\s*[:：]\s*(\d+)\s*$")


def parse_score(text: str) -> tuple[int, int]:
    """
    从 ``\"2:1\"`` 或 ``\"2：1\"`` 解析为主客队进球数。
    无法解析时抛出 ``ValueError``。
    """
    m = _SCORE_RE.match(text)
    if not m:
        raise ValueError(f"无法解析比分: {text!r}")
    return int(m.group(1)), int(m.group(2))


def format_percentage(value: float, *, digits: int = 1) -> str:
    """将 0~1 的小数格式化为百分比字符串，例如 0.523 -> ``52.3%``。"""
    return f"{value * 100:.{digits}f}%"


def safe_json_loads(text: str, default: Any = None) -> Any:
    """解析 JSON；失败时返回 ``default``，不抛出异常。"""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default
