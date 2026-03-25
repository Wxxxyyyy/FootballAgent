# -*- coding: utf-8 -*-
"""
赛前分析数据的跨模块同步状态

server_api.py 调用 notify_pre_match() 存入数据并唤醒等待方
advance_predictor.py 调用 wait_for_pre_match() 阻塞等待数据到达

两者在同一进程中共享本模块的全局字典，通过 threading.Event 实现同步。
"""

import threading
from typing import Optional

_cache: dict[str, dict] = {}
_events: dict[str, threading.Event] = {}
_lock = threading.Lock()


def _key(home: str, away: str) -> str:
    return f"{home.strip().lower()}|{away.strip().lower()}"


def wait_for_pre_match(home: str, away: str, timeout: float = 120) -> Optional[dict]:
    """
    阻塞等待 OpenClaw 赛前分析数据到达

    由预测流水线调用，会在收到 /receive_data 推送后返回
    """
    key = _key(home, away)

    with _lock:
        if key in _cache:
            return _cache.pop(key)
        event = threading.Event()
        _events[key] = event

    success = event.wait(timeout=timeout)

    with _lock:
        _events.pop(key, None)
        if success:
            return _cache.pop(key, None)
    return None


def notify_pre_match(home: str, away: str, data: dict):
    """
    通知等待方数据已到达

    由 server_api.py 的 /receive_data 端点调用
    """
    key = _key(home, away)
    with _lock:
        _cache[key] = data
        event = _events.get(key)
    if event:
        event.set()
