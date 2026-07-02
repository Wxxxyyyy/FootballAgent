# -*- coding: utf-8 -*-
"""
链路健康看门狗（宿主机侧，由 systemd timer 每 5 分钟运行一次）

检查项:
  1. 预测服务  http://127.0.0.1:8000/health
  2. OpenClaw  http://127.0.0.1:9000/health（含调度器运行状态）
  3. 关键容器  football_mysql / football_neo4j / football_openclaw 是否 running

告警策略（防刷屏）:
  - 状态翻转时告警: 正常→故障 发告警, 故障→恢复 发恢复通知
  - 持续故障期间每 30 分钟重复提醒一次
  - 状态记录在 data/watchdog_state.json

推送通道复用 Bark（自建优先+公共兜底+重试），与 docker/openclaw/notifier 同一套逻辑。
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

STATE_FILE = os.path.join(PROJECT_ROOT, "data", "watchdog_state.json")
REALERT_INTERVAL = 30 * 60  # 持续故障期间重复告警间隔（秒）

BARK_KEY = os.getenv("BARK_KEY", "")
PUBLIC_BARK_BASE = "https://api.day.app"
# 宿主机访问自建 bark-server 走本机映射端口
BARK_SERVER = os.getenv("BARK_SERVER_HOST", "").rstrip("/")


def _send_alert(title: str, body: str) -> bool:
    """发送告警推送（自建优先+公共兜底，各重试2次）"""
    if not BARK_KEY:
        return False
    bases = ([BARK_SERVER] if BARK_SERVER else []) + [PUBLIC_BARK_BASE]
    payload = {
        "title": title, "body": body,
        "sound": "alarm", "group": "系统告警", "level": "timeSensitive",
    }
    for base in bases:
        for attempt in range(2):
            try:
                r = httpx.post(f"{base}/{BARK_KEY}/", json=payload, timeout=15)
                if r.status_code == 200 and r.json().get("code") == 200:
                    return True
            except Exception:
                pass
            time.sleep(2)
    return False


# ══════════════════ 检查项 ══════════════════

def check_predictor() -> tuple:
    try:
        r = httpx.get("http://127.0.0.1:8000/health", timeout=8)
        if r.status_code == 200 and r.json().get("status") == "ok":
            return True, "ok"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, f"{type(e).__name__}"


def check_openclaw() -> tuple:
    try:
        r = httpx.get("http://127.0.0.1:9000/health", timeout=8)
        data = r.json()
        if r.status_code != 200 or data.get("status") != "ok":
            return False, f"HTTP {r.status_code}"
        if not data.get("scheduler_running"):
            return False, "调度器已停止"
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}"


def check_containers() -> tuple:
    required = ["football_mysql", "football_neo4j", "football_openclaw"]
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=15,
        ).stdout.split()
        missing = [c for c in required if c not in out]
        if missing:
            return False, f"容器未运行: {', '.join(missing)}"
        return True, "ok"
    except Exception as e:
        return False, f"docker 检查失败: {type(e).__name__}"


CHECKS = {
    "预测服务(8000)": check_predictor,
    "OpenClaw(9000)": check_openclaw,
    "Docker容器": check_containers,
}


# ══════════════════ 状态管理与主流程 ══════════════════

def _load_state() -> dict:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def main() -> int:
    state = _load_state()
    now = time.time()
    now_str = datetime.now().strftime("%H:%M:%S")
    any_down = False

    for name, check in CHECKS.items():
        ok, detail = check()
        prev = state.get(name, {"ok": True, "last_alert": 0})

        if not ok:
            any_down = True
            first_failure = prev["ok"]
            need_realert = (now - prev.get("last_alert", 0)) >= REALERT_INTERVAL
            if first_failure or need_realert:
                sent = _send_alert(
                    f"🚨 {name} 故障",
                    f"{detail}\n时间: {now_str}"
                    + ("" if first_failure else "\n(持续故障中，每30分钟提醒)"),
                )
                print(f"[watchdog] {name} DOWN ({detail}) 告警{'已发' if sent else '发送失败'}")
                state[name] = {"ok": False, "last_alert": now}
            else:
                state[name] = prev  # 故障中但未到重复告警时间
        else:
            if not prev["ok"]:
                _send_alert(f"✅ {name} 已恢复", f"时间: {now_str}")
                print(f"[watchdog] {name} RECOVERED")
            state[name] = {"ok": True, "last_alert": 0}

    _save_state(state)
    print(f"[watchdog] {now_str} 检查完成: {'有故障' if any_down else '全部正常'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
