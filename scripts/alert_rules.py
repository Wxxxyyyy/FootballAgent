# -*- coding: utf-8 -*-
"""
LLM 告警规则引擎（宿主机侧，由 systemd timer 每 10 分钟运行）

第一性原理:
  llm_calls 表已经在记录每次 LLM 调用（model/latency/tokens/success/trace_id），
  但"记录"≠"监控"。监控 = 记录 + 主动检查 + 超阈值告警。
  本脚本把已有数据闭环：定期检查 llm_calls 表，超阈值时 Bark 推送。

告警规则（5 条核心指标）:
  1. 近 1h 成功率 < 70%          → LLM 频繁失败，需排查
  2. 近 1h 平均延迟 > 30s        → LLM 响应慢，影响用户体验
  3. 近 24h token 消耗 > 500k   → 成本异常，可能有死循环
  4. 近 1h 调用次数 = 0 且非深夜 → 服务可能假死
  5. 近 24h 错误数 > 10          → 错误堆积，需关注

防刷屏策略:
  - 状态翻转时告警（正常→故障 / 故障→恢复）
  - 持续故障期间每 60 分钟重复提醒
  - 状态记录在 data/alert_state.json

推送通道: 复用 health_watchdog 的 Bark 推送逻辑（自建优先+公共兜底+重试）
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta

import pymysql
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("alert_rules")

STATE_FILE = os.path.join(PROJECT_ROOT, "data", "alert_state.json")
REALERT_INTERVAL = 60 * 60  # 持续故障期间重复告警间隔（秒）

# 告警阈值
THRESHOLDS = {
    "success_rate_1h": 0.70,       # 近 1h 成功率低于此值告警
    "avg_latency_1h_ms": 30000,    # 近 1h 平均延迟超过此值告警（30s）
    "token_24h_limit": 500000,     # 近 24h token 消耗超过此值告警
    "error_24h_limit": 10,         # 近 24h 错误数超过此值告警
}

# Bark 推送
BARK_KEY = os.getenv("BARK_KEY", "")
PUBLIC_BARK_BASE = "https://api.day.app"
BARK_SERVER = os.getenv("BARK_SERVER_HOST", "").rstrip("/")


# ═══════════════════════════════════════════════════════════════
#  Bark 推送（复用 watchdog 逻辑）
# ═══════════════════════════════════════════════════════════════

def _send_alert(title: str, body: str) -> bool:
    """发送告警推送（自建优先+公共兜底，各重试2次）"""
    if not BARK_KEY:
        logger.warning("BARK_KEY 未配置，跳过告警推送")
        return False
    import httpx
    bases = ([BARK_SERVER] if BARK_SERVER else []) + [PUBLIC_BARK_BASE]
    payload = {
        "title": title, "body": body,
        "sound": "alarm", "group": "LLM告警", "level": "timeSensitive",
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


# ═══════════════════════════════════════════════════════════════
#  指标采集
# ═══════════════════════════════════════════════════════════════

def _db():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "football123"),
        database=os.getenv("MYSQL_DATABASE", "football_agent"),
        charset="utf8mb4",
        connect_timeout=3,
    )


def collect_metrics() -> dict:
    """从 llm_calls 表采集告警所需指标

    Returns:
        {
            "success_rate_1h": float,      # 近1h成功率
            "avg_latency_1h_ms": float,   # 近1h平均延迟(ms)
            "call_count_1h": int,          # 近1h调用次数
            "token_24h": int,             # 近24h token总消耗
            "error_24h": int,             # 近24h错误数
            "call_count_24h": int,         # 近24h调用次数
        }
    """
    try:
        conn = _db()
        cursor = conn.cursor()
        now = datetime.now()

        # 近1h 指标
        h1_ago = now - timedelta(hours=1)
        cursor.execute(
            """SELECT
                COUNT(*) as total,
                COALESCE(SUM(success), 0) as ok,
                COALESCE(AVG(latency_ms), 0) as avg_ms,
                COALESCE(SUM(prompt_tokens), 0) + COALESCE(SUM(completion_tokens), 0) as tokens
               FROM llm_calls WHERE called_at >= %s""",
            (h1_ago.strftime("%Y-%m-%d %H:%M:%S"),),
        )
        row = cursor.fetchone()
        total_1h, ok_1h, avg_ms_1h, tokens_1h = row

        # 近24h 指标
        h24_ago = now - timedelta(hours=24)
        cursor.execute(
            """SELECT
                COUNT(*) as total,
                COALESCE(SUM(CASE WHEN success=0 THEN 1 ELSE 0 END), 0) as errors,
                COALESCE(SUM(prompt_tokens), 0) + COALESCE(SUM(completion_tokens), 0) as tokens
               FROM llm_calls WHERE called_at >= %s""",
            (h24_ago.strftime("%Y-%m-%d %H:%M:%S"),),
        )
        row = cursor.fetchone()
        total_24h, errors_24h, tokens_24h = row

        cursor.close()
        conn.close()

        return {
            "success_rate_1h": float(ok_1h) / float(total_1h) if total_1h > 0 else 1.0,
            "avg_latency_1h_ms": float(avg_ms_1h) if total_1h > 0 else 0.0,
            "call_count_1h": int(total_1h),
            "token_24h": int(tokens_24h),
            "error_24h": int(errors_24h),
            "call_count_24h": int(total_24h),
        }
    except Exception as e:
        # 表不存在时视为"无数据"（表是惰性创建的，首次 log_llm_call 才建表）
        # 返回默认正常状态，不触发告警
        logger.info(f"指标采集跳过（表可能不存在）: {e}")
        return {
            "success_rate_1h": 1.0,
            "avg_latency_1h_ms": 0.0,
            "call_count_1h": 0,
            "token_24h": 0,
            "error_24h": 0,
            "call_count_24h": 0,
        }


# ═══════════════════════════════════════════════════════════════
#  告警规则评估
# ═══════════════════════════════════════════════════════════════

def evaluate_rules(metrics: dict) -> list:
    """评估告警规则，返回触发的告警列表

    Returns:
        [{"rule": "success_rate_1h", "value": 0.65, "threshold": 0.70,
          "message": "近1h成功率 65% 低于阈值 70%"}, ...]
    """
    if not metrics:
        return []

    alerts = []

    # 规则1: 近1h成功率
    sr = metrics["success_rate_1h"]
    if metrics["call_count_1h"] > 0 and sr < THRESHOLDS["success_rate_1h"]:
        alerts.append({
            "rule": "success_rate_1h",
            "value": round(sr, 4),
            "threshold": THRESHOLDS["success_rate_1h"],
            "message": f"近1h LLM成功率 {sr:.0%} 低于阈值 {THRESHOLDS['success_rate_1h']:.0%} "
                       f"({metrics['call_count_1h']}次调用)",
        })

    # 规则2: 近1h平均延迟
    lat = metrics["avg_latency_1h_ms"]
    if metrics["call_count_1h"] > 0 and lat > THRESHOLDS["avg_latency_1h_ms"]:
        alerts.append({
            "rule": "avg_latency_1h",
            "value": round(lat, 0),
            "threshold": THRESHOLDS["avg_latency_1h_ms"],
            "message": f"近1h LLM平均延迟 {lat/1000:.1f}s 超过阈值 {THRESHOLDS['avg_latency_1h_ms']/1000:.0f}s",
        })

    # 规则3: 近24h token消耗
    tk = metrics["token_24h"]
    if tk > THRESHOLDS["token_24h_limit"]:
        alerts.append({
            "rule": "token_24h",
            "value": tk,
            "threshold": THRESHOLDS["token_24h_limit"],
            "message": f"近24h token消耗 {tk:,} 超过阈值 {THRESHOLDS['token_24h_limit']:,}",
        })

    # 规则4: 近1h无调用且非深夜（22:00-08:00 视为深夜，无调用正常）
    now_hour = datetime.now().hour
    if metrics["call_count_1h"] == 0 and not (22 <= now_hour or now_hour < 8):
        alerts.append({
            "rule": "no_calls_1h",
            "value": 0,
            "threshold": 1,
            "message": "近1h无LLM调用，服务可能假死（当前非深夜时段）",
        })

    # 规则5: 近24h错误数
    err = metrics["error_24h"]
    if err > THRESHOLDS["error_24h_limit"]:
        alerts.append({
            "rule": "error_24h",
            "value": err,
            "threshold": THRESHOLDS["error_24h_limit"],
            "message": f"近24h LLM错误数 {err} 超过阈值 {THRESHOLDS['error_24h_limit']}",
        })

    return alerts


# ═══════════════════════════════════════════════════════════════
#  状态管理与主流程
# ═══════════════════════════════════════════════════════════════

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
    """主流程: 采集指标 → 评估规则 → 告警推送"""
    logger.info("=" * 50)
    logger.info("LLM 告警规则检查开始")

    metrics = collect_metrics()
    if not metrics:
        logger.error("指标采集失败，跳过本次检查")
        return 1

    logger.info(f"指标: 成功率1h={metrics['success_rate_1h']:.0%} "
                f"延迟1h={metrics['avg_latency_1h_ms']/1000:.1f}s "
                f"调用1h={metrics['call_count_1h']} "
                f"token24h={metrics['token_24h']:,} "
                f"错误24h={metrics['error_24h']}")

    alerts = evaluate_rules(metrics)
    state = _load_state()
    now = time.time()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not alerts:
        # 全部正常
        recovered = []
        for rule in ["success_rate_1h", "avg_latency_1h", "token_24h", "no_calls_1h", "error_24h"]:
            prev = state.get(rule, {"ok": True, "last_alert": 0})
            if not prev.get("ok", True):
                recovered.append(rule)

        if recovered:
            _send_alert("✅ LLM告警已恢复", f"时间: {now_str}\n已恢复: {', '.join(recovered)}")
            logger.info(f"告警恢复: {recovered}")

        for rule in ["success_rate_1h", "avg_latency_1h", "token_24h", "no_calls_1h", "error_24h"]:
            state[rule] = {"ok": True, "last_alert": 0}
        _save_state(state)
        logger.info("全部指标正常，无告警")
        return 0

    # 有告警
    for alert in alerts:
        rule = alert["rule"]
        prev = state.get(rule, {"ok": True, "last_alert": 0})
        first_failure = prev.get("ok", True)
        need_realert = (now - prev.get("last_alert", 0)) >= REALERT_INTERVAL

        if first_failure or need_realert:
            sent = _send_alert(
                f"🚨 LLM告警: {alert['message']}",
                f"时间: {now_str}\n规则: {rule}\n当前值: {alert['value']}\n阈值: {alert['threshold']}"
                + ("" if first_failure else "\n(持续告警中，每60分钟提醒)"),
            )
            logger.info(f"告警 [{rule}]: {alert['message']} 推送{'成功' if sent else '失败'}")
            state[rule] = {"ok": False, "last_alert": now}
        else:
            state[rule] = prev  # 持续告警中，未到重复时间

    _save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
