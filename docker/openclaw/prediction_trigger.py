# -*- coding: utf-8 -*-
"""
OpenClaw 预测触发模块

负责在赛前特定时间点触发预测流程:
  - 赛前3.5h: 第一次预测（初盘+实时赔率 + 赛前信息第一档）
  - 赛前2.5h: 第二次预测（赔率变动追踪）
  - 赛前1.5h: 第三次预测（赔率+情报更新）
  - 赛前0.5h: 最终预测（最新赔率+官方首发+第二档）

预测通过 HTTP 请求发送到主预测系统（advance_predictor）。

去重机制:
  - 使用文件持久化（不怕容器重启）
  - 按 match_id + trigger_point 组合判断是否已触发
"""

import os
import json
import time
import logging
from datetime import datetime, date as date_type
from typing import Optional

import httpx

from odds_scraper import fetch_match_list, get_snapshot_manager
from notifier import push_prediction

logger = logging.getLogger(__name__)

# 预测系统地址（主应用）
PREDICTOR_API_URL = os.getenv("PREDICTOR_API_URL", "http://host.docker.internal:8000")

# 预测触发时间点（赛前N小时，四档）
TRIGGER_POINTS = [3.5, 2.5, 1.5, 0.5]
TIER2_THRESHOLD_HOURS = 1.0    # 赛前1h内切换到第二档（官方首发）

# 触发窗口（单位：小时）— 调度每30分钟一次，窗口设为15分钟确保不重复
# 关键逻辑：调度间隔30分钟，窗口±15分钟，保证每个触发点只被一次调度命中
TRIGGER_WINDOW = 0.25  # 距离触发点±15分钟内触发

# 持久化去重文件目录
TRIGGERED_STATE_DIR = os.getenv("TRIGGERED_STATE_DIR", "/app/data/triggered_state")


# ═══════════════════════════════════════════════════════════════
#  持久化去重（替代内存字典，容器重启不丢失）
# ═══════════════════════════════════════════════════════════════

def _state_file_path() -> str:
    """获取当天的去重状态文件路径"""
    os.makedirs(TRIGGERED_STATE_DIR, exist_ok=True)
    today = date_type.today().isoformat()
    return os.path.join(TRIGGERED_STATE_DIR, f"triggered_{today}.json")


def _load_triggered_state() -> dict:
    """加载当天的去重状态"""
    path = _state_file_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_triggered_state(state: dict):
    """保存去重状态到文件"""
    path = _state_file_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except IOError as e:
        logger.error(f"保存去重状态失败: {e}")


def _should_trigger(match_id: str, trigger_point: float) -> bool:
    """
    判断是否应该触发预测（基于持久化文件去重）

    去重 key: match_id + trigger_point
    """
    state = _load_triggered_state()
    triggered_points = state.get(match_id, [])
    for t in triggered_points:
        if abs(t - trigger_point) < 0.3:  # 同一触发点（容差18分钟）视为重复
            return False
    return True


def _mark_triggered(match_id: str, trigger_point: float):
    """标记某场比赛在某个触发点已完成预测（持久化到文件）"""
    state = _load_triggered_state()
    if match_id not in state:
        state[match_id] = []
    state[match_id].append(trigger_point)
    _save_triggered_state(state)


def _unmark_triggered(match_id: str, trigger_point: float):
    """移除某场比赛某个触发点的标记（预测失败时调用，允许下次重试）"""
    state = _load_triggered_state()
    if match_id in state:
        state[match_id] = [t for t in state[match_id] if abs(t - trigger_point) >= 0.3]
        if not state[match_id]:
            del state[match_id]
        _save_triggered_state(state)


def _cleanup_old_state_files():
    """清理3天前的去重状态文件"""
    if not os.path.exists(TRIGGERED_STATE_DIR):
        return
    today = date_type.today()
    for fname in os.listdir(TRIGGERED_STATE_DIR):
        if not fname.startswith("triggered_") or not fname.endswith(".json"):
            continue
        try:
            date_str = fname.replace("triggered_", "").replace(".json", "")
            file_date = date_type.fromisoformat(date_str)
            if (today - file_date).days > 3:
                os.remove(os.path.join(TRIGGERED_STATE_DIR, fname))
                logger.info(f"清理旧状态文件: {fname}")
        except (ValueError, OSError):
            pass


# ═══════════════════════════════════════════════════════════════
#  预测触发
# ═══════════════════════════════════════════════════════════════

def trigger_prediction(match: dict, tier: int = 1, trigger_point: float = 0) -> Optional[dict]:
    """
    触发单场比赛的预测

    通过 HTTP 请求发送到主预测系统

    Args:
        match: 比赛信息 {"match_id", "home_team", "away_team", ...}
        tier: 1=第一档, 2=第二档
        trigger_point: 触发时间点（用于推送标题）

    Returns:
        预测结果（如果预测系统同步返回）
    """
    match_id = match["match_id"]
    home = match["home_team"]
    away = match["away_team"]
    date = match.get("date", "")

    # 更新赔率快照
    mgr = get_snapshot_manager()
    mgr.update(match_id, match.get("kickoff_time"))
    odds = mgr.get_odds(match_id)

    payload = {
        "match_id": match_id,
        "home_team": home,
        "away_team": away,
        "date": date,
        "tier": tier,
        "kickoff_time": match.get("kickoff_time"),
        "odds": odds,
        "trigger_time": datetime.now().isoformat(),
    }

    logger.info(f"[预测触发] {home} vs {away} (tier={tier}, trigger_point={trigger_point}h, match_id={match_id})")

    try:
        # 发送到预测系统
        resp = httpx.post(
            f"{PREDICTOR_API_URL}/api/predict",
            json=payload,
            timeout=300,  # 预测可能需要5分钟
        )
        if resp.status_code == 200:
            result = resp.json()
            logger.info(f"[预测完成] {home} vs {away}: {result.get('llm_analysis', {}).get('wdl_prediction', {}).get('primary', '?')}")
            # 推送结果到用户手机
            push_prediction(
                home_team=home,
                away_team=away,
                prediction_result=result,
                tier=tier,
                hours_to_kickoff=trigger_point,
            )
            return result
        else:
            logger.error(f"预测系统返回 {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.error(f"触发预测失败: {e}")

    return None


# ═══════════════════════════════════════════════════════════════
#  主调度循环
# ═══════════════════════════════════════════════════════════════

def prediction_loop():
    """
    预测调度主循环（由 APScheduler 每30分钟调用）

    逻辑:
      1. 获取比赛列表（带重试）
      2. 更新即将开赛的赔率快照
      3. 遍历每场比赛，在赛前 3.5h / 2.5h / 1.5h / 0.5h 四个时间点触发预测
         - 赛前 > 1h: 第一档（缺阵预判+惯用阵型）
         - 赛前 ≤ 1h: 第二档（尝试获取官方首发，爬不到则用惯用阵型兜底）
      4. 每场比赛每个触发点只触发一次（持久化去重）
    """
    logger.info("=== 预测调度 tick ===")

    # 获取比赛列表（带重试，避免网络抖动导致漏赛）
    matches = None
    for attempt in range(3):
        matches = fetch_match_list()
        if matches:
            break
        logger.warning(f"获取比赛列表失败，第{attempt+1}次重试...")
        time.sleep(5)

    if not matches:
        logger.warning("获取比赛列表失败（已重试3次），跳过本次调度")
        return

    logger.info(f"获取到 {len(matches)} 场比赛，其中未完赛: "
                f"{sum(1 for m in matches if not m.get('finished'))} 场")

    # 更新赔率快照
    mgr = get_snapshot_manager()
    mgr.update_all_upcoming(matches)

    # 清理旧状态文件
    _cleanup_old_state_files()

    triggered_count = 0
    skipped_count = 0

    for m in matches:
        if m.get("finished"):
            continue

        hours = m.get("hours_to_kickoff")
        if hours is None or hours < 0:
            continue

        match_id = m["match_id"]

        # 找最接近的触发时间点（且当前时间在触发窗口内）
        nearest_point = None
        min_diff = float('inf')
        for tp in TRIGGER_POINTS:
            diff = abs(hours - tp)
            if diff < min_diff:
                min_diff = diff
                nearest_point = tp

        # 触发窗口：距触发点±15分钟内触发（调度间隔30分钟，确保每个触发点只被一次调度命中）
        if min_diff > TRIGGER_WINDOW:
            continue

        # 持久化去重：避免重复触发
        if not _should_trigger(match_id, nearest_point):
            skipped_count += 1
            continue

        # 确定档位：赛前≤1h用第二档，>1h用第一档
        tier = 2 if hours <= TIER2_THRESHOLD_HOURS else 1

        logger.info(f"[触发预测] {m['home_team']} vs {m['away_team']} "
                    f"(距开赛{hours:.1f}h, 触发点={nearest_point}h, tier={tier})")

        # ★ 关键：先标记为已触发，再执行预测
        # 避免预测耗时期间（可能5分钟），下一次调度又触发同一场比赛
        _mark_triggered(match_id, nearest_point)

        result = trigger_prediction(m, tier=tier, trigger_point=nearest_point)

        if result is not None:
            triggered_count += 1
        else:
            # 预测失败：移除标记，允许下次重试
            _unmark_triggered(match_id, nearest_point)
            logger.warning(f"[预测失败] {m['home_team']} vs {m['away_team']} 将在下次调度重试")

    logger.info(f"=== 预测调度 tick 完成: 触发{triggered_count}场, 跳过(已触发){skipped_count}场 ===")
