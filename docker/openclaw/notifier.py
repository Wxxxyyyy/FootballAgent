# -*- coding: utf-8 -*-
"""
预测结果推送模块 — Bark (iOS)

将预测结果推送到用户 iPhone，通过 Bark APP 的 iOS 推送通知。

Bark API: https://api.day.app/{key}/推送内容

去重机制:
  推送层自身维护一份已推送记录（最后一道防线），
  即使上游 prediction_trigger 的去重出问题，也不会重复推送。
"""

import os
import logging
import json
from datetime import date as date_type

import httpx

logger = logging.getLogger(__name__)

# Bark device key（从环境变量读取，避免硬编码）
BARK_KEY = os.getenv("BARK_KEY", "")
BARK_ENABLED = os.getenv("BARK_ENABLED", "true").lower() == "true"

# Bark API 地址
BARK_API_BASE = "https://api.day.app"

# 推送去重状态目录
PUSH_STATE_DIR = os.getenv("PUSH_STATE_DIR", "/app/data/push_state")


def _push_state_file() -> str:
    """获取当天推送去重状态文件"""
    os.makedirs(PUSH_STATE_DIR, exist_ok=True)
    today = date_type.today().isoformat()
    return os.path.join(PUSH_STATE_DIR, f"pushed_{today}.json")


def _load_push_state() -> set:
    """加载已推送的 key 集合"""
    path = _push_state_file()
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except (json.JSONDecodeError, IOError):
        return set()


def _save_push_state(state: set):
    """保存推送去重状态"""
    path = _push_state_file()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(state), f, ensure_ascii=False)
    except IOError as e:
        logger.error(f"保存推送去重状态失败: {e}")


def _is_already_pushed(home_team: str, away_team: str, trigger_point: float) -> bool:
    """检查某场比赛某个触发点是否已推送过"""
    key = f"{home_team}_vs_{away_team}_{trigger_point:.1f}"
    state = _load_push_state()
    return key in state


def _mark_pushed(home_team: str, away_team: str, trigger_point: float):
    """标记已推送"""
    key = f"{home_team}_vs_{away_team}_{trigger_point:.1f}"
    state = _load_push_state()
    state.add(key)
    _save_push_state(state)


def _bark_url(path: str) -> str:
    """拼接 Bark API URL"""
    key = os.getenv("BARK_KEY", BARK_KEY)
    return f"{BARK_API_BASE}/{key}/{path}"


def push_prediction(
    home_team: str,
    away_team: str,
    prediction_result: dict,
    tier: int = 1,
    hours_to_kickoff: float = 0.0,
) -> bool:
    """
    推送预测结果到用户 iPhone

    Args:
        home_team: 主队名
        away_team: 客队名
        prediction_result: 预测系统返回的结果字典
        tier: 预测档位 1/2
        hours_to_kickoff: 触发点时间（用于标题显示，如 3.5、2.5、1.5）

    Returns:
        True=推送成功, False=推送失败或未启用
    """
    if not BARK_ENABLED:
        logger.debug("Bark 推送已禁用")
        return False

    if not BARK_KEY:
        logger.warning("Bark key 未配置，跳过推送")
        return False

    # 推送层去重（最后一道防线）
    if _is_already_pushed(home_team, away_team, hours_to_kickoff):
        logger.info(f"[Bark] 跳过重复推送: {home_team} vs {away_team} (触发点={hours_to_kickoff:.1f}h)")
        return False

    # 提取预测结果中的关键信息
    llm_analysis = prediction_result.get("llm_analysis", {}) or {}
    wdl = llm_analysis.get("wdl_prediction", {}) or {}
    score_predictions = llm_analysis.get("score_predictions", []) or []
    upset = llm_analysis.get("upset_prediction") or {}

    # 构建推送标题和内容
    # hours_to_kickoff 这里表示触发点（如 3.5, 2.5, 1.5, 0.5）
    if tier == 2:
        tier_label = "最终预测"
    else:
        # 使用触发点的整数/半整数显示，避免 3.4999 这类浮点显示问题
        tier_label = f"{hours_to_kickoff:.1f}h"
    title = f"预测 {tier_label} | {home_team} vs {away_team}"

    body_parts = []

    # 胜平负
    wdl_primary = wdl.get("primary", "")
    wdl_secondary = wdl.get("secondary", "")
    wdl_conf = wdl.get("confidence", "N/A")
    pri_label = {"H": "主胜", "D": "平局", "A": "客胜"}.get(wdl_primary, wdl_primary)
    sec_label = {"H": "主胜", "D": "平局", "A": "客胜"}.get(wdl_secondary, wdl_secondary)
    body_parts.append(f"胜负: {pri_label}")
    if sec_label:
        body_parts.append(f"次选: {sec_label}")
    body_parts.append(f"置信度: {wdl_conf}")
    body_parts.append(f"")

    # 正常比分（2个）
    if score_predictions:
        body_parts.append(f"正常比分:")
        for i, s in enumerate(score_predictions[:2], 1):
            score_str = s.get("score", "未知")
            prob = s.get("prob", "")
            prob_str = f" ({prob:.0%})" if isinstance(prob, float) else ""
            body_parts.append(f"{i}. {score_str}{prob_str}")
        body_parts.append(f"")

    # 爆冷比分（1个）
    if upset and upset.get("score"):
        upset_score = upset.get("score", "")
        upset_result = upset.get("result", "")
        result_label = {"H": "主胜", "D": "平局", "A": "客胜"}.get(upset_result, upset_result)
        body_parts.append(f"⚠️ 爆冷可能: {upset_score} ({result_label})")
        body_parts.append(f"")

    # 关键依据（取前2条）
    key_points = llm_analysis.get("key_points", [])
    if not key_points:
        # 尝试从 score_predictions 和 upset 的 reason 中提取
        reasons = []
        for s in score_predictions[:2]:
            r = s.get("reason", "")
            if r:
                reasons.append(r)
        if upset and upset.get("reason"):
            reasons.append(f"爆冷: {upset['reason']}")
        key_points = reasons
    if key_points:
        body_parts.append(f"关键依据:")
        for i, point in enumerate(key_points[:2], 1):
            body_parts.append(f"{i}. {point}")
        body_parts.append(f"")

    # 触发时间
    trigger_time = prediction_result.get("trigger_time", "")
    if trigger_time:
        body_parts.append(f"⏰ {trigger_time}")

    body = "\n".join(body_parts)

    # 发送推送（Bark 支持 URL 参数式 POST）
    try:
        # 使用 Bark 的高级参数：title + body，支持响铃
        payload = {
            "title": title,
            "body": body,
            "sound": "minuet",  # 默认铃声
            "group": "FootballAgent",  # 消息分组
        }

        resp = httpx.post(
            _bark_url(""),  # https://api.day.app/{key}/
            json=payload,
            timeout=10,
        )

        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == 200:
                logger.info(f"[Bark] 推送成功: {home_team} vs {away_team} (触发点={hours_to_kickoff:.1f}h)")
                # 标记已推送，防止重复
                _mark_pushed(home_team, away_team, hours_to_kickoff)
                return True
            else:
                logger.warning(f"[Bark] 推送失败: {data}")
                return False
        else:
            logger.warning(f"[Bark] HTTP {resp.status_code}: {resp.text[:200]}")
            return False

    except Exception as e:
        logger.error(f"[Bark] 推送异常: {e}")
        return False


def push_simple(title: str, body: str) -> bool:
    """简单文本推送（用于测试或系统通知）"""
    if not BARK_ENABLED:
        return False
    if not BARK_KEY:
        return False

    try:
        resp = httpx.post(
            _bark_url(""),
            json={
                "title": title,
                "body": body,
                "sound": "minuet",
                "group": "FootballAgent",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("code") == 200
        return False
    except Exception:
        return False
