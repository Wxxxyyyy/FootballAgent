# -*- coding: utf-8 -*-
"""
LLM 推理预测模型 - 调用 Kimi 2.5 进行赛前综合分析

输入: ML 基准概率 + 近5场 + 历史交锋 + 赔率
输出: 结构化预测结果 (JSON)
  - 胜平负预测 (两种最可能结果)
  - 大小球预测
  - 三个比分预测
  - 综合分析文字
"""

import os
import re
import json
from typing import Optional
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


# ═══════════════════════════════════════════════════════════════
#  System Prompt
# ═══════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
你是一名顶级足球赛事分析师，擅长结合统计模型、赛事数据和赔率信息进行赛前预测。

分析原则（按优先级排序）:
1. 【最高优先级】赛前情报（战意/出线形势/轮换/伤停）是判断的核心依据
   - 特别关注"已出线/已淘汰/轮换练兵/生死战/打平即可"等战意因素
   - 当战意因素与 ML 概率矛盾时，以战意因素为准
2. ML 模型概率是基于历史赔率训练的基准参考（验证集准确率约54.5%）
   - 仅在无战意因素干扰时作为基础判断
   - 赔率反映市场综合判断，但市场可能未充分反映轮换/战意
3. 蒙特卡洛模拟基于泊松分布，提供比分分布参考
4. 近期战绩反映球队当前状态
5. 历史交锋体现两队之间的相生相克关系
6. 赔率反映了市场对比赛的综合判断

输出要求: 必须严格以 JSON 格式输出，结构如下:
{
  "wdl_prediction": {
    "primary": "H 或 D 或 A（综合所有信息后的最可能结果）",
    "secondary": "H 或 D 或 A（次可能结果，与 primary 不同）",
    "confidence": "高 或 中 或 低"
  },
  "key_points": [
    "关键依据1（最重要，优先引用赛前情报：伤停/首发/战意/轮换/新闻软信号）",
    "关键依据2",
    "关键依据3"
  ],
  "ml_prediction": {
    "result": "H 或 D 或 A",
    "confidence": "高/中/低",
    "reason": "基于ML模型概率的理由（50字内）"
  },
  "monte_carlo_prediction": {
    "result": "H 或 D 或 A",
    "most_likely_score": "主队:客队",
    "reason": "基于蒙特卡洛模拟的理由（50字内）"
  },
  "score_predictions": [
    {"score": "主队:客队", "prob": 0.15, "reason": "正常比分预测1"},
    {"score": "主队:客队", "prob": 0.12, "reason": "正常比分预测2"}
  ],
  "upset_prediction": null,
  "overall_analysis": "综合分析（100-200字）"
}

关于 wdl_prediction 与 key_points（这两项是推送给用户的核心，务必填写）:
- wdl_prediction 是你综合 ML/蒙特卡洛/近况/交锋/赛前情报后的【最终判断】，不要照抄某一个模型。
- score_predictions 的比分也必须是综合判断（以蒙特卡洛分布为基础，但要结合战意/伤停/情报修正），不要照抄蒙特卡洛。
- key_points 给 2-4 条，必须【优先】体现赛前情报（伤停、官方首发、战意/出线形势、轮换、新闻软信号）；
  只有在确实没有任何情报时，才退而引用模型概率与近期战绩。严禁编造情报里没有的内容。

爆冷预测规则:
- 如果你认为存在爆冷可能（冷门方胜率≥15%），则填写 upset_prediction:
  "upset_prediction": {
    "result": "H 或 D 或 A（爆冷结果）",
    "score": "主队:客队（爆冷比分）",
    "reason": "爆冷理由（80字内）"
  }
- 如果没有爆冷可能，设为 null
- 爆冷结果必须与你的正常预测不同

注意:
- H=主胜, D=平局, A=客胜
- score 格式严格为 "主队进球:客队进球"（用英文冒号）
- prob 为 0-1 之间的小数
- 所有分析必须基于提供的数据
- 严禁编造球员名字、伤停名单、轮换细节等情报中未明确提及的内容
- 只能引用赛前情报中出现的具体信息，不得自行推测或脑补"""


# ═══════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════

def predict_with_llm(
    home_team: str,
    away_team: str,
    date: str,
    ml_result: dict,
    home_last_5: list,
    away_last_5: list,
    h2h_records: list,
    odds_info: dict,
    upset_signals: dict = None,
    pre_match_intel_summary: str = "",
    monte_carlo_result: dict = None,
) -> dict:
    """
    调用 Kimi 2.5 进行综合赛前分析

    Args:
        pre_match_intel_summary: 赛前情报摘要文本（伤停/首发/新闻/教练风格）
        monte_carlo_result: 蒙特卡洛模拟结果（比分分布）

    Returns:
        包含 wdl_prediction / score_predictions /
        overall_analysis / upset_alert / raw_response 的字典
    """
    prompt = _build_prompt(
        home_team, away_team, date,
        ml_result, home_last_5, away_last_5,
        h2h_records, odds_info, upset_signals,
        pre_match_intel_summary,
        monte_carlo_result,
    )

    print(f"  [LLM] 调用 {os.getenv('LLM_MODEL_DEEPSEEK_PRO', 'deepseek-v4-pro-202606')}...")

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )

        # 默认使用 DeepSeek V4 Pro
        model_name = os.getenv("LLM_MODEL_DEEPSEEK_PRO", "deepseek-v4-pro-202606")

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )

        raw_text = response.choices[0].message.content
        print(f"  [LLM] 收到响应 ({len(raw_text)} 字符)")

        parsed = _parse_response(raw_text)
        # 归一化：保证 wdl_prediction / key_points 存在（notifier 推送依赖这两项）
        parsed = _normalize_output(parsed, ml_result, monte_carlo_result, pre_match_intel_summary)
        parsed["raw_response"] = raw_text
        return parsed

    except Exception as e:
        print(f"  [LLM] 调用失败: {e}")
        import traceback
        traceback.print_exc()
        # 兜底：LLM 不可用时，用 ML 概率 + 蒙特卡洛合成可推送的结果，保证赛前仍能收到预测
        fallback = _fallback_from_models(ml_result, monte_carlo_result, str(e))
        fallback["raw_response"] = ""
        return fallback


# ═══════════════════════════════════════════════════════════════
#  Prompt 构建
# ═══════════════════════════════════════════════════════════════

def _build_prompt(home_team, away_team, date,
                  ml_result, home_last_5, away_last_5,
                  h2h_records, odds_info, upset_signals=None,
                  pre_match_intel_summary="", monte_carlo_result=None) -> str:
    parts = []

    # 比赛信息
    parts.append(
        f"## 比赛信息\n"
        f"- 主队: {home_team}\n"
        f"- 客队: {away_team}\n"
        f"- 日期: {date}"
    )

    # ML 模型预测
    if ml_result and "error" not in ml_result:
        parts.append(
            f"## 机器学习模型预测（LightGBM 19维，验证集准确率54.5%）\n"
            f"- 主胜概率: {ml_result.get('home_win_prob', 'N/A')}\n"
            f"- 平局概率: {ml_result.get('draw_prob', 'N/A')}\n"
            f"- 客胜概率: {ml_result.get('away_win_prob', 'N/A')}\n"
            f"- 模型倾向: {_wdl_label(ml_result.get('wdl_prediction', ''))}\n"
            f"- 赔率来源: {ml_result.get('odds_source', 'N/A')}"
        )
    else:
        err = ml_result.get("error", "未知") if ml_result else "无数据"
        parts.append(f"## 机器学习模型预测\n无法获取（{err}）")

    # 蒙特卡洛模拟
    if monte_carlo_result:
        mc = monte_carlo_result
        score_dist_str = ", ".join(
            f"{s}:{p:.1%}" for s, p in mc.get("score_distribution", {}).items()
        )
        parts.append(
            f"## 蒙特卡洛模拟（{mc.get('n_simulations', 10000)}次泊松模拟）\n"
            f"- 最可能比分: {mc.get('most_likely_score', 'N/A')}\n"
            f"- 预期进球: 主队{mc.get('expected_goals_home', 'N/A')} 客队{mc.get('expected_goals_away', 'N/A')}\n"
            f"- Top5比分: {score_dist_str}\n"
            f"- 模拟胜平负: 主胜{mc.get('home_win_prob', 'N/A')} 平{mc.get('draw_prob', 'N/A')} 客胜{mc.get('away_win_prob', 'N/A')}"
        )

    # 主队近5场
    if home_last_5:
        lines = [f"## {home_team} 最近5场比赛"]
        for m in home_last_5:
            lines.append(_format_match(m))
        parts.append("\n".join(lines))

    # 客队近5场
    if away_last_5:
        lines = [f"## {away_team} 最近5场比赛"]
        for m in away_last_5:
            lines.append(_format_match(m))
        parts.append("\n".join(lines))

    # 历史交锋（仅有数据时才提供，最多5场）
    if h2h_records:
        lines = [f"## {home_team} vs {away_team} 历史交锋（知识图谱）"]
        for r in h2h_records[:5]:
            lines.append(
                f"- {r.get('date', '?')} [{r.get('season', '?')}] "
                f"{r.get('result', '?')} (总进球: {r.get('total_goals', '?')})"
            )
        parts.append("\n".join(lines))

    # 赔率
    if odds_info:
        if isinstance(odds_info, dict) and "bookmakers" in odds_info:
            lines = ["## 市场赔率（多家庄家）"]
            for bk, bd in odds_info["bookmakers"].items():
                title = bd.get("title", bk)
                mkts = bd.get("markets", {})
                h2h = mkts.get("h2h", {})
                if h2h:
                    lines.append(f"- {title}: {json.dumps(h2h, ensure_ascii=False)}")
            parts.append("\n".join(lines))
        else:
            parts.append(
                f"## 赔率信息\n"
                f"{json.dumps(odds_info, ensure_ascii=False, indent=2)}"
            )

    # 爆冷信号
    if upset_signals and upset_signals.get("has_risk") and upset_signals.get("signals"):
        lines = [
            f"## 爆冷信号分析（系统检测到 {len(upset_signals['signals'])} 个信号）",
            f"- 热门方: {upset_signals.get('favorite_name', '?')}",
            f"- 冷门方: {upset_signals.get('underdog_name', '?')}",
        ]
        for s in upset_signals["signals"]:
            lines.append(f"- [{s['severity']}] {s['type']}: {s['desc']}")
        lines.append(
            "\n请结合以上爆冷信号，判断是否需要触发爆冷预警（upset_alert）。"
            "如果你认为爆冷概率≥15%，请填写 upset_alert 字段。"
        )
        parts.append("\n".join(lines))

    # 赛前情报（伤停/首发/新闻/教练风格）
    if pre_match_intel_summary:
        parts.append(
            f"## 赛前情报（伤停/首发/新闻/教练风格）\n{pre_match_intel_summary}\n"
            f"请特别关注核心球员缺阵、队内冲突等赔率未反映的突发因素，"
            f"这些是爆冷的重要信号。"
        )

    parts.append(
        f"\n请基于以上所有数据，给出你对 {home_team} vs {away_team} 的赛前预测。\n"
        f"严格按照 system prompt 要求的 JSON 格式输出，不要添加其他文字。"
    )

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════

def _format_match(m: dict) -> str:
    d = m.get("date", "?")
    ht = m.get("home_team", "?")
    at = m.get("away_team", "?")
    hg = m.get("home_goals", "?")
    ag = m.get("away_goals", "?")
    lg = m.get("league", "")
    return f"- {d} | {ht} {hg}:{ag} {at} ({lg})"


def _wdl_label(code: str) -> str:
    return {"H": "主胜", "D": "平局", "A": "客胜"}.get(code, code)


def _parse_response(text: str) -> dict:
    """从 LLM 输出中提取 JSON"""
    # 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # markdown code block
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 找最外层 { ... }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return {
        "parse_error": "无法解析 LLM JSON 输出",
        "overall_analysis": text,
    }


# ═══════════════════════════════════════════════════════════════
#  输出归一化 / 兜底（保证 notifier 需要的 wdl_prediction / key_points 存在）
# ═══════════════════════════════════════════════════════════════

def _to_float(v, default: float = 0.0) -> float:
    """把概率值统一成 0-1 float，兼容 '52.1%' / '0.52' / 0.52。"""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace("%", "")
        try:
            f = float(s)
            return f / 100.0 if "%" in v else f
        except ValueError:
            return default
    return default


def _wdl_from_probs(ml_result: Optional[dict]):
    """从 ML 概率推 (primary, secondary, confidence)；无数据返回 (None, None, None)。"""
    if not ml_result or "error" in ml_result:
        return None, None, None
    probs = {
        "H": _to_float(ml_result.get("home_win_prob")),
        "D": _to_float(ml_result.get("draw_prob")),
        "A": _to_float(ml_result.get("away_win_prob")),
    }
    if sum(probs.values()) <= 0:
        return None, None, None
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    primary, p1 = ranked[0]
    secondary, p2 = ranked[1]
    gap = p1 - p2
    confidence = "高" if gap >= 0.20 else ("中" if gap >= 0.08 else "低")
    return primary, secondary, confidence


def _synth_key_points(parsed: dict, intel_summary: str = "") -> list:
    """合成关键依据：优先模型/比分给出的 reason。"""
    points = []
    mlp = parsed.get("ml_prediction") or {}
    mcp = parsed.get("monte_carlo_prediction") or {}
    for r in (mlp.get("reason"), mcp.get("reason")):
        if r and r not in points:
            points.append(str(r))
    for s in (parsed.get("score_predictions") or [])[:2]:
        r = s.get("reason") if isinstance(s, dict) else None
        if r and r not in points:
            points.append(str(r))
    if not points:
        oa = parsed.get("overall_analysis")
        if oa:
            points.append(str(oa)[:80])
    return points[:3]


def _normalize_output(parsed: dict, ml_result: Optional[dict] = None,
                      monte_carlo_result: Optional[dict] = None,
                      intel_summary: str = "") -> dict:
    """保证 parsed 含可用的 wdl_prediction 与 key_points（LLM 漏填时合成）。"""
    if not isinstance(parsed, dict):
        return parsed

    wdl = parsed.get("wdl_prediction")
    valid = isinstance(wdl, dict) and wdl.get("primary") in ("H", "D", "A")
    if not valid:
        # 1) 优先用 LLM 已给的 ml_prediction / monte_carlo_prediction 结果
        mlp = parsed.get("ml_prediction") or {}
        mcp = parsed.get("monte_carlo_prediction") or {}
        primary = mlp.get("result") if mlp.get("result") in ("H", "D", "A") else None
        secondary = mcp.get("result") if mcp.get("result") in ("H", "D", "A") else None
        confidence = mlp.get("confidence")
        # 2) 退回 ML 概率合成
        if not primary or not secondary or not confidence:
            p, s, c = _wdl_from_probs(ml_result)
            primary = primary or p
            secondary = secondary or (s if s != primary else None) or s
            confidence = confidence or c
        if primary:
            if not secondary or secondary == primary:
                secondary = next((x for x in ("H", "D", "A") if x != primary), "D")
            parsed["wdl_prediction"] = {
                "primary": primary,
                "secondary": secondary,
                "confidence": confidence or "中",
            }

    kps = parsed.get("key_points")
    if not isinstance(kps, list) or not kps:
        parsed["key_points"] = _synth_key_points(parsed, intel_summary)

    return parsed


def _fallback_from_models(ml_result: Optional[dict],
                          monte_carlo_result: Optional[dict],
                          err: str) -> dict:
    """LLM 不可用时，用 ML + 蒙特卡洛合成可推送结果，确保赛前仍能收到预测。"""
    primary, secondary, confidence = _wdl_from_probs(ml_result)
    mc = monte_carlo_result or {}

    # 比分：取蒙特卡洛 top2
    score_predictions = []
    dist = mc.get("score_distribution") or {}
    if dist:
        for score, prob in list(dist.items())[:2]:
            score_predictions.append({
                "score": str(score).replace("-", ":"),
                "prob": _to_float(prob),
                "reason": "蒙特卡洛模拟高频比分",
            })
    elif mc.get("most_likely_score"):
        score_predictions.append({
            "score": str(mc["most_likely_score"]).replace("-", ":"),
            "prob": 0.0,
            "reason": "蒙特卡洛最可能比分",
        })

    key_points = ["⚠️ LLM 暂不可用，以下为 ML+蒙特卡洛 模型综合结果（未含赛前新闻分析）"]
    if mc.get("most_likely_score"):
        key_points.append(f"蒙特卡洛最可能比分 {mc['most_likely_score']}")

    return {
        "wdl_prediction": {
            "primary": primary or "D",
            "secondary": secondary or "H",
            "confidence": confidence or "低",
        },
        "score_predictions": score_predictions,
        "upset_prediction": None,
        "key_points": key_points,
        "overall_analysis": f"LLM 调用失败（{err}），已降级为模型综合输出。",
        "error": err,
        "degraded": True,
    }
