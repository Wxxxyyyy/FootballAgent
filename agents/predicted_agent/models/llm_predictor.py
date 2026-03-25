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

分析原则:
1. ML 模型概率是基于历史赔率训练的基准参考，但不是唯一依据
2. 近期战绩反映球队当前状态，赢球势头或连败低迷都很重要
3. 历史交锋体现两队之间的相生相克关系
4. 赔率反映了市场对比赛的综合判断
5. 主客场优势也是重要因素

输出要求: 必须严格以 JSON 格式输出，结构如下:
{
  "wdl_prediction": {
    "primary": "H 或 D 或 A",
    "primary_prob": 0.45,
    "primary_reason": "简要理由（中文，50字以内）",
    "secondary": "H 或 D 或 A",
    "secondary_prob": 0.30,
    "secondary_reason": "简要理由（中文，50字以内）",
    "confidence": "高/中/低"
  },
  "ou_prediction": {
    "result": "Over 或 Under",
    "prob": 0.55,
    "reason": "简要理由（中文，50字以内）",
    "confidence": "高/中/低"
  },
  "score_predictions": [
    {"score": "主队进球:客队进球", "prob": 0.15, "reason": "简要说明"},
    {"score": "主队进球:客队进球", "prob": 0.12, "reason": "简要说明"},
    {"score": "主队进球:客队进球", "prob": 0.10, "reason": "简要说明"}
  ],
  "overall_analysis": "综合分析（中文，100-200字）",
  "upset_alert": null
}

爆冷预警规则:
- 当数据中提供了"爆冷信号分析"段落且存在信号时，你需要综合判断是否真的有爆冷风险
- 如果你认为确实存在爆冷可能（概率≥15%），则填写 upset_alert 字段:
  "upset_alert": {
    "triggered": true,
    "level": "高危/中危",
    "upset_wdl": "H 或 D 或 A（爆冷结果）",
    "upset_reason": "爆冷理由（中文，80字以内）",
    "upset_scores": [
      {"score": "主队进球:客队进球", "reason": "简要说明"},
      {"score": "主队进球:客队进球", "reason": "简要说明"}
    ]
  }
- 如果你认为爆冷可能性很低，设为 null 即可
- upset_scores 给出两个爆冷比分（可以是平局）
- upset_wdl 必须与你的 primary 预测不同（否则不算爆冷）

注意:
- H=主胜, D=平局, A=客胜
- score 格式严格为 "主队进球:客队进球"（用英文冒号）
- prob 为 0-1 之间的小数
- 所有分析必须基于提供的数据"""


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
) -> dict:
    """
    调用 Kimi 2.5 进行综合赛前分析

    Returns:
        包含 wdl_prediction / ou_prediction / score_predictions /
        overall_analysis / upset_alert / raw_response 的字典
    """
    prompt = _build_prompt(
        home_team, away_team, date,
        ml_result, home_last_5, away_last_5,
        h2h_records, odds_info, upset_signals,
    )

    print(f"  [LLM] 调用 Kimi 2.5 ({os.getenv('LLM_MODEL_KIMI_NAME', 'kimi-k2.5')})...")

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )

        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL_KIMI_NAME", "kimi-k2.5"),
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
        parsed["raw_response"] = raw_text
        return parsed

    except Exception as e:
        print(f"  [LLM] 调用失败: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "raw_response": ""}


# ═══════════════════════════════════════════════════════════════
#  Prompt 构建
# ═══════════════════════════════════════════════════════════════

def _build_prompt(home_team, away_team, date,
                  ml_result, home_last_5, away_last_5,
                  h2h_records, odds_info, upset_signals=None) -> str:
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
            f"## 机器学习模型预测（RandomForest，基于历史赔率训练）\n"
            f"- 主胜概率: {ml_result.get('home_win_prob', 'N/A')}\n"
            f"- 平局概率: {ml_result.get('draw_prob', 'N/A')}\n"
            f"- 客胜概率: {ml_result.get('away_win_prob', 'N/A')}\n"
            f"- 大2.5球概率: {ml_result.get('over25_prob', 'N/A')}\n"
            f"- 小2.5球概率: {ml_result.get('under25_prob', 'N/A')}\n"
            f"- 模型倾向: {_wdl_label(ml_result.get('wdl_prediction', ''))}, "
            f"{ml_result.get('ou_prediction', '')}\n"
            f"- 赔率来源: {ml_result.get('odds_source', 'N/A')}"
        )
    else:
        err = ml_result.get("error", "未知") if ml_result else "无数据"
        parts.append(f"## 机器学习模型预测\n无法获取（{err}）")

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

    # 历史交锋
    if h2h_records:
        lines = [f"## {home_team} vs {away_team} 历史交锋（知识图谱）"]
        for r in h2h_records:
            lines.append(
                f"- {r.get('date', '?')} [{r.get('season', '?')}] "
                f"{r.get('result', '?')} (总进球: {r.get('total_goals', '?')})"
            )
        parts.append("\n".join(lines))
    else:
        parts.append(
            f"## 历史交锋\n未找到 {home_team} vs {away_team} 的交锋记录"
        )

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
