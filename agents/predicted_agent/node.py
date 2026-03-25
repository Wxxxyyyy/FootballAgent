# -*- coding: utf-8 -*-
"""
预测 Agent 节点

流程:
  1. 从用户消息中提取球队名(中/英文) + 日期
  2. 若提取到两支球队 -> 启动赛前预测流水线
  3. 若信息不足 -> 设置对话锁，等待用户补充
  4. 将预测结果格式化后写入 raw_agent_response -> 交给 summary_agent 包装
"""

import re
from datetime import datetime, timedelta
from typing import Optional

from agents.states import AgentState


# ═══════════════════════════════════════════════════════════════
#  球队名提取（复用 neo4j_tools 的中英文映射）
# ═══════════════════════════════════════════════════════════════

_name_map: Optional[dict] = None


def _load_name_map() -> dict:
    """加载中英文球队名映射（懒加载单例）"""
    global _name_map
    if _name_map is not None:
        return _name_map

    import os
    import csv
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_path = os.path.join(project_root, "data", "English2Chinese", "\u4e2d\u82f1\u6587\u5bf9\u7167.csv")

    _name_map = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                en = row.get("ClubName", "").strip()
                if not en:
                    continue
                _name_map[en] = en
                _name_map[en.lower()] = en

                zh = row.get("ClubNameZh", "").strip()
                if zh:
                    _name_map[zh] = en

                alias_raw = row.get("AliasZh", "").strip()
                if alias_raw:
                    for a in alias_raw.replace("\uff0c", ",").split(","):
                        a = a.strip()
                        if a and a != zh:
                            _name_map[a] = en
    except Exception as e:
        print(f"[predicted_agent] \u52a0\u8f7d\u7403\u961f\u540d\u6620\u5c04\u5931\u8d25: {e}")

    return _name_map


def _extract_teams(text: str) -> list[str]:
    """从文本中提取球队英文名列表，按在原文中出现的先后顺序返回"""
    nm = _load_name_map()
    sorted_keys = sorted(nm.keys(), key=len, reverse=True)

    found = []
    used_ranges = []

    for key in sorted_keys:
        pos = text.find(key)
        if pos == -1:
            continue
        end = pos + len(key)
        if any(s < end and pos < e for s, e in used_ranges):
            continue
        en = nm[key]
        if en not in [t[1] for t in found]:
            found.append((pos, en))
            used_ranges.append((pos, end))

    found.sort(key=lambda x: x[0])
    return [name for _, name in found]


def _extract_date(text: str) -> Optional[str]:
    """从文本中提取比赛日期 -> YYYY-MM-DD"""
    # 精确日期 "2026-03-20" 或 "2026/03/20" 或 "3月20日"
    m = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    m = re.search(r'(\d{1,2})\u6708(\d{1,2})[\u65e5\u53f7]?', text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        year = datetime.now().year
        return f"{year}-{month:02d}-{day:02d}"

    today = datetime.now()
    if re.search(r'\u4eca\u5929|\u4eca\u665a', text):
        return today.strftime("%Y-%m-%d")
    if re.search(r'\u660e\u5929|\u660e\u665a', text):
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    if re.search(r'\u540e\u5929', text):
        return (today + timedelta(days=2)).strftime("%Y-%m-%d")

    weekday_map = {
        "\u5468\u4e00": 0, "\u5468\u4e8c": 1, "\u5468\u4e09": 2, "\u5468\u56db": 3,
        "\u5468\u4e94": 4, "\u5468\u516d": 5, "\u5468\u65e5": 6, "\u5468\u5929": 6,
        "\u661f\u671f\u4e00": 0, "\u661f\u671f\u4e8c": 1, "\u661f\u671f\u4e09": 2,
        "\u661f\u671f\u56db": 3, "\u661f\u671f\u4e94": 4, "\u661f\u671f\u516d": 5,
        "\u661f\u671f\u5929": 6, "\u661f\u671f\u65e5": 6,
    }
    for label, wd in weekday_map.items():
        prefix = "\u672c" if re.search(rf'\u672c{label}|\u8fd9{label}', text) else ""
        if prefix or label in text:
            diff = (wd - today.weekday()) % 7
            if diff == 0:
                diff = 7
            target = today + timedelta(days=diff)
            return target.strftime("%Y-%m-%d")

    return None


# ═══════════════════════════════════════════════════════════════
#  预测结果 -> 可读文本
# ═══════════════════════════════════════════════════════════════

def _format_prediction(result: dict) -> str:
    """将结构化预测结果转为 summary_agent 可消费的文本"""
    home = result.get("home_team", "?")
    away = result.get("away_team", "?")
    date = result.get("date", "")

    lines = [
        f"\u3010\u8d5b\u524d\u9884\u6d4b\u5206\u6790\u3011{home} vs {away}"
        + (f" ({date})" if date else ""),
        "",
    ]

    # ML 模型
    ml = result.get("ml_prediction", {})
    if ml and "error" not in ml:
        lines.append("## ML \u6a21\u578b\u57fa\u51c6\u6982\u7387")
        lines.append(f"- \u4e3b\u80dc\u6982\u7387: {ml.get('home_win_prob', 'N/A')}")
        lines.append(f"- \u5e73\u5c40\u6982\u7387: {ml.get('draw_prob', 'N/A')}")
        lines.append(f"- \u5ba2\u80dc\u6982\u7387: {ml.get('away_win_prob', 'N/A')}")
        lines.append(f"- \u5927 2.5 \u7403\u6982\u7387: {ml.get('over25_prob', 'N/A')}")
        lines.append(f"- \u5c0f 2.5 \u7403\u6982\u7387: {ml.get('under25_prob', 'N/A')}")
        lines.append(f"- \u8d54\u7387\u6765\u6e90: {ml.get('odds_source', 'N/A')}")
        lines.append("")

    # LLM 分析
    llm = result.get("llm_analysis", {})
    if llm and "error" not in llm:
        wdl = llm.get("wdl_prediction", {})
        if wdl:
            pri_label = {"H": "\u4e3b\u80dc", "D": "\u5e73\u5c40", "A": "\u5ba2\u80dc"}.get(
                wdl.get("primary", ""), wdl.get("primary", ""))
            sec_label = {"H": "\u4e3b\u80dc", "D": "\u5e73\u5c40", "A": "\u5ba2\u80dc"}.get(
                wdl.get("secondary", ""), wdl.get("secondary", ""))
            lines.append("## \u80dc\u5e73\u8d1f\u9884\u6d4b")
            lines.append(f"- \u6700\u53ef\u80fd: {pri_label} "
                         f"(\u6982\u7387 ~{wdl.get('primary_prob', '?')}) "
                         f"- {wdl.get('primary_reason', '')}")
            lines.append(f"- \u6b21\u53ef\u80fd: {sec_label} "
                         f"(\u6982\u7387 ~{wdl.get('secondary_prob', '?')}) "
                         f"- {wdl.get('secondary_reason', '')}")
            lines.append(f"- \u7f6e\u4fe1\u5ea6: {wdl.get('confidence', '?')}")
            lines.append("")

        ou = llm.get("ou_prediction", {})
        if ou:
            ou_label = "\u5927 2.5 \u7403" if ou.get("result") == "Over" else "\u5c0f 2.5 \u7403"
            lines.append("## \u5927\u5c0f\u7403\u9884\u6d4b")
            lines.append(f"- \u9884\u6d4b: {ou_label} (\u6982\u7387 ~{ou.get('prob', '?')})")
            lines.append(f"- \u7406\u7531: {ou.get('reason', '')}")
            lines.append(f"- \u7f6e\u4fe1\u5ea6: {ou.get('confidence', '?')}")
            lines.append("")

        scores = llm.get("score_predictions", [])
        if scores:
            lines.append("## \u6bd4\u5206\u9884\u6d4b")
            for i, s in enumerate(scores, 1):
                lines.append(f"- {i}. {s.get('score', '?')} "
                             f"(\u6982\u7387 {s.get('prob', '?')}) "
                             f"- {s.get('reason', '')}")
            lines.append("")

        analysis = llm.get("overall_analysis", "")
        if analysis:
            lines.append("## \u7efc\u5408\u5206\u6790")
            lines.append(analysis)
            lines.append("")

    elif llm and "error" in llm:
        lines.append(f"## LLM \u5206\u6790\n\u8c03\u7528\u5931\u8d25: {llm['error']}")
        lines.append("")

    # 爆冷预警
    upset = llm.get("upset_alert") if llm else None
    if upset and isinstance(upset, dict) and upset.get("triggered"):
        upset_wdl_label = {"H": "\u4e3b\u80dc", "D": "\u5e73\u5c40", "A": "\u5ba2\u80dc"}.get(
            upset.get("upset_wdl", ""), upset.get("upset_wdl", "?"))
        level = upset.get("level", "")
        lines.append(f"## \u26a0\ufe0f \u7206\u51b7\u9884\u8b66 [{level}]")
        lines.append(f"- \u7206\u51b7\u7ed3\u679c: {upset_wdl_label}")
        lines.append(f"- \u7406\u7531: {upset.get('upset_reason', '')}")
        upset_scores = upset.get("upset_scores", [])
        if upset_scores:
            lines.append("- \u7206\u51b7\u6bd4\u5206:")
            for us in upset_scores:
                lines.append(f"  - {us.get('score', '?')} - {us.get('reason', '')}")

        # 补充系统检测到的信号明细
        sig_data = result.get("upset_signals", {})
        sigs = sig_data.get("signals", [])
        if sigs:
            lines.append("- \u89e6\u53d1\u4fe1\u53f7:")
            for s in sigs:
                lines.append(f"  - [{s['severity']}] {s['type']}: {s['desc']}")
        lines.append("")

    # 数据来源
    oc = result.get("openclaw_summary", {})
    h2h = result.get("h2h_records", [])
    lines.append("## \u6570\u636e\u6765\u6e90")
    lines.append(f"- \u8d54\u7387 + \u8fd15\u573a: OpenClaw"
                 f" ({'已获取' if oc.get('status') == 'ok' else '未获取'})")
    lines.append(f"- \u5386\u53f2\u4ea4\u950b: Neo4j \u77e5\u8bc6\u56fe\u8c31"
                 f" ({len(h2h)} \u6761\u8bb0\u5f55)")
    lines.append(f"- \u8017\u65f6: {result.get('elapsed_seconds', '?')}s")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  节点函数
# ═══════════════════════════════════════════════════════════════

def predicted_agent_node(state: AgentState) -> dict:
    """
    预测 Agent 节点

    从用户消息提取球队 + 日期 -> 调用赛前预测流水线 -> 输出结果
    若信息不足则设置对话锁等待用户补充
    """
    user_msg = state["messages"][-1].content if state["messages"] else ""
    dialog_state = state.get("dialog_state", "normal")

    print("=" * 60)
    print(f"[predicted_agent] \u8fdb\u5165\u9884\u6d4b\u8282\u70b9")
    print(f"  \u7528\u6237\u8f93\u5165: '{user_msg}'")
    print(f"  \u5bf9\u8bdd\u72b6\u6001: {dialog_state}")
    print("=" * 60)

    # ── 提取球队和日期 ──
    teams = _extract_teams(user_msg)
    date = _extract_date(user_msg)

    print(f"  \u63d0\u53d6\u7403\u961f: {teams}")
    print(f"  \u63d0\u53d6\u65e5\u671f: {date}")

    # ── 信息不足: 要求用户提供 ──
    if len(teams) < 2:
        hint_parts = []
        if len(teams) == 0:
            hint_parts.append("\u8bf7\u63d0\u4f9b\u4e3b\u961f\u548c\u5ba2\u961f\u540d\u79f0")
        elif len(teams) == 1:
            hint_parts.append(f"\u5df2\u8bc6\u522b\u5230 {teams[0]}\uff0c\u8bf7\u63d0\u4f9b\u5bf9\u624b\u7403\u961f\u540d\u79f0")
        if not date:
            hint_parts.append("\u53ef\u4ee5\u8865\u5145\u6bd4\u8d5b\u65e5\u671f")

        msg = (
            f"\u9700\u8981\u66f4\u591a\u4fe1\u606f\u624d\u80fd\u8fdb\u884c\u9884\u6d4b\u3002\n"
            f"{'；'.join(hint_parts)}\u3002\n\n"
            f"\u793a\u4f8b: \u201c\u5e2e\u6211\u9884\u6d4b\u8d5b\u524d Arsenal vs Chelsea 3\u670820\u65e5\u7684\u6bd4\u8d5b\u201d\n"
            f"\u6216: \u201c\u62dc\u4ec1\u5bf9\u591a\u7279\u5468\u516d\u7684\u6bd4\u8d5b\u201d"
        )

        return {
            "raw_agent_response": msg,
            "dialog_state": "waiting_prediction_input",
        }

    home_team = teams[0]
    away_team = teams[1]

    # ── 执行赛前预测 ──
    try:
        from agents.predicted_agent.advance_predictor import get_predictor
        predictor = get_predictor()
        result = predictor.predict(home_team, away_team, date)
        formatted = _format_prediction(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        formatted = (
            f"\u8d5b\u524d\u9884\u6d4b\u6267\u884c\u5f02\u5e38: {e}\n\n"
            f"\u8bf7\u786e\u8ba4:\n"
            f"1. OpenClaw \u65e7\u7535\u8111\u5df2\u542f\u52a8\n"
            f"2. \u4e2d\u7ee7\u7ad9\u5df2\u542f\u52a8\n"
            f"3. SSH \u53cd\u5411\u96a7\u9053\u5df2\u5f00\u542f"
        )

    return {
        "raw_agent_response": formatted,
        "dialog_state": "normal",
    }
