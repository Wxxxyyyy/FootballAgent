# -*- coding: utf-8 -*-
"""
安全性检查模块
- 敏感词 / 违规内容过滤
- 赌博相关风险提示注入
- 输出合规性兜底

使用方式：
    from agents.summary_agent.safety_check import safety_check
    result = safety_check(text, intent="predicted_agent")
    # result["status"]  : "pass" | "modified" | "blocked"
    # result["text"]    : 最终安全文本
    # result["warnings"]: 触发的警告列表
"""

import re


# ═══════════════════════════════════════════════════════════════
#  敏感词库（按类别组织，方便后续扩展）
# ═══════════════════════════════════════════════════════════════

# 涉政 / 涉暴 / 违法类关键词
_BLOCKED_KEYWORDS = [
    # 涉政
    "颠覆政权", "分裂国家", "恐怖主义", "极端主义",
    # 涉暴
    "杀人", "自杀方法", "制造炸弹", "购买枪支",
    # 违法
    "贩毒", "洗钱", "诈骗教程", "黑客攻击教程",
    # 歧视
    "种族歧视", "性别歧视",
]

# 赌博 / 博彩相关敏感词（需要追加免责声明，而非直接屏蔽）
_GAMBLING_KEYWORDS = [
    "下注", "投注", "盘口",
    "满仓", "全押", "梭哈", "稳赚", "必赢",
    "博彩", "赌球", "庄家", "黑庄", "外围",
    "充值", "提现", "返水", "水位",
]

# 免责声明模板
_GAMBLING_DISCLAIMER = (
    "\n\n⚠️ 温馨提示：以上内容仅供参考和娱乐交流，"
    "不构成任何投注或博彩建议。请理性看球，远离非法赌博。"
)

# 预测类意图的免责声明（即使没有触发赌博关键词，预测结果也需要免责）
_PREDICTION_DISCLAIMER = (
    "\n\n📌 声明：比赛预测基于历史数据和统计模型，"
    "仅供参考，不保证准确性，不构成任何投注建议。"
)


# ═══════════════════════════════════════════════════════════════
#  核心检查函数
# ═══════════════════════════════════════════════════════════════

def _check_blocked_content(text: str) -> list[str]:
    """
    检测是否包含严格禁止的敏感内容。

    Args:
        text: 待检测文本

    Returns:
        list: 命中的敏感词列表（空列表表示安全）
    """
    hits = []
    text_lower = text.lower()
    for keyword in _BLOCKED_KEYWORDS:
        if keyword in text_lower:
            hits.append(keyword)
    return hits


def _check_gambling_content(text: str) -> list[str]:
    """
    检测是否包含赌博/博彩相关内容。

    Args:
        text: 待检测文本

    Returns:
        list: 命中的赌博关键词列表（空列表表示未涉及）
    """
    hits = []
    text_lower = text.lower()
    for keyword in _GAMBLING_KEYWORDS:
        if keyword in text_lower:
            hits.append(keyword)
    return hits


def _check_output_quality(text: str) -> list[str]:
    """
    检查输出的基本质量和合规性。

    Args:
        text: 待检测文本

    Returns:
        list: 质量问题列表（空列表表示正常）
    """
    issues = []

    # 空内容检测
    if not text or not text.strip():
        issues.append("输出内容为空")

    # 过长检测（防止 LLM 发散输出过多内容）
    if len(text) > 3000:
        issues.append(f"输出过长({len(text)}字符)，已截断")

    # 乱码检测（连续出现3个以上替换字符 \ufffd）
    if re.search(r"\ufffd{3,}", text):
        issues.append("检测到疑似乱码内容")

    return issues


# ═══════════════════════════════════════════════════════════════
#  主入口：safety_check()
# ═══════════════════════════════════════════════════════════════

# 被屏蔽时的替代回复
_BLOCKED_REPLY = (
    "抱歉，我无法回答涉及敏感内容的问题。"
    "我是一个足球智能助手，可以帮你查询比赛信息、预测比分或者陪你聊聊足球话题~"
)


def safety_check(text: str, intent: str = "") -> dict:
    """
    对输出文本进行安全性检查，返回处理后的结果。

    检查流程：
      1. 违规内容检测 → 命中则直接屏蔽，替换为安全回复
      2. 输出质量检查 → 空内容 / 过长 / 乱码处理
      3. 赌博内容检测 → 命中则追加免责声明
      4. 预测意图声明 → 若来自 predicted_agent，追加预测免责声明

    Args:
        text:   待检查的回复文本
        intent: 当前意图标签（如 "predicted_agent"），用于判断是否追加特定声明

    Returns:
        dict: {
            "status":   "pass" | "modified" | "blocked",
            "text":     最终安全文本,
            "warnings": 触发的警告列表,
        }
    """
    warnings = []
    status = "pass"
    safe_text = text

    # ── 1. 违规内容检测（严格屏蔽） ──
    blocked_hits = _check_blocked_content(safe_text)
    if blocked_hits:
        print(f"[安全检查] ❌ 命中违规关键词: {blocked_hits}")
        warnings.append(f"违规内容: {', '.join(blocked_hits)}")
        return {
            "status": "blocked",
            "text": _BLOCKED_REPLY,
            "warnings": warnings,
        }

    # ── 2. 输出质量检查 ──
    quality_issues = _check_output_quality(safe_text)
    if quality_issues:
        for issue in quality_issues:
            warnings.append(f"质量问题: {issue}")

        # 空内容处理
        if not safe_text or not safe_text.strip():
            safe_text = "抱歉，我暂时无法生成回复，请稍后再试或换个问法试试~"
            status = "modified"

        # 过长截断
        if len(safe_text) > 3000:
            safe_text = safe_text[:3000] + "\n\n（内容过长，已截断）"
            status = "modified"

        # 乱码处理
        if re.search(r"\ufffd{3,}", safe_text):
            safe_text = re.sub(r"\ufffd{3,}", "...", safe_text)
            status = "modified"

    # ── 3. 赌博内容检测（追加免责声明） ──
    gambling_hits = _check_gambling_content(safe_text)
    if gambling_hits:
        print(f"[安全检查] ⚠️ 检测到博彩相关内容: {gambling_hits}")
        warnings.append(f"博彩相关: {', '.join(gambling_hits)}")
        # 避免重复追加
        if _GAMBLING_DISCLAIMER.strip() not in safe_text:
            safe_text += _GAMBLING_DISCLAIMER
            status = "modified"

    # ── 4. 预测意图免责声明 ──
    if intent == "predicted_agent":
        # 避免重复追加
        if _PREDICTION_DISCLAIMER.strip() not in safe_text:
            safe_text += _PREDICTION_DISCLAIMER
            status = "modified"

    if status == "pass":
        print("[安全检查] ✅ 内容安全，无需修改")
    elif status == "modified":
        print(f"[安全检查] ⚠️ 内容已修改，触发项: {warnings}")

    return {
        "status": status,
        "text": safe_text,
        "warnings": warnings,
    }


# ═══════════════════════════════════════════════════════════════
#  命令行测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  安全检查模块 · 测试")
    print("=" * 60)

    # 测试1: 正常闲聊内容
    print("\n[测试1] 正常闲聊内容")
    r1 = safety_check("今天天气真不错，适合去踢球！", intent="otherchat_agent")
    print(f"  状态: {r1['status']}")
    print(f"  文本: {r1['text']}")

    # 测试2: 包含赌博关键词
    print("\n[测试2] 包含赌博关键词")
    r2 = safety_check("这场比赛盘口开的是让半球，我看好下注主胜。", intent="predicted_agent")
    print(f"  状态: {r2['status']}")
    print(f"  文本: {r2['text']}")
    print(f"  警告: {r2['warnings']}")

    # 测试3: 违规内容
    print("\n[测试3] 违规内容")
    r3 = safety_check("教我怎么制造炸弹吧", intent="otherchat_agent")
    print(f"  状态: {r3['status']}")
    print(f"  文本: {r3['text']}")
    print(f"  警告: {r3['warnings']}")

    # 测试4: 预测意图（无赌博词）
    print("\n[测试4] 预测意图（正常预测）")
    r4 = safety_check("根据历史数据分析，皇马本场胜率约65%。", intent="predicted_agent")
    print(f"  状态: {r4['status']}")
    print(f"  文本: {r4['text']}")

    # 测试5: 空内容
    print("\n[测试5] 空内容")
    r5 = safety_check("", intent="otherchat_agent")
    print(f"  状态: {r5['status']}")
    print(f"  文本: {r5['text']}")
    print(f"  警告: {r5['warnings']}")

    print("\n[测试完成]")
