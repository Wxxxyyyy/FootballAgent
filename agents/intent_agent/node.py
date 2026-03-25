# -*- coding: utf-8 -*-
"""
意图识别节点
- 加载微调后的 BERT 模型，分析用户输入的意图
- 根据置信度阈值进行路由分发：
  - predicted_agent   （比分/赛果预测类）
  - information_agent （信息查询类）
  - otherchat_agent   （闲聊/其他 & 低置信度兜底）
"""

import os
import sys

# 确保项目根目录在 sys.path 中，以便导入 intent 模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from intent.predict import IntentClassifier

# ═══════════════════════════════════════════════════════════════
#  置信度阈值 —— 低于此值时触发兜底逻辑，强制路由到 otherchat_agent
# ═══════════════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD = 0.7

# 兜底意图标签
FALLBACK_INTENT = "otherchat_agent"

# ─── 全局单例：避免重复加载模型 ──────────────────────────────────
_classifier_instance: IntentClassifier | None = None


def _get_classifier() -> IntentClassifier:
    """
    获取意图分类器的全局单例。
    首次调用时加载模型，后续调用复用同一实例，节省显存/内存。
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance


# ═══════════════════════════════════════════════════════════════
#  核心路由函数
# ═══════════════════════════════════════════════════════════════

def intent_route(user_input: str) -> dict:
    """
    意图识别路由函数 —— 意图识别节点的核心逻辑。

    流程：
      1. 调用 BERT 模型预测用户输入的意图和置信度
      2. 若 confidence >= CONFIDENCE_THRESHOLD (0.7)，返回模型预测的 intent
      3. 若 confidence <  CONFIDENCE_THRESHOLD (0.7)，触发兜底，强制返回 'otherchat_agent'

    Args:
        user_input: 用户输入的原始文本

    Returns:
        dict: {
            "intent":      最终路由的意图标签（str）,
            "confidence":  模型原始置信度（float）,
            "all_scores":  各类别概率分布（dict）,
            "is_fallback": 是否触发了兜底逻辑（bool）
        }
    """
    classifier = _get_classifier()

    # 调用 BERT 模型进行预测
    prediction = classifier.predict(user_input)

    raw_intent = prediction["intent"]
    confidence = prediction["confidence"]
    all_scores = prediction["all_scores"]

    # 置信度阈值判断
    if confidence >= CONFIDENCE_THRESHOLD:
        # 高置信度 —— 信任模型预测结果
        final_intent = raw_intent
        is_fallback = False
    else:
        # 低置信度 —— 模型不确定，触发兜底逻辑，强制路由到闲聊/其他
        final_intent = FALLBACK_INTENT
        is_fallback = True

    result = {
        "intent": final_intent,
        "confidence": confidence,
        "all_scores": all_scores,
        "is_fallback": is_fallback,
    }

    # 打印路由日志
    fallback_tag = " [兜底]" if is_fallback else ""
    print(f"[意图识别] 输入: '{user_input}'")
    print(f"  → 模型预测: {raw_intent} (置信度: {confidence:.4f})")
    print(f"  → 最终路由: {final_intent}{fallback_tag}")
    print(f"  → 概率分布: {all_scores}")

    return result


# ═══════════════════════════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  意图识别节点 · 路由测试")
    print(f"  置信度阈值: {CONFIDENCE_THRESHOLD}")
    print("=" * 70)

    # 测试用例：覆盖三类意图 + 模糊表达触发兜底
    test_cases = [
        # ── 预测类（predicted_agent）──
        "这周末皇马打巴萨你觉得谁能赢？",
        "帮我预测一下阿森纳对切尔西的比分",
        "拜仁下场比赛能赢吗",
        "利物浦对曼城你看好谁",

        # ── 信息查询类（information_agent）──
        "曼联上赛季的战绩怎么样",
        "介绍一下巴塞罗那的历史",
        "皇马和巴萨上赛季交手几次",
        "AC米兰是哪个联赛的",

        # ── 闲聊/其他类（otherchat_agent）──
        "你好呀",
        "今天天气怎么样",
        "给我讲个笑话",
        "你是谁开发的",

        # ── 模糊表达（期望触发兜底逻辑）──
        "随便说说",
        "嗯",
        "哦",
        "啊？",
        "1234567",
        "...",
    ]

    print()
    for i, text in enumerate(test_cases, 1):
        print(f"[测试 {i:02d}]")
        result = intent_route(text)
        status = "✅ 正常分发" if not result["is_fallback"] else "⚠️ 兜底触发"
        print(f"  状态: {status}")
        print("-" * 70)
