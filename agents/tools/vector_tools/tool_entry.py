# -*- coding: utf-8 -*-
"""
向量知识检索工具 · @tool 统一入口
════════════════════════════════════
对外暴露 search_knowledge_base 函数，供 LangChain/LangGraph Agent 调用。

流程:
  1. 接收用户问题
  2. 调用 retriever.search_team_profiles 进行向量检索（含两道防线）
  3. 格式化结果为易于大模型 Summary 节点阅读的文本
"""

from langchain_core.tools import tool
from agents.tools.vector_tools.retriever import search_team_profiles


# ═══════════════════════════════════════════════════════════════
#  @tool 统一入口
# ═══════════════════════════════════════════════════════════════

@tool
def search_knowledge_base(question: str) -> str:
    """
    球队知识库检索工具 —— 用于查询球队的背景介绍、历史底蕴、别名等非结构化文本知识。

    输入的问题必须包含明确的球队名称或关键描述。
    适合回答如"阿森纳的历史底蕴"、"哪支球队联赛不败夺冠"、"皇马的背景介绍"等问题。

    数据来源: 五大联赛（英超/西甲/意甲/德甲/法甲）共 ~100 支球队的中文简介。

    Args:
        question: 用户的自然语言查询问题（应包含球队名称或具体描述）

    Returns:
        str: 格式化的检索结果文本，或未找到相关信息的提示
    """
    print(f"\n{'='*60}")
    print(f"[Vector Tool] 收到查询: {question}")
    print(f"{'='*60}")

    # ── 调用检索器（含两道防线） ──
    results = search_team_profiles(question)

    # ── 检索为空（被阈值拦截） ──
    if not results:
        print("[Vector Tool] 检索结果为空（全部被阈值拦截或无匹配）")
        return "[向量检索] 知识库中未找到与该问题高度相关的球队简介。"

    # ── 格式化为易于大模型阅读的文本 ──
    lines = []
    lines.append(f"[向量检索] 找到以下 {len(results)} 条相关球队信息：\n")

    for i, item in enumerate(results, 1):
        zh_name = item["club_name_zh"]
        en_name = item["club_name"]
        alias = item["alias_zh"]
        league = item["league"]
        intro = item["intro"]
        dist = item["distance"]

        # 组装球队标识行
        name_tag = f"{zh_name}({en_name})"
        if alias:
            name_tag += f"，别名: {alias}"

        lines.append(f"{i}. [{name_tag}] — 联赛: {league} | 相关度: {dist:.4f}")
        lines.append(f"   {intro}")
        lines.append("")

    formatted = "\n".join(lines).strip()
    print(f"[Vector Tool] 返回 {len(results)} 条结果")
    return formatted


# ═══════════════════════════════════════════════════════════════
#  命令行测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  向量知识检索工具 · 交互式测试")
    print("=" * 60)
    print("  输入自然语言问题检索球队知识库，输入 q 退出")
    print("=" * 60)

    while True:
        try:
            print("\n>> 请输入查询: ", end="", flush=True)
            q = sys.stdin.readline()
            if not q:
                break
            q = q.strip()
        except KeyboardInterrupt:
            print("\n再见！")
            break

        if q.lower() in ("q", "quit", "exit"):
            break

        if not q:
            continue

        result = search_knowledge_base.invoke(q)
        print(f"\n{result}")

