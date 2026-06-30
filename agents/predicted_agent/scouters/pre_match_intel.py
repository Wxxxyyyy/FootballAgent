# -*- coding: utf-8 -*-
"""
赛前情报聚合器（Pre-Match Intel）

统一调度各采集器，聚合输出结构化赛前情报，接入 advance_predictor 流程。

情报维度:
  1. 伤停 + 红黄牌停赛（injury_suspension_scouter）
  2. 首发阵容两档（lineup_scouter）
     - 一档: 赛前1h外 → 核心缺阵 + 惯用阵型
     - 二档: 赛前1h内 → 官方首发 + 阵型
  3. 赛前新闻 + 软信号（news_scouter）
  4. 教练风格 + 惯用阵型（coach_style_scouter）

调用方式:
    intel = PreMatchIntel()
    report = intel.gather("Brazil", "Germany", date="2026-06-25", hours_to_kickoff=3)
    # hours_to_kickoff > 1 → 第一档; <= 1 → 第二档
"""

import logging
import time
from datetime import datetime
from typing import Optional

from agents.predicted_agent.scouters.national_team_config import (
    resolve_national_team, get_team_info, to_chinese,
)
from agents.predicted_agent.scouters.injury_suspension_scouter import (
    get_injuries_and_suspensions, summarize_absences,
)
from agents.predicted_agent.scouters.lineup_scouter import (
    get_lineup_intel, summarize_lineup,
)
from agents.predicted_agent.scouters.news_scouter import (
    get_pre_match_news, summarize_news,
)
from agents.predicted_agent.scouters.coach_style_scouter import (
    get_coach_style, compare_styles,
)

logger = logging.getLogger(__name__)


class PreMatchIntel:
    """赛前情报聚合器"""

    def __init__(self, timeout_per_source: int = 20):
        """
        Args:
            timeout_per_source: 每个数据源的最大超时时间（秒）
        """
        self.timeout = timeout_per_source

    def gather(
        self,
        home_team: str,
        away_team: str,
        date: str = None,
        hours_to_kickoff: float = 24,
    ) -> dict:
        """
        采集完整赛前情报

        Args:
            home_team: 主队名（中文/英文/别名均可）
            away_team: 客队名
            date: 比赛日期 YYYY-MM-DD
            hours_to_kickoff: 距离开赛还有多少小时
                              >1 → 第一档阵容, <=1 → 第二档阵容

        Returns:
            {
                "home_team": "Brazil",
                "away_team": "Germany",
                "date": "2026-06-25",
                "tier": 1 | 2,
                "lineup_intel": {...},       # 首发阵容情报
                "coach_style": {...},        # 教练风格对比
                "news": {                    # 赛前新闻
                    "home": {...},
                    "away": {...},
                },
                "summary": "...",            # 文本摘要（供 LLM 消费）
                "elapsed_seconds": float,
            }
        """
        ts_start = time.time()

        # ── 统一球队名 ──
        home_en = resolve_national_team(home_team) or home_team
        away_en = resolve_national_team(away_team) or away_team

        home_zh = to_chinese(home_en) or home_en
        away_zh = to_chinese(away_en) or away_en

        # 确定档位
        tier = 2 if hours_to_kickoff <= 1 else 1
        tier_desc = "第二档（官方首发）" if tier == 2 else "第一档（缺阵预判）"

        print("=" * 60)
        print(f"[赛前情报] {home_zh} vs {away_zh} ({date or '未指定'})")
        print(f"[赛前情报] 阵容档位: {tier_desc}（距开赛 {hours_to_kickoff}h）")
        print("=" * 60)

        # ── 1. 首发阵容情报 ──
        print("\n[1/4] 采集首发阵容情报...")
        try:
            lineup_intel = get_lineup_intel(home_en, away_en, date, tier=tier)
        except Exception as e:
            logger.error(f"[赛前情报] 首发阵容采集失败: {e}")
            lineup_intel = {"tier": tier, "home": {}, "away": {}}

        # ── 2. 教练风格对比 ──
        print("[2/4] 采集教练风格...")
        try:
            coach_matchup = compare_styles(home_en, away_en)
        except Exception as e:
            logger.error(f"[赛前情报] 教练风格采集失败: {e}")
            coach_matchup = {"home": None, "away": None, "tactical_matchup": ""}

        # ── 3. 赛前新闻（双方）──
        print("[3/4] 采集赛前新闻...")
        news_data = {}
        for side, team_en in [("home", home_en), ("away", away_en)]:
            try:
                news_data[side] = get_pre_match_news(team_en, limit=8)
            except Exception as e:
                logger.error(f"[赛前情报] {team_en} 新闻采集失败: {e}")
                news_data[side] = {"news": [], "soft_signals": []}

        # ── 4. 生成文本摘要 ──
        print("[4/4] 生成情报摘要...")
        summary = self._build_summary(
            home_en, away_en, home_zh, away_zh,
            lineup_intel, coach_matchup, news_data,
        )

        elapsed = time.time() - ts_start
        print(f"\n[赛前情报] 采集完成，耗时 {elapsed:.1f}s")

        return {
            "home_team": home_en,
            "away_team": away_en,
            "home_team_zh": home_zh,
            "away_team_zh": away_zh,
            "date": date,
            "tier": tier,
            "lineup_intel": lineup_intel,
            "coach_style": coach_matchup,
            "news": news_data,
            "summary": summary,
            "elapsed_seconds": round(elapsed, 1),
        }

    # ═══════════════════════════════════════════════════════════════
    #  情报摘要生成（供 LLM 消费）
    # ═══════════════════════════════════════════════════════════════

    def _build_summary(
        self,
        home_en: str, away_en: str,
        home_zh: str, away_zh: str,
        lineup_intel: dict,
        coach_matchup: dict,
        news_data: dict,
    ) -> str:
        """
        生成结构化情报摘要文本

        这段文本会直接注入 LLM 的 prompt，作为赛前情报输入。
        """
        lines = [
            f"═══ 赛前情报（{home_zh} vs {away_zh}）═══",
            "",
        ]

        # ── 阵容情报 ──
        tier = lineup_intel.get("tier", 1)
        tier_label = "官方首发" if tier == 2 else "缺阵预判"
        lines.append(f"【阵容情报 - {tier_label}】")
        lines.append(summarize_lineup(home_en, away_en, lineup_intel))
        lines.append("")

        # ── 教练风格 ──
        lines.append("【教练风格对比】")
        home_coach = coach_matchup.get("home", {})
        away_coach = coach_matchup.get("away", {})
        if home_coach and home_coach.get("data_available"):
            lines.append(
                f"{home_zh} - {home_coach['coach']}: "
                f"惯用{home_coach['preferred_formation']}, "
                f"风格{home_coach['style']}({home_coach['tendency']})"
            )
        if away_coach and away_coach.get("data_available"):
            lines.append(
                f"{away_zh} - {away_coach['coach']}: "
                f"惯用{away_coach['preferred_formation']}, "
                f"风格{away_coach['style']}({away_coach['tendency']})"
            )
        matchup = coach_matchup.get("tactical_matchup", "")
        if matchup:
            lines.append(f"战术对碰: {matchup}")
        lines.append("")

        # ── 赛前新闻 ──
        lines.append("【赛前新闻】")
        home_news = news_data.get("home", {})
        away_news = news_data.get("away", {})
        lines.append(summarize_news(home_en, home_news, max_items=4))
        lines.append("")
        lines.append(summarize_news(away_en, away_news, max_items=4))
        lines.append("")

        # ── 软信号汇总 ──
        home_signals = home_news.get("soft_signals", [])
        away_signals = away_news.get("soft_signals", [])
        if home_signals or away_signals:
            lines.append("【软信号汇总】")
            if home_signals:
                lines.append(f"{home_zh}:")
                for s in home_signals:
                    lines.append(f"  [{s['severity']}] {s['type']}: {s['desc']}")
            if away_signals:
                lines.append(f"{away_zh}:")
                for s in away_signals:
                    lines.append(f"  [{s['severity']}] {s['type']}: {s['desc']}")
            lines.append("")

        lines.append("═══ 情报结束 ═══")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════
    #  便捷方法：只采集伤停（用于 advance_predictor 的爆冷信号）
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def get_absences_quick(team_en: str) -> dict:
        """快速获取单支球队伤停信息（不爬新闻/首发）"""
        return get_injuries_and_suspensions(team_en)

    @staticmethod
    def get_coach_quick(team_en: str) -> dict:
        """快速获取单支球队教练风格"""
        return get_coach_style(team_en)


# ======================== 模块级单例 ========================

_intel: Optional[PreMatchIntel] = None


def get_intel() -> PreMatchIntel:
    """获取 PreMatchIntel 单例"""
    global _intel
    if _intel is None:
        _intel = PreMatchIntel()
    return _intel


# ======================== 测试 ========================

if __name__ == "__main__":
    print("=== 赛前情报聚合器测试 ===\n")

    intel = PreMatchIntel()

    # 第一档测试（赛前1h外）
    report = intel.gather("Brazil", "Germany", date="2026-06-25", hours_to_kickoff=24)
    print("\n" + "=" * 60)
    print("情报摘要:")
    print("=" * 60)
    print(report["summary"])
