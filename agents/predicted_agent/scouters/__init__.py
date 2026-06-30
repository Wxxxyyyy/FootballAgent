# -*- coding: utf-8 -*-
"""
赛前情报采集模块（Scouters）

针对世界杯国家队比赛，采集以下赛前情报：
  1. 伤停 + 红黄牌停赛（Transfermarkt）
  2. 首发阵容（两档：一档停赛+伤员预判 / 二档官方首发+阵型）
  3. 赛前新闻（懂球帝，队内冲突/不和等软信息）
  4. 教练风格 + 惯用阵型（静态数据 + Transfermarkt 补充）

统一通过 pre_match_intel.PreMatchIntel 聚合输出，接入 advance_predictor 流程。
"""

from agents.predicted_agent.scouters.pre_match_intel import PreMatchIntel

__all__ = ["PreMatchIntel"]
