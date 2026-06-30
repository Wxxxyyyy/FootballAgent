# -*- coding: utf-8 -*-
"""
首发阵容采集器（两档）

第一档（赛前1小时外）:
  - 不预测官方首发，而是基于伤停+停赛生成"核心缺阵名单"
  - 配合教练惯用阵型，让 LLM 预判阵容影响
  - 数据来源: injury_suspension_scouter + coach_style_scouter

第二档（赛前1小时内）:
  - 爬取官方首发阵容 + 阵型
  - 数据来源: 直播吧 / FotMob（赛前60-90分钟发布）

输出格式:
  {
    "tier": 1,                        # 1=赛前1h外, 2=赛前1h内
    "home": {
      "formation": "4-3-3",           # 阵型（一档用惯用阵型，二档用实际阵型）
      "starting_xi": [...],           # 首发11人（仅二档有完整数据）
      "key_absences": [...],          # 核心缺阵名单（一档主要数据）
      "is_official": false,           # 是否官方首发
    },
    "away": {...}
  }
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from agents.predicted_agent.scouters.national_team_config import get_team_info, to_chinese
from agents.predicted_agent.scouters.injury_suspension_scouter import get_injuries_and_suspensions
from agents.predicted_agent.scouters.coach_style_scouter import get_coach_style

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 15
MAX_RETRIES = 2

_ZB8_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/125.0.0.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}


# ======================== 第一档：核心缺阵 + 惯用阵型 ========================

def get_tier1_lineup_intel(team_en: str) -> dict:
    """
    第一档：赛前1小时外的阵容情报

    不预测具体首发，而是提供:
      1. 核心缺阵名单（伤停+停赛）
      2. 教练惯用阵型（供 LLM 预判）
    """
    # 获取伤停+停赛
    absence_data = get_injuries_and_suspensions(team_en)
    key_absences = absence_data.get("key_absences", [])

    # 获取教练惯用阵型
    coach_style = get_coach_style(team_en)
    formation = coach_style.get("preferred_formation", "未知") if coach_style else "未知"

    return {
        "formation": formation,
        "formation_source": "教练惯用阵型",
        "starting_xi": [],  # 一档不预测具体首发
        "key_absences": key_absences,
        "is_official": False,
        "tier": 1,
    }


# ======================== 第二档：官方首发爬取 ========================

def _fetch_zhibo8_lineup(home_en: str, away_en: str, date: str = None) -> dict:
    """
    爬取直播吧官方首发阵容

    直播吧在赛前60-90分钟发布首发阵容，
    比赛页面 URL 需要根据赛程确定。

    此处先实现通用解析逻辑，具体比赛页面 URL 由调用方提供或通过搜索获取。
    """
    # 搜索直播吧世界杯比赛页面
    home_zh = to_chinese(home_en) or home_en
    away_zh = to_chinese(away_en) or away_en
    search_keyword = f"{home_zh} {away_zh}"

    # 直播吧搜索接口
    search_url = f"https://www.zhibo8.com/search/?q={search_keyword}"
    logger.info(f"[首发] 搜索直播吧比赛: {search_url}")

    match_url = None
    try:
        resp = httpx.get(search_url, headers=_ZB8_HEADERS, timeout=REQUEST_TIMEOUT,
                         follow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 从搜索结果中找比赛链接
        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(strip=True)
            if home_zh in text and away_zh in text and "/zuqiu/" in href:
                if not href.startswith("http"):
                    href = f"https://www.zhibo8.com{href}"
                match_url = href
                break
    except Exception as e:
        logger.warning(f"[首发] 直播吧搜索失败: {e}")

    if not match_url:
        logger.info(f"[首发] 未找到直播吧比赛页面，可能比赛尚未临近")
        return None

    # 爬取比赛页面的首发阵容
    logger.info(f"[首发] 爬取直播吧比赛页: {match_url}")
    try:
        resp = httpx.get(match_url, headers=_ZB8_HEADERS, timeout=REQUEST_TIMEOUT,
                         follow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"[首发] 直播吧比赛页请求失败: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    return _parse_zhibo8_lineup(soup, home_zh, away_zh)


def _parse_zhibo8_lineup(soup: BeautifulSoup, home_zh: str, away_zh: str) -> dict:
    """解析直播吧比赛页面的首发阵容"""
    home_xi = []
    away_xi = []
    home_formation = ""
    away_formation = ""

    # 直播吧首发阵容通常在 class 含 "lineup" 的元素中
    lineup_section = soup.find("div", class_=re.compile("lineup|formation|zhenrong"))
    if not lineup_section:
        # 尝试其他选择器
        lineup_section = soup.find("div", {"id": re.compile("lineup|formation")})

    if not lineup_section:
        logger.info("[首发] 直播吧页面未找到首发阵容（可能尚未公布）")
        return None

    # 解析主队阵容
    home_block = lineup_section.find("div", class_=re.compile("home|left|homeTeam"))
    if home_block:
        # 阵型
        form_tag = home_block.find(class_=re.compile("formation|form|zhenxing"))
        if form_tag:
            home_formation = form_tag.get_text(strip=True)

        # 球员名单
        for player_tag in home_block.find_all(class_=re.compile("player|name")):
            name = player_tag.get_text(strip=True)
            if name and len(name) <= 20 and name not in home_xi:
                home_xi.append(name)

    # 解析客队阵容
    away_block = lineup_section.find("div", class_=re.compile("away|right|awayTeam"))
    if away_block:
        form_tag = away_block.find(class_=re.compile("formation|form|zhenxing"))
        if form_tag:
            away_formation = form_tag.get_text(strip=True)

        for player_tag in away_block.find_all(class_=re.compile("player|name")):
            name = player_tag.get_text(strip=True)
            if name and len(name) <= 20 and name not in away_xi:
                away_xi.append(name)

    if not home_xi and not away_xi:
        logger.info("[首发] 直播吧页面阵容解析为空（可能尚未公布）")
        return None

    return {
        "home_formation": home_formation,
        "home_xi": home_xi[:11],
        "away_formation": away_formation,
        "away_xi": away_xi[:11],
    }


def _fetch_fotmob_lineup(home_en: str, away_en: str) -> dict:
    """
    FotMob 补充源（英文）

    FotMob 有比赛页面包含首发阵容，但需要知道比赛 ID。
    此处作为直播吧的补充源，后续完善。
    """
    # TODO: FotMob 需要比赛 ID，可通过 API 搜索获取
    logger.info("[首发] FotMob 源待完善，跳过")
    return None


def _fetch_dqd_lineup(home_en: str, away_en: str) -> dict:
    """
    从懂球帝爬取官方首发阵容

    懂球帝赛前会发布包含首发阵容的文章，
    文章正文中有"XX首发：1-球员名、2-球员名..."格式。

    Returns:
        {"home_formation": "", "home_xi": [...], "away_formation": "", "away_xi": [...]}
        或 None
    """
    from agents.predicted_agent.scouters.news_scouter import (
        _fetch_dqd_news_list, _fetch_article_detail,
    )

    home_zh = to_chinese(home_en) or home_en
    away_zh = to_chinese(away_en) or away_en

    # 获取懂球帝头条新闻
    news_list = _fetch_dqd_news_list(pages=3)

    # 筛选包含两队名 + "首发"的新闻
    lineup_article = None
    for news in news_list:
        title = news.get("title", "")
        if "首发" in title and (home_zh in title or away_zh in title):
            lineup_article = news
            break

    if not lineup_article:
        logger.info("[首发] 懂球帝未找到首发新闻")
        return None

    # 获取文章详情
    article_id = lineup_article.get("id")
    if not article_id:
        return None

    logger.info(f"[首发] 从懂球帝获取首发文章: {lineup_article.get('title', '')}")
    detail = _fetch_article_detail(article_id)
    if not detail:
        logger.info("[首发] 懂球帝文章详情为空")
        return None

    return _parse_dqd_lineup(detail, home_zh, away_zh)


def _parse_dqd_lineup(text: str, home_zh: str, away_zh: str) -> dict:
    """
    从懂球帝文章详情中解析双方首发名单

    文章格式示例:
        厄瓜多尔首发：
        1-埃尔南-加林兹、3-因卡皮耶、4-奥多涅斯...
        德国首发：
        1-诺伊尔、2-吕迪格、5-胡梅尔斯...
    """
    home_xi = []
    away_xi = []

    for team_zh, xi_list in [(home_zh, home_xi), (away_zh, away_xi)]:
        # 精确匹配 "队名首发：" 后面的球员列表（队名和首发之间不能有太多其他文字）
        pattern = rf'{re.escape(team_zh)}首发[：:]\s*(.*?)(?:\n\s*\n|\n[^、\d\s]|替补|主教练|$)'
        m = re.search(pattern, text, re.DOTALL)
        if m:
            players_text = m.group(1).strip()
            # 用 、, ， 分割球员
            parts = re.split(r'[、,，\n]', players_text)
            for p in parts:
                p = p.strip()
                # 去掉号码前缀 "1-"
                p = re.sub(r'^\d+-', '', p)
                # 去掉多余空白
                p = p.strip()
                if p and len(p) <= 20 and p not in xi_list:
                    xi_list.append(p)

    if not home_xi and not away_xi:
        logger.info("[首发] 懂球帝文章中未解析到首发名单")
        return None

    logger.info(f"[首发] 解析成功: {home_zh} {len(home_xi)}人, {away_zh} {len(away_xi)}人")

    return {
        "home_formation": "",
        "home_xi": home_xi[:11],
        "away_formation": "",
        "away_xi": away_xi[:11],
    }


# ======================== 对外接口 ========================

def get_lineup_intel(home_en: str, away_en: str, date: str = None,
                     tier: int = 1) -> dict:
    """
    获取首发阵容情报

    Args:
        home_en: 主队英文标准名
        away_en: 客队英文标准名
        date: 比赛日期 YYYY-MM-DD
        tier: 1=赛前1h外（缺阵名单+惯用阵型）, 2=赛前1h内（官方首发）

    Returns:
        {
            "tier": 1 | 2,
            "home": {
                "formation": "...",
                "starting_xi": [...],     # 二档才有
                "key_absences": [...],
                "is_official": bool,
            },
            "away": {...}
        }
    """
    if tier == 1:
        # ── 第一档：核心缺阵 + 惯用阵型 ──
        logger.info(f"[首发] 第一档: {home_en} vs {away_en}（缺阵名单+惯用阵型）")
        home_intel = get_tier1_lineup_intel(home_en)
        away_intel = get_tier1_lineup_intel(away_en)

        return {
            "tier": 1,
            "home": home_intel,
            "away": away_intel,
        }

    else:
        # ── 第二档：尝试官方首发 ──
        logger.info(f"[首发] 第二档: {home_en} vs {away_en}（官方首发）")

        # 先尝试懂球帝（赛前会发布首发阵容文章）
        official = _fetch_dqd_lineup(home_en, away_en)
        if not official:
            # 补充源: 直播吧
            official = _fetch_zhibo8_lineup(home_en, away_en, date)
        if not official:
            # 补充源: FotMob
            official = _fetch_fotmob_lineup(home_en, away_en)

        if official:
            # 官方首发获取成功，合并伤停信息
            home_absence = get_injuries_and_suspensions(home_en)
            away_absence = get_injuries_and_suspensions(away_en)

            return {
                "tier": 2,
                "home": {
                    "formation": official.get("home_formation", ""),
                    "formation_source": "官方首发",
                    "starting_xi": official.get("home_xi", []),
                    "key_absences": home_absence.get("key_absences", []),
                    "is_official": True,
                },
                "away": {
                    "formation": official.get("away_formation", ""),
                    "formation_source": "官方首发",
                    "starting_xi": official.get("away_xi", []),
                    "key_absences": away_absence.get("key_absences", []),
                    "is_official": True,
                },
            }
        else:
            # 官方首发未获取到，用惯用阵型作为预测首发（保持 tier=2）
            logger.info("[首发] 官方首发未获取到，使用惯用阵型作为预测首发")
            home_intel = get_tier1_lineup_intel(home_en)
            away_intel = get_tier1_lineup_intel(away_en)
            return {
                "tier": 2,
                "home": home_intel,
                "away": away_intel,
                "note": "官方首发未获取到，使用惯用阵型作为预测首发",
            }


def summarize_lineup(home_en: str, away_en: str, data: dict) -> str:
    """
    生成阵容情报的文本摘要（供 LLM 消费）
    """
    home_zh = to_chinese(home_en) or home_en
    away_zh = to_chinese(away_en) or away_en
    tier = data.get("tier", 1)

    lines = []

    for side, team_en, team_zh in [("home", home_en, home_zh), ("away", away_en, away_zh)]:
        side_data = data.get(side, {})
        formation = side_data.get("formation", "未知")
        is_official = side_data.get("is_official", False)
        absences = side_data.get("key_absences", [])
        xi = side_data.get("starting_xi", [])

        tag = "官方首发" if is_official else "惯用阵型"
        lines.append(f"{team_zh}队（{tag}，阵型: {formation}）:")

        if xi:
            lines.append(f"  首发: {', '.join(xi[:11])}")

        if absences:
            abs_parts = []
            for ab in absences:
                abs_parts.append(f"{ab['importance']}{ab['player']}({ab['reason']})")
            lines.append(f"  缺阵: {', '.join(abs_parts)}")
        else:
            lines.append(f"  缺阵: 无重要缺阵")

    return "\n".join(lines)


# ======================== 测试 ========================

if __name__ == "__main__":
    print("=== 首发阵容采集测试 ===\n")

    # 第一档测试
    print("--- 第一档（赛前1h外）---")
    data = get_lineup_intel("Brazil", "Germany", tier=1)
    print(summarize_lineup("Brazil", "Germany", data))
    print()

    # 第二档测试
    print("--- 第二档（赛前1h内）---")
    data = get_lineup_intel("Brazil", "Germany", tier=2)
    print(summarize_lineup("Brazil", "Germany", data))
