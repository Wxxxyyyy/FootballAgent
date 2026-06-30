# -*- coding: utf-8 -*-
"""
伤停 + 红黄牌停赛采集器

数据源:
  懂球帝 API
    1. 头条新闻列表: api.dongqiudi.com/app/tabs/iphone/1.json
    2. 伤停情报文章: dongqiudi.com/article/{id}
    3. 从文章正文中用正则提取【XXX缺阵】格式的伤停信息

输出格式:
  {
    "injuries": [
      {
        "player": "鲁本·迪亚斯",
        "position": "主力后卫",
        "reason": "复出",            # 伤情描述
        "status": "已恢复",          # 缺阵/存疑/已恢复
        "importance": "主力",
        "source": "懂球帝"
      },
    ],
    "suspensions": [],
    "key_absences": [
      {"player": "...", "reason": "伤/停", "importance": "核心"},
    ]
  }
"""

import logging
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from agents.predicted_agent.scouters.national_team_config import get_team_info, to_chinese

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 15

# 懂球帝请求头（模拟 iPhone 客户端）
_DQD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                  "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                  "Version/17.0 Mobile/15E148 Safari/604.1",
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

# 懂球帝头条新闻 API
_DQD_NEWS_API = "http://api.dongqiudi.com/app/tabs/iphone/1.json"
# 懂球帝文章详情 URL 模板
_DQD_ARTICLE_URL = "https://www.dongqiudi.com/article/{article_id}"

# 伤停情报文章标题关键词
_INJURY_TITLE_KEYWORDS = ["伤停", "伤情", "缺阵", "伤员", "出战成疑", "复出"]


# ======================== 懂球帝 API ========================

def _fetch_dqd_news_list(pages: int = 3) -> list[dict]:
    """调用懂球帝 API 获取头条新闻列表"""
    all_articles = []
    url = _DQD_NEWS_API

    for page in range(pages):
        try:
            resp = httpx.get(url, headers=_DQD_HEADERS, timeout=REQUEST_TIMEOUT,
                             follow_redirects=True)
            if resp.status_code != 200:
                break

            import json
            data = json.loads(resp.text)
            articles = data.get("articles", [])

            for a in articles:
                all_articles.append({
                    "id": a.get("id"),
                    "title": a.get("title", ""),
                    "url": a.get("share", ""),
                })

            next_url = data.get("next")
            if not next_url:
                break
            url = next_url

        except Exception as e:
            logger.error(f"[伤停] 懂球帝API请求失败: {e}")
            break

    return all_articles


def _fetch_article_detail(article_id: int) -> str:
    """获取懂球帝文章全文"""
    url = _DQD_ARTICLE_URL.format(article_id=article_id)
    try:
        resp = httpx.get(url, headers=_DQD_HEADERS, timeout=REQUEST_TIMEOUT,
                         follow_redirects=True)
        if resp.status_code != 200:
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")
        content = soup.find("div", class_=re.compile("detail|article|content|body"))
        if content:
            return content.get_text(strip=True, separator="\n")

        paragraphs = soup.find_all("p")
        return "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

    except Exception as e:
        logger.warning(f"[伤停] 获取文章 {article_id} 失败: {e}")
        return ""


# ======================== 伤停情报文章查找 ========================

def _find_injury_reports(news_list: list[dict], team_zh: str = None) -> list[dict]:
    """
    从新闻列表中找伤停情报文章

    匹配规则: 标题含伤停关键词，且（如果指定了球队名）标题含球队名或"世界杯"
    """
    reports = []
    for news in news_list:
        title = news.get("title", "")

        # 标题必须含伤停关键词
        if not any(kw in title for kw in _INJURY_TITLE_KEYWORDS):
            continue

        # 如果指定了球队名，标题或球队需要匹配
        if team_zh:
            # 伤停情报通常是"世界杯四场伤停情报"这种，涵盖多场比赛
            # 只要标题含"世界杯"或"伤停"就获取全文，然后从全文中提取对应球队
            if "世界杯" in title or "伤停" in title:
                reports.append(news)
        else:
            reports.append(news)

    return reports


# ======================== 伤停信息解析 ========================

# 伤停信息正则模式
# 格式: 【葡萄牙缺阵】鲁本·迪亚斯(主力后卫/复出)
# 格式: 【加纳缺阵】阿蒂-齐吉(主力门将/肌肉伤出战成疑) 帕尔特伊(主力中场/复出)
# 格式: 【葡萄牙缺阵】无
# 注意: 每个【XXX缺阵】的内容在同一行，不跨行匹配

_ABSENCE_PATTERN = re.compile(
    r'【(.+?)缺阵】([^\n]+)',
)

# 单个球员信息: 球员名(位置/状态) 或 球员名(位置/状态) 球员名(位置/状态)
_PLAYER_PATTERN = re.compile(
    r'([^\s()（）]+)\s*[（(]([^）)]+)[）)]',
)


def _parse_injury_report(text: str, team_en: str) -> dict:
    """
    从伤停情报文章正文中解析指定球队的伤停信息

    Args:
        text: 文章全文
        team_en: 国家队英文标准名

    Returns:
        {
            "injuries": [...],
            "suspensions": [...],
            "key_absences": [...]
        }
    """
    team_info = get_team_info(team_en)
    if not team_info:
        return {"injuries": [], "suspensions": [], "key_absences": []}

    # 构建球队匹配关键词
    team_keywords = [team_en]
    if team_info.get("zh"):
        team_keywords.append(team_info["zh"])
    if team_info.get("dqd_team"):
        team_keywords.append(team_info["dqd_team"])
    # 常见简称
    team_keywords.extend(team_info.get("alias", []))

    injuries = []
    suspensions = []

    # 提取所有【XXX缺阵】块
    for match in _ABSENCE_PATTERN.finditer(text):
        team_name_in_text = match.group(1).strip()
        absence_content = match.group(2).strip()

        # 判断这个块是否属于目标球队
        is_target_team = any(kw in team_name_in_text for kw in team_keywords)
        if not is_target_team:
            continue

        # 如果是"无"，跳过
        if absence_content == "无" or not absence_content:
            continue

        # 解析球员信息
        for player_match in _PLAYER_PATTERN.finditer(absence_content):
            player_name = player_match.group(1).strip()
            player_info = player_match.group(2).strip()

            # 解析位置和状态: "主力后卫/复出" 或 "主力门将/肌肉伤出战成疑"
            parts = player_info.split("/")
            position = parts[0].strip() if parts else ""
            reason = parts[1].strip() if len(parts) > 1 else ""

            # 判断状态
            status = "缺阵"
            if "复出" in reason or "回归" in reason or "伤愈" in reason:
                status = "已恢复"
            elif "成疑" in reason or "存疑" in reason:
                status = "存疑"
            elif "停赛" in reason or "黄牌" in reason or "红牌" in reason:
                status = "停赛"

            # 判断重要性
            importance = "主力"
            if "核心" in position or "核心" in reason:
                importance = "核心"
            elif "替补" in position:
                importance = "替补"

            # 区分伤停和停赛
            if "停赛" in status:
                suspensions.append({
                    "player": player_name,
                    "reason": reason or "停赛",
                    "source": "懂球帝",
                })
            else:
                injuries.append({
                    "player": player_name,
                    "position": position,
                    "reason": reason or "受伤",
                    "status": status,
                    "importance": importance,
                    "source": "懂球帝",
                })

    # 汇总核心缺阵名单
    key_absences = []
    for inj in injuries:
        if inj["status"] in ("缺阵", "存疑"):
            key_absences.append({
                "player": inj["player"],
                "reason": f"伤: {inj['reason']}",
                "importance": inj["importance"],
                "status": inj["status"],
            })
    for sus in suspensions:
        key_absences.append({
            "player": sus["player"],
            "reason": sus["reason"],
            "importance": "主力",
            "status": "停赛",
        })

    # 按重要性排序
    importance_order = {"核心": 0, "主力": 1, "替补": 2}
    key_absences.sort(key=lambda x: importance_order.get(x["importance"], 3))

    return {
        "injuries": injuries,
        "suspensions": suspensions,
        "key_absences": key_absences,
    }


# ======================== 对外接口 ========================

def get_injuries_and_suspensions(team_en: str) -> dict:
    """
    获取球队伤停 + 停赛信息

    流程:
      1. 从懂球帝API获取头条新闻列表
      2. 筛选标题含"伤停"的文章
      3. 获取文章全文
      4. 用正则提取【XXX缺阵】格式的伤停信息

    Args:
        team_en: 国家队英文标准名（如 "Portugal"）

    Returns:
        {
            "injuries": [...],
            "suspensions": [...],
            "key_absences": [...]
        }
    """
    team_info = get_team_info(team_en)
    if not team_info:
        logger.warning(f"[伤停] 未知球队: {team_en}")
        return {"injuries": [], "suspensions": [], "key_absences": []}

    team_zh = team_info.get("zh", team_en)

    # 1. 获取新闻列表
    news_list = _fetch_dqd_news_list(pages=3)

    # 2. 找伤停情报文章
    injury_reports = _find_injury_reports(news_list, team_zh=team_zh)
    logger.info(f"[伤停] 找到 {len(injury_reports)} 篇伤停情报文章")

    if not injury_reports:
        logger.info(f"[伤停] 未找到伤停情报文章")
        return {"injuries": [], "suspensions": [], "key_absences": []}

    # 3. 获取文章全文并解析
    all_injuries = []
    all_suspensions = []
    all_key_absences = []

    for report in injury_reports[:3]:  # 最多解析3篇文章
        article_id = report.get("id")
        if not article_id:
            continue

        detail = _fetch_article_detail(article_id)
        if not detail:
            continue

        result = _parse_injury_report(detail, team_en)
        all_injuries.extend(result.get("injuries", []))
        all_suspensions.extend(result.get("suspensions", []))
        all_key_absences.extend(result.get("key_absences", []))

    # 去重（同一球员可能出现在多篇文章中）
    seen_players = set()
    unique_injuries = []
    for inj in all_injuries:
        if inj["player"] not in seen_players:
            seen_players.add(inj["player"])
            unique_injuries.append(inj)

    seen_players_sus = set()
    unique_suspensions = []
    for sus in all_suspensions:
        if sus["player"] not in seen_players_sus:
            seen_players_sus.add(sus["player"])
            unique_suspensions.append(sus)

    # key_absences 去重
    seen_absence = set()
    unique_absences = []
    for ab in all_key_absences:
        if ab["player"] not in seen_absence:
            seen_absence.add(ab["player"])
            unique_absences.append(ab)

    importance_order = {"核心": 0, "主力": 1, "替补": 2}
    unique_absences.sort(key=lambda x: importance_order.get(x["importance"], 3))

    logger.info(f"[伤停] {team_zh}: {len(unique_injuries)}名伤员, "
                f"{len(unique_suspensions)}名停赛, "
                f"{len(unique_absences)}名核心缺阵")

    return {
        "injuries": unique_injuries,
        "suspensions": unique_suspensions,
        "key_absences": unique_absences,
    }


def summarize_absences(team_en: str, data: dict) -> str:
    """
    生成球队缺阵名单的文本摘要（供 LLM 消费）
    """
    team_zh = to_chinese(team_en) or team_en
    key_absences = data.get("key_absences", [])

    if not key_absences:
        return f"{team_zh}队目前无重要缺阵球员"

    parts = []
    for ab in key_absences:
        level = ab.get("importance", "")
        player = ab.get("player", "?")
        reason = ab.get("reason", "")
        status = ab.get("status", "")
        parts.append(f"{level}{player}（{reason}，{status}）")

    return f"{team_zh}队缺阵情况：{'、'.join(parts)}"


# ======================== 测试 ========================

if __name__ == "__main__":
    print("=== 伤停采集测试（懂球帝API）===\n")

    for team in ["Portugal", "Brazil", "England"]:
        print(f"--- {team} ---")
        data = get_injuries_and_suspensions(team)
        print(f"  伤员: {len(data['injuries'])} 人")
        for inj in data["injuries"][:5]:
            print(f"    {inj['player']} - {inj['position']} - {inj['reason']} ({inj['status']})")
        print(f"  停赛: {len(data['suspensions'])} 人")
        print(f"  核心缺阵: {len(data['key_absences'])} 人")
        print(f"  摘要: {summarize_absences(team, data)}")
        print()
