# -*- coding: utf-8 -*-
"""
赛前新闻采集器

数据源:
  懂球帝 API（api.dongqiudi.com）
    1. 头条新闻列表: api.dongqiudi.com/app/tabs/iphone/1.json
    2. 文章详情: dongqiudi.com/article/{id}

采集内容:
  - 队内冲突 / 球员不和 / 教练下课传闻等"软信息"
  - 赛前前瞻 / 战术分析
  - 伤情更新 / 阵容预测

输出格式:
  {
    "news": [
      {
        "title": "...",
        "summary": "...",
        "source": "懂球帝",
        "url": "...",
        "published_at": "2026-06-23",
        "relevance_tags": ["队内冲突", "伤情"],
      },
    ],
    "soft_signals": [
      {"type": "队内不和", "desc": "...", "severity": "中"},
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

# 懂球帝 API 请求头（模拟 iPhone 客户端）
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


# ======================== 软信号关键词 ========================

SOFT_SIGNAL_KEYWORDS = {
    "轮换意图": {
        "keywords": ["轮换", "练兵", "替补", "轮休", "雪藏", "主力休息",
                     "大面积轮换", "休整", "放弃", "保守"],
        "severity": "高",
    },
    "出线形势": {
        "keywords": ["提前出线", "锁定小组第一", "锁定第一", "已出线", "已晋级",
                     "已淘汰", "出局", "打平即可", "必须取胜", "生死战",
                     "无关紧要", "无欲无求", "争出线", "保出线"],
        "severity": "高",
    },
    "队内冲突": {
        "keywords": ["内讧", "冲突", "矛盾", "不和", "争吵", "打架", "更衣室", "分裂"],
        "severity": "高",
    },
    "教练危机": {
        "keywords": ["下课", "辞职", "解雇", "帅位", "离任", "信任危机"],
        "severity": "中",
    },
    "球员离队": {
        "keywords": ["转会", "离队", "不满", "想走", "提交转会"],
        "severity": "中",
    },
    "士气低落": {
        "keywords": ["低迷", "信心不足", "压力", "崩溃", "士气"],
        "severity": "中",
    },
    "伤病更新": {
        "keywords": ["伤", "缺阵", "退出", "无法上场", "复出", "伤停"],
        "severity": "低",
    },
    "场外因素": {
        "keywords": ["旅行", "罢工", "政治", "安全", "签证", "行程"],
        "severity": "中",
    },
}


def _extract_soft_signals(text: str) -> list[dict]:
    """从新闻文本中提取软信号"""
    signals = []
    for signal_type, config in SOFT_SIGNAL_KEYWORDS.items():
        for kw in config["keywords"]:
            if kw in text:
                signals.append({
                    "type": signal_type,
                    "desc": f"新闻提及: {kw}",
                    "severity": config["severity"],
                })
                break
    return signals


# ======================== 懂球帝 API ========================

def _fetch_dqd_news_list(pages: int = 3) -> list[dict]:
    """
    调用懂球帝 API 获取头条新闻列表

    Args:
        pages: 获取几页新闻（每页20条）

    Returns:
        新闻列表，每条含 id, title, description, comments_total, share, published_at
    """
    all_articles = []
    url = _DQD_NEWS_API

    for page in range(pages):
        try:
            resp = httpx.get(url, headers=_DQD_HEADERS, timeout=REQUEST_TIMEOUT,
                             follow_redirects=True)
            if resp.status_code != 200:
                logger.warning(f"[新闻] 懂球帝API返回 {resp.status_code}")
                break

            import json
            data = json.loads(resp.text)
            articles = data.get("articles", [])

            for a in articles:
                all_articles.append({
                    "id": a.get("id"),
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "comments_total": a.get("comments_total", 0),
                    "url": a.get("share", ""),
                    "published_at": "",
                })

            # 获取下一页 URL
            next_url = data.get("next")
            if not next_url:
                break
            url = next_url

        except Exception as e:
            logger.error(f"[新闻] 懂球帝API请求失败: {e}")
            break

    logger.info(f"[新闻] 懂球帝API获取 {len(all_articles)} 条新闻")
    return all_articles


def _fetch_article_detail(article_id: int) -> str:
    """
    获取懂球帝文章全文

    Args:
        article_id: 文章ID

    Returns:
        文章正文文本
    """
    url = _DQD_ARTICLE_URL.format(article_id=article_id)
    try:
        resp = httpx.get(url, headers=_DQD_HEADERS, timeout=REQUEST_TIMEOUT,
                         follow_redirects=True)
        if resp.status_code != 200:
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")

        # 懂球帝文章正文通常在 class 含 detail/content/article 的 div 中
        content = soup.find("div", class_=re.compile("detail|article|content|body"))
        if content:
            return content.get_text(strip=True, separator="\n")

        # 兜底：找所有段落
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        return text

    except Exception as e:
        logger.warning(f"[新闻] 获取文章 {article_id} 失败: {e}")
        return ""


# ======================== 球队新闻筛选 ========================

def _filter_news_by_team(news_list: list[dict], team_en: str) -> list[dict]:
    """
    从新闻列表中筛选与指定球队相关的新闻

    匹配规则: 标题或描述中包含球队中文名/英文名/别名
    """
    team_info = get_team_info(team_en)
    if not team_info:
        return []

    # 构建匹配关键词
    keywords = [team_en, team_en.lower()]
    if team_info.get("zh"):
        keywords.append(team_info["zh"])
    if team_info.get("dqd_team"):
        keywords.append(team_info["dqd_team"])
    for alias in team_info.get("alias", []):
        keywords.append(alias)

    filtered = []
    for news in news_list:
        text = f"{news.get('title', '')} {news.get('description', '')}"
        if any(kw in text for kw in keywords):
            # 提取软信号标签
            relevance_tags = []
            for signal_type, config in SOFT_SIGNAL_KEYWORDS.items():
                for kw in config["keywords"]:
                    if kw in text:
                        relevance_tags.append(signal_type)
                        break

            news_copy = dict(news)
            news_copy["source"] = "懂球帝"
            news_copy["relevance_tags"] = relevance_tags
            filtered.append(news_copy)

    return filtered


# ======================== 对外接口 ========================

def get_pre_match_news(team_en: str, limit: int = 8) -> dict:
    """
    获取球队赛前新闻 + 软信号提取

    Args:
        team_en: 国家队英文标准名（如 "Brazil"）
        limit: 最多获取几条相关新闻

    Returns:
        {
            "news": [...],
            "soft_signals": [...]
        }
    """
    team_zh = to_chinese(team_en) or team_en

    # 1. 获取懂球帝头条新闻列表
    news_list = _fetch_dqd_news_list(pages=3)

    # 2. 按球队名筛选
    team_news = _filter_news_by_team(news_list, team_en)

    # 3. 获取前几条新闻的详情摘要
    for news in team_news[:limit]:
        article_id = news.get("id")
        if article_id:
            detail = _fetch_article_detail(article_id)
            if detail:
                # 取前200字符作为摘要
                news["summary"] = detail[:200].replace("\n", " ").strip()

    # 4. 汇总软信号
    all_soft_signals = []
    for item in team_news:
        text = f"{item.get('title', '')} {item.get('summary', '')}"
        signals = _extract_soft_signals(text)
        all_soft_signals.extend(signals)
        if not item.get("relevance_tags"):
            item["relevance_tags"] = [s["type"] for s in signals]

    # 软信号去重
    seen_types = set()
    unique_signals = []
    for sig in all_soft_signals:
        if sig["type"] not in seen_types:
            seen_types.add(sig["type"])
            unique_signals.append(sig)

    return {
        "news": team_news[:limit],
        "soft_signals": unique_signals,
    }


def summarize_news(team_en: str, data: dict, max_items: int = 5) -> str:
    """
    生成球队赛前新闻的文本摘要（供 LLM 消费）
    """
    team_zh = to_chinese(team_en) or team_en
    news = data.get("news", [])[:max_items]
    soft_signals = data.get("soft_signals", [])

    if not news and not soft_signals:
        return f"{team_zh}队近期无特殊赛前新闻"

    parts = []
    if news:
        parts.append(f"{team_zh}队近期新闻:")
        for i, n in enumerate(news, 1):
            tags = f" [{','.join(n.get('relevance_tags', []))}]" if n.get("relevance_tags") else ""
            parts.append(f"  {i}. {n['title']}{tags}")
            if n.get("summary"):
                parts.append(f"     {n['summary'][:100]}")

    if soft_signals:
        parts.append(f"  软信号:")
        for sig in soft_signals:
            parts.append(f"    [{sig['severity']}] {sig['type']}: {sig['desc']}")

    return "\n".join(parts)


# ======================== 测试 ========================

if __name__ == "__main__":
    print("=== 赛前新闻采集测试（懂球帝API）===\n")

    for team in ["Portugal", "Brazil"]:
        print(f"--- {team} ---")
        data = get_pre_match_news(team, limit=5)
        print(f"  新闻: {len(data['news'])} 条")
        for n in data["news"][:3]:
            print(f"    [{n.get('source', '?')}] {n['title']}")
            if n.get("summary"):
                print(f"      摘要: {n['summary'][:80]}")
        print(f"  软信号: {len(data['soft_signals'])} 个")
        for s in data["soft_signals"]:
            print(f"    [{s['severity']}] {s['type']}: {s['desc']}")
        print()
