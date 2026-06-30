# -*- coding: utf-8 -*-
"""
OpenClaw 每日比赛结果同步

北京时间15:00自动执行，获取已完赛的世界杯比赛结果，
同步到 MySQL（全部比赛）+ Neo4j（48队交锋）。
"""

import os
import re
import logging
import httpx
from bs4 import BeautifulSoup
from typing import Optional

logger = logging.getLogger(__name__)

TITAN007_SCORE_URL = os.getenv("WORLDCUP_SCORE_URL", "https://2026.titan007.com/")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
}

# titan007 中文队名 → 标准英文名
ZH_TO_EN = {
    "墨西哥": "Mexico", "南非": "South Africa", "韩国": "South Korea",
    "捷克": "Czech Republic", "加拿大": "Canada",
    "波黑": "Bosnia & Herzegovina", "卡塔尔": "Qatar", "瑞士": "Switzerland",
    "巴西": "Brazil", "摩洛哥": "Morocco", "海地": "Haiti", "苏格兰": "Scotland",
    "美国": "USA", "巴拉圭": "Paraguay", "澳大利亚": "Australia", "土耳其": "Turkey",
    "德国": "Germany", "库拉索": "Curacao", "科特迪瓦": "Ivory Coast",
    "厄瓜多尔": "Ecuador", "荷兰": "Netherlands", "日本": "Japan",
    "瑞典": "Sweden", "突尼斯": "Tunisia", "比利时": "Belgium",
    "埃及": "Egypt", "伊朗": "Iran", "新西兰": "New Zealand",
    "西班牙": "Spain", "佛得角": "Cape Verde", "沙特阿拉伯": "Saudi Arabia",
    "乌拉圭": "Uruguay", "法国": "France", "塞内加尔": "Senegal",
    "伊拉克": "Iraq", "挪威": "Norway", "阿根廷": "Argentina",
    "阿尔及利亚": "Algeria", "奥地利": "Austria", "约旦": "Jordan",
    "葡萄牙": "Portugal", "刚果民主共和国": "DR Congo",
    "乌兹别克斯坦": "Uzbekistan", "哥伦比亚": "Colombia",
    "英格兰": "England", "克罗地亚": "Croatia", "加纳": "Ghana", "巴拿马": "Panama",
}


def normalize_team(name: str) -> str:
    return ZH_TO_EN.get(name.strip(), name.strip())


def fetch_finished_matches() -> list[dict]:
    """从 titan007 获取已完赛比赛"""
    try:
        resp = httpx.get(TITAN007_SCORE_URL, headers=HEADERS, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            return []
    except:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    matches = []

    for tr in soup.find_all("tr", id=re.compile(r"^(tr\d+_|near_tr_)\d{7}$")):
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue

        date_str = ""
        home = ""
        away = ""
        finished = False
        score_idx = None

        for i, td in enumerate(tds):
            cls = td.get("class", [""])[0] if td.get("class") else ""
            text = td.get_text(strip=True)
            if cls == "gamedate":
                date_str = text
            if cls == "score":
                score_idx = i
            if "完" in text:
                finished = True

        if not finished or score_idx is None:
            continue

        score_text = tds[score_idx].get_text(strip=True)
        if "-" not in score_text:
            continue

        parts = score_text.split("-")
        try:
            hg = int(parts[0].strip())
            ag = int(parts[1].strip())
        except ValueError:
            continue

        # 中文队名转标准英文名（修复: 之前误调用未定义的 normalize）
        home = normalize_team(tds[score_idx - 1].get_text(strip=True)) if score_idx > 0 else ""
        away = normalize_team(tds[score_idx + 1].get_text(strip=True)) if score_idx + 1 < len(tds) else ""
        if not home or not away:
            continue

        result = "H" if hg > ag else ("A" if hg < ag else "D")
        matches.append({
            "home": home, "away": away,
            "home_goals": hg, "away_goals": ag,
            "result": result, "date": date_str,
            "competition": "World Cup 2026",
        })

    return matches


def sync_to_mysql(matches: list[dict]) -> int:
    """写入 MySQL（全部比赛）"""
    import pymysql

    conn = pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "football123"),
        database=os.getenv("MYSQL_DATABASE", "football_agent"),
        charset="utf8mb4",
    )
    cursor = conn.cursor()

    inserted = 0
    for m in matches:
        date_sorted = None
        try:
            parts = m["date"].split(" ")[0]
            date_sorted = f"2026-{parts}"
        except:
            pass
        try:
            cursor.execute(
                """INSERT IGNORE INTO intl_matches
                   (match_date, home_team, away_team, home_goals, away_goals,
                    result, competition, season, match_date_sorted)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (m["date"], m["home"], m["away"], m["home_goals"], m["away_goals"],
                 m["result"], m["competition"], "2026", date_sorted),
            )
            inserted += cursor.rowcount
        except:
            pass

    conn.commit()
    cursor.close()
    conn.close()
    return inserted


def sync_to_neo4j(matches: list[dict]) -> int:
    """写入 Neo4j（仅48队交锋）"""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URL", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"),
              os.getenv("NEO4J_PASSWORD", "football123")),
    )

    # 48队标准英文名集合
    team_48 = set(ZH_TO_EN.values())
    imported = 0

    with driver.session() as s:
        for m in matches:
            if m["home"] not in team_48 or m["away"] not in team_48:
                continue
            result_text = f"{m['home_goals']}-{m['away_goals']}"
            result_desc = f"{m['home']}胜" if m["result"] == "H" else (
                f"{m['away']}胜" if m["result"] == "A" else "平局")
            s.run("""
                MERGE (h:NationalTeam {name: $home})
                MERGE (a:NationalTeam {name: $away})
                MERGE (h)-[r:PLAYED_AGAINST {match_date: $date, season: '2026'}]->(a)
                SET r.competition = $comp, r.match_result = $result,
                    r.result_desc = $desc, r.home_goals = $hg,
                    r.away_goals = $ag, r.total_goals = $total
            """, home=m["home"], away=m["away"], date=m["date"],
                comp=m["competition"], result=result_text, desc=result_desc,
                hg=m["home_goals"], ag=m["away_goals"],
                total=m["home_goals"] + m["away_goals"])
            imported += 1

    driver.close()
    return imported


def daily_sync():
    """每日同步入口"""
    logger.info("=== 每日比赛结果同步开始 ===")
    matches = fetch_finished_matches()
    if not matches:
        logger.info("无已完赛比赛")
        return

    logger.info(f"获取到 {len(matches)} 场已完赛比赛")

    try:
        mysql_count = sync_to_mysql(matches)
        logger.info(f"MySQL 写入: {mysql_count} 场")
    except Exception as e:
        logger.error(f"MySQL 同步失败: {e}")

    try:
        neo4j_count = sync_to_neo4j(matches)
        logger.info(f"Neo4j 写入: {neo4j_count} 场")
    except Exception as e:
        logger.error(f"Neo4j 同步失败: {e}")

    logger.info("=== 每日同步完成 ===")
