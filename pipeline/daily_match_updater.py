# -*- coding: utf-8 -*-
"""
每日比赛结果更新

世界杯期间，每天从 titan007 获取当日已完赛的比赛结果，
MERGE 到 Neo4j，保持近5场数据最新。

用法:
  # 手动更新当天比赛
  python -m pipeline.daily_match_updater

  # 或通过 scheduler.py 每天定时执行
"""

import os
import re
import logging
import httpx
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

TITAN007_URL = os.getenv("TITAN007_BASE_URL", "https://2026.titan007.com/")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
}

# titan007 中文队名 → 标准英文名（和11v11.com/Neo4j一致）
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


def normalize_team_name(name: str) -> str:
    """中文队名 → 标准英文名，如果不在映射表中则返回原名"""
    return ZH_TO_EN.get(name.strip(), name.strip())


def fetch_finished_matches() -> list[dict]:
    """
    从 titan007 获取今日已完赛的比赛

    返回: [{"home", "away", "home_goals", "away_goals", "date", "competition"}, ...]
    """
    try:
        resp = httpx.get(TITAN007_URL, headers=HEADERS, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            logger.warning(f"titan007 返回 {resp.status_code}")
            return []
    except Exception as e:
        logger.error(f"获取比赛列表失败: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    matches = []

    # 找所有比赛行（tr1_ 和 near_tr_ 两种格式）
    for tr in soup.find_all("tr", id=re.compile(r"^(tr\d+_|near_tr_)\d{7}$")):
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue

        # 找 score td 定位主客队
        score_idx = None
        date_str = ""
        finished = False

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

        # 提取比分和队名
        score_text = tds[score_idx].get_text(strip=True)
        if "-" not in score_text:
            continue

        parts = score_text.split("-")
        if len(parts) != 2:
            continue

        try:
            hg = int(parts[0].strip())
            ag = int(parts[1].strip())
        except ValueError:
            continue

        home = normalize_team_name(tds[score_idx - 1].get_text(strip=True)) if score_idx > 0 else ""
        away = normalize_team_name(tds[score_idx + 1].get_text(strip=True)) if score_idx + 1 < len(tds) else ""

        if not home or not away:
            continue

        # 计算结果
        if hg > ag:
            result = "H"
        elif hg < ag:
            result = "A"
        else:
            result = "D"

        matches.append({
            "home": home,
            "away": away,
            "home_goals": hg,
            "away_goals": ag,
            "result": result,
            "date": date_str,
            "competition": "World Cup 2026",
        })

    logger.info(f"获取到 {len(matches)} 场已完赛比赛")
    return matches


def update_neo4j(matches: list[dict]):
    """将比赛结果 MERGE 到 Neo4j（只导入48队之间的交锋）"""
    from neo4j import GraphDatabase
    from dotenv import load_dotenv
    from pipeline.national_team_neo4j_loader import normalize_team_name
    load_dotenv()

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URL", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"),
              os.getenv("NEO4J_PASSWORD", "football123"))
    )

    imported = 0
    skipped = 0
    with driver.session() as s:
        for m in matches:
            # 标准化队名
            home_std = normalize_team_name(m["home"])
            away_std = normalize_team_name(m["away"])

            # 只导入48队之间的交锋
            if not home_std or not away_std:
                skipped += 1
                continue

            result_text = f"{m['home_goals']}-{m['away_goals']}"
            result_desc = f"{home_std}胜" if m["result"] == "H" else (
                f"{away_std}胜" if m["result"] == "A" else "平局"
            )

            s.run("""
                MERGE (h:NationalTeam {name: $home})
                MERGE (a:NationalTeam {name: $away})
                MERGE (h)-[r:PLAYED_AGAINST {match_date: $date, season: '2026'}]->(a)
                SET r.competition = $comp,
                    r.match_result = $result,
                    r.result_desc = $desc,
                    r.home_goals = $hg,
                    r.away_goals = $ag,
                    r.total_goals = $total
            """, home=home_std, away=away_std, date=m["date"],
                comp=m["competition"],
                result=result_text, desc=result_desc,
                hg=m["home_goals"], ag=m["away_goals"],
                total=m["home_goals"] + m["away_goals"])
            imported += 1

    driver.close()
    logger.info(f"更新 Neo4j: {imported}场, 跳过(非48队) {skipped}场")
    return imported


def update_mysql(matches: list[dict]):
    """将比赛结果写入 MySQL（全部比赛，含非48队对手）"""
    import pymysql
    from dotenv import load_dotenv
    load_dotenv()

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
    skipped = 0
    for m in matches:
        # 解析日期为可排序格式
        date_str = m["date"]
        date_sorted = None
        try:
            parts = date_str.split(" ")[0]  # MM-DD
            date_sorted = f"2026-{parts}"
        except:
            pass

        # 去重检查：同一主队+客队+日期+比分视为已存在
        if date_sorted:
            cursor.execute(
                """SELECT COUNT(*) as cnt FROM intl_matches
                   WHERE home_team = %s AND away_team = %s
                     AND home_goals = %s AND away_goals = %s
                     AND match_date_sorted = %s""",
                (m["home"], m["away"], m["home_goals"], m["away_goals"], date_sorted),
            )
            if cursor.fetchone()["cnt"] > 0:
                skipped += 1
                continue

        try:
            cursor.execute(
                """INSERT INTO intl_matches
                   (match_date, home_team, away_team, home_goals, away_goals,
                    result, competition, season, match_date_sorted)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (date_str, m["home"], m["away"], m["home_goals"], m["away_goals"],
                 m["result"], m["competition"], "2026", date_sorted),
            )
            inserted += cursor.rowcount
        except Exception as e:
            logger.warning(f"MySQL插入失败: {e}")

    conn.commit()
    cursor.close()
    conn.close()
    logger.info(f"更新 MySQL: 新增 {inserted}场, 跳过重复 {skipped}场")
    return inserted


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 60)
    print("  每日比赛结果更新")
    print("=" * 60)

    # 1. 获取今日已完赛比赛
    matches = fetch_finished_matches()
    if not matches:
        print("今日无已完赛比赛")
        return

    print(f"\n获取到 {len(matches)} 场已完赛比赛:")
    for m in matches:
        print(f"  {m['home']} {m['home_goals']}-{m['away_goals']} {m['away']}")

    # 2. 更新 MySQL（全部比赛）+ Neo4j（48队交锋）
    try:
        update_mysql(matches)
    except Exception as e:
        print(f"⚠️ MySQL 更新失败: {e}")

    try:
        update_neo4j(matches)
    except Exception as e:
        print(f"⚠️ Neo4j 更新失败: {e}")

    print(f"\n✅ 更新完成")

    print("=" * 60)


if __name__ == "__main__":
    main()
