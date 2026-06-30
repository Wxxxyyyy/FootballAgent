# -*- coding: utf-8 -*-
"""
11v11.com 国家队历史比赛爬虫

从 11v11.com 爬取48支世界杯参赛队近10年的历史比赛记录，
导入 Neo4j 补充国家队交锋数据。

数据源: 11v11.com/teams/{slug}/tab/matches/season/{year}/
爬取方式: httpx 直接请求（无需JS渲染）

用法:
  python -m pipeline.intl_match_scraper
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Optional
from collections import Counter

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════════
#  48 支国家队在 11v11.com 的 slug
# ═══════════════════════════════════════════════════════════════

TEAM_SLUGS = {
    "Mexico": "mexico", "South Africa": "south-africa",
    "South Korea": "korea-republic", "Czech Republic": "czech-republic",
    "Canada": "canada", "Bosnia & Herzegovina": "bosnia-and-herzegovina",
    "Qatar": "qatar", "Switzerland": "switzerland",
    "Brazil": "brazil", "Morocco": "morocco",
    "Haiti": "haiti", "Scotland": "scotland",
    "USA": "usa", "Paraguay": "paraguay",
    "Australia": "australia", "Turkey": "turkey",
    "Germany": "germany", "Curacao": "curacao",
    "Ivory Coast": "ivory-coast", "Ecuador": "ecuador",
    "Netherlands": "netherlands", "Japan": "japan",
    "Sweden": "sweden", "Tunisia": "tunisia",
    "Belgium": "belgium", "Egypt": "egypt",
    "Iran": "iran", "New Zealand": "new-zealand",
    "Spain": "spain", "Cape Verde": "cape-verde-islands",
    "Saudi Arabia": "saudi-arabia", "Uruguay": "uruguay",
    "France": "france", "Senegal": "senegal",
    "Iraq": "iraq", "Norway": "norway",
    "Argentina": "argentina", "Algeria": "algeria",
    "Austria": "austria", "Jordan": "jordan",
    "Portugal": "portugal", "DR Congo": "congo-dr",
    "Uzbekistan": "uzbekistan", "Colombia": "colombia",
    "England": "england", "Croatia": "croatia",
    "Ghana": "ghana", "Panama": "panama",
}

# 要爬取的赛季（近10年）
SEASONS = list(range(2017, 2027))  # 2017~2026

BASE_URL = "https://www.11v11.com/teams/{slug}/tab/matches/season/{year}/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
}

# 11v11 队名 → 标准英文名 映射（处理差异）
NAME_MAP = {
    "Korea Republic": "South Korea",
    "Congo DR": "DR Congo",
    "Bosnia-Herzegovina": "Bosnia & Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Ivory Coast": "Ivory Coast",
    "USA": "USA",
    "United States": "USA",
}


def normalize_name(name: str) -> str:
    """标准化队名"""
    name = name.strip()
    return NAME_MAP.get(name, name)


# ═══════════════════════════════════════════════════════════════
#  爬取单队单赛季
# ═══════════════════════════════════════════════════════════════

def scrape_season(slug: str, year: int) -> list[dict]:
    """爬取单支球队单个赛季的比赛"""
    url = BASE_URL.format(slug=slug, year=year)
    matches = []

    try:
        resp = httpx.get(url, headers=HEADERS, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            return matches

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        if not table:
            return matches

        rows = table.find_all("tr")
        for row in rows[1:]:  # 跳过表头
            cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if len(cells) < 5:
                continue

            date_str = cells[0]
            match_str = cells[1]  # 格式: "Brazil v Chile" 或 "Chile v Brazil"
            result = cells[2]     # W/L/D
            score = cells[3]      # "3-0"
            competition = cells[4]

            # 跳过未完赛的比赛（没有比分）
            if not score or "-" not in score:
                continue

            # 解析比分
            score_m = re.match(r"(\d+)\s*[-–]\s*(\d+)", score)
            if not score_m:
                continue
            hg = int(score_m.group(1))
            ag = int(score_m.group(2))

            # 解析队名（格式: "Home v Away"）
            teams = match_str.split(" v ")
            if len(teams) != 2:
                continue
            home = normalize_name(teams[0])
            away = normalize_name(teams[1])

            # 计算结果
            if hg > ag:
                result_code = "H"
            elif hg < ag:
                result_code = "A"
            else:
                result_code = "D"

            matches.append({
                "home": home,
                "away": away,
                "home_goals": hg,
                "away_goals": ag,
                "result": result_code,
                "date": date_str,
                "competition": competition,
                "season": str(year),
            })

    except Exception as e:
        logger.error(f"爬取 {slug} {year} 失败: {e}")

    return matches


# ═══════════════════════════════════════════════════════════════
#  爬取单队全部赛季
# ═══════════════════════════════════════════════════════════════

def scrape_team(team_en: str, slug: str) -> list[dict]:
    """爬取单支球队所有赛季"""
    all_matches = []
    for year in SEASONS:
        matches = scrape_season(slug, year)
        all_matches.extend(matches)
        logger.info(f"  {team_en} {year}: {len(matches)}场")
        time.sleep(0.3)  # 礼貌延迟
    return all_matches


# ═══════════════════════════════════════════════════════════════
#  批量爬取全部48队
# ═══════════════════════════════════════════════════════════════

def scrape_all_teams() -> list[dict]:
    """爬取全部48支球队近10年比赛"""
    all_matches = []
    total = len(TEAM_SLUGS)

    for i, (team_en, slug) in enumerate(TEAM_SLUGS.items(), 1):
        logger.info(f"[{i}/{total}] 爬取 {team_en}...")
        matches = scrape_team(team_en, slug)
        all_matches.extend(matches)
        logger.info(f"  合计: {len(matches)}场")

    # 去重
    seen = set()
    unique = []
    for m in all_matches:
        key = (m["home"], m["away"], m["date"], m["home_goals"], m["away_goals"])
        if key not in seen:
            seen.add(key)
            unique.append(m)

    logger.info(f"总计: {len(all_matches)}场（去重后 {len(unique)} 场）")
    return unique


# ═══════════════════════════════════════════════════════════════
#  导入 Neo4j
# ═══════════════════════════════════════════════════════════════

def import_to_neo4j(matches: list[dict]):
    """将比赛数据导入 Neo4j"""
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
            # 确保两队都是48支参赛队之一
            home_std = normalize_team_name(m["home"])
            away_std = normalize_team_name(m["away"])

            if not home_std or not away_std:
                skipped += 1
                continue

            # 如果队名和标准名不同但能匹配，用标准名
            if home_std != m["home"]:
                m["home"] = home_std
            if away_std != m["away"]:
                m["away"] = away_std

            result_text = f"{m['home_goals']}-{m['away_goals']}"
            result_desc = f"{m['home']}胜" if m["result"] == "H" else (
                f"{m['away']}胜" if m["result"] == "A" else "平局"
            )

            s.run("""
                MATCH (h:NationalTeam {name: $home}),
                      (a:NationalTeam {name: $away})
                MERGE (h)-[r:PLAYED_AGAINST {
                    match_date: $date,
                    season: $season
                }]->(a)
                SET r.competition = $comp,
                    r.match_result = $result,
                    r.result_desc = $desc,
                    r.home_goals = $hg,
                    r.away_goals = $ag,
                    r.total_goals = $total
            """, home=m["home"], away=m["away"], date=m["date"],
                season=m.get("season", "11v11"),
                comp=m["competition"],
                result=result_text, desc=result_desc,
                hg=m["home_goals"], ag=m["away_goals"],
                total=m["home_goals"] + m["away_goals"])
            imported += 1

    driver.close()
    logger.info(f"导入 Neo4j: {imported}场, 跳过 {skipped}场")
    return imported


# ═══════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 60)
    print("  11v11.com 国家队历史比赛爬虫")
    print(f"  48支球队 × {len(SEASONS)}个赛季 ({SEASONS[0]}-{SEASONS[-1]})")
    print("=" * 60)

    # 1. 爬取
    print("\n[1/2] 爬取48支球队历史比赛...")
    matches = scrape_all_teams()

    if not matches:
        print("❌ 未爬取到比赛数据")
        return

    # 统计
    comp_counts = Counter(m["competition"] for m in matches)
    print(f"\n比赛统计:")
    for comp, cnt in comp_counts.most_common(15):
        print(f"  {comp}: {cnt}场")

    # 保存备份
    backup_path = os.path.join(PROJECT_ROOT, "data", "intl_matches_11v11.json")
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)
    print(f"\n备份: {backup_path}")

    # 2. 导入 Neo4j
    print(f"\n[2/2] 导入 Neo4j...")
    try:
        import_to_neo4j(matches)
    except Exception as e:
        print(f"❌ Neo4j 导入失败: {e}")

    print("=" * 60)


if __name__ == "__main__":
    main()
