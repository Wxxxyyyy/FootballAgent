# -*- coding: utf-8 -*-
"""
懂球帝国家队历史比赛爬虫

从懂球帝网站爬取48支世界杯参赛队的历史比赛记录，
补充 Neo4j 中的国家队交锋数据（含友谊赛/洲际赛等）。

数据源: dongqiudi.com/team/{team_id}（赛程页面）
爬取方式: Playwright 渲染 JS → 提取赛程数据

用法:
  python -m pipeline.dongqiudi_match_scraper
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════════
#  48 支国家队在懂球帝的 team_id
# ═══════════════════════════════════════════════════════════════

# 中文名 → (懂球帝team_id, 标准英文名)
DQD_TEAM_MAP = {
    # A组
    "墨西哥": (1278, "Mexico"),
    "南非": (1753, "South Africa"),
    "韩国": (1181, "South Korea"),
    "捷克": (453, "Czech Republic"),
    # B组
    "加拿大": (303, "Canada"),
    "波黑": (219, "Bosnia & Herzegovina"),
    "卡塔尔": (1542, "Qatar"),
    "瑞士": (1931, "Switzerland"),
    # C组
    "巴西": (269, "Brazil"),
    "摩洛哥": (1289, "Morocco"),
    "海地": (916, "Haiti"),
    "苏格兰": (1683, "Scotland"),
    # D组
    "美国": (2008, "USA"),
    "巴拉圭": (1405, "Paraguay"),
    "澳大利亚": (87, "Australia"),
    "土耳其": (1977, "Turkey"),
    # E组
    "德国": (868, "Germany"),
    "库拉索": (1332, "Curacao"),
    "科特迪瓦": (454, "Ivory Coast"),
    "厄瓜多尔": (510, "Ecuador"),
    # F组
    "荷兰": (1331, "Netherlands"),
    "日本": (1146, "Japan"),
    "瑞典": (1904, "Sweden"),
    "突尼斯": (1941, "Tunisia"),
    # G组
    "比利时": (203, "Belgium"),
    "埃及": (511, "Egypt"),
    "伊朗": (986, "Iran"),
    "新西兰": (1341, "New Zealand"),
    # H组
    "西班牙": (1869, "Spain"),
    "佛得角": (304, "Cape Verde"),
    "沙特阿拉伯": (1640, "Saudi Arabia"),
    "乌拉圭": (2026, "Uruguay"),
    # I组
    "法国": (789, "France"),
    "塞内加尔": (1684, "Senegal"),
    "伊拉克": (987, "Iraq"),
    "挪威": (1389, "Norway"),
    # J组
    "阿根廷": (67, "Argentina"),
    "阿尔及利亚": (13, "Algeria"),
    "奥地利": (108, "Austria"),
    "约旦": (1147, "Jordan"),
    # K组
    "葡萄牙": (1540, "Portugal"),
    "刚果": (366, "DR Congo"),
    "乌兹别克斯坦": (2027, "Uzbekistan"),
    "哥伦比亚": (364, "Colombia"),
    # L组
    "英格兰": (627, "England"),
    "克罗地亚": (396, "Croatia"),
    "加纳": (869, "Ghana"),
    "巴拿马": (1393, "Panama"),
}


# ═══════════════════════════════════════════════════════════════
#  Playwright 爬取单队赛程
# ═══════════════════════════════════════════════════════════════

def scrape_team_matches(team_id: int, team_name_zh: str, team_name_en: str) -> list[dict]:
    """
    用 Playwright 爬取单支球队的比赛记录

    返回: [{"home", "away", "home_goals", "away_goals", "date", "competition", "result"}, ...]
    """
    from playwright.sync_api import sync_playwright

    matches = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
        page = browser.new_page()
        page.set_default_timeout(20000)

        try:
            url = f"https://www.dongqiudi.com/team/{team_id}"
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(3000)

            # 点击赛程tab
            page.click("text=赛程")
            page.wait_for_timeout(5000)

            # 获取页面HTML，用正则提取比赛数据
            html = page.evaluate("() => document.body.innerHTML")

            # 提取比赛信息
            # HTML格式: class="tp-schedule-row__teams"> 主队 比分 客队 Played
            # 日期在附近的元素中: 3/26 20:00
            # 赛事名称在附近: 友谊赛 / 世界杯 等

            # 方法：找所有 "队名A X - Y 队名B Played" 的模式
            team_match_pattern = re.findall(
                r'tp-schedule-row__teams[^>]*>\s*([^<]+?)\s+(\d+)\s*-\s*(\d+)\s+([^<]+?)\s+Played',
                html
            )

            # 同时提取日期和赛事
            date_pattern = re.findall(r'(\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2})', html)
            # 赛事名称通常在比赛前（友谊赛、世界杯等）
            comp_pattern = re.findall(r'(友谊赛|世界杯|World Cup|欧洲杯|美洲杯|非洲杯|亚洲杯|世预赛|欧预赛|预选赛|Qualifying|Qualification|Confederations|联合会杯| Nations League|国联赛)', html, re.IGNORECASE)

            for i, (home, hg, ag, away) in enumerate(team_match_pattern):
                home = home.strip()
                away = away.strip()

                # 标准化队名
                home_en = _normalize_team_name(home) or home
                away_en = _normalize_team_name(away) or away

                # 日期
                date_str = date_pattern[i] if i < len(date_pattern) else ""

                # 赛事
                competition = comp_pattern[i] if i < len(comp_pattern) else "未知"

                # 计算结果
                hg_int = int(hg)
                ag_int = int(ag)
                result = "H" if hg_int > ag_int else ("A" if hg_int < ag_int else "D")

                matches.append({
                    "home": home_en,
                    "away": away_en,
                    "home_goals": hg_int,
                    "away_goals": ag_int,
                    "result": result,
                    "date": date_str,
                    "competition": competition,
                })

            # 备选方案：如果正则没匹配到，用innerText解析
            if not matches:
                body_text = page.inner_text("body")
                # 找 "队名 比分 队名 Played" 格式
                text_matches = re.findall(
                    r'([^\n]+?)\s+(\d+)\s*-\s*(\d+)\s+([^\n]+?)\s+Played',
                    body_text
                )
                for home, hg, ag, away in text_matches:
                    home = home.strip().split('\n')[-1]  # 取最后一部分
                    away = away.strip().split('\n')[0]   # 取第一部分
                    home_en = _normalize_team_name(home) or home
                    away_en = _normalize_team_name(away) or away
                    result = "H" if int(hg) > int(ag) else ("A" if int(hg) < int(ag) else "D")
                    matches.append({
                        "home": home_en,
                        "away": away_en,
                        "home_goals": int(hg),
                        "away_goals": int(ag),
                        "result": result,
                        "date": "",
                        "competition": "未知",
                    })

        except Exception as e:
            logger.error(f"爬取 {team_name_zh} (id={team_id}) 失败: {e}")
        finally:
            browser.close()

    return matches


def _normalize_team_name(name: str) -> Optional[str]:
    """将中文队名标准化为英文"""
    from pipeline.national_team_neo4j_loader import normalize_team_name
    return normalize_team_name(name)


# ═══════════════════════════════════════════════════════════════
#  批量爬取所有48队
# ═══════════════════════════════════════════════════════════════

def scrape_all_teams() -> list[dict]:
    """爬取全部48支球队的历史比赛"""
    all_matches = []
    total = len(DQD_TEAM_MAP)

    for i, (zh_name, (team_id, en_name)) in enumerate(DQD_TEAM_MAP.items(), 1):
        logger.info(f"[{i}/{total}] 爬取 {zh_name} ({en_name}, id={team_id})...")
        try:
            matches = scrape_team_matches(team_id, zh_name, en_name)
            logger.info(f"  获取到 {len(matches)} 场比赛")
            all_matches.extend(matches)
        except Exception as e:
            logger.error(f"  失败: {e}")

        time.sleep(1)  # 礼貌延迟

    # 去重（同一场比赛可能从两队各爬一次）
    seen = set()
    unique = []
    for m in all_matches:
        key = (m["home"], m["away"], m["date"], m["home_goals"], m["away_goals"])
        if key not in seen:
            seen.add(key)
            unique.append(m)

    logger.info(f"总计: {len(all_matches)} 场（去重后 {len(unique)} 场）")
    return unique


# ═══════════════════════════════════════════════════════════════
#  导入 Neo4j
# ═══════════════════════════════════════════════════════════════

def import_to_neo4j(matches: list[dict]):
    """将比赛数据导入 Neo4j"""
    from neo4j import GraphDatabase
    from dotenv import load_dotenv
    load_dotenv()

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URL", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"),
              os.getenv("NEO4J_PASSWORD", "football123"))
    )

    imported = 0
    with driver.session() as s:
        for m in matches:
            # 确保两个队都是48支参赛队之一
            from pipeline.national_team_neo4j_loader import normalize_team_name
            home_std = normalize_team_name(m["home"])
            away_std = normalize_team_name(m["away"])

            if not home_std or not away_std:
                continue

            result_text = f"{m['home_goals']}-{m['away_goals']}"
            result_desc = f"{home_std}胜" if m["result"] == "H" else (
                f"{away_std}胜" if m["result"] == "A" else "平局"
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
            """, home=home_std, away=away_std, date=m["date"],
                season="dqd", comp=m["competition"],
                result=result_text, desc=result_desc,
                hg=m["home_goals"], ag=m["away_goals"],
                total=m["home_goals"] + m["away_goals"])
            imported += 1

    driver.close()
    logger.info(f"导入 Neo4j: {imported} 场")
    return imported


# ═══════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 60)
    print("  懂球帝国家队历史比赛爬虫")
    print("=" * 60)

    # 1. 爬取
    print("\n[1/2] 爬取48支球队赛程...")
    matches = scrape_all_teams()

    if not matches:
        print("❌ 未爬取到比赛数据")
        return

    # 统计
    from collections import Counter
    comp_counts = Counter(m["competition"] for m in matches)
    print(f"\n比赛统计:")
    for comp, cnt in comp_counts.most_common():
        print(f"  {comp}: {cnt}场")

    # 保存到JSON（备份）
    backup_path = os.path.join(PROJECT_ROOT, "data", "dongqiudi_matches.json")
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)
    print(f"\n备份保存: {backup_path}")

    # 2. 导入 Neo4j
    print(f"\n[2/2] 导入 Neo4j...")
    try:
        import_to_neo4j(matches)
    except Exception as e:
        print(f"❌ Neo4j 导入失败: {e}")
        print(f"  比赛数据已保存到 {backup_path}，可稍后手动导入")

    print("=" * 60)


if __name__ == "__main__":
    main()
