# -*- coding: utf-8 -*-
"""
titan007 2022 世界杯赔率爬虫

从 titan007.com 爬取 2022 卡塔尔世界杯全部 64 场比赛的:
  - Bet365 初盘赔率（B365H/D/A）
  - Bet365 终盘(即时)赔率（B365CH/D/CA）
  - 比赛结果（FTHG/FTAG/FTR）

输出: agents/predicted_agent/models/saved/wc2022_test.csv
       列格式与 football-data.co.uk 联赛 CSV 兼容，可直接喂给 build_features()

爬取流程（已验证可行）:
  1. 2022.titan007.com 比分页面 → 从 scheduleStr JS 数据提取所有 match_id
  2. 1x2d.titan007.com/{match_id}.js → 解析 Bet365 初盘+即时赔率（352 家博彩公司，Bet365 是第一条）
  3. 比分页面本身已包含比赛结果

用法:
  python -m agents.predicted_agent.scripts.wc2022_odds_scraper
"""

import os
import re
import time
import sys
import httpx
import pandas as pd

# 输出路径（与 statistical_model.py 中的 WC2022_TEST_PATH 一致）
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "saved"
)
OUTPUT_PATH = os.path.join(MODEL_DIR, "wc2022_test.csv")

# titan007 数据源
SCORE_PAGE_URL = "https://2022.titan007.com/"
ODDS_JS_URL = "https://1x2d.titan007.com/{match_id}.js"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://op1.titan007.com/",
}

# Bet365 在 titan007 博彩公司列表中的标识关键词
BET365_KEYWORDS = ["Bet 365", "bet365", "Bet365"]


# ═══════════════════════════════════════════════════════════════
#  Step 1: 从比分页面提取所有比赛 + match_id
# ═══════════════════════════════════════════════════════════════

def fetch_score_page() -> str:
    """获取 2022 世界杯比分页面 HTML"""
    print(f"[Step 1] 获取比分页面: {SCORE_PAGE_URL}")
    resp = httpx.get(SCORE_PAGE_URL, headers=HEADERS, timeout=20, follow_redirects=True)
    print(f"  状态码: {resp.status_code}, 长度: {len(resp.text)} 字符")
    if resp.status_code != 200:
        raise RuntimeError(f"比分页面请求失败: {resp.status_code}")
    return resp.text


def parse_matches_from_score_page(html: str) -> list[dict]:
    """
    从比分页面 HTML 解析比赛列表

    titan007 比分页面用 HTML 表格展示全部比赛:
      <tr id="tr1_{match_id}" index="N">
        <td class="gamedate">日期</td>
        <td id="time_{match_id}"><font color=red><b>完</b></font></td>  <!-- 完赛标记 -->
        <td class="home"><a>主队</a></td>
        <td><a>2-2</a></td>  <!-- 比分 -->
        <td class="guest"><a>客队</a></td>
        ...
      </tr>

    返回: [{"match_id", "date", "home_team", "away_team",
            "home_goals", "away_goals", "finished"}, ...]
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    matches = []

    # 找所有 id="tr1_{match_id}" 的 tr 元素
    for tr in soup.find_all("tr", id=re.compile(r"^tr1_\d{7}$")):
        tr_id = tr.get("id", "")
        match_id = tr_id.replace("tr1_", "")
        if not match_id.isdigit():
            continue

        tds = tr.find_all("td")
        if len(tds) < 5:
            continue

        # 提取各字段
        date = ""
        home_team = ""
        away_team = ""
        score_text = ""
        finished = False

        for td in tds:
            cls = td.get("class", [""])[0] if td.get("class") else ""
            td_text = td.get_text(strip=True)

            if cls == "gamedate":
                date = td_text
            elif cls == "home":
                home_team = td_text
            elif cls == "guest":
                away_team = td_text
            elif td_text and re.match(r"^\d+-\d+$", td_text):
                # 比分格式: 2-2 或 2-1
                score_text = td_text

            # 检查完赛标记（含"完"字）
            if "完" in td_text:
                finished = True

        # 解析比分
        home_goals = None
        away_goals = None
        if score_text and "-" in score_text:
            parts = score_text.split("-")
            if len(parts) == 2:
                try:
                    home_goals = int(parts[0])
                    away_goals = int(parts[1])
                except ValueError:
                    pass

        if not home_team or not away_team:
            continue

        matches.append({
            "match_id": match_id,
            "date": date,
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": home_goals if finished else None,
            "away_goals": away_goals if finished else None,
            "finished": finished,
        })

    print(f"  解析到 {len(matches)} 场比赛（含未完赛）")
    finished_count = sum(1 for m in matches if m["finished"])
    print(f"  其中已完赛: {finished_count} 场")
    return matches


# ═══════════════════════════════════════════════════════════════
#  Step 2: 从赔率 JS 文件提取 Bet365 初盘 + 终盘
# ═══════════════════════════════════════════════════════════════

def fetch_odds_js(match_id: str) -> str:
    """获取单场比赛的赔率 JS 数据文件"""
    url = ODDS_JS_URL.format(match_id=match_id)
    headers = {**HEADERS, "Referer": f"https://op1.titan007.com/oddslist/{match_id}.htm"}
    resp = httpx.get(url, headers=headers, timeout=20, follow_redirects=True)
    if resp.status_code != 200:
        return ""
    return resp.content.decode("utf-8", errors="ignore")


def parse_bet365_odds(js_content: str) -> dict:
    """
    从赔率 JS 文件解析 Bet365 的初盘 + 即时(终盘)赔率

    JS 数据格式:
      var game=Array("field1|field2|...|fieldN","...",...);

    每条记录字段（| 分隔）:
      [0]  博彩公司ID
      [2]  博彩公司名称
      [3]  初盘主胜赔率
      [4]  初盘平局赔率
      [5]  初盘客胜赔率
      [10] 即时主胜赔率
      [11] 即时平局赔率
      [12] 即时客胜赔率

    返回: {"b365h", "b365d", "b365a", "b365ch", "b365cd", "b365ca"} 或 None
    """
    if not js_content:
        return None

    # 提取 game 数组
    game_match = re.search(r"var\s+game\s*=\s*Array\((.+?)\);", js_content, re.DOTALL)
    if not game_match:
        return None

    game_str = game_match.group(1)
    # 每条记录用双引号包裹
    entries = re.findall(r'"([^"]+)"', game_str)

    for entry in entries:
        fields = entry.split("|")
        if len(fields) < 13:
            continue

        company = fields[2] if len(fields) > 2 else ""
        # 匹配 Bet365
        if any(kw.lower() in company.lower() for kw in BET365_KEYWORDS):
            try:
                return {
                    "b365h": float(fields[3]),
                    "b365d": float(fields[4]),
                    "b365a": float(fields[5]),
                    "b365ch": float(fields[10]),
                    "b365cd": float(fields[11]),
                    "b365ca": float(fields[12]),
                }
            except (ValueError, IndexError):
                continue

    return None


# ═══════════════════════════════════════════════════════════════
#  Step 3: 组装 CSV
# ═══════════════════════════════════════════════════════════════

def build_csv_rows(matches: list[dict]) -> list[dict]:
    """爬取所有完赛比赛的赔率，组装成 CSV 行"""
    rows = []
    total = len(matches)
    success = 0
    failed = 0

    for i, m in enumerate(matches, 1):
        if not m["finished"]:
            continue

        print(f"[{i}/{total}] {m['home_team']} vs {m['away_team']} "
              f"({m['home_goals']}:{m['away_goals']})...", end=" ")

        # 爬取赔率
        js = fetch_odds_js(m["match_id"])
        odds = parse_bet365_odds(js)

        if odds is None:
            print("❌ 未找到 Bet365 赔率")
            failed += 1
            time.sleep(0.3)
            continue

        # 确定 FTR（全场结果）
        hg, ag = m["home_goals"], m["away_goals"]
        if hg > ag:
            ftr = "H"
        elif hg < ag:
            ftr = "A"
        else:
            ftr = "D"

        row = {
            "Date": m["date"],
            "HomeTeam": m["home_team"],
            "AwayTeam": m["away_team"],
            "FTHG": hg,
            "FTAG": ag,
            "FTR": ftr,
            "B365H": odds["b365h"],
            "B365D": odds["b365d"],
            "B365A": odds["b365a"],
            "B365CH": odds["b365ch"],
            "B365CD": odds["b365cd"],
            "B365CA": odds["b365ca"],
        }
        rows.append(row)
        success += 1
        print(f"✅ 初盘 {odds['b365h']}/{odds['b365d']}/{odds['b365a']} "
              f"终盘 {odds['b365ch']}/{odds['b365cd']}/{odds['b365ca']}")

        time.sleep(0.3)  # 礼貌延迟

    print(f"\n[完成] 成功 {success} 场, 失败 {failed} 场")
    return rows


def main():
    """主入口：爬取 2022 世界杯全部比赛赔率并保存为 CSV"""
    print("=" * 60)
    print("  2022 世界杯赔率爬虫 (titan007.com)")
    print("=" * 60)

    # Step 1: 获取比分页面
    html = fetch_score_page()

    # Step 2: 解析比赛列表
    matches = parse_matches_from_score_page(html)
    if not matches:
        print("❌ 未解析到任何比赛")
        sys.exit(1)

    # Step 3: 爬取赔率
    rows = build_csv_rows(matches)
    if not rows:
        print("❌ 未能爬取到任何赔率数据")
        sys.exit(1)

    # Step 4: 保存 CSV
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[保存] {len(df)} 场比赛已保存到: {OUTPUT_PATH}")
    print(f"  列: {list(df.columns)}")
    print(f"  示例:")
    print(df.head(3).to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
