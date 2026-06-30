# -*- coding: utf-8 -*-
"""
OpenClaw 赔率爬虫

从 titan007.com 爬取 Bet365 赔率，使用 httpx 直接请求 JS 数据文件（无需 Playwright）。

数据源:
  1. 比分页面 {year}.titan007.com → 获取比赛列表 + match_id
  2. 赔率JS文件 1x2d.titan007.com/{match_id}.js → Bet365 初盘+即时赔率

赔率快照管理:
  - 第一次调用: 记录初盘 + 实时赔率
  - 后续调用: 只更新终盘（实时赔率）
"""

import os
import re
import time
import json
import logging
from datetime import datetime
from typing import Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

TITAN007_SCORE_URL = os.getenv("WORLDCUP_SCORE_URL", "https://2026.titan007.com/")
ODDS_JS_URL = "https://1x2d.titan007.com/{match_id}.js"
BET365_KEYWORDS = ["Bet 365", "bet365", "Bet365"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}


# ═══════════════════════════════════════════════════════════════
#  比赛列表获取
# ═══════════════════════════════════════════════════════════════

def fetch_match_list() -> list[dict]:
    """
    从 titan007 比分页面获取比赛列表

    返回: [{"match_id", "home_team", "away_team", "date", "kickoff_time",
            "hours_to_kickoff", "finished", "score"}, ...]
    """
    try:
        resp = httpx.get(TITAN007_SCORE_URL, headers=HEADERS, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            logger.warning(f"比分页面返回 {resp.status_code}")
            return []
    except Exception as e:
        logger.error(f"获取比赛列表失败: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    matches = []
    seen_match_ids = set()  # 按 match_id 去重，避免同一场比赛多行重复
    now = datetime.now()

    for tr in soup.find_all("tr", id=re.compile(r"^(tr\d+_|near_tr_)\d{7}$")):
        tr_id = tr.get("id", "")
        match_id = re.sub(r"^(tr\d+_|near_tr_)", "", tr_id)
        if not match_id.isdigit():
            continue

        # 同一 match_id 只取第一行，跳过后续重复行
        if match_id in seen_match_ids:
            continue
        seen_match_ids.add(match_id)

        tds = tr.find_all("td")
        if len(tds) < 5:
            continue

        date_str = ""
        home_team = ""
        away_team = ""
        score = ""
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

        # 通过 score td 定位主客队
        if score_idx is not None:
            score = tds[score_idx].get_text(strip=True)
            if score_idx > 0:
                home_team = tds[score_idx - 1].get_text(strip=True)
            if score_idx + 1 < len(tds):
                away_team = tds[score_idx + 1].get_text(strip=True)

        if not home_team or not away_team:
            continue

        # 解析开赛时间
        kickoff_time = None
        hours_to_kickoff = None
        try:
            year = now.year
            kickoff_time = datetime.strptime(f"{year}-{date_str}", "%Y-%m-%d %H:%M")
            hours_to_kickoff = (kickoff_time - now).total_seconds() / 3600
        except (ValueError, TypeError):
            pass

        matches.append({
            "match_id": match_id,
            "home_team": home_team,
            "away_team": away_team,
            "date": date_str,
            "kickoff_time": kickoff_time.isoformat() if kickoff_time else None,
            "hours_to_kickoff": hours_to_kickoff,
            "finished": finished,
            "score": score,
        })

    return matches


# ═══════════════════════════════════════════════════════════════
#  赔率爬取
# ═══════════════════════════════════════════════════════════════

def fetch_bet365_odds(match_id: str) -> Optional[dict]:
    """
    从 titan007 爬取单场比赛的 Bet365 初盘 + 即时赔率

    数据源: 1x2d.titan007.com/{match_id}.js
    JS文件中 Bet365 是第一条记录，字段用 | 分隔:
      fields[3-5] = 初盘赔率, fields[10-12] = 即时赔率

    返回: {"b365h", "b365d", "b365a", "b365ch", "b365cd", "b365ca",
           "home_team", "away_team"} 或 None
    """
    url = ODDS_JS_URL.format(match_id=match_id)
    headers = {**HEADERS, "Referer": f"https://op1.titan007.com/oddslist/{match_id}.htm"}

    try:
        resp = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            return None
    except Exception as e:
        logger.error(f"爬取赔率失败 match_id={match_id}: {e}")
        return None

    content = resp.content.decode("utf-8", errors="ignore")

    # 提取队名
    home_team = ""
    away_team = ""
    home_m = re.search(r'var hometeam_cn="([^"]+)"', content)
    away_m = re.search(r'var guestteam_cn="([^"]+)"', content)
    if home_m:
        home_team = home_m.group(1)
    if away_m:
        away_team = away_m.group(1)

    # 提取 game 数组
    game_match = re.search(r'var\s+game\s*=\s*Array\((.+?)\);', content, re.DOTALL)
    if not game_match:
        return None

    entries = re.findall(r'"([^"]+)"', game_match.group(1))

    for entry in entries:
        fields = entry.split("|")
        if len(fields) < 13:
            continue

        company = fields[2] if len(fields) > 2 else ""
        if any(kw.lower() in company.lower() for kw in BET365_KEYWORDS):
            try:
                return {
                    "b365h": float(fields[3]),
                    "b365d": float(fields[4]),
                    "b365a": float(fields[5]),
                    "b365ch": float(fields[10]),
                    "b365cd": float(fields[11]),
                    "b365ca": float(fields[12]),
                    "home_team": home_team,
                    "away_team": away_team,
                }
            except (ValueError, IndexError) as e:
                logger.warning(f"解析 Bet365 赔率失败: {e}")
                continue

    return None


# ═══════════════════════════════════════════════════════════════
#  赔率快照管理
# ═══════════════════════════════════════════════════════════════

class OddsSnapshotManager:
    """赔率快照管理器（容器内文件存储）"""

    def __init__(self, snapshot_dir: str = "/app/data/odds_snapshots"):
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def _path(self, match_id: str) -> str:
        return os.path.join(self.snapshot_dir, f"{match_id}.json")

    def load(self, match_id: str) -> Optional[dict]:
        path = self._path(match_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return None

    def update(self, match_id: str, kickoff_time: str = None) -> Optional[dict]:
        """
        更新赔率快照

        第一次调用: 记录初盘 + 实时赔率
        后续调用: 只更新终盘（实时赔率）
        """
        odds = fetch_bet365_odds(match_id)
        if odds is None:
            return None

        now = datetime.now().isoformat()
        snapshot = self.load(match_id)

        if snapshot is None:
            # 第一次：记录初盘 + 实时
            snapshot = {
                "match_id": match_id,
                "home_team": odds.get("home_team", ""),
                "away_team": odds.get("away_team", ""),
                "kickoff_time": kickoff_time,
                "opening_odds": {
                    "b365h": odds["b365h"], "b365d": odds["b365d"], "b365a": odds["b365a"],
                    "timestamp": now,
                },
                "closing_odds": {
                    "b365h": odds["b365ch"], "b365d": odds["b365cd"], "b365a": odds["b365ca"],
                    "timestamp": now,
                },
                "history": [{
                    "b365h": odds["b365h"], "b365d": odds["b365d"], "b365a": odds["b365a"],
                    "timestamp": now, "type": "opening",
                }, {
                    "b365h": odds["b365ch"], "b365d": odds["b365cd"], "b365a": odds["b365ca"],
                    "timestamp": now, "type": "live",
                }],
            }
            logger.info(f"[赔率快照] 首次记录 {match_id}: "
                        f"初盘 {odds['b365h']}/{odds['b365d']}/{odds['b365a']} → "
                        f"实时 {odds['b365ch']}/{odds['b365cd']}/{odds['b365ca']}")
        else:
            # 后续：只更新终盘
            snapshot["closing_odds"] = {
                "b365h": odds["b365ch"], "b365d": odds["b365cd"], "b365a": odds["b365ca"],
                "timestamp": now,
            }
            snapshot["history"].append({
                "b365h": odds["b365ch"], "b365d": odds["b365cd"], "b365a": odds["b365ca"],
                "timestamp": now, "type": "live",
            })

        # 保存
        with open(self._path(match_id), "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

        return snapshot

    def get_odds(self, match_id: str) -> Optional[dict]:
        """获取初盘+终盘赔率"""
        snapshot = self.load(match_id)
        if snapshot is None:
            return None
        opening = snapshot.get("opening_odds", {})
        closing = snapshot.get("closing_odds", {})
        return {
            "B365H": opening.get("b365h"),
            "B365D": opening.get("b365d"),
            "B365A": opening.get("b365a"),
            "B365CH": closing.get("b365h"),
            "B365CD": closing.get("b365d"),
            "B365CA": closing.get("b365a"),
            "home_team": snapshot.get("home_team", ""),
            "away_team": snapshot.get("away_team", ""),
        }

    def update_all_upcoming(self, matches: list[dict]):
        """批量更新即将开赛的赔率快照"""
        updated = 0
        for m in matches:
            if m.get("finished"):
                continue
            hours = m.get("hours_to_kickoff")
            if hours is None or hours < -2 or hours > 48:
                continue
            try:
                self.update(m["match_id"], m.get("kickoff_time"))
                updated += 1
                time.sleep(0.3)
            except Exception as e:
                logger.error(f"更新赔率快照失败 {m['match_id']}: {e}")
        return updated


# ═══════════════════════════════════════════════════════════════
#  模块级单例
# ═══════════════════════════════════════════════════════════════

_snapshot_mgr: Optional[OddsSnapshotManager] = None


def get_snapshot_manager() -> OddsSnapshotManager:
    global _snapshot_mgr
    if _snapshot_mgr is None:
        _snapshot_mgr = OddsSnapshotManager()
    return _snapshot_mgr
