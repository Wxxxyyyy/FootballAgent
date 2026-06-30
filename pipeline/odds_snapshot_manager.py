# -*- coding: utf-8 -*-
"""
赔率快照管理模块（Odds Snapshot Manager）

功能:
  1. 管理每场比赛的赔率历史快照
  2. 第一次调用: 爬取初盘 + 当时的实时赔率
  3. 后续每1h: 只爬实时赔率（更新"终盘"）
  4. 预测时: 读取初盘(首次记录) + 最新实时赔率(终盘)

数据源: titan007.com
  - 比分页面: {year}.titan007.com → 提取 match_id
  - 赔率JS文件: 1x2d.titan007.com/{match_id}.js → Bet365 初盘+即时赔率

存储格式: data/odds_snapshots/{match_id}.json
  {
    "match_id": "2906973",
    "home_team": "葡萄牙",
    "away_team": "乌兹别克斯坦",
    "kickoff_time": "2026-06-24T01:00:00",
    "opening_odds": {           # 初盘（第一次记录，不再更新）
      "b365h": 1.22, "b365d": 6.00, "b365a": 10.00,
      "timestamp": "2026-06-23T10:00:00"
    },
    "closing_odds": {           # 终盘（每次更新实时赔率时覆盖）
      "b365h": 1.14, "b365d": 8.00, "b365a": 17.00,
      "timestamp": "2026-06-24T00:00:00"
    },
    "history": [                # 赔率变化历史
      {"b365h": 1.22, "b365d": 6.00, "b365a": 10.00, "timestamp": "2026-06-23T10:00:00"},
      {"b365h": 1.18, "b365d": 7.00, "b365a": 13.00, "timestamp": "2026-06-23T11:00:00"},
      ...
    ]
  }

用法:
  from pipeline.odds_snapshot_manager import OddsSnapshotManager
  mgr = OddsSnapshotManager()
  mgr.update_snapshot("2906973")  # 爬取并更新赔率快照
  odds = mgr.get_odds("2906973")  # 获取初盘+终盘
  # odds = {"b365h": 1.22, "b365d": 6.00, "b365a": 10.00,
  #         "b365ch": 1.14, "b365cd": 8.00, "b365ca": 17.00}
"""

import os
import json
import re
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = os.getenv(
    "ODDS_SNAPSHOT_DIR",
    str(PROJECT_ROOT / "data" / "odds_snapshots"),
)

# titan007 赔率JS文件URL
ODDS_JS_URL = "https://1x2d.titan007.com/{match_id}.js"

# Bet365 在 titan007 博彩公司列表中的标识
BET365_KEYWORDS = ["Bet 365", "bet365", "Bet365"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://op1.titan007.com/",
}


class OddsSnapshotManager:
    """赔率快照管理器"""

    def __init__(self, snapshot_dir: str = None):
        self.snapshot_dir = snapshot_dir or SNAPSHOT_DIR
        os.makedirs(self.snapshot_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    #  爬取单场赔率
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def fetch_odds_from_titan007(match_id: str) -> Optional[dict]:
        """
        从 titan007 爬取单场比赛的 Bet365 初盘 + 即时赔率

        Args:
            match_id: titan007 match_id（7位数字）

        Returns:
            {
                "b365h": 初盘主胜, "b365d": 初盘平局, "b365a": 初盘客胜,
                "b365ch": 即时主胜, "b365cd": 即时平局, "b365ca": 即时客胜,
                "home_team": "队名", "away_team": "队名",
            }
            或 None（爬取失败）
        """
        url = ODDS_JS_URL.format(match_id=match_id)
        headers = {**HEADERS, "Referer": f"https://op1.titan007.com/oddslist/{match_id}.htm"}

        try:
            resp = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
            if resp.status_code != 200:
                logger.warning(f"titan007 赔率页面返回 {resp.status_code}: match_id={match_id}")
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
            logger.warning(f"未找到赔率数据: match_id={match_id}")
            return None

        entries = re.findall(r'"([^"]+)"', game_match.group(1))

        for entry in entries:
            fields = entry.split("|")
            if len(fields) < 13:
                continue

            company = fields[2] if len(fields) > 2 else ""
            # 匹配 Bet365
            if any(kw.lower() in company.lower() for kw in BET365_KEYWORDS):
                try:
                    return {
                        "b365h": float(fields[3]),   # 初盘主胜
                        "b365d": float(fields[4]),   # 初盘平局
                        "b365a": float(fields[5]),   # 初盘客胜
                        "b365ch": float(fields[10]), # 即时主胜
                        "b365cd": float(fields[11]), # 即时平局
                        "b365ca": float(fields[12]), # 即时客胜
                        "home_team": home_team,
                        "away_team": away_team,
                    }
                except (ValueError, IndexError) as e:
                    logger.warning(f"解析 Bet365 赔率失败: {e}")
                    continue

        logger.warning(f"未找到 Bet365 赔率: match_id={match_id}")
        return None

    # ═══════════════════════════════════════════════════════════
    #  快照管理
    # ═══════════════════════════════════════════════════════════

    def _snapshot_path(self, match_id: str) -> str:
        """获取快照文件路径"""
        return os.path.join(self.snapshot_dir, f"{match_id}.json")

    def load_snapshot(self, match_id: str) -> Optional[dict]:
        """加载已保存的赔率快照"""
        path = self._snapshot_path(match_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载快照失败 {match_id}: {e}")
            return None

    def update_snapshot(self, match_id: str, kickoff_time: str = None) -> Optional[dict]:
        """
        更新单场比赛的赔率快照

        逻辑:
          - 如果快照不存在（第一次调用）:
            → 记录初盘 + 实时赔率（初盘 = titan007 的初盘，终盘 = 即时赔率）
          - 如果快照已存在（后续调用）:
            → 只更新终盘（实时赔率），初盘保持不变

        Args:
            match_id: titan007 match_id
            kickoff_time: 开赛时间 ISO 格式（可选）

        Returns:
            更新后的快照 dict
        """
        # 爬取最新赔率
        odds = self.fetch_odds_from_titan007(match_id)
        if odds is None:
            return None

        now = datetime.now().isoformat()
        snapshot = self.load_snapshot(match_id)

        if snapshot is None:
            # 第一次调用：记录初盘 + 实时赔率
            snapshot = {
                "match_id": match_id,
                "home_team": odds.get("home_team", ""),
                "away_team": odds.get("away_team", ""),
                "kickoff_time": kickoff_time,
                "opening_odds": {
                    "b365h": odds["b365h"],
                    "b365d": odds["b365d"],
                    "b365a": odds["b365a"],
                    "timestamp": now,
                },
                "closing_odds": {
                    "b365h": odds["b365ch"],
                    "b365d": odds["b365cd"],
                    "b365a": odds["b365ca"],
                    "timestamp": now,
                },
                "history": [
                    {
                        "b365h": odds["b365h"],
                        "b365d": odds["b365d"],
                        "b365a": odds["b365a"],
                        "timestamp": now,
                        "type": "opening",
                    },
                    {
                        "b365h": odds["b365ch"],
                        "b365d": odds["b365cd"],
                        "b365a": odds["b365ca"],
                        "timestamp": now,
                        "type": "live",
                    },
                ],
            }
            logger.info(f"[赔率快照] 首次记录 {match_id}: "
                        f"初盘 {odds['b365h']}/{odds['b365d']}/{odds['b365a']} → "
                        f"实时 {odds['b365ch']}/{odds['b365cd']}/{odds['b365ca']}")
        else:
            # 后续调用：只更新终盘
            snapshot["closing_odds"] = {
                "b365h": odds["b365ch"],
                "b365d": odds["b365cd"],
                "b365a": odds["b365ca"],
                "timestamp": now,
            }
            snapshot["history"].append({
                "b365h": odds["b365ch"],
                "b365d": odds["b365cd"],
                "b365a": odds["b365ca"],
                "timestamp": now,
                "type": "live",
            })
            logger.info(f"[赔率快照] 更新终盘 {match_id}: "
                        f"{odds['b365ch']}/{odds['b365cd']}/{odds['b365ca']}")

        # 保存
        self._save_snapshot(match_id, snapshot)
        return snapshot

    def _save_snapshot(self, match_id: str, snapshot: dict):
        """保存快照到文件"""
        path = self._snapshot_path(match_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

    # ═══════════════════════════════════════════════════════════
    #  获取赔率（供 ML 模型使用）
    # ═══════════════════════════════════════════════════════════

    def get_odds(self, match_id: str) -> Optional[dict]:
        """
        获取单场比赛的初盘 + 终盘赔率（供 ML 模型使用）

        Returns:
            {
                "B365H": 初盘主胜, "B365D": 初盘平局, "B365A": 初盘客胜,
                "B365CH": 终盘主胜, "B365CD": 终盘平局, "B365CA": 终盘客胜,
                "home_team": "队名", "away_team": "队名",
            }
            或 None（快照不存在）
        """
        snapshot = self.load_snapshot(match_id)
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

    def get_odds_change(self, match_id: str) -> Optional[dict]:
        """
        获取赔率变化（终盘 - 初盘）

        Returns:
            {"move_h": float, "move_d": float, "move_a": float}
        """
        odds = self.get_odds(match_id)
        if odds is None or not odds.get("B365H"):
            return None

        return {
            "move_h": odds["B365CH"] - odds["B365H"],
            "move_d": odds["B365CD"] - odds["B365D"],
            "move_a": odds["B365CA"] - odds["B365A"],
        }

    # ═══════════════════════════════════════════════════════════
    #  批量更新
    # ═══════════════════════════════════════════════════════════

    def update_all_upcoming(self, matches: list[dict]):
        """
        批量更新即将开始的比赛的赔率快照

        Args:
            matches: fetch_today_matches() 返回的比赛列表
        """
        updated = 0
        for m in matches:
            if m.get("finished"):
                continue
            hours = m.get("hours_to_kickoff")
            if hours is None or hours < 0 or hours > 48:
                continue

            match_id = m["match_id"]
            kickoff = m.get("kickoff_time")

            try:
                self.update_snapshot(match_id, kickoff_time=kickoff)
                updated += 1
                time.sleep(0.5)  # 礼貌延迟
            except Exception as e:
                logger.error(f"更新赔率快照失败 {match_id}: {e}")

        logger.info(f"批量更新赔率快照完成: {updated}/{len(matches)} 场")
        return updated


# ═══════════════════════════════════════════════════════════════
#  模块级单例
# ═══════════════════════════════════════════════════════════════

_manager: Optional[OddsSnapshotManager] = None


def get_manager() -> OddsSnapshotManager:
    """获取 OddsSnapshotManager 单例"""
    global _manager
    if _manager is None:
        _manager = OddsSnapshotManager()
    return _manager


# ═══════════════════════════════════════════════════════════════
#  命令行入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import sys

    if len(sys.argv) > 1:
        match_id = sys.argv[1]
        mgr = OddsSnapshotManager()

        if len(sys.argv) > 2 and sys.argv[2] == "get":
            # 获取赔率
            odds = mgr.get_odds(match_id)
            if odds:
                print(f"\n{odds['home_team']} vs {odds['away_team']}")
                print(f"初盘: {odds['B365H']}/{odds['B365D']}/{odds['B365A']}")
                print(f"终盘: {odds['B365CH']}/{odds['B365CD']}/{odds['B365CA']}")
                change = mgr.get_odds_change(match_id)
                if change:
                    print(f"变化: 主{change['move_h']:+.2f} 平{change['move_d']:+.2f} 客{change['move_a']:+.2f}")
            else:
                print("快照不存在")
        else:
            # 更新快照
            print(f"爬取 match_id={match_id} 的赔率...")
            result = mgr.update_snapshot(match_id)
            if result:
                print(f"\n{result['home_team']} vs {result['away_team']}")
                print(f"初盘: {result['opening_odds']['b365h']}/{result['opening_odds']['b365d']}/{result['opening_odds']['b365a']}")
                print(f"终盘: {result['closing_odds']['b365h']}/{result['closing_odds']['b365d']}/{result['closing_odds']['b365a']}")
                print(f"历史记录: {len(result['history'])} 条")
            else:
                print("爬取失败")
    else:
        print("用法:")
        print("  python odds_snapshot_manager.py {match_id}        # 爬取/更新赔率快照")
        print("  python odds_snapshot_manager.py {match_id} get    # 查看已保存的赔率")
