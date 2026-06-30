# -*- coding: utf-8 -*-
"""
赔率爬虫 - 基于球探网(titan007.com)的世界杯赔率采集

数据源:
  1. 比分页面 2026.titan007.com    → 获取比赛列表 + match_id
  2. 欧赔页面 op1.titan007.com      → Bet365 胜平负赔率
  3. 大小球页面 vip.titan007.com    → Bet365 大小球盘口+水位

为什么用球探网:
  Bet365 官网有 Cloudflare 反爬，球探网聚合了 Bet365 赔率，
  在国内服务器访问速度快(0.07s)，数据完整含胜平负+大小球+亚盘。

庄家编号:
  "36*" = Bet365（球探网固定编号）
  "澳*" = 澳门, "伟*" = 威廉希尔, "易胜*" = 易胜博 等

输出格式: 统一为 B365H/B365D/B365A/B365>2.5/B365<2.5/AHh
"""

import os
import re
import time
import threading
import logging
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from playwright.sync_api import sync_playwright, Browser, Page
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("openclaw.scraper")

# ═══════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════

# 球探网 2026 世界杯比分页面
SCORE_URL = os.getenv(
    "WORLDCUP_SCORE_URL",
    "https://2026.titan007.com/",
)

# 欧赔页面模板
EURO_ODDS_URL = "https://op1.titan007.com/oddslist/{match_id}.htm"

# 大小球页面模板
OVER_UNDER_URL = "https://vip.titan007.com/OverDown_n.aspx?id={match_id}"

# Bet365 在球探网的庄家编号
BET365_CODE = "36"

# 页面超时
PAGE_TIMEOUT = 20000
AJAX_WAIT = 6000  # AJAX 加载等待时间

# 每次循环最多抓取的比赛数（避免太慢）
MAX_MATCHES_PER_LOOP = 15


# ═══════════════════════════════════════════════════════════════
#  赔率爬虫
# ═══════════════════════════════════════════════════════════════

class OddsScraper:
    """球探网赔率爬虫"""

    def __init__(self):
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._lock = threading.Lock()  # Playwright 不是线程安全的
        self._init_browser()

    def _init_browser(self):
        """初始化 Playwright 浏览器"""
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
            )
            logger.info("Playwright Chromium 浏览器已启动")
        except Exception as e:
            logger.error(f"浏览器启动失败: {e}")
            raise

    # ═══════════════════════════════════════════════════════════
    #  Step 1: 从比分页面获取比赛列表
    # ═══════════════════════════════════════════════════════════

    def scrape_match_list(self) -> list[dict]:
        """
        从球探网比分页面获取世界杯比赛列表（用 httpx，不需要 JS 渲染）

        返回:
          [
            {
              "match_id": "2906973",
              "home": "葡萄牙",
              "away": "乌兹别克斯坦",
              "kickoff": "2026-06-23 15:00",
              "status": "upcoming" | "finished",
              "score": None | "2:0",
              "stage": "小组赛",
            },
            ...
          ]
        """
        try:
            logger.info(f"正在访问比分页面: {SCORE_URL}")
            resp = httpx.get(SCORE_URL, timeout=15, follow_redirects=True)
            if resp.status_code != 200:
                logger.error(f"比分页面 HTTP {resp.status_code}")
                return []

            soup = BeautifulSoup(resp.text, "lxml")
            matches = self._parse_match_table_bs(soup)
            logger.info(f"比分页面解析到 {len(matches)} 场比赛")
            return matches

        except Exception as e:
            logger.error(f"比分页面爬取失败: {e}")
            return []

    def _parse_match_table_bs(self, soup: BeautifulSoup) -> list[dict]:
        """用 BeautifulSoup 解析比分页面的比赛列表表格"""
        matches = []

        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue

            # 检查表头是否含"主队"/"客队"
            header_text = rows[0].get_text()
            if "主队" not in header_text and "客队" not in header_text:
                continue

            for row in rows[1:]:
                try:
                    match = self._parse_match_row_bs(row)
                    if match:
                        matches.append(match)
                except Exception:
                    continue

        return matches

    def _parse_match_row_bs(self, row) -> Optional[dict]:
        """用 BeautifulSoup 解析单行比赛数据"""
        cells = row.find_all("td")
        if len(cells) < 6:
            return None

        # 提取所有单元格文本
        texts = [c.get_text(strip=True) for c in cells]

        # 找"欧"链接提取 match_id
        links = row.find_all("a")
        match_id = None
        home_name = None
        away_name = None

        for a in links:
            href = a.get("href", "")
            text = a.get_text(strip=True)

            # 从欧赔链接提取 match_id
            m = re.search(r'/oddslist/(\d+)\.htm', href)
            if m:
                match_id = m.group(1)

            # 主队/客队链接（指向 team/Summary/）
            if '/team/Summary/' in href and text:
                if home_name is None:
                    home_name = text
                elif away_name is None:
                    away_name = text

        if not match_id or not home_name or not away_name:
            return None

        # 解析状态和比分
        status_text = ""
        score_text = ""
        time_text = ""
        stage_text = ""

        for t in texts:
            if t in ("未", "未开赛", "赛前"):
                status_text = "upcoming"
            elif t in ("完", "完场", "已完场"):
                status_text = "finished"
            elif t in ("进行", "进行中", "中场"):
                status_text = "live"
            if re.match(r'^\d+[:\-]\d+$', t):
                score_text = t
            if re.match(r'^\d{1,2}:\d{2}$', t) or re.match(r'^\d{2}-\d{2}\s', t):
                time_text = t
            if re.match(r'^第\d+轮|^小组|^[A-H]组', t):
                stage_text = t

        if not status_text:
            status_text = "finished" if score_text else "upcoming"

        return {
            "match_id": match_id,
            "home": home_name,
            "away": away_name,
            "kickoff": time_text,
            "status": status_text,
            "score": score_text if score_text else None,
            "stage": stage_text or "世界杯",
        }

    # ═══════════════════════════════════════════════════════════
    #  Step 2: 从欧赔页面获取胜平负赔率
    # ═══════════════════════════════════════════════════════════

    def scrape_euro_odds(self, match_id: str) -> Optional[dict]:
        """
        从欧赔页面获取 Bet365 胜平负赔率

        返回: {"B365H": 1.50, "B365D": 4.33, "B365A": 6.50} 或 None
        """
        url = EURO_ODDS_URL.format(match_id=match_id)

        with self._lock:
            page = self._browser.new_page()
            page.set_default_timeout(PAGE_TIMEOUT)

            try:
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_timeout(AJAX_WAIT)

                odds = self._parse_euro_odds(page)
                if odds:
                    logger.info(f"  [欧赔] {match_id}: H={odds['B365H']} D={odds['B365D']} A={odds['B365A']}")
                return odds

            except Exception as e:
                logger.error(f"  [欧赔] {match_id} 爬取失败: {e}")
                return None
            finally:
                page.close()

    def _parse_euro_odds(self, page: Page) -> Optional[dict]:
        """
        解析欧赔表格，找 Bet365("36*") 的赔率

        表格结构(Table 8):
          列: [空, 庄家名, 主胜, 平, 客胜, 主胜率%, 平局率%, 客胜率%, 返还率%, 凯利1, 凯利2, 凯利3, 时间, 操作]
        """
        tables = page.query_selector_all("table")
        for table in tables:
            rows = table.query_selector_all("tr")
            if len(rows) < 3:
                continue

            for row in rows:
                cells = row.query_selector_all("td")
                if len(cells) < 5:
                    continue

                # 庄家名在第2列（索引1）
                bookmaker = cells[1].inner_text().strip() if len(cells) > 1 else ""

                # 查找 Bet365（"36" 开头）
                if not bookmaker.startswith(BET365_CODE):
                    continue

                # 赔率在第3-5列（索引2,3,4）
                try:
                    b365h = float(cells[2].inner_text().strip())
                    b365d = float(cells[3].inner_text().strip())
                    b365a = float(cells[4].inner_text().strip())

                    # 验证赔率合理性
                    if b365h < 1.01 or b365d < 1.01 or b365a < 1.01:
                        continue

                    return {"B365H": b365h, "B365D": b365d, "B365A": b365a}
                except (ValueError, IndexError):
                    continue

        return None

    # ═══════════════════════════════════════════════════════════
    #  Step 3: 从大小球页面获取大小球赔率
    # ═══════════════════════════════════════════════════════════

    def scrape_over_under(self, match_id: str) -> Optional[dict]:
        """
        从大小球页面获取 Bet365 大小球数据

        球探网大小球格式: "盘口 大球水位 小球水位"
          例: "3/3.5 1.00 0.79" → 盘口3.25, 大球赔率2.00, 小球赔率1.79

        水位转十进制赔率: decimal_odds = water + 1

        返回: {"B365>2.5": 2.00, "B365<2.5": 1.79, "AHh": 3.25} 或 None
        """
        url = OVER_UNDER_URL.format(match_id=match_id)

        with self._lock:
            page = self._browser.new_page()
            page.set_default_timeout(PAGE_TIMEOUT)

            try:
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_timeout(AJAX_WAIT)

                ou = self._parse_over_under(page)
                if ou:
                    logger.info(f"  [大小球] {match_id}: 盘口={ou['AHh']} 大={ou['B365>2.5']} 小={ou['B365<2.5']}")
                return ou

            except Exception as e:
                logger.error(f"  [大小球] {match_id} 爬取失败: {e}")
                return None
            finally:
                page.close()

    def _parse_over_under(self, page: Page) -> Optional[dict]:
        """
        解析大小球表格，找 Bet365("36*") 的最新数据

        表格结构(Table 7):
          表头: [澳*, Crow*, 36*, 易胜*, ...]
          每行: 某个时间点各庄家的大小球数据
          数据格式: "3/3.5 1.00 0.79" (盘口 大球水位 小球水位)

        Bet365 "36*" 在第3列（索引2）
        """
        tables = page.query_selector_all("table")
        for table in tables:
            rows = table.query_selector_all("tr")
            if len(rows) < 2:
                continue

            # 确认表头含 "36*"
            header_cells = rows[0].query_selector_all("th, td")
            bet365_col = -1
            for i, cell in enumerate(header_cells):
                if cell.inner_text().strip().startswith(BET365_CODE):
                    bet365_col = i
                    break

            if bet365_col < 0:
                continue

            # 遍历数据行，找第一行有 Bet365 数据的（最新的）
            for row in rows[1:]:
                cells = row.query_selector_all("td")
                if bet365_col >= len(cells):
                    continue

                cell_text = cells[bet365_col].inner_text().strip()
                if not cell_text:
                    continue

                # 解析 "3/3.5 1.00 0.79" 或 "2.5 0.95 0.85"
                ou = self._parse_ou_text(cell_text)
                if ou:
                    return ou

        return None

    @staticmethod
    def _parse_ou_text(text: str) -> Optional[dict]:
        """
        解析大小球单元格文本

        格式: "3/3.5 1.00 0.79" 或 "2.5 0.95\\xa00.85"
        返回: {"B365>2.5": 大球赔率, "B365<2.5": 小球赔率, "AHh": 盘口}
        """
        # 清理文本（替换不换行空格等）
        text = text.replace("\xa0", " ").replace("\n", " ").strip()

        # 匹配盘口和水位
        # 盘口: "3/3.5" 或 "2.5" 或 "2/2.5" 或 "3"
        # 水位: 两个小数
        m = re.match(
            r'(\d+(?:/\d+)?)\s+([\d.]+)\s+([\d.]+)',
            text
        )
        if not m:
            return None

        handicap_str = m.group(1)
        over_water = float(m.group(2))
        under_water = float(m.group(3))

        # 盘口转数值: "3/3.5" → 3.25, "2.5" → 2.5, "2/2.5" → 2.25
        if "/" in handicap_str:
            parts = handicap_str.split("/")
            ahh = (float(parts[0]) + float(parts[1])) / 2
        else:
            ahh = float(handicap_str)

        # 水位转十进制赔率
        over_odds = round(over_water + 1.0, 2)
        under_odds = round(under_water + 1.0, 2)

        return {
            "B365>2.5": over_odds,
            "B365<2.5": under_odds,
            "AHh": ahh,
        }

    # ═══════════════════════════════════════════════════════════
    #  组合: 获取完整赔率
    # ═══════════════════════════════════════════════════════════

    def scrape_worldcup_odds(self) -> list[dict]:
        """
        完整爬取流程:
          1. 从比分页面获取比赛列表
          2. 对 upcoming 比赛抓取欧赔 + 大小球
          3. 返回统一格式的赔率数据
        """
        # Step 1: 获取比赛列表
        match_list = self.scrape_match_list()
        if not match_list:
            logger.warning("比分页面未获取到比赛")
            return []

        # 只抓 upcoming 比赛
        upcoming = [m for m in match_list if m["status"] == "upcoming"]
        # 已完赛的比赛也保留（用于录入实际结果）
        finished = [m for m in match_list if m["status"] == "finished"]

        logger.info(f"比赛列表: {len(upcoming)} 场未开赛, {len(finished)} 场已完赛")

        # 限制每次抓取数量
        to_scrape = upcoming[:MAX_MATCHES_PER_LOOP]
        logger.info(f"本次抓取 {len(to_scrape)} 场 upcoming 比赛")

        # Step 2: 逐场抓取赔率
        results = []

        # 已完赛比赛直接加入（有比分无赔率）
        for m in finished:
            results.append({
                "match_id": m["match_id"],
                "home": m["home"],
                "away": m["away"],
                "kickoff": m["kickoff"],
                "stage": m["stage"],
                "status": "finished",
                "score": m["score"],
                "odds": {},
            })

        # 抓取 upcoming 比赛赔率
        for m in to_scrape:
            logger.info(f"抓取赔率: {m['home']} vs {m['away']} (id={m['match_id']})")

            # 抓欧赔
            euro = self.scrape_euro_odds(m["match_id"])

            # 抓大小球
            ou = self.scrape_over_under(m["match_id"])

            # 合并赔率
            odds = {}
            if euro:
                odds.update(euro)
            if ou:
                odds.update(ou)

            # 如果赔率不完整，补默认值
            odds.setdefault("B365H", 0)
            odds.setdefault("B365D", 0)
            odds.setdefault("B365A", 0)
            odds.setdefault("B365>2.5", 1.90)
            odds.setdefault("B365<2.5", 1.90)
            odds.setdefault("AHh", 0.0)

            results.append({
                "match_id": m["match_id"],
                "home": m["home"],
                "away": m["away"],
                "kickoff": m["kickoff"],
                "stage": m["stage"],
                "status": "upcoming",
                "score": None,
                "odds": odds,
            })

        logger.info(f"爬取完成: {len(results)} 场比赛")
        return results

    # ═══════════════════════════════════════════════════════════
    #  辅助
    # ═══════════════════════════════════════════════════════════

    def _dismiss_popups(self, page: Page):
        """关闭弹窗"""
        popup_selectors = [
            "button:has-text('关闭')",
            "button:has-text('确定')",
            ".close",
            "[class*=close]",
            "[class*=popup] button",
        ]
        for selector in popup_selectors:
            try:
                btn = page.query_selector(selector)
                if btn and btn.is_visible():
                    btn.click()
                    time.sleep(0.5)
                    break
            except Exception:
                continue

    def close(self):
        """关闭浏览器"""
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        logger.info("Playwright 浏览器已关闭")


# ═══════════════════════════════════════════════════════════════
#  定时采集循环
# ═══════════════════════════════════════════════════════════════

class OddsLoop:
    """赔率定时采集循环（后台线程运行）"""

    def __init__(
        self,
        scraper: OddsScraper,
        cache: dict,
        cache_lock: threading.Lock,
        interval: int = 900,
    ):
        self.scraper = scraper
        self.cache = cache
        self.cache_lock = cache_lock
        self.interval = interval
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._prev_odds: dict[str, dict] = {}

    def start(self):
        """启动后台循环"""
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="odds_loop"
        )
        self._thread.start()
        logger.info(f"赔率采集循环已启动，间隔 {self.interval}s")

    def stop(self):
        """停止循环"""
        self._running = False

    def _run(self):
        """循环主体"""
        while self._running:
            try:
                matches = self.scraper.scrape_worldcup_odds()
                if matches:
                    shifts = self._detect_shifts(matches)

                    with self.cache_lock:
                        self.cache.clear()
                        self.cache.update({
                            "fetch_time": datetime.now().isoformat(),
                            "matches": matches,
                            "odds_shifts": shifts,
                        })

                    upcoming_count = sum(1 for m in matches if m["status"] == "upcoming")
                    logger.info(
                        f"采集成功: {len(matches)} 场比赛 "
                        f"({upcoming_count} 场未开赛), {len(shifts)} 个赔率变动"
                    )
                else:
                    logger.warning("采集到 0 场比赛")
            except Exception as e:
                logger.error(f"采集循环异常: {e}")

            time.sleep(self.interval)

    def _detect_shifts(self, matches: list[dict]) -> list[dict]:
        """检测赔率变动（与上次快照对比，变动>10%记录）"""
        shifts = []
        current = {}

        for m in matches:
            mid = m["match_id"]
            odds = m.get("odds", {})
            current[mid] = odds

            prev = self._prev_odds.get(mid)
            if prev:
                for key in ["B365H", "B365D", "B365A"]:
                    old_val = prev.get(key, 0)
                    new_val = odds.get(key, 0)
                    if old_val > 0 and new_val > 0:
                        change_pct = abs(new_val - old_val) / old_val
                        if change_pct > 0.10:
                            shifts.append({
                                "match_id": mid,
                                "home": m["home"],
                                "away": m["away"],
                                "odds_key": key,
                                "old_value": old_val,
                                "new_value": new_val,
                                "change_pct": round(change_pct, 4),
                            })
                            logger.info(
                                f"  [变动] {m['home']} vs {m['away']} "
                                f"{key}: {old_val} -> {new_val} ({change_pct:.1%})"
                            )

        self._prev_odds = current
        return shifts
