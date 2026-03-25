# -*- coding: utf-8 -*-
"""
球队名称映射 —— 基于 data/中英文对照.csv
- 中文官方名 / 球迷简称 / 英文名 → 统一映射为 CSV 中的英文标准名
- 英文名 → 中文官方名 / 球迷简称
"""

import os
import csv
from typing import Optional, Dict, List

# ======================== CSV 路径 ========================

_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "English2Chinese", "中英文对照.csv"
)

# ======================== 联赛中英映射 ========================

LEAGUE_ZH = {
    "England": "英超", "France": "法甲",
    "Germany": "德甲", "Italy": "意甲", "Spain": "西甲",
}

# ======================== 加载 CSV 构建映射表 ========================

# 任意名称(小写) → 英文标准名
_any_to_en: Dict[str, str] = {}
# 英文标准名 → {league, zh, alias}
_en_to_info: Dict[str, dict] = {}


def _load():
    """读取 CSV，构建双向映射字典"""
    if _any_to_en:  # 已加载过
        return
    with open(_CSV_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            en = row["ClubName"].strip()
            league = row["League"].strip()
            zh = row["ClubNameZh"].strip()
            alias = row["AliasZh"].strip()

            _en_to_info[en] = {"league": league, "zh": zh, "alias": alias}

            # 注册所有可识别的名称（不区分大小写）
            for name in {en, zh, alias}:
                _any_to_en[name.lower()] = en


_load()  # 模块导入时自动加载

# ======================== 对外接口 ========================


def resolve(name: str) -> Optional[str]:
    """任意名称 → 英文标准名，识别不了返回 None"""
    return _any_to_en.get(name.strip().lower())


def to_chinese(en_name: str, alias: bool = False) -> Optional[str]:
    """英文名 → 中文（alias=True 返回球迷简称，否则返回官方名）"""
    info = _en_to_info.get(en_name)
    if info is None:
        return None
    return info["alias"] if alias else info["zh"]


def to_english(zh_name: str) -> Optional[str]:
    """中文名/球迷简称 → 英文标准名"""
    return resolve(zh_name)


def get_league(name: str) -> Optional[str]:
    """任意名称 → 所属联赛英文（England/France/...）"""
    en = resolve(name)
    return _en_to_info[en]["league"] if en else None


def get_league_zh(name: str) -> Optional[str]:
    """任意名称 → 所属联赛中文（英超/法甲/...）"""
    league = get_league(name)
    return LEAGUE_ZH.get(league) if league else None


def all_teams() -> List[str]:
    """返回所有英文标准名列表"""
    return list(_en_to_info.keys())


def teams_by_league(league: str) -> List[str]:
    """按联赛获取球队列表（支持中英文联赛名）"""
    # 中文联赛名 → 英文
    en_league = {v: k for k, v in LEAGUE_ZH.items()}.get(league, league)
    return [en for en, info in _en_to_info.items() if info["league"] == en_league]


# ======================== 测试 ========================

if __name__ == "__main__":
    print(f"共加载 {len(_en_to_info)} 支球队，{len(_any_to_en)} 条名称映射\n")

    tests = [
        "曼联", "曼城", "皇马", "巴萨", "拜仁", "国米",
        "Arsenal", "热刺", "药厂", "蓝军", "黄潜", "红狼",
        "Man United", "巴黎", "门兴", "紫百合",
    ]
    for t in tests:
        en = resolve(t)
        if en:
            zh = to_chinese(en)
            alias = to_chinese(en, alias=True)
            league = get_league_zh(t)
            print(f"  '{t}' → {en} | {zh} | {alias} | {league}")
        else:
            print(f"  '{t}' → ❌ 未识别")
