# -*- coding: utf-8 -*-
"""
国家队配置 —— 世界杯参赛队中英文名映射 + 各数据源标识

数据源标识说明:
  - tm_slug:  Transfermarkt 球队 URL slug（用于拼接伤停/阵容页）
  - tm_id:    Transfermarkt 球队 ID
  - dqd_team: 懂球帝球队中文名（用于新闻搜索）
  - zhibo8:   直播吧球队标识（用于首发阵容）

注意: tm_slug / tm_id 需根据 Transfermarkt 实际页面确认，
      此处提供常见强队的默认值，未确认的留空，运行时会跳过该数据源。
"""

# ======================== 国家队映射表 ========================
# key = 英文标准名（与预测流程一致）
# 2026 世界杯 48 队，此处先收录常见强队 + 已知队伍，后续可扩充

NATIONAL_TEAMS = {
    # ── 南美 ──
    "Brazil":       {"zh": "巴西",     "alias": ["桑巴军团"],         "tm_slug": "selecao",           "tm_id": "3433",  "dqd_team": "巴西",     "group": "C"},
    "Argentina":    {"zh": "阿根廷",   "alias": ["潘帕斯雄鹰"],       "tm_slug": "argentinien",       "tm_id": "3437",  "dqd_team": "阿根廷",   "group": "A"},
    "Uruguay":      {"zh": "乌拉圭",   "alias": ["天蓝军团"],         "tm_slug": "uruguay",           "tm_id": "3440",  "dqd_team": "乌拉圭",   "group": "E"},
    "Colombia":     {"zh": "哥伦比亚", "alias": ["咖啡农夫"],         "tm_slug": "kolumbien",         "tm_id": "3438",  "dqd_team": "哥伦比亚", "group": "D"},
    "Ecuador":      {"zh": "厄瓜多尔", "alias": [],                  "tm_slug": "ecuador",           "tm_id": "3439",  "dqd_team": "厄瓜多尔", "group": "B"},
    "Paraguay":     {"zh": "巴拉圭",   "alias": [],                  "tm_slug": "paraguay",          "tm_id": "3442",  "dqd_team": "巴拉圭",   "group": "B"},
    "Peru":         {"zh": "秘鲁",     "alias": [],                  "tm_slug": "peru",              "tm_id": "3443",  "dqd_team": "秘鲁",     "group": "A"},
    "Chile":        {"zh": "智利",     "alias": [],                  "tm_slug": "chile",             "tm_id": "3436",  "dqd_team": "智利",     "group": "F"},

    # ── 欧洲 ──
    "Germany":      {"zh": "德国",     "alias": ["日耳曼战车"],       "tm_slug": "deutschland",       "tm_id": "3266",  "dqd_team": "德国",     "group": "G"},
    "France":       {"zh": "法国",     "alias": ["高卢雄鸡"],         "tm_slug": "frankreich",        "tm_id": "3377",  "dqd_team": "法国",     "group": "D"},
    "Spain":        {"zh": "西班牙",   "alias": ["斗牛士"],           "tm_slug": "spanien",           "tm_id": "3375",  "dqd_team": "西班牙",   "group": "F"},
    "England":      {"zh": "英格兰",   "alias": ["三狮军团"],         "tm_slug": "england",           "tm_id": "3299",  "dqd_team": "英格兰",   "group": "B"},
    "Portugal":     {"zh": "葡萄牙",   "alias": ["五盾军团"],         "tm_slug": "portugal",          "tm_id": "3300",  "dqd_team": "葡萄牙",   "group": "H"},
    "Netherlands":  {"zh": "荷兰",     "alias": ["橙衣军团"],         "tm_slug": "niederlande",       "tm_id": "3378",  "dqd_team": "荷兰",     "group": "G"},
    "Belgium":      {"zh": "比利时",   "alias": ["欧洲红魔"],         "tm_slug": "belgien",           "tm_id": "3376",  "dqd_team": "比利时",   "group": "C"},
    "Italy":        {"zh": "意大利",   "alias": ["蓝衣军团"],         "tm_slug": "italien",           "tm_id": "3374",  "dqd_team": "意大利",   "group": "E"},
    "Croatia":      {"zh": "克罗地亚", "alias": ["格子军团"],         "tm_slug": "kroatien",          "tm_id": "3536",  "dqd_team": "克罗地亚", "group": "A"},
    "Switzerland":  {"zh": "瑞士",     "alias": [],                  "tm_slug": "schweiz",           "tm_id": "3384",  "dqd_team": "瑞士",     "group": "C"},
    "Denmark":      {"zh": "丹麦",     "alias": [],                  "tm_slug": "danemark",          "tm_id": "3430",  "dqd_team": "丹麦",     "group": "D"},
    "Serbia":       {"zh": "塞尔维亚", "alias": [],                  "tm_slug": "serbien",           "tm_id": "3431",  "dqd_team": "塞尔维亚", "group": "E"},
    "Poland":       {"zh": "波兰",     "alias": [],                  "tm_slug": "polen",             "tm_id": "3432",  "dqd_team": "波兰",     "group": "H"},
    "Austria":      {"zh": "奥地利",   "alias": [],                  "tm_slug": "osterreich",        "tm_id": "3382",  "dqd_team": "奥地利",   "group": "F"},
    "Ukraine":      {"zh": "乌克兰",   "alias": [],                  "tm_slug": "ukraine",           "tm_id": "3670",  "dqd_team": "乌克兰",   "group": "H"},
    "Turkey":       {"zh": "土耳其",   "alias": ["星月军团"],         "tm_slug": "turkei",            "tm_id": "3379",  "dqd_team": "土耳其",   "group": "G"},
    "Sweden":       {"zh": "瑞典",     "alias": ["北欧海盗"],         "tm_slug": "schweden",          "tm_id": "3373",  "dqd_team": "瑞典",     "group": "G"},
    "Norway":       {"zh": "挪威",     "alias": [],                  "tm_slug": "norwegen",          "tm_id": "3430",  "dqd_team": "挪威",     "group": "G"},
    "Czech Republic":{"zh": "捷克",     "alias": [],                  "tm_slug": "tschechien",        "tm_id": "3430",  "dqd_team": "捷克",     "group": "A"},
    "Bosnia & Herzegovina": {"zh": "波黑", "alias": [],               "tm_slug": "bosnien-herzegowina","tm_id": "8075", "dqd_team": "波黑",    "group": "B"},
    "Scotland":     {"zh": "苏格兰",   "alias": [],                  "tm_slug": "schottland",        "tm_id": "3380",  "dqd_team": "苏格兰",   "group": "C"},
    "Ivory Coast":  {"zh": "科特迪瓦", "alias": [],                  "tm_slug": "elfenbeinkueste",   "tm_id": "4225",  "dqd_team": "科特迪瓦", "group": "E"},
    "Curacao":      {"zh": "库拉索",   "alias": [],                  "tm_slug": "curacao",           "tm_id": "19047", "dqd_team": "库拉索",   "group": "E"},
    "Cape Verde":   {"zh": "佛得角",   "alias": [],                  "tm_slug": "kap-verde",         "tm_id": "15353", "dqd_team": "佛得角",   "group": "F"},
    "Jordan":       {"zh": "约旦",     "alias": ["约旦"],             "tm_slug": "jordanien",         "tm_id": "4227",  "dqd_team": "约旦",     "group": "K"},
    "DR Congo":     {"zh": "刚果民主共和国","alias": ["刚果金"],       "tm_slug": "kongo-kinshasa",    "tm_id": "4226",  "dqd_team": "刚果民主共和国","group": "I"},
    "Uzbekistan":   {"zh": "乌兹别克斯坦","alias": [],                "tm_slug": "usbekistan",        "tm_id": "4228",  "dqd_team": "乌兹别克斯坦","group": "I"},
    "Ghana":        {"zh": "加纳",     "alias": ["黑星"],             "tm_slug": "ghana",             "tm_id": "3444",  "dqd_team": "加纳",     "group": "J"},
    "Haiti":        {"zh": "海地",     "alias": [],                  "tm_slug": "haiti",            "tm_id": "4229",  "dqd_team": "海地",     "group": "C"},

    # ── 非洲 ──
    "Morocco":      {"zh": "摩洛哥",   "alias": ["亚特拉斯雄狮"],     "tm_slug": "marokko",           "tm_id": "3446",  "dqd_team": "摩洛哥",   "group": "A"},
    "Senegal":      {"zh": "塞内加尔", "alias": ["特兰加雄狮"],       "tm_slug": "senegal",           "tm_id": "4215",  "dqd_team": "塞内加尔", "group": "B"},
    "Nigeria":      {"zh": "尼日利亚", "alias": ["超级雄鹰"],         "tm_slug": "nigeria",           "tm_id": "3444",  "dqd_team": "尼日利亚", "group": "C"},
    "Egypt":        {"zh": "埃及",     "alias": ["法老"],             "tm_slug": "agypten",           "tm_id": "3640",  "dqd_team": "埃及",     "group": "D"},
    "Tunisia":      {"zh": "突尼斯",   "alias": [],                  "tm_slug": "tunesien",          "tm_id": "3645",  "dqd_team": "突尼斯",   "group": "E"},
    "Algeria":      {"zh": "阿尔及利亚","alias": [],                  "tm_slug": "algerien",          "tm_id": "4213",  "dqd_team": "阿尔及利亚","group": "F"},
    "South Africa": {"zh": "南非",     "alias": [],                  "tm_slug": "sudafrika",         "tm_id": "4216",  "dqd_team": "南非",     "group": "G"},
    "Cameroon":     {"zh": "喀麦隆",   "alias": [],                  "tm_slug": "kamerun",           "tm_id": "4212",  "dqd_team": "喀麦隆",   "group": "H"},

    # ── 亚洲 ──
    "Japan":        {"zh": "日本",     "alias": ["蓝武士"],           "tm_slug": "japan",             "tm_id": "3435",  "dqd_team": "日本",     "group": "A"},
    "South Korea":  {"zh": "韩国",     "alias": ["太极虎"],           "tm_slug": "sudkorea",          "tm_id": "3441",  "dqd_team": "韩国",     "group": "B"},
    "Iran":         {"zh": "伊朗",     "alias": [],                  "tm_slug": "iran",              "tm_id": "3445",  "dqd_team": "伊朗",     "group": "C"},
    "Australia":    {"zh": "澳大利亚", "alias": ["袋鼠军团"],         "tm_slug": "australien",        "tm_id": "3434",  "dqd_team": "澳大利亚", "group": "D"},
    "Saudi Arabia": {"zh": "沙特阿拉伯","alias": ["沙特"],            "tm_slug": "saudi-arabien",     "tm_id": "4218",  "dqd_team": "沙特阿拉伯","group": "E"},
    "Qatar":        {"zh": "卡塔尔",   "alias": [],                  "tm_slug": "katar",             "tm_id": "4217",  "dqd_team": "卡塔尔",   "group": "F"},
    "China PR":     {"zh": "中国",     "alias": ["国足"],             "tm_slug": "china",             "tm_id": "4219",  "dqd_team": "中国",     "group": "G"},
    "Iraq":         {"zh": "伊拉克",   "alias": [],                  "tm_slug": "irak",              "tm_id": "4220",  "dqd_team": "伊拉克",   "group": "H"},

    # ── 中北美 ──
    "USA":          {"zh": "美国",     "alias": ["星条旗"],           "tm_slug": "usa",               "tm_id": "3505",  "dqd_team": "美国",     "group": "A"},
    "Mexico":       {"zh": "墨西哥",   "alias": ["仙人掌"],           "tm_slug": "mexiko",            "tm_id": "3447",  "dqd_team": "墨西哥",   "group": "B"},
    "Canada":       {"zh": "加拿大",   "alias": [],                  "tm_slug": "kanada",            "tm_id": "3448",  "dqd_team": "加拿大",   "group": "C"},
    "Costa Rica":   {"zh": "哥斯达黎加","alias": [],                  "tm_slug": "costa-rica",        "tm_id": "4214",  "dqd_team": "哥斯达黎加","group": "D"},
    "Panama":       {"zh": "巴拿马",   "alias": [],                  "tm_slug": "panama",            "tm_id": "4221",  "dqd_team": "巴拿马",   "group": "E"},
    "Honduras":     {"zh": "洪都拉斯", "alias": [],                  "tm_slug": "honduras",          "tm_id": "4222",  "dqd_team": "洪都拉斯", "group": "F"},

    # ── 大洋洲 ──
    "New Zealand":  {"zh": "新西兰",   "alias": [],                  "tm_slug": "neuseeland",        "tm_id": "4223",  "dqd_team": "新西兰",   "group": "G"},
    "Fiji":         {"zh": "斐济",     "alias": [],                  "tm_slug": "fidschi",           "tm_id": "4224",  "dqd_team": "斐济",     "group": "H"},
}

# ======================== 反向映射：中文/别名 → 英文标准名 ========================

_zh_to_en: dict = {}
_alias_to_en: dict = {}

for _en, _info in NATIONAL_TEAMS.items():
    _zh_to_en[_info["zh"]] = _en
    for _alias in _info.get("alias", []):
        _alias_to_en[_alias] = _en
    # 英文名本身也注册（大小写不敏感）
    _alias_to_en[_en.lower()] = _en


def resolve_national_team(name: str) -> str | None:
    """
    任意名称（中文/英文/别名）→ 英文标准名

    Returns:
        英文标准名（如 "Brazil"），识别不了返回 None
    """
    if not name:
        return None
    name = name.strip()
    # 直接匹配英文标准名
    if name in NATIONAL_TEAMS:
        return name
    # 中文官方名
    if name in _zh_to_en:
        return _zh_to_en[name]
    # 别名（大小写不敏感）
    lower = name.lower()
    if lower in _alias_to_en:
        return _alias_to_en[lower]
    return None


def get_team_info(en_name: str) -> dict | None:
    """英文标准名 → 球队完整信息字典"""
    return NATIONAL_TEAMS.get(en_name)


def to_chinese(en_name: str) -> str | None:
    """英文标准名 → 中文名"""
    info = NATIONAL_TEAMS.get(en_name)
    return info["zh"] if info else None


def all_national_teams() -> list[str]:
    """返回所有国家队英文标准名"""
    return list(NATIONAL_TEAMS.keys())


# ======================== Transfermarkt 通用配置 ========================

TM_BASE_URL = "https://www.transfermarkt.com"
TM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/125.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def tm_injuries_url(tm_slug: str, tm_id: str) -> str:
    """拼接 Transfermarkt 球队伤停页 URL"""
    return f"{TM_BASE_URL}/{tm_slug}/verletzungen/verein/{tm_id}"


def tm_squad_url(tm_slug: str, tm_id: str) -> str:
    """拼接 Transfermarkt 球队阵容页 URL"""
    return f"{TM_BASE_URL}/{tm_slug}/kader/verein/{tm_id}"


def tm_profile_url(tm_slug: str, tm_id: str) -> str:
    """拼接 Transfermarkt 球队主页 URL"""
    return f"{TM_BASE_URL}/{tm_slug}/startseite/verein/{tm_id}"


# ======================== 测试 ========================

if __name__ == "__main__":
    print(f"共加载 {len(NATIONAL_TEAMS)} 支国家队\n")

    tests = ["巴西", "德国", "Brazil", "英格兰", "三狮军团", "日本", "国足", "Argentina"]
    for t in tests:
        en = resolve_national_team(t)
        if en:
            info = get_team_info(en)
            print(f"  '{t}' → {en} | {info['zh']} | 小组{info['group']} | TM:{info['tm_slug']}/{info['tm_id']}")
        else:
            print(f"  '{t}' → ❌ 未识别")
