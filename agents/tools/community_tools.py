# -*- coding: utf-8 -*-
"""
社区常用工具集（Community Tools）
═══════════════════════════════════
打包社区现成工具 + 通用小工具，补充自定义工具未覆盖的通用能力。
注册进 ReAct 的 TOOLS/_TOOL_MAP 后即可被 LLM 自主调度。

依赖（可选，缺了对应工具会降级为提示信息，不崩）：
  - duckduckgo-search  → web_search        实时联网搜索
  - wikipedia          → wikipedia_search  百科检索
  - httpx（已有）      → weather_query     天气查询（wttr.in，免 API Key）
"""

from datetime import datetime

from langchain_core.tools import tool

# ═══════════════════════════════════════════════════════════════
#  1. 联网搜索（DuckDuckGo，免 API Key）
# ═══════════════════════════════════════════════════════════════
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    _ddg = DuckDuckGoSearchRun()
    _HAS_DDG = True
except Exception:  # noqa: BLE001
    _HAS_DDG = False


@tool
def web_search(query: str) -> str:
    """实时联网搜索：查询最新新闻、赛前动态、转会、伤停等"当下发生"的时效信息。
    仅在内部数据库/知识库查不到的实时信息时使用，常规战绩/交锋优先走内部工具。"""
    if not _HAS_DDG:
        return "[web_search 未启用] 请 pip install duckduckgo-search"
    try:
        return _ddg.run(query)
    except Exception as e:  # noqa: BLE001
        return f"[web_search 失败] {type(e).__name__}: {e}"


# ═══════════════════════════════════════════════════════════════
#  2. 维基百科检索（球队/球员/赛事通用背景知识）
# ═══════════════════════════════════════════════════════════════
try:
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    _wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(
            lang="zh", top_k_results=2, doc_content_chars_max=2000
        )
    )
    _HAS_WIKI = True
except Exception:  # noqa: BLE001
    _HAS_WIKI = False


@tool
def wikipedia_search(query: str) -> str:
    """维基百科检索：查询球队、球员、赛事、历史等通用背景知识。"""
    if not _HAS_WIKI:
        return "[wikipedia_search 未启用] 请 pip install wikipedia"
    try:
        return _wiki.run(query)
    except Exception as e:  # noqa: BLE001
        return f"[wikipedia_search 失败] {type(e).__name__}: {e}"


# ═══════════════════════════════════════════════════════════════
#  3. 天气查询（wttr.in，免 API Key）
# ═══════════════════════════════════════════════════════════════
@tool
def weather_query(city: str) -> str:
    """查询指定城市的实时天气（天气现象/温度/风速/湿度）。
    用于评估比赛日天气（雨/大风/高温）对赛事发挥的潜在影响。入参为城市英文名。"""
    try:
        import httpx
        resp = httpx.get(f"https://wttr.in/{city}?format=j1", timeout=10)
        data = resp.json()
        cur = data["current_condition"][0]
        desc = cur["weatherDesc"][0]["value"]
        return (
            f"{city} 当前天气：{desc}，温度 {cur['temp_C']}°C"
            f"（体感 {cur['FeelsLikeC']}°C），风速 {cur['windspeedKmph']}km/h，"
            f"湿度 {cur['humidity']}%"
        )
    except Exception as e:  # noqa: BLE001
        return f"[weather_query 失败] {type(e).__name__}: {e}"


# ═══════════════════════════════════════════════════════════════
#  4. 当前日期时间（"今天/本周/最近"类查询的时间锚点）
# ═══════════════════════════════════════════════════════════════
@tool
def current_datetime(dummy: str = "") -> str:
    """获取当前日期和时间，用于解析"今天/明天/本周/最近"等相对时间查询。"""
    now = datetime.now()
    return now.strftime("当前时间: %Y-%m-%d %H:%M:%S（%A）")


# ═══════════════════════════════════════════════════════════════
#  5. 文本翻译（deep-translator，免 API Key）
# ═══════════════════════════════════════════════════════════════
try:
    from deep_translator import GoogleTranslator  # noqa: F401
    _HAS_TRANSLATOR = True
except Exception:  # noqa: BLE001
    _HAS_TRANSLATOR = False


@tool
def translate(text: str, target_lang: str = "zh-CN") -> str:
    """文本翻译：将外文（英文/西语等）新闻、资讯、赔率分析翻译成中文。
    入参 text 为待翻译文本，target_lang 为目标语言（默认 zh-CN 中文）。"""
    if not _HAS_TRANSLATOR:
        return "[translate 未启用] 请 pip install deep-translator"
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception as e:  # noqa: BLE001
        return f"[translate 失败] {type(e).__name__}: {e}"
