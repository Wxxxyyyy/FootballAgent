# -*- coding: utf-8 -*-
"""
动态记忆引擎（football.md 索引 + memory/*.md 分类文件）
═══════════════════════════════════════════════════════════════
按 Claude Code 记忆机制裁剪实现：
  - 1 个总静态文件 data/memory/football.md：静态规则 + 记忆索引（常驻 prompt）
  - N 个动态记忆文件 data/memory/*.md：每条记忆一个文件，只分 4 类
        user      用户画像（喜欢的球队、常驻联赛……）
        feedback  行为偏好（用户明确要求怎么/不要怎么回答）
        project   项目动态（某队伤情、刚做的预测结论等随时间变化的事实）
        reference 外部指针（某类信息要去哪里查）
  - 召回不用向量库：读索引一句话描述 → 小模型做选择题挑 top-K → 注入正文
  - 写入走后台 extractMemories：判断有无值得记的 → 归类 → 写文件 → 更新索引
"""

import os
import re
import glob
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ─── 路径 ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEMORY_DIR = os.path.join(PROJECT_ROOT, "data", "memory")
FOOTBALL_MD = os.path.join(MEMORY_DIR, "football.md")

VALID_TYPES = ("user", "feedback", "project", "reference")

# ═══════════════════════════════════════════════════════════════
#  静态规则（football.md，常驻 prompt，带缓存）
# ═══════════════════════════════════════════════════════════════

_static_cache: str | None = None


def get_static_rules() -> str:
    """读取 football.md 静态规则 + 记忆索引（带缓存，索引更新后失效重读）"""
    global _static_cache
    if _static_cache is None:
        try:
            with open(FOOTBALL_MD, encoding="utf-8") as f:
                _static_cache = f.read()
        except Exception as e:
            logger.warning(f"[Memory] 读取 football.md 失败: {e}")
            _static_cache = ""
    return _static_cache


def _invalidate_cache() -> None:
    global _static_cache
    _static_cache = None


# ═══════════════════════════════════════════════════════════════
#  动态记忆文件扫描与解析
# ═══════════════════════════════════════════════════════════════

def _parse_frontmatter(content: str) -> dict:
    """解析 md 文件头部 --- ... --- 之间的 name/type/description"""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not m:
        return {}
    fm = {}
    for line in m.group(1).split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    return fm


def _scan_memories() -> list[dict]:
    """扫描 data/memory/*.md（不含 football.md），返回各记忆的 frontmatter + 路径"""
    memories = []
    for path in sorted(glob.glob(os.path.join(MEMORY_DIR, "*.md"))):
        if os.path.basename(path) == "football.md":
            continue
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            fm = _parse_frontmatter(content)
            memories.append({
                "path": path,
                "name": fm.get("name", os.path.basename(path)[:-3]),
                "type": fm.get("type", "project"),
                "description": fm.get("description", ""),
                "created": fm.get("created", ""),
            })
        except Exception:  # noqa: BLE001
            continue
    return memories


# ═══════════════════════════════════════════════════════════════
#  召回（cc 做法：索引常驻 + 小模型做选择题，不用向量库）
# ═══════════════════════════════════════════════════════════════

def recall_memories(query: str, top_k: int = 3) -> str:
    """
    根据当前问题召回相关记忆正文。
    流程：扫描索引（只给 name/type/description）→ 小模型选 top-K → 读入选记忆正文拼接返回。
    无相关记忆返回空串。
    """
    memories = _scan_memories()
    if not memories:
        return ""

    catalog = "\n".join(
        f"{i}. [{m['type']}] {m['name']}：{m['description']}"
        for i, m in enumerate(memories)
    )
    prompt = f"""以下是可选的记忆清单（编号 + 类型 + 一句话描述）：
{catalog}

用户当前问题：{query}

请判断哪些记忆与当前问题真正相关。规则：
- 只选确实能用上的，宁缺毋滥；不确定的不要选。
- 完全无关则只输出"无"。
- 最多选 {top_k} 条。
只输出编号，用逗号分隔（如 "0,2"），不要输出任何解释。"""

    try:
        from common.llm_select import llm_call
        resp = llm_call(prompt, force_fallback=True)  # 本地小模型，零成本
        selected = _parse_indices(getattr(resp, "content", str(resp)), len(memories))
    except Exception as e:
        logger.warning(f"[Memory] 召回选择失败: {e}")
        return ""

    if not selected:
        return ""

    parts = []
    for idx in selected[:top_k]:
        try:
            with open(memories[idx]["path"], encoding="utf-8") as f:
                parts.append(f.read().strip())
        except Exception:  # noqa: BLE001
            continue
    return "\n\n".join(parts)


def _parse_indices(text: str, n: int) -> list[int]:
    """从小模型输出解析编号，过滤越界；输出"无"返回空"""
    if "无" in text:
        return []
    indices = []
    for tok in re.findall(r"\d+", text):
        i = int(tok)
        if 0 <= i < n and i not in indices:
            indices.append(i)
    return indices


# ═══════════════════════════════════════════════════════════════
#  写入（extractMemories：LLM 判断 → 归类 → 写文件 → 更新索引）
# ═══════════════════════════════════════════════════════════════

def extract_and_write(conversation_text: str, max_new: int = 2) -> None:
    """
    后台提取记忆。分析对话，若有值得长期记住的信息则写入新 .md 并更新索引。
    只记代码/数据里推不出来的信息；绝大多数轮次返回"无"，不落盘。
    """
    if not conversation_text.strip():
        return

    prompt = f"""分析下面这段对话，判断是否包含值得长期记住的信息。只记四类：
- user（用户画像：喜欢的球队、常驻联赛、身份等）
- feedback（行为偏好：用户明确要求怎么回答/不要怎么回答）
- project（项目动态：某队伤情、刚做的预测结论等会随时间变化的事实）
- reference（外部指针：某类信息要去哪里查）

对话记录：
{conversation_text}

规则：
- 没有值得记的，只输出"无"。
- 有则输出 JSON 数组，每项含 name/description/type/why/how_to_apply/content 六个字段。
- type 必须是 user/feedback/project/reference 之一。
- 只记"从代码或数据里推不出来"的信息；常识不记；最多 {max_new} 条。
- 输出必须是合法 JSON 数组或"无"，不要输出其他内容。"""

    try:
        from common.llm_select import llm_call
        resp = llm_call(prompt, force_fallback=True)
        raw = getattr(resp, "content", str(resp)).strip()
    except Exception as e:
        logger.warning(f"[Memory] 提取调用失败: {e}")
        return

    if "无" in raw or not raw:
        return

    items = _parse_json_array(raw)
    if not items:
        return

    os.makedirs(MEMORY_DIR, exist_ok=True)
    for item in items[:max_new]:
        try:
            _write_memory_file(item)
        except Exception as e:
            logger.warning(f"[Memory] 写记忆失败: {e}")

    _rebuild_index()


def _parse_json_array(raw: str) -> list[dict]:
    """从 LLM 输出解析 JSON 数组（容忍 ```json 包裹）"""
    import json as _json
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
    try:
        data = _json.loads(raw)
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def _write_memory_file(item: dict) -> None:
    """把一条记忆写成带 frontmatter 的 md 文件"""
    mtype = item.get("type", "project")
    if mtype not in VALID_TYPES:
        mtype = "project"
    name = str(item.get("name", "未命名记忆")).strip()
    description = str(item.get("description", "")).strip()
    why = str(item.get("why", "")).strip()
    how = str(item.get("how_to_apply", "")).strip()
    content = str(item.get("content", "")).strip()
    created = datetime.now().strftime("%Y-%m-%d")

    # 文件名：类型_清洗后的名字
    safe = re.sub(r"[^\w一-龥]+", "_", name)[:40]
    filename = f"{mtype}_{safe}.md"
    path = os.path.join(MEMORY_DIR, filename)

    md = f"""---
name: {name}
description: {description}
type: {mtype}
created: {created}
---
{content}

**Why:** {why}

**How to apply:** {how}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info(f"[Memory] 写入记忆 [{mtype}] {name} -> {filename}")


def _rebuild_index() -> None:
    """根据当前所有动态记忆，重建 football.md 末尾的记忆索引区"""
    memories = _scan_memories()
    lines = ["## 记忆索引（动态记忆目录，自动维护）", ""]
    if not memories:
        lines.append("- [暂无，随 extractMemories 写入自动更新]")
    else:
        for m in memories:
            lines.append(f"- [{m['type']}] {m['name']}：{m['description']}")
    index_text = "\n".join(lines)

    try:
        with open(FOOTBALL_MD, encoding="utf-8") as f:
            content = f.read()
    except Exception:
        content = "# football.md — 足球 Agent 静态规则\n"

    if "## 记忆索引" in content:
        head = content.split("## 记忆索引")[0].rstrip() + "\n\n"
        new_content = head + index_text + "\n"
    else:
        new_content = content.rstrip() + "\n\n" + index_text + "\n"

    try:
        with open(FOOTBALL_MD, "w", encoding="utf-8") as f:
            f.write(new_content)
        _invalidate_cache()
    except Exception as e:
        logger.warning(f"[Memory] 更新索引失败: {e}")
