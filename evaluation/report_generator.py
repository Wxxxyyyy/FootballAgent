# -*- coding: utf-8 -*-
"""
评估报告：将多模块指标汇总为 JSON 与 Markdown 文件。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class ReportGenerator:
    """构造时传入嵌套 dict（如 accuracy / backtest / profit 各一节），再写出文件。"""

    def __init__(self, sections: Mapping[str, Any], *, title: str = "足球预测评估报告") -> None:
        self.sections = dict(sections)
        self.title = title

    def to_json(self, path: str | Path) -> None:
        """写入 UTF-8 JSON，ensure_ascii=False 保留中文。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"title": self.title, "sections": self.sections}
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def to_markdown(self, path: str | Path) -> None:
        """生成一级标题 + 各节二级标题与缩进列表行。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = [f"# {self.title}", ""]
        for name, body in self.sections.items():
            lines.append(f"## {name}")
            lines.append("")
            lines.extend(_markdown_body(body, indent=0))
            lines.append("")
        with p.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def build(self) -> dict[str, Any]:
        """返回与 JSON 内容一致的字典，便于接口直接返回。"""
        return {"title": self.title, "sections": self.sections}


def _markdown_body(obj: Any, indent: int) -> list[str]:
    pad = "  " * indent
    out: list[str] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if isinstance(v, (Mapping, list)):
                out.append(f"{pad}- **{k}**:")
                out.extend(_markdown_body(v, indent + 1))
            else:
                out.append(f"{pad}- **{k}**: {v}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, (Mapping, list)):
                out.append(f"{pad}- [{i}]")
                out.extend(_markdown_body(item, indent + 1))
            else:
                out.append(f"{pad}- {item}")
    else:
        out.append(f"{pad}{obj}")
    return out
