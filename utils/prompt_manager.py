# -*- coding: utf-8 -*-
"""
Prompt 模板加载器 —— 从项目根目录 `prompt_templates/` 读取 YAML。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# 项目根目录（utils/ 的上级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATES_DIR = _PROJECT_ROOT / "prompt_templates"


class PromptManager:
    """从 YAML 加载 name / version / system_prompt / user_prompt_template 等字段。"""

    @staticmethod
    def load(skill_name: str, version: str = "v1") -> dict[str, Any]:
        """
        读取 ``prompt_templates/{skill_name}_{version}.yaml``，UTF-8 解码。

        Args:
            skill_name: 模板逻辑名，如 ``football_analyst``、``memory_flush``
            version:    版本后缀，默认 ``v1``

        Returns:
            YAML 解析后的字典（可含额外自定义键）。
        """
        filename = f"{skill_name}_{version}.yaml"
        path = _TEMPLATES_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Prompt 模板不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Prompt YAML 根节点必须为 mapping: {path}")

        return data
