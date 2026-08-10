# -*- coding: utf-8 -*-
"""
模型版本管理契约测试。

核心契约:
  1. generate_version 返回 v 开头的版本号
  2. register_version 注册后能查到
  3. list_versions 返回所有版本
  4. set_current / get_current_version 版本切换正确
  5. get_model_path 优先返回版本路径，回退到旧路径
"""

import os
import json
import tempfile
import pytest

from agents.predicted_agent.models import registry


class TestVersionGeneration:
    def test_generate_version_starts_with_v(self):
        """版本号必须以 v 开头"""
        v = registry.generate_version()
        assert v.startswith("v"), f"版本号应以 v 开头: {v}"

    def test_generate_version_is_unique(self):
        """两次生成不应完全相同（含时间戳）"""
        v1 = registry.generate_version()
        import time
        time.sleep(1.1)  # 版本号精度到秒，需等超过1秒
        v2 = registry.generate_version()
        assert v1 != v2, "版本号应含时间戳保证唯一"


class TestRegistry:
    def test_register_and_list(self, tmp_path, monkeypatch):
        """注册后能查到"""
        # 重定向 MODEL_DIR 到临时目录
        monkeypatch.setattr(registry, "MODEL_DIR", str(tmp_path))
        monkeypatch.setattr(registry, "REGISTRY_FILE", str(tmp_path / "registry.json"))

        v = registry.generate_version()
        registry.register_version(
            version=v,
            metrics={"val_accuracy": 0.545},
            data_range="test",
        )

        versions = registry.list_versions()
        assert len(versions) == 1
        assert versions[0]["version"] == v

    def test_first_version_becomes_current(self, tmp_path, monkeypatch):
        """第一个版本自动设为 current"""
        monkeypatch.setattr(registry, "MODEL_DIR", str(tmp_path))
        monkeypatch.setattr(registry, "REGISTRY_FILE", str(tmp_path / "registry.json"))

        v = registry.generate_version()
        registry.register_version(version=v, metrics={})
        assert registry.get_current_version() == v

    def test_set_current_version(self, tmp_path, monkeypatch):
        """切换当前版本"""
        monkeypatch.setattr(registry, "MODEL_DIR", str(tmp_path))
        monkeypatch.setattr(registry, "REGISTRY_FILE", str(tmp_path / "registry.json"))

        v1 = registry.generate_version()
        v2 = "v_test_002"
        registry.register_version(version=v1, metrics={})
        registry.register_version(version=v2, metrics={})

        # 切换到 v2
        assert registry.set_current_version(v2) is True
        assert registry.get_current_version() == v2

    def test_set_current_nonexistent_returns_false(self, tmp_path, monkeypatch):
        """切换到不存在的版本应返回 False"""
        monkeypatch.setattr(registry, "MODEL_DIR", str(tmp_path))
        monkeypatch.setattr(registry, "REGISTRY_FILE", str(tmp_path / "registry.json"))

        assert registry.set_current_version("nonexistent") is False

    def test_get_model_path_fallback(self, tmp_path, monkeypatch):
        """无版本时回退到旧路径"""
        monkeypatch.setattr(registry, "MODEL_DIR", str(tmp_path))
        monkeypatch.setattr(registry, "REGISTRY_FILE", str(tmp_path / "registry.json"))

        path = registry.get_model_path()
        assert path.endswith("wdl_model.pkl")

    def test_get_model_path_for_version(self, tmp_path, monkeypatch):
        """指定版本返回对应路径"""
        monkeypatch.setattr(registry, "MODEL_DIR", str(tmp_path))
        monkeypatch.setattr(registry, "REGISTRY_FILE", str(tmp_path / "registry.json"))

        v = "v_test_path"
        version_dir = registry.register_version(version=v, metrics={})
        # 创建模型文件
        with open(os.path.join(version_dir, "wdl_model.pkl"), "w") as f:
            f.write("dummy")

        path = registry.get_model_path(v)
        assert v in path
        assert path.endswith("wdl_model.pkl")
