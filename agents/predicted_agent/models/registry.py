# -*- coding: utf-8 -*-
"""
模型版本注册中心
══════════════════════════════════════════════════════════════

第一性原理:
  模型版本管理的本质 = 回答"当前线上用的是哪个模型？它表现如何？能否回滚？"
  当前问题: 只有一份 wdl_model.pkl，训练新模型直接覆盖旧模型，无法对比/回滚。

设计方案:
  每次训练保存到 saved/{version}/ 目录，含:
    - wdl_model.pkl      (模型权重)
    - model_meta.json    (训练指标: 准确率/特征/样本数)
    - version_info.json   (版本信息: 版本号/创建时间/git commit/训练数据范围)

  registry.json 记录所有版本及当前线上版本，支持:
    - list_versions()    — 列出所有版本
    - get_current()      — 获取当前线上版本
    - get_model_path(v)  — 获取指定版本的模型路径
    - set_current(v)     — 切换线上版本（回滚/灰度）

版本命名: v{timestamp} (如 v20260703_172000)
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Optional

# 模型存储根目录
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
REGISTRY_FILE = os.path.join(MODEL_DIR, "registry.json")


# ═══════════════════════════════════════════════════════════════
#  版本信息管理
# ═══════════════════════════════════════════════════════════════

def _load_registry() -> dict:
    """加载注册表"""
    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"current_version": None, "versions": []}


def _save_registry(registry: dict):
    """保存注册表"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def generate_version() -> str:
    """生成版本号: v{YYYYMMDD_HHMMSS}"""
    return "v" + datetime.now().strftime("%Y%m%d_%H%M%S")


def get_version_dir(version: str) -> str:
    """获取版本目录路径"""
    return os.path.join(MODEL_DIR, version)


# ═══════════════════════════════════════════════════════════════
#  注册/查询 API
# ═══════════════════════════════════════════════════════════════

def register_version(
    version: str,
    metrics: dict,
    data_range: str = "",
    git_commit: str = "",
) -> str:
    """注册一个新版本

    Args:
        version: 版本号 (如 v20260703_172000)
        metrics: 训练指标 (val_accuracy, cv_accuracy, train_samples 等)
        data_range: 训练数据范围描述 (如 "五大联赛 2021-2024")
        git_commit: 当前 git commit hash

    Returns: 版本目录路径
    """
    registry = _load_registry()

    version_info = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "metrics": metrics,
        "data_range": data_range,
        "git_commit": git_commit,
    }

    # 更新或添加版本记录
    versions = registry.get("versions", [])
    versions = [v for v in versions if v["version"] != version]
    versions.append(version_info)
    registry["versions"] = versions

    # 如果是第一个版本，设为 current
    if registry.get("current_version") is None:
        registry["current_version"] = version

    _save_registry(registry)

    version_dir = get_version_dir(version)
    os.makedirs(version_dir, exist_ok=True)

    # 保存版本信息到版本目录
    with open(os.path.join(version_dir, "version_info.json"), "w", encoding="utf-8") as f:
        json.dump(version_info, f, ensure_ascii=False, indent=2)

    return version_dir


def list_versions() -> list:
    """列出所有版本（按创建时间倒序）"""
    registry = _load_registry()
    versions = registry.get("versions", [])
    return sorted(versions, key=lambda v: v.get("created_at", ""), reverse=True)


def get_current_version() -> Optional[str]:
    """获取当前线上版本号"""
    registry = _load_registry()
    return registry.get("current_version")


def set_current_version(version: str) -> bool:
    """切换当前线上版本（用于回滚/灰度）

    Returns: True=成功, False=版本不存在
    """
    registry = _load_registry()
    versions = [v["version"] for v in registry.get("versions", [])]
    if version not in versions:
        return False
    registry["current_version"] = version
    _save_registry(registry)
    return True


def get_model_path(version: Optional[str] = None) -> str:
    """获取模型文件路径

    Args:
        version: 指定版本；None=当前线上版本；找不到则回退到旧的 saved/wdl_model.pkl

    Returns: wdl_model.pkl 的完整路径
    """
    if version is None:
        version = get_current_version()

    if version:
        path = os.path.join(get_version_dir(version), "wdl_model.pkl")
        if os.path.exists(path):
            return path

    # 回退到旧路径（兼容无版本管理的模型）
    return os.path.join(MODEL_DIR, "wdl_model.pkl")


def get_meta_path(version: Optional[str] = None) -> str:
    """获取模型元信息文件路径"""
    if version is None:
        version = get_current_version()

    if version:
        path = os.path.join(get_version_dir(version), "model_meta.json")
        if os.path.exists(path):
            return path

    return os.path.join(MODEL_DIR, "model_meta.json")


# ═══════════════════════════════════════════════════════════════
#  CLI 入口
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="模型版本管理")
    parser.add_argument("--list", action="store_true", help="列出所有版本")
    parser.add_argument("--current", action="store_true", help="查看当前版本")
    parser.add_argument("--set", type=str, help="切换当前版本")
    args = parser.parse_args()

    if args.list:
        versions = list_versions()
        if not versions:
            print("暂无已注册版本")
            return
        current = get_current_version()
        print(f"{'版本':<20s} {'创建时间':<26s} {'验证准确率':<10s} {'状态'}")
        print("-" * 70)
        for v in versions:
            acc = v.get("metrics", {}).get("val_accuracy", "N/A")
            status = "← 当前" if v["version"] == current else ""
            print(f"{v['version']:<20s} {v.get('created_at','N/A'):<26s} {str(acc):<10s} {status}")

    elif args.current:
        v = get_current_version()
        print(f"当前线上版本: {v}" if v else "未设置线上版本")

    elif args.set:
        ok = set_current_version(args.set)
        print(f"已切换到 {args.set}" if ok else f"版本 {args.set} 不存在")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
