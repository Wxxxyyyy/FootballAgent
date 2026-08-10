# -*- coding: utf-8 -*-
"""
A/B 测试框架
══════════════════════════════════════════════════════════════

第一性原理:
  新策略上线时不能靠猜，要用数据说话。
  A/B 测试 = 把流量分桶，对照组用旧策略，实验组用新策略，
  收集足够样本后用统计检验判断差异是否显著。

方案:
  1. 用户/请求按 stable hash 分桶（同一用户始终在同一桶，避免交叉污染）
  2. 每次预测记录所属实验和桶
  3. 达到最小样本量后计算 p-value，判断差异是否显著

支持:
  - 创建实验: create_experiment(name, variants, sample_size_per_variant)
  - 分桶: assign_variant(user_id, experiment_name) → "control" | "treatment"
  - 记录结果: record_outcome(experiment, variant, success)
  - 分析: analyze_experiment(name) → p-value, 显著性结论

存储:
  实验配置和结果落 MySQL ab_experiments / ab_outcomes 表
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
import logging
from datetime import datetime
from typing import Optional

import pymysql

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ab_testing")


def _db():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "football123"),
        database=os.getenv("MYSQL_DATABASE", "football_agent"),
        charset="utf8mb4",
        connect_timeout=3,
    )


# ═══════════════════════════════════════════════════════════════
#  表结构
# ═══════════════════════════════════════════════════════════════

_ddl_done = False

DDL = """
CREATE TABLE IF NOT EXISTS ab_experiments (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(64) NOT NULL UNIQUE COMMENT '实验名',
  description TEXT COMMENT '实验描述',
  variants JSON COMMENT '["control", "treatment"]',
  sample_size_per_variant INT DEFAULT 100 COMMENT '每组最小样本量',
  status VARCHAR(16) DEFAULT 'running' COMMENT 'running / completed',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  completed_at DATETIME NULL
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT='A/B 测试实验配置'
"""

DDL_OUTCOMES = """
CREATE TABLE IF NOT EXISTS ab_outcomes (
  id INT AUTO_INCREMENT PRIMARY KEY,
  experiment_name VARCHAR(64) NOT NULL,
  variant VARCHAR(32) NOT NULL COMMENT 'control / treatment',
  user_id VARCHAR(128) COMMENT '用户或请求标识',
  success TINYINT NOT NULL COMMENT '1=成功, 0=失败',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  KEY idx_exp (experiment_name),
  KEY idx_exp_var (experiment_name, variant)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
"""


# ═══════════════════════════════════════════════════════════════
#  分桶（stable hash）
# ═══════════════════════════════════════════════════════════════

def assign_variant(user_id: str, experiment_name: str, variants: list = None) -> str:
    """为用户分配实验桶（同一用户始终在同一桶）

    使用 stable hash 确保同一 user_id + experiment_name 组合始终返回相同结果。
    """
    if variants is None:
        variants = ["control", "treatment"]

    # 用 experiment_name 做 salt，避免不同实验桶分配相关性
    key = f"{experiment_name}:{user_id}"
    hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return variants[hash_val % len(variants)]


# ═══════════════════════════════════════════════════════════════
#  实验管理
# ═══════════════════════════════════════════════════════════════

def create_experiment(name: str, description: str, variants: list, sample_size: int = 100) -> bool:
    """创建新实验

    Args:
        name: 实验名（唯一）
        description: 实验描述
        variants: 变体列表 ["control", "treatment"]
        sample_size: 每组最小样本量
    """
    try:
        conn = _db()
        cursor = conn.cursor()
        cursor.execute(DDL)
        cursor.execute(
            "INSERT IGNORE INTO ab_experiments (name, description, variants, sample_size_per_variant) "
            "VALUES (%s, %s, %s, %s)",
            (name, description, json.dumps(variants), sample_size),
        )
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"实验已创建: {name} (变体: {variants}, 最小样本: {sample_size})")
        return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"创建实验失败: {e}")
        return False


def get_experiment(name: str) -> Optional[dict]:
    """获取实验配置"""
    try:
        conn = _db()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT * FROM ab_experiments WHERE name = %s", (name,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row and isinstance(row.get("variants"), str):
            row["variants"] = json.loads(row["variants"])
        return row
    except Exception as e:
        logger.error(f"获取实验失败: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  结果记录
# ═══════════════════════════════════════════════════════════════

def record_outcome(experiment_name: str, variant: str, user_id: str, success: bool) -> bool:
    """记录一次实验结果"""
    try:
        conn = _db()
        cursor = conn.cursor()
        cursor.execute(DDL_OUTCOMES)
        cursor.execute(
            "INSERT INTO ab_outcomes (experiment_name, variant, user_id, success) VALUES (%s, %s, %s, %s)",
            (experiment_name, variant, user_id, int(success)),
        )
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"记录结果失败: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
#  统计分析
# ═══════════════════════════════════════════════════════════════

def analyze_experiment(name: str) -> dict:
    """分析实验结果

    使用卡方检验判断两组成功率差异是否统计显著。

    Returns:
        {
            "name": ...,
            "variants": {...},
            "sample_sizes": {"control": 100, "treatment": 100},
            "success_rates": {"control": 0.45, "treatment": 0.55},
            "p_value": 0.03,
            "is_significant": True,
            "conclusion": "..."
        }
    """
    try:
        conn = _db()
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        exp = get_experiment(name)
        if not exp:
            return {"error": "实验不存在"}

        # 获取各变体的成功/失败数
        cursor.execute(
            "SELECT variant, COUNT(*) as total, SUM(success) as success FROM ab_outcomes "
            "WHERE experiment_name = %s GROUP BY variant",
            (name,),
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            return {"name": name, "error": "无结果数据"}

        stats = {}
        for row in rows:
            stats[row["variant"]] = {
                "total": row["total"],
                "success": row["success"],
                "rate": row["success"] / row["total"] if row["total"] > 0 else 0,
            }

        # 卡方检验（2x2 列联表）
        from scipy.stats import chi2_contingency

        # 构建 2x2 矩阵: [[success_control, fail_control], [success_treatment, fail_treatment]]
        variants_list = list(stats.keys())
        if len(variants_list) < 2:
            return {"name": name, "stats": stats, "error": "需要至少2个变体"}

        v1, v2 = variants_list[0], variants_list[1]
        s1, t1 = stats[v1]["success"], stats[v1]["total"]
        s2, t2 = stats[v2]["success"], stats[v2]["total"]

        contingency = [[s1, t1 - s1], [s2, t2 - s2]]
        chi2, p_value, _, _ = chi2_contingency(contingency)

        is_significant = p_value < 0.05
        min_sample = exp.get("sample_size_per_variant", 100)
        has_enough_data = t1 >= min_sample and t2 >= min_sample

        if not has_enough_data:
            conclusion = f"样本不足（需每组 {min_sample}，当前 {v1}={t1}, {v2}={t2}），继续收集"
        elif is_significant:
            better = v1 if stats[v1]["rate"] > stats[v2]["rate"] else v2
            conclusion = f"差异显著 (p={p_value:.4f})，{better} 表现更好"
        else:
            conclusion = f"差异不显著 (p={p_value:.4f})，两组表现相近"

        return {
            "name": name,
            "stats": {v: {"total": s["total"], "success": s["success"],
                          "rate": round(s["rate"], 4)} for v, s in stats.items()},
            "p_value": round(float(p_value), 4),
            "is_significant": is_significant,
            "has_enough_data": has_enough_data,
            "conclusion": conclusion,
        }
    except ImportError:
        return {"error": "scipy 未安装，无法进行统计检验"}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
#  CLI 入口
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="A/B 测试框架")
    parser.add_argument("--create", nargs=3, metavar=("NAME", "DESC", "VARIANTS"),
                        help="创建实验: NAME DESC 'control,treatment'")
    parser.add_argument("--analyze", type=str, help="分析实验结果")
    args = parser.parse_args()

    if args.create:
        name, desc, variants_str = args.create
        variants = variants_str.split(",")
        create_experiment(name, desc, variants)
    elif args.analyze:
        result = analyze_experiment(args.analyze)
        import pprint
        pprint.pprint(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
