# -*- coding: utf-8 -*-
"""
赛后准确率自动跟踪（宿主机侧，由 systemd timer 每天 15:30 运行）

数据闭环: prediction → outcome → metric
  1. 扫描 data/predictions/*_tier*.json（每份 = 一次触发点的预测）
  2. 从 OpenClaw /matches/today 获取已完赛比分（match_id 直接关联，无需队名映射）
  3. 逐份评估: LLM 胜平负命中 / ML 命中 / 蒙特卡洛命中 / 比分命中 / Brier 分
  4. 结果落 MySQL prediction_outcomes 表（幂等: 按预测文件名去重）
  5. 周日推送"本周预测战绩"到手机（Bark）

用法:
  python3 scripts/accuracy_tracker.py            # 常规: 评估新完赛比赛（周日自动附带周报）
  python3 scripts/accuracy_tracker.py --report   # 强制推送战绩报告（不限周日）
  python3 scripts/accuracy_tracker.py --dry-run  # 只打印评估结果，不写库不推送
"""

import os
import re
import sys
import json
import glob
import time
import logging
from datetime import datetime, timedelta

import httpx
import pymysql
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("accuracy_tracker")

PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "data", "predictions")
OPENCLAW_URL = os.getenv("OPENCLAW_URL", "http://127.0.0.1:9000")

BARK_KEY = os.getenv("BARK_KEY", "")
PUBLIC_BARK_BASE = "https://api.day.app"

RESULT_LABEL = {"H": "主胜", "D": "平局", "A": "客胜"}


# ══════════════════ MySQL ══════════════════

def _db():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "football123"),
        database=os.getenv("MYSQL_DATABASE", "football_agent"),
        charset="utf8mb4",
    )


DDL = """
CREATE TABLE IF NOT EXISTS prediction_outcomes (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  match_id      VARCHAR(20)  NOT NULL,
  home_team     VARCHAR(64),
  away_team     VARCHAR(64),
  tier          TINYINT,
  pred_file     VARCHAR(255) NOT NULL,
  prediction_time DATETIME NULL,
  -- 三方预测
  pred_primary   CHAR(1),  -- LLM 首选 (H/D/A)
  pred_secondary CHAR(1),  -- LLM 次选
  confidence     VARCHAR(8),
  ml_pred        CHAR(1),  -- ML 模型 argmax
  mc_pred        CHAR(1),  -- 蒙特卡洛 argmax
  mc_score       VARCHAR(16),  -- 蒙特卡洛最可能比分
  llm_scores     JSON NULL,    -- LLM 比分预测(前2)
  -- 实际结果
  actual_score   VARCHAR(16),
  actual_result  CHAR(1),
  -- 命中标记
  llm_hit        TINYINT,  -- LLM 首选命中
  secondary_hit  TINYINT,  -- LLM 首选或次选命中
  ml_hit         TINYINT,
  mc_hit         TINYINT,
  mc_score_hit   TINYINT,  -- 蒙特卡洛比分精确命中
  llm_score_hit  TINYINT,  -- LLM 比分(前2)精确命中
  -- 概率质量
  brier_ml       FLOAT NULL,  -- ML 概率 Brier 分（越小越好）
  brier_mc       FLOAT NULL,
  evaluated_at   DATETIME,
  UNIQUE KEY uq_pred_file (pred_file)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
"""


# ══════════════════ 数据获取 ══════════════════

def fetch_finished_results() -> dict:
    """从 OpenClaw 拉已完赛比分 → {match_id: {"score", "hg", "ag", "result"}}"""
    try:
        r = httpx.get(f"{OPENCLAW_URL}/matches/today", timeout=30)
        matches = r.json().get("matches", [])
    except Exception as e:
        logger.error(f"获取比赛列表失败: {e}")
        return {}

    results = {}
    for m in matches:
        if not m.get("finished"):
            continue
        score = (m.get("score") or "").strip()
        mm = re.match(r"^(\d+)\s*-\s*(\d+)$", score)
        if not mm:
            continue
        hg, ag = int(mm.group(1)), int(mm.group(2))
        results[str(m["match_id"])] = {
            "score": f"{hg}-{ag}",
            "hg": hg,
            "ag": ag,
            "result": "H" if hg > ag else ("A" if hg < ag else "D"),
        }
    return results


def load_predictions() -> list:
    """加载所有真实预测文件（match_id 为纯数字，排除测试文件）"""
    preds = []
    for path in sorted(glob.glob(os.path.join(PREDICTIONS_DIR, "*_tier*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        if not str(d.get("match_id", "")).isdigit():
            continue  # 测试/手动验证文件
        d["_file"] = os.path.basename(path)
        preds.append(d)
    return preds


# ══════════════════ 评估 ══════════════════

def _argmax_wdl(probs: dict) -> str:
    """{"home_win_prob","draw_prob","away_win_prob"} → H/D/A"""
    trio = [
        ("H", probs.get("home_win_prob") or 0),
        ("D", probs.get("draw_prob") or 0),
        ("A", probs.get("away_win_prob") or 0),
    ]
    return max(trio, key=lambda x: x[1])[0]


def _brier(probs: dict, actual: str):
    """三向 Brier 分: sum((p_i - y_i)^2)，范围 0~2，越小越好"""
    try:
        p = {
            "H": float(probs.get("home_win_prob") or 0),
            "D": float(probs.get("draw_prob") or 0),
            "A": float(probs.get("away_win_prob") or 0),
        }
    except (TypeError, ValueError):
        return None
    if not any(p.values()):
        return None
    return round(sum((p[k] - (1.0 if k == actual else 0.0)) ** 2 for k in "HDA"), 4)


def _norm_score(s: str) -> str:
    """'1:0' / '1-0' / '1 - 0' → '1-0'"""
    return re.sub(r"\s*[:\-]\s*", "-", str(s or "").strip())


def evaluate(pred: dict, outcome: dict) -> dict:
    """单份预测 vs 实际结果 → 评估行"""
    actual = outcome["result"]
    llm = pred.get("llm_analysis") or {}
    wdl = llm.get("wdl_prediction") or {}
    ml = pred.get("ml_prediction") or {}
    mc = pred.get("monte_carlo") or {}

    primary = wdl.get("primary") or ""
    secondary = wdl.get("secondary") or ""
    confidence = wdl.get("confidence") or ""
    if not primary:
        # 旧格式兼容（2026-07-02 之前 llm_analysis 无 wdl_prediction）:
        # 取 LLM 对 ML/蒙卡的判读结果作为其最终倾向
        legacy = llm.get("ml_prediction") or llm.get("monte_carlo_prediction") or {}
        if isinstance(legacy, dict):
            primary = legacy.get("result") or ""
            confidence = confidence or legacy.get("confidence") or ""
    ml_pred = _argmax_wdl(ml) if "home_win_prob" in ml else ""
    mc_pred = _argmax_wdl(mc) if "home_win_prob" in mc else ""
    mc_score = _norm_score(mc.get("most_likely_score"))

    llm_scores = []
    for s in (llm.get("score_predictions") or [])[:2]:
        sc = _norm_score(s.get("score")) if isinstance(s, dict) else _norm_score(s)
        if sc:
            llm_scores.append(sc)

    pred_time = None
    try:
        pred_time = datetime.fromisoformat(pred.get("prediction_time", "")).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        pass

    return {
        "match_id": str(pred["match_id"]),
        "home_team": pred.get("home_team", ""),
        "away_team": pred.get("away_team", ""),
        "tier": int(pred.get("tier") or 1),
        "pred_file": pred["_file"],
        "prediction_time": pred_time,
        "pred_primary": primary[:1],
        "pred_secondary": secondary[:1],
        "confidence": confidence[:8],
        "ml_pred": ml_pred,
        "mc_pred": mc_pred,
        "mc_score": mc_score,
        "llm_scores": json.dumps(llm_scores, ensure_ascii=False),
        "actual_score": outcome["score"],
        "actual_result": actual,
        "llm_hit": int(primary == actual),
        "secondary_hit": int(actual in (primary, secondary)),
        "ml_hit": int(ml_pred == actual) if ml_pred else None,
        "mc_hit": int(mc_pred == actual) if mc_pred else None,
        "mc_score_hit": int(mc_score == outcome["score"]) if mc_score else None,
        "llm_score_hit": int(outcome["score"] in llm_scores) if llm_scores else None,
        "brier_ml": _brier(ml, actual),
        "brier_mc": _brier(mc, actual),
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def run_evaluation(dry_run: bool = False) -> int:
    """主流程: 评估所有未入库的已完赛预测。返回新增行数。"""
    results = fetch_finished_results()
    logger.info(f"已完赛比赛: {len(results)} 场")
    preds = load_predictions()
    logger.info(f"预测文件: {len(preds)} 份")
    if not results or not preds:
        return 0

    conn = _db()
    cursor = conn.cursor()
    cursor.execute(DDL)
    cursor.execute("SELECT pred_file FROM prediction_outcomes")
    done = {row[0] for row in cursor.fetchall()}

    new_rows = 0
    for pred in preds:
        if pred["_file"] in done:
            continue
        outcome = results.get(str(pred["match_id"]))
        if not outcome:
            continue  # 未完赛或已不在页面上
        row = evaluate(pred, outcome)
        if dry_run:
            logger.info(f"[dry-run] {row['home_team']} vs {row['away_team']} "
                        f"tier{row['tier']} 预测={row['pred_primary']} 实际={row['actual_result']} "
                        f"命中={'✓' if row['llm_hit'] else '✗'}")
            new_rows += 1
            continue
        cols = ", ".join(row.keys())
        ph = ", ".join(["%s"] * len(row))
        cursor.execute(
            f"INSERT IGNORE INTO prediction_outcomes ({cols}) VALUES ({ph})",
            list(row.values()),
        )
        new_rows += cursor.rowcount

    if not dry_run:
        conn.commit()
    cursor.close()
    conn.close()
    logger.info(f"新增评估: {new_rows} 条")
    return new_rows


# ══════════════════ 战绩报告 ══════════════════

def build_report(days: int = 7) -> str:
    """聚合近 N 天战绩。逐场口径: 每场取最终预测(tier2 优先，否则最晚一份)。"""
    conn = _db()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

    # 每场的最终预测（tier 最大 → 时间最晚）
    cursor.execute("""
        SELECT o.* FROM prediction_outcomes o
        JOIN (
            SELECT match_id, MAX(CONCAT(tier, '|', COALESCE(prediction_time, ''))) AS mx
            FROM prediction_outcomes
            WHERE evaluated_at >= %s OR prediction_time >= %s
            GROUP BY match_id
        ) t ON o.match_id = t.match_id
           AND CONCAT(o.tier, '|', COALESCE(o.prediction_time, '')) = t.mx
    """, (since, since))
    finals = cursor.fetchall()

    # 全量行（所有触发档）用于三方对比
    cursor.execute(
        "SELECT * FROM prediction_outcomes WHERE evaluated_at >= %s OR prediction_time >= %s",
        (since, since),
    )
    all_rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not finals:
        return f"近{days}天暂无已评估的预测"

    n = len(finals)
    llm_hits = sum(r["llm_hit"] or 0 for r in finals)
    sec_hits = sum(r["secondary_hit"] or 0 for r in finals)
    score_hits = sum(1 for r in finals if r["llm_score_hit"] or r["mc_score_hit"])

    def _rate(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (0, 0)

    llm_all, _ = _rate(all_rows, "llm_hit")
    ml_all, _ = _rate(all_rows, "ml_hit")
    mc_all, _ = _rate(all_rows, "mc_hit")
    brier_ml = [r["brier_ml"] for r in all_rows if r["brier_ml"] is not None]
    brier_mc = [r["brier_mc"] for r in all_rows if r["brier_mc"] is not None]

    lines = [
        f"📊 近{days}天战绩（{n}场，按最终预测）",
        f"胜平负: {llm_hits}/{n} = {llm_hits/n:.0%}",
        f"含次选: {sec_hits}/{n} = {sec_hits/n:.0%}",
        f"比分命中: {score_hits}/{n}",
        "",
        f"三方对比（全部{len(all_rows)}次触发）:",
        f"LLM {llm_all:.0%} | ML {ml_all:.0%} | 蒙卡 {mc_all:.0%}",
    ]
    if brier_ml and brier_mc:
        lines.append(f"Brier: ML {sum(brier_ml)/len(brier_ml):.3f} | 蒙卡 {sum(brier_mc)/len(brier_mc):.3f}")

    # 最近战绩明细（最多5场）
    lines.append("")
    recent = sorted(finals, key=lambda r: r.get("prediction_time") or datetime.min, reverse=True)[:5]
    for r in recent:
        mark = "✅" if r["llm_hit"] else ("🟡" if r["secondary_hit"] else "❌")
        lines.append(f"{mark} {r['home_team']} {r['actual_score']} {r['away_team']} "
                     f"(预测{RESULT_LABEL.get(r['pred_primary'], '?')})")
    return "\n".join(lines)


def push_report(body: str) -> bool:
    """Bark 推送战绩（公共服务器 + 重试）"""
    if not BARK_KEY:
        logger.warning("BARK_KEY 未配置，跳过推送")
        return False
    payload = {
        "title": "⚽ 预测战绩周报",
        "body": body,
        "sound": "minuet",
        "group": "预测战绩",
    }
    for attempt in range(1, 4):
        try:
            r = httpx.post(f"{PUBLIC_BARK_BASE}/{BARK_KEY}/", json=payload, timeout=20)
            if r.status_code == 200 and r.json().get("code") == 200:
                logger.info("战绩推送成功")
                return True
        except Exception as e:
            logger.warning(f"第{attempt}次推送失败: {e}")
        time.sleep(2 ** attempt)
    logger.error("战绩推送最终失败")
    return False


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    force_report = "--report" in sys.argv

    new_rows = run_evaluation(dry_run=dry_run)

    is_sunday = datetime.now().weekday() == 6
    if force_report or (is_sunday and not dry_run):
        report = build_report(days=7)
        print(report)
        if not dry_run:
            push_report(report)
    elif new_rows:
        logger.info("非周日，跳过推送（--report 可强制）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
