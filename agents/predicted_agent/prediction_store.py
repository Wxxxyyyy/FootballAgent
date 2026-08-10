# -*- coding: utf-8 -*-
"""
预测结果精简存储（prediction_records 表）
═══════════════════════════════════════════════════════════════
每场预测完成后，只把**结论**（胜平负/比分/大小球/爆冷等级）落库，
**不存推理过程**——用于 L3 压缩清掉长预测文本后的历史召回。

  - save_prediction_result()  预测成功后写入一条精简记录（best-effort，不阻断主流程）
  - query_recent_predictions() 读取最近记录
  - read_prediction           @tool，注册进 ReAct，供"上次预测那场…"类问题召回
"""

import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

_DB = os.getenv("MYSQL_DATABASE", "football_agent")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS prediction_records (
  id          INT AUTO_INCREMENT PRIMARY KEY,
  match_date  VARCHAR(32),
  home_team   VARCHAR(64),
  away_team   VARCHAR(64),
  wdl_result  VARCHAR(16)  COMMENT '主胜/平局/客胜',
  wdl_prob    VARCHAR(16)  COMMENT '胜平负主结论概率',
  ou_result   VARCHAR(16)  COMMENT '大球/小球',
  ou_prob     VARCHAR(16),
  score_pred  VARCHAR(128) COMMENT 'Top 比分，逗号分隔',
  upset_level VARCHAR(16)  COMMENT '爆冷等级：无/轻度/重度',
  created_at  DATETIME,
  KEY idx_match (home_team, away_team),
  KEY idx_created (created_at)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
"""


def _connect():
    import pymysql
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "football123"),
        database=_DB,
        charset="utf8mb4",
        connect_timeout=3,
    )


def _ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(_CREATE_TABLE)
    conn.commit()


# ─── 从 predict() 返回的完整结果中提取精简结论 ────────────────
def _extract_concise(result: dict) -> dict:
    llm = result.get("llm_analysis") or {}
    wdl = llm.get("wdl_prediction") or {}
    ou = llm.get("ou_prediction") or {}
    scores = llm.get("score_predictions") or []
    upset = llm.get("upset_alert") or {}

    wdl_map = {"H": "主胜", "D": "平局", "A": "客胜"}
    ou_map = {"Over": "大球", "Under": "小球"}

    score_list = [s.get("score", "") for s in scores if isinstance(s, dict)]
    score_pred = ", ".join([s for s in score_list if s][:3])

    return {
        "match_date": str(result.get("date") or ""),
        "home_team": result.get("home_team", ""),
        "away_team": result.get("away_team", ""),
        "wdl_result": wdl_map.get(wdl.get("primary"), wdl.get("primary", "")),
        "wdl_prob": str(wdl.get("primary_prob", "")),
        "ou_result": ou_map.get(ou.get("result"), ou.get("result", "")),
        "ou_prob": str(ou.get("prob", "")),
        "score_pred": score_pred,
        "upset_level": (upset.get("level", "") if upset.get("triggered") else "无"),
    }


def save_prediction_result(result: dict) -> bool:
    """预测成功后写入一条精简结论。任何失败静默降级，绝不影响预测主流程。"""
    try:
        # 预测失败/降级（无有效 llm_analysis）不记录
        llm = result.get("llm_analysis")
        if not isinstance(llm, dict) or "error" in llm:
            return False
        c = _extract_concise(result)
        conn = _connect()
        try:
            _ensure_table(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO prediction_records
                       (match_date, home_team, away_team, wdl_result, wdl_prob,
                        ou_result, ou_prob, score_pred, upset_level, created_at)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        c["match_date"], c["home_team"], c["away_team"],
                        c["wdl_result"], c["wdl_prob"], c["ou_result"], c["ou_prob"],
                        c["score_pred"], c["upset_level"], datetime.now(),
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        logger.info(f"[预测存储] 已记录 {c['home_team']} vs {c['away_team']} → {c['wdl_result']}")
        return True
    except Exception as e:
        logger.warning(f"[预测存储] 写入失败（已忽略，不影响主流程）: {e}")
        return False


def query_recent_predictions(limit: int = 10) -> list[dict]:
    """读取最近的预测记录（精简结论）"""
    try:
        conn = _connect()
        try:
            _ensure_table(conn)
            with conn.cursor(pymysql_dict_cursor()) as cur:
                cur.execute(
                    """SELECT match_date, home_team, away_team, wdl_result, wdl_prob,
                              ou_result, ou_prob, score_pred, upset_level, created_at
                       FROM prediction_records ORDER BY id DESC LIMIT %s""",
                    (limit,),
                )
                return cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"[预测存储] 查询失败: {e}")
        return []


def pymysql_dict_cursor():
    import pymysql
    return pymysql.cursors.DictCursor


# ─── ReAct 工具：历史预测召回 ────────────────────────────────
from langchain_core.tools import tool  # noqa: E402


@tool
def read_prediction(dummy: str = "") -> str:
    """查询最近的赛前预测记录（只有结论，无推理过程）。
    用于回答"上次预测的那场结果如何/你预测谁会赢"这类回顾历史预测的问题。"""
    records = query_recent_predictions(limit=10)
    if not records:
        return "暂无历史预测记录。"
    lines = ["最近的赛前预测记录（按时间倒序）："]
    for r in records:
        date = r.get("match_date") or str(r.get("created_at", ""))[:10]
        lines.append(
            f"- {date} {r['home_team']} vs {r['away_team']}："
            f"{r['wdl_result']}({r['wdl_prob']})，大小球 {r['ou_result']}({r['ou_prob']})，"
            f"参考比分 {r['score_pred']}，爆冷 {r['upset_level']}"
        )
    return "\n".join(lines)
