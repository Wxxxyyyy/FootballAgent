# -*- coding: utf-8 -*-
"""
LLM 调用日志（落 MySQL，替代原 observability/llm_usage_tracker 的内存实现）

目的: 回答"LLM 每次调用是否成功、多快、消耗多少 token"，
与 prediction_outcomes 表合起来构成 LLM 增益评测的完整证据链
（准确率提升多少 × 花了多少成本）。

设计原则: 纯旁路。写日志失败只打 warning，绝不影响预测主链路。
"""

import os
import logging
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

_ddl_done = False
_ddl_lock = threading.Lock()

DDL = """
CREATE TABLE IF NOT EXISTS llm_calls (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  called_at     DATETIME NOT NULL,
  model         VARCHAR(64),
  home_team     VARCHAR(64),
  away_team     VARCHAR(64),
  latency_ms    INT,
  prompt_tokens INT NULL,
  completion_tokens INT NULL,
  success       TINYINT,       -- 1=正常返回, 0=异常走了兜底
  error         VARCHAR(255),
  trace_id      VARCHAR(32) NULL COMMENT '关联 traces 表的 trace_id（链路追踪用）',
  KEY idx_called_at (called_at),
  KEY idx_trace (trace_id)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
"""

# 兼容旧表: 若表已存在但缺 trace_id 列，自动补列（首次连接时执行一次）
_ALTER_ADD_TRACE_ID = "ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS trace_id VARCHAR(32) NULL COMMENT '关联 traces 表的 trace_id'"
_ALTER_ADD_TRACE_IDX = "ALTER TABLE llm_calls ADD INDEX IF NOT EXISTS idx_trace (trace_id)"


def _connect():
    import pymysql

    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "football123"),
        database=os.getenv("MYSQL_DATABASE", "football_agent"),
        charset="utf8mb4",
        connect_timeout=3,
    )


def log_llm_call(
    model: str,
    home_team: str = "",
    away_team: str = "",
    latency_ms: int = 0,
    prompt_tokens=None,
    completion_tokens=None,
    success: bool = True,
    error: str = "",
) -> bool:
    """记录一次 LLM 调用（best-effort，任何失败静默降级）"""
    global _ddl_done
    try:
        conn = _connect()
        cursor = conn.cursor()
        if not _ddl_done:
            with _ddl_lock:
                if not _ddl_done:
                    cursor.execute(DDL)
                    # 兼容旧表: 补 trace_id 列（MySQL 8.0+ 支持 IF NOT EXISTS）
                    try:
                        cursor.execute(_ALTER_ADD_TRACE_ID)
                        cursor.execute(_ALTER_ADD_TRACE_IDX)
                    except Exception:
                        pass  # 旧版 MySQL 不支持 IF NOT EXISTS，列已存在时会报错，忽略
                    _ddl_done = True

        # 获取当前上下文的 trace_id（由 common.tracer 注入），关联 traces 表
        trace_id = None
        try:
            from common.tracer import get_current_trace_id
            trace_id = get_current_trace_id()
        except Exception:
            pass

        cursor.execute(
            """INSERT INTO llm_calls
               (called_at, model, home_team, away_team, latency_ms,
                prompt_tokens, completion_tokens, success, error, trace_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                (model or "")[:64],
                (home_team or "")[:64],
                (away_team or "")[:64],
                int(latency_ms),
                prompt_tokens,
                completion_tokens,
                int(bool(success)),
                (error or "")[:255],
                trace_id,
            ),
        )
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:  # noqa: BLE001 — 旁路日志绝不阻断预测
        logger.warning(f"[llm_call_log] 写入失败（忽略）: {e}")
        return False
