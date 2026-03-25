# -*- coding: utf-8 -*-
"""
服务器端 API - 双向通信枢纽

数据流向:
  正向（接收）: 旧电脑(OpenClaw) → 本地中继 → 本服务器
  反向（请求）: 本服务器 → 本地中继 → 旧电脑(OpenClaw)

端口: 8000
"""

import sys
import os
from fastapi import FastAPI, HTTPException
import httpx
import uvicorn
import uuid
import json
import asyncio
import threading
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
from contextlib import asynccontextmanager

# 将项目根目录加入 sys.path，确保能 import pipeline 模块
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==================== 配置 ====================

# 中继站地址（通过 SSH 反向端口转发连接到本地电脑的 5000 端口）
# 需要在本地电脑执行: ssh -R 15000:localhost:5000 user@server
# 这样服务器上的 localhost:15000 就映射到本地电脑的 5000 端口
RELAY_URL = "http://localhost:15000"

# 同步任务默认超时（秒）
TASK_TIMEOUT = 30.0

# 接收数据存储目录
DATA_DIR = Path(__file__).parent.parent / "data" / "openclaw_received"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 异步任务管理 ====================

# 异步任务结果暂存: task_id → {"content": ..., "received_at": ...}
_task_results: dict[str, Any] = {}
# 异步任务等待信号: task_id → asyncio.Event
_task_events: dict[str, asyncio.Event] = {}


# ==================== FastAPI 应用 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print(f"⚽ 服务器 API 启动 | 中继站: {RELAY_URL}")
    yield
    print("⚽ 服务器 API 关闭")

app = FastAPI(
    title="Football Agent 服务器 API",
    description="双向通信枢纽 - 接收 OpenClaw 数据 & 向 OpenClaw 发送任务",
    lifespan=lifespan,
)


# ==============================================================
#  正向接口: 接收数据（OpenClaw → 中继 → 服务器）
# ==============================================================

@app.get("/")
async def root():
    """健康检查"""
    return {
        "status": "running",
        "message": "⚽ Football Agent 服务器 API 已就绪",
        "relay_url": RELAY_URL,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/receive_data")
async def receive_data(data: dict):
    """
    统一数据接收入口（正向通道）

    支持的 data_type:
      - realtime_odds   : 实时赔率
      - daily_matches   : 每日已结束比赛
      - injury_report   : 伤病报告
      - recent_matches  : 近期比赛记录
      - csv_bulk        : CSV 批量数据（兼容旧格式）
      - realtime        : 旧实时数据格式（兼容）
      - task_result     : 异步任务结果回传
    """
    data_type = data.get("data_type", "unknown")
    task_id = data.get("task_id")
    content = data.get("content")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---------- 异步任务结果回传 ----------
    if data_type == "task_result" and task_id:
        _task_results[task_id] = {"content": content, "received_at": ts}
        if task_id in _task_events:
            _task_events[task_id].set()
        print(f"📨 [{ts}] 异步任务结果 -> task_id: {task_id[:8]}...")
        return {"status": "success", "message": "任务结果已接收", "task_id": task_id}

    # ---------- 实时赔率 ----------
    if data_type == "realtime_odds":
        match_info = content.get("match", "未知比赛") if isinstance(content, dict) else "N/A"
        odds = content.get("odds", "N/A") if isinstance(content, dict) else "N/A"
        print(f"⚡ [{ts}] 实时赔率 -> {match_info}, 赔率: {odds}")
        _save_received_data(data, "realtime_odds")
        # TODO: 写入 Redis 缓存，供 prediction_agent 实时读取
        return {"status": "success", "message": "实时赔率已接收"}

    # ---------- 每日已结束比赛（兼容 football_daily_report） ----------
    if data_type in ("daily_matches", "football_daily_report"):
        return _handle_daily_matches(ts, content, data)

    # ---------- 赛前分析（OpenClaw 回传） ----------
    if data_type == "pre_match_analysis":
        return _handle_pre_match_analysis(ts, content, data)

    # ---------- 伤病报告 ----------
    if data_type == "injury_report":
        team = content.get("team", "未知") if isinstance(content, dict) else "N/A"
        print(f"🏥 [{ts}] 伤病报告 -> 球队: {team}")
        _save_received_data(data, "injury_report")
        if isinstance(content, dict):
            injuries = content.get("injuries", [])
            for inj in injuries:
                player = inj.get("player", "未知") if isinstance(inj, dict) else str(inj)
                status_text = inj.get("status", "") if isinstance(inj, dict) else ""
                print(f"       - {player}: {status_text}")
        return {"status": "success", "message": "伤病报告已接收"}

    # ---------- 近期比赛 ----------
    if data_type == "recent_matches":
        team = content.get("team", "未知") if isinstance(content, dict) else "N/A"
        matches = content.get("matches", []) if isinstance(content, dict) else []
        print(f"📊 [{ts}] 近期比赛 -> 球队: {team}, {len(matches)} 场")
        _save_received_data(data, "recent_matches")
        for m in matches[:5]:
            if isinstance(m, dict):
                print(f"       - {m.get('home_team', '?')} vs {m.get('away_team', '?')}: {m.get('home_score', '?')}-{m.get('away_score', '?')}")
        return {"status": "success", "message": "近期比赛数据已接收"}

    # ---------- CSV 批量（兼容旧格式）----------
    if data_type == "csv_bulk":
        row_count = len(content) if isinstance(content, list) else 0
        print(f"📁 [{ts}] 批量数据 -> 共 {row_count} 条")
        _save_received_data(data, "csv_bulk")
        return {"status": "success", "message": f"收到 {row_count} 条批量数据"}

    # ---------- 旧 realtime 格式（兼容）----------
    if data_type == "realtime":
        match_info = content.get("match", "未知比赛") if isinstance(content, dict) else "N/A"
        odds = content.get("odds", "N/A") if isinstance(content, dict) else "N/A"
        print(f"⚡ [{ts}] 实时数据(旧) -> {match_info}, 赔率: {odds}")
        return {"status": "success", "message": "实时数据已接收"}

    # ---------- 未知类型（仍然保存以便查看） ----------
    print(f"❓ [{ts}] 未知数据类型: {data_type}")
    print(f"   完整数据预览: {json.dumps(data, ensure_ascii=False, default=str)[:500]}")
    _save_received_data(data, f"unknown_{data_type}")
    return {"status": "warning", "message": f"未知数据类型: {data_type}，数据已保存到文件"}


def _save_received_data(data: dict, prefix: str):
    """将接收到的数据保存为 JSON 文件，方便后续查看和调试"""
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = DATA_DIR / f"{prefix}_{ts_file}.json"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        print(f"   💾 数据已保存: {filename.name}")
    except Exception as e:
        print(f"   ⚠️ 保存失败: {e}")


def _handle_pre_match_analysis(ts: str, content, data: dict) -> dict:
    """
    处理 OpenClaw 赛前分析数据

    流程:
      1. 保存原始 JSON
      2. 通过 pre_match_state 通知等待中的预测流水线
    """
    _save_received_data(data, "pre_match_analysis")

    home = content.get("home_team", "?") if isinstance(content, dict) else "?"
    away = content.get("away_team", "?") if isinstance(content, dict) else "?"
    print(f"📋 [{ts}] 赛前分析数据 -> {home} vs {away}")

    # 通知等待中的预测模块
    if isinstance(content, dict) and home != "?" and away != "?":
        try:
            from api.pre_match_state import notify_pre_match
            notify_pre_match(home, away, content)
            print(f"   📡 已通知预测模块")
        except Exception as e:
            print(f"   ⚠️ 通知预测模块失败: {e}")

    return {"status": "success", "message": f"赛前分析数据已接收 ({home} vs {away})"}


def _handle_daily_matches(ts: str, content, data: dict) -> dict:
    """
    处理每日比赛数据

    流程:
      1. 保存原始 JSON 作为凭证
      2. 解析并打印摘要
      3. 在后台线程触发 MySQL + Neo4j 增量入库
    """
    _save_received_data(data, "daily_matches")

    # 检查是否有实际比赛数据
    total_matches = 0
    if isinstance(content, dict):
        inner = content.get("data", content)
        total_matches = inner.get("total_matches", 0)

        # 打印摘要
        date = inner.get("date", content.get("date", "未知日期"))
        leagues = inner.get("leagues", {})
        for code, info in leagues.items():
            if isinstance(info, dict):
                league_name = info.get("league_name", code)
                match_count = info.get("match_count", 0)
                match_list = info.get("matches", [])
                if match_count > 0:
                    print(f"📅 [{ts}] {league_name} ({date}) -> {match_count} 场")
                    for m in match_list[:3]:
                        if isinstance(m, dict):
                            home = m.get("HomeTeam", "?")
                            away = m.get("AwayTeam", "?")
                            hs = m.get("FTHG", "?")
                            aws = m.get("FTAG", "?")
                            print(f"       ⚽ {home} {hs}-{aws} {away}")
                    if match_count > 3:
                        print(f"       ... 还有 {match_count - 3} 场")

    if total_matches == 0:
        print(f"📅 [{ts}] 该日期无比赛数据，跳过入库")
        return {"status": "success", "message": "无比赛数据（当日无赛事）"}

    # 后台线程执行入库，不阻塞 HTTP 响应
    _trigger_ingestion_background(data)

    return {
        "status": "success",
        "message": f"收到 {total_matches} 场比赛数据，正在后台入库 MySQL + Neo4j",
    }


def _trigger_ingestion_background(data: dict):
    """在后台线程中执行 MySQL + Neo4j 增量入库"""
    def _worker():
        try:
            from pipeline.openclaw_ingestion import ingest_openclaw_data
            result = ingest_openclaw_data(data)
            print(f"📦 后台入库完成: {json.dumps(result, ensure_ascii=False)}")
        except Exception as e:
            print(f"📦 后台入库异常: {e}")
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


@app.post("/receive_odds")
async def receive_odds(data: dict):
    """兼容旧接口，内部转发到统一入口"""
    return await receive_data(data)


# ==============================================================
#  反向接口: 向 OpenClaw 发送任务（服务器 → 中继 → OpenClaw）
# ==============================================================

async def _send_to_openclaw(
    task_type: str,
    params: Optional[dict] = None,
    timeout: Optional[float] = None,
    async_mode: bool = False,
) -> dict:
    """
    底层函数: 通过中继链向 OpenClaw 发送任务

    Args:
        task_type:  任务类型 (ping / fetch_daily_matches / query_injury / ...)
        params:     任务参数
        timeout:    超时秒数
        async_mode: 是否异步模式（OpenClaw 立即确认，结果通过正向通道回传）
    Returns:
        dict: OpenClaw 返回的响应
    """
    if timeout is None:
        timeout = TASK_TIMEOUT

    task_id = str(uuid.uuid4())
    payload = {
        "task_id": task_id,
        "task_type": task_type,
        "params": params or {},
        "async_mode": async_mode,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{RELAY_URL}/relay_to_openclaw", json=payload)
            resp.raise_for_status()
            return {
                "status": "success",
                "task_id": task_id,
                "openclaw_response": resp.json(),
            }
    except httpx.TimeoutException:
        return {"status": "timeout", "task_id": task_id, "message": "请求超时"}
    except httpx.ConnectError:
        return {
            "status": "error",
            "task_id": task_id,
            "message": f"无法连接中继站 ({RELAY_URL})，请检查 SSH 端口转发是否开启",
        }
    except Exception as e:
        return {"status": "error", "task_id": task_id, "message": str(e)}


@app.post("/request_openclaw")
async def request_openclaw(task: dict):
    """
    同步模式: 发送任务到 OpenClaw 并直接等待 HTTP 响应

    请求体:
    {
        "task_type": "ping | fetch_daily_matches | query_injury
                      | query_recent_matches | start_realtime_odds
                      | stop_realtime_odds",
        "params": { ... }
    }
    """
    task_type = task.get("task_type")
    params = task.get("params", {})

    if not task_type:
        raise HTTPException(status_code=400, detail="缺少 task_type 参数")

    return await _send_to_openclaw(task_type, params)


@app.post("/request_openclaw_async")
async def request_openclaw_async(task: dict):
    """
    异步模式: 发送任务后阻塞等待 OpenClaw 通过正向通道回传结果
    适用于耗时较长的任务（爬取大量数据等）

    请求体:
    {
        "task_type": "fetch_daily_matches | ...",
        "params": { ... },
        "timeout": 60          (可选, 等待超时秒数)
    }
    """
    task_type = task.get("task_type")
    params = task.get("params", {})
    wait_timeout = task.get("timeout", TASK_TIMEOUT)

    if not task_type:
        raise HTTPException(status_code=400, detail="缺少 task_type 参数")

    task_id = str(uuid.uuid4())

    # 注册等待信号
    event = asyncio.Event()
    _task_events[task_id] = event

    # 向 OpenClaw 发送异步任务
    payload = {
        "task_id": task_id,
        "task_type": task_type,
        "params": params,
        "async_mode": True,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{RELAY_URL}/relay_to_openclaw", json=payload)
            resp.raise_for_status()
    except Exception as e:
        _task_events.pop(task_id, None)
        return {"status": "error", "task_id": task_id, "message": f"发送任务失败: {e}"}

    # 阻塞等待正向通道回传结果
    try:
        await asyncio.wait_for(event.wait(), timeout=wait_timeout)
        result = _task_results.pop(task_id, None)
        return {"status": "success", "task_id": task_id, "result": result}
    except asyncio.TimeoutError:
        return {
            "status": "timeout",
            "task_id": task_id,
            "message": f"等待结果超时（{wait_timeout}秒）",
        }
    finally:
        _task_events.pop(task_id, None)
        _task_results.pop(task_id, None)


# ==============================================================
#  测试 & 诊断接口
# ==============================================================

@app.get("/test/ping_openclaw")
async def test_ping_openclaw():
    """测试反向通道: 服务器 → 中继 → OpenClaw → 返回"""
    result = await _send_to_openclaw("ping", timeout=10.0)
    return {"test": "ping_openclaw", "result": result}


@app.get("/test/connection_status")
async def test_connection_status():
    """检查全链路连通状态"""
    status = {
        "server": "running",
        "relay_url": RELAY_URL,
        "timestamp": datetime.now().isoformat(),
    }

    # 检查中继站
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{RELAY_URL}/")
        status["relay"] = "connected" if resp.status_code == 200 else f"http {resp.status_code}"
    except Exception as e:
        status["relay"] = f"disconnected ({type(e).__name__})"

    # 检查 OpenClaw（通过中继）
    try:
        result = await _send_to_openclaw("ping", timeout=8.0)
        if result.get("status") == "success":
            status["openclaw"] = "connected"
        else:
            status["openclaw"] = result.get("message", "unknown error")
    except Exception as e:
        status["openclaw"] = f"disconnected ({type(e).__name__})"

    return status


# ==============================================================
#  启动
# ==============================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
