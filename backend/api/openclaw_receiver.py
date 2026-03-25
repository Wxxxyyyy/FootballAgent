# -*- coding: utf-8 -*-
"""
OpenClaw 数据接收接口
- 接收实时赔率数据（JSON 字典格式）
- 接收 CSV 批量比赛数据（JSON 列表格式）
- 后续对接 Agent 推理 和 数据库写入
"""
from fastapi import APIRouter
from datetime import datetime

router = APIRouter(prefix="/openclaw", tags=["OpenClaw数据接收"])


@router.get("/")
async def root():
    """健康检查接口"""
    return {"message": "⚽ OpenClaw 数据接收服务已就绪，正在等待数据..."}


@router.post("/receive_odds")
async def receive_odds(data: dict):
    """
    接收 OpenClaw 推送的赔率/比赛数据

    数据格式:
    - data_type: "realtime" (实时赔率) 或 "csv_bulk" (CSV批量数据)
    - content: 对应的数据内容
    """
    # 获取数据类型和内容
    data_type = data.get("data_type", "unknown")
    content = data.get("content")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. 处理实时赔率数据 (JSON 字典格式)
    if data_type == "realtime":
        match_info = content.get("match", "未知比赛")
        odds = content.get("odds", "N/A")
        print(f"⚡ [{timestamp}] 收到实时数据 -> 比赛: {match_info}, 赔率: {odds}")
        # TODO: 对接 Redis 缓存 + prediction_agent 实时预测
        return {"status": "success", "message": "实时赔率已接收"}

    # 2. 处理 CSV 批量数据 (JSON 列表格式)
    elif data_type == "csv_bulk":
        row_count = len(content) if isinstance(content, list) else 0
        print(f"📁 [{timestamp}] 收到批量 CSV 数据 -> 共 {row_count} 条记录")
        # TODO: 对接 MySQL/Neo4j 数据库写入逻辑
        return {"status": "success", "message": f"成功接收 {row_count} 条批量数据"}

    # 3. 处理未知格式
    else:
        print(f"❓ [{timestamp}] 收到未知格式数据: {data}")
        return {"status": "error", "message": "未知数据类型"}
