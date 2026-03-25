# -*- coding: utf-8 -*-
"""
Neo4j 异步驱动管理（AsyncGraphDatabase）
用于赛事关系图谱、球员/俱乐部关联查询等。
"""
from typing import Any, Optional

from neo4j import AsyncGraphDatabase

from .config import get_settings

# neo4j 驱动具体类型随版本略有差异，此处用 Any 避免类型存根不一致
_driver: Optional[Any] = None


def get_driver() -> Any:
    """
    懒加载全局异步驱动（单例）。

    - ``max_connection_pool_size=10``：控制与集群的并发会话上限。
    - 认证使用 ``(username, password)`` 元组，与 Neo4j 4.x/5.x 默认配置一致。
    - 长驻进程内勿重复创建 driver，避免池耗尽。
    """
    global _driver
    if _driver is None:
        settings = get_settings()
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_url,
            auth=(settings.neo4j_username, settings.neo4j_password),
            max_connection_pool_size=10,
        )
    return _driver


async def close_driver() -> None:
    """
    异步关闭驱动并清空池。

    与单次查询会话不同，driver 应在应用生命周期结束时显式 close，
    否则可能拖慢进程退出。
    """
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


async def health_check() -> bool:
    """
    执行 ``RETURN 1 AS ok`` 探测 Bolt 与认证。

    若库未就绪或凭据错误，捕获异常并返回 False，便于编排层剔除实例。
    """
    drv = get_driver()
    try:
        async with drv.session() as session:
            result = await session.run("RETURN 1 AS ok")
            record = await result.single()
            if record is None:
                return False
            val = record["ok"]
            return val == 1
    except Exception:
        return False
