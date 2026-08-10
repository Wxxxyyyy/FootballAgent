# -*- coding: utf-8 -*-
"""
单测公共配置: 在任何被测模块导入前固定环境。

关键点:
  - REDIS_PORT 指向不存在的端口 → 强制 redis_cache 走降级路径，
    同时让去重测试真实覆盖"Redis 挂了退回文件"的兜底行为。
  - PUSH/TRIGGER 状态目录指向临时目录，测试互不污染、不碰生产数据。
  - docker/openclaw 加入 sys.path（容器代码在宿主机直接可测）。
"""

import os
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 必须在 import 被测模块之前设置（它们在模块级读取这些环境变量）
os.environ["REDIS_HOST"] = "127.0.0.1"
os.environ["REDIS_PORT"] = "6390"  # 不存在的端口 → 降级
os.environ["REDIS_TIMEOUT"] = "0.3"
os.environ["BARK_ENABLED"] = "true"
os.environ["BARK_KEY"] = "unittest-key"
os.environ["BARK_SERVER"] = ""
os.environ["PUSH_STATE_DIR"] = tempfile.mkdtemp(prefix="push_state_")
os.environ["TRIGGERED_STATE_DIR"] = tempfile.mkdtemp(prefix="trig_state_")
# MQ 指向不存在的端口 → 强制 publish 走降级路径（返回 False 不抛异常）
os.environ["MQ_HOST"] = "127.0.0.1"
os.environ["MQ_PORT"] = "5699"  # 不存在的端口

for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "docker", "openclaw"),
          os.path.join(PROJECT_ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
