# =============================================================
# Football Agent 业务层镜像（预测服务 / MQ 消费者 / 运维脚本共用）
# 构建: docker compose build
# =============================================================
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# 系统依赖：git（GitPython 模型版本管理）、curl（健康检查）、
# build-essential + libta-lib-dev（TA-Lib 无预编译 wheel 时从源码构建）
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl build-essential libta-lib-dev \
    && rm -rf /var/lib/apt/lists/*

# 先单独安装 CPU 版 PyTorch（避免默认源拉取数 GB 的 CUDA 依赖），再装全量依赖
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# 拷贝业务代码（内含 ML 模型权重 agents/predicted_agent/models/saved/ 与球队映射表）
COPY . .

# 默认入口：预测服务（与 systemd 时代的 ExecStart 一致；其他服务在 compose 里覆盖 command）
CMD ["python", "-m", "api.prediction_api"]
