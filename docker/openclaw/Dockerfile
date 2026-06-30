# OpenClaw 爬虫服务 Dockerfile
# 轻量级：使用 httpx 直接爬 titan007 JS数据文件，无需 Playwright 浏览器

FROM python:3.11-slim

# 系统依赖（含 tzdata，让容器系统时间跟随 Asia/Shanghai）
RUN apt-get update && apt-get install -y --no-install-recommends curl tzdata \
    && ln -snf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 数据目录（赔率快照）
RUN mkdir -p /app/data/odds_snapshots

EXPOSE 9000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

# 启动服务
CMD ["python", "app.py"]
