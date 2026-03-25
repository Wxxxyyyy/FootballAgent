# ⚽ Football Agent - 常用命令

# ---------- 服务启动 ----------
.PHONY: run
run:  ## 启动后端服务
	cd backend && python run.py

.PHONY: dev
dev:  ## 开发模式启动（热重载）
	cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000

.PHONY: frontend
frontend:  ## 启动前端
	cd frontend && npm run dev

# ---------- Docker ----------
.PHONY: docker-up
docker-up:  ## 启动所有依赖服务（MySQL/Neo4j/Redis）
	docker-compose up -d

.PHONY: docker-down
docker-down:  ## 停止所有依赖服务
	docker-compose down

# ---------- 数据管道 ----------
.PHONY: init-db
init-db:  ## 初始化所有数据库
	python -m backend.scripts.init_mysql
	python -m backend.scripts.init_neo4j
	python -m backend.scripts.init_vector_db

.PHONY: load-data
load-data:  ## 导入CSV数据到数据库
	python -m pipeline.data_preprocess
	python -m pipeline.mysql_loader
	python -m pipeline.neo4j_loader
	python -m pipeline.vector_loader

# ---------- 数据库迁移 ----------
.PHONY: migrate
migrate:  ## 生成数据库迁移
	alembic revision --autogenerate -m "$(msg)"

.PHONY: upgrade
upgrade:  ## 执行数据库迁移
	alembic upgrade head

# ---------- 测试 ----------
.PHONY: test
test:  ## 运行所有测试
	pytest tests/ -v

# ---------- 帮助 ----------
.PHONY: help
help:  ## 显示帮助
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

