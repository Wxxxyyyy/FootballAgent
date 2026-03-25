# ⚽ Football Agent — 基于 OpenClaw 的足球赛事智能 Agent

基于 **LangGraph** 构建的多 Agent 足球赛事分析系统，覆盖赛前预测、数据查询、智能闲聊三大场景。

## 系统架构

```
用户输入 → Intent Node (BERT) → 条件路由
                                    │
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
          预测 Agent          信息查询 Agent         闲聊 Agent
          │                   │                      │
          ├─ OpenClaw 赔率    ├─ Text2SQL (MySQL)    └─ Ollama 本地
          ├─ ML 赔率模型      ├─ Text2Cypher (Neo4j)
          ├─ Neo4j 交锋       └─ 向量检索 (ChromaDB)
          ├─ 爆冷信号检测
          └─ Kimi 2.5 分析
                 │                  │                  │
                 └──────────────────┼──────────────────┘
                                    ▼
                            Summary Agent
                         LLM 润色 + 安全检查
                                    ▼
                              最终回复
```

## 核心特性

### 🧠 意图识别
- 微调 **bert-base-chinese** 做三分类，CPU 推理 12ms，F1 = 0.96
- 置信度阈值 0.7 + 兜底机制，低于阈值自动转闲聊引导

### 🔮 赛前预测
- **OpenClaw 三级网络链路**：服务器 ↔ SSH 隧道 ↔ 本地中继 ↔ WiFi ↔ 爬虫端
- **ML 赔率基座模型**：RandomForest 12 维特征（Bet365 赔率 + 隐含概率），WDL 准确率 52.1%
- **Neo4j 历史交锋**：图谱查询两队近 5 次交锋
- **5 维爆冷信号检测**：近况反差 / 状态断崖 / 交锋克制 / 赛程疲劳 / 火力冲击
- **Kimi 2.5 综合分析**：输出结构化 JSON（胜平负 + 大小球 + 比分 + 爆冷预警）

### 📊 数据查询
- **Text2SQL** + **Text2Cypher** + **向量检索** 三通道
- 每条通道四道安全防线（读写隔离 → Schema 校验 → EXPLAIN 语法 → 强制 LIMIT）
- 失败自动重试纠错，Text2SQL 命中率 78% → 92%

### 🗄️ 三层数据存储
- **MySQL**：结构化比赛数据（比分、赔率、技术统计）
- **Neo4j**：球队关系图谱（96 节点 + 9500 PLAYED_AGAINST 关系）
- **ChromaDB**：球队底蕴向量知识库（bge-m3 Embedding）

### 🛡️ 工程化实践
- **Redis**：预测缓存（命中率 34%）、会话缓存、滑动窗口限流、入库消息队列
- **Docker Compose**：MySQL / Neo4j / Redis 一键编排
- **LLM 多模型调度 + 熔断降级**：5 个远程模型可切换，失败自动降级本地 Ollama
- **LangFuse**：全链路追踪（Agent 调用链 / Token 用量 / 延迟分布）
- **安全合规**：敏感词拦截 + 赌博风险检测 + 预测免责声明自动注入

## 项目结构

```
footballAgent/
├── agents/                     # Agent 核心逻辑
│   ├── graph_builder.py        #   LangGraph 主图（意图路由 → 子 Agent → 总结）
│   ├── states.py               #   全局状态定义 (AgentState)
│   ├── intent_agent/           #   意图识别（BERT 微调）
│   ├── predicted_agent/        #   赛前预测（OpenClaw + ML + Neo4j + LLM）
│   ├── information_agent/      #   数据查询（Planner → 三通道工具调度）
│   ├── otherchat_agent/        #   智能闲聊
│   ├── summary_agent/          #   总结输出 + 安全检查
│   └── tools/                  #   工具集
│       ├── mysql_tools/        #     Text2SQL + 四道安全防线
│       ├── neo4j_tools/        #     Text2Cypher + 四道安全防线
│       └── vector_tools/       #     ChromaDB 向量检索
├── api/                        # 三级通信
│   ├── server_api.py           #   服务器端（数据接收 + 任务下发）
│   ├── relay_api.py            #   本地中继站
│   ├── openclaw_client.py      #   OpenClaw 爬虫端
│   └── pre_match_state.py      #   跨线程同步（threading.Event）
├── backend/                    # 后端服务
│   ├── core/                   #   基础设施（config / DB / Redis / 日志 / 安全）
│   ├── services/               #   业务服务（缓存 / LLM / 用户 / 会话）
│   └── api/                    #   API 路由（聊天 / 预测 / 比赛 / 认证）
├── pipeline/                   # 数据管道
│   ├── data_preprocess.py      #   CSV 预处理
│   ├── mysql_loader.py         #   MySQL 批量导入
│   ├── neo4j_loader.py         #   Neo4j 批量导入
│   ├── vector_loader.py        #   向量库导入
│   ├── openclaw_ingestion.py   #   OpenClaw 增量入库
│   └── scheduler.py            #   APScheduler 定时任务
├── intent/                     # 意图识别模型
│   ├── train.py                #   BERT 微调训练
│   ├── predict.py              #   推理接口
│   └── data/                   #   训练 / 验证 / 测试数据
├── common/                     # 公共模块
│   ├── llm_select.py           #   LLM 统一调度 + 降级
│   ├── team_mapping.py         #   球队名中英文映射
│   ├── constants.py            #   全局常量
│   ├── exceptions.py           #   自定义异常
│   └── utils.py                #   通用工具函数
├── evaluation/                 # 评测框架
│   ├── metrics.py              #   评估指标（Brier / ROI / 命中率）
│   ├── accuracy_evaluator.py   #   准确率评估器
│   ├── backtest.py             #   历史回测引擎
│   └── report_generator.py     #   评估报告生成
├── observability/              # 可观测性
│   ├── langfuse_tracer.py      #   LangFuse 链路追踪
│   ├── llm_usage_tracker.py    #   LLM 用量统计
│   └── alert_rules.py          #   告警规则引擎
├── configs/                    # 配置文件
│   ├── mysql_schema.sql        #   MySQL 建表
│   ├── neo4j_schema.cypher     #   Neo4j 约束与索引
│   └── redis.conf              #   Redis 配置
├── data/                       # 数据目录
│   ├── ori_data/               #   原始 CSV（football-data.co.uk）
│   ├── processed/              #   清洗后 CSV（五大联赛 5 赛季）
│   ├── team_profiles/          #   球队简介 JSON
│   └── English2Chinese/        #   中英文对照表
├── docs/                       # 项目文档
├── tests/                      # 测试
├── docker-compose.yml          # 容器编排
├── requirements.txt            # Python 依赖
├── Makefile                    # 常用命令
└── .env.example                # 环境变量模板
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/yourname/footballAgent.git
cd footballAgent

# 创建 conda 环境
conda create -n football python=3.11
conda activate football
pip install -r requirements.txt

# 复制环境变量模板
cp .env.example .env
# 编辑 .env 填入 API Key、数据库密码等
```

### 2. 启动中间件

```bash
docker-compose up -d   # 拉起 MySQL / Neo4j / Redis
```

### 3. 初始化数据

```bash
make load-data         # 导入 CSV 到 MySQL + Neo4j
make init-vector       # 导入球队简介到 ChromaDB
```

### 4. 运行

```bash
# 终端交互模式
python agents/graph_builder.py

# 启动 API 服务
python api/server_api.py
```

## 技术栈

| 类别 | 技术 |
|------|------|
| Agent 框架 | LangGraph |
| 意图识别 | bert-base-chinese (HuggingFace) |
| ML 模型 | scikit-learn RandomForest |
| LLM | Kimi 2.5 / Qwen / GLM (DashScope) + Ollama |
| 数据库 | MySQL 8.0 + Neo4j 5.x + ChromaDB |
| 缓存 | Redis 7 |
| Web 框架 | FastAPI |
| Embedding | bge-m3 (BAAI) |
| 容器 | Docker Compose |
| 可观测性 | LangFuse |

## 数据覆盖

- **五大联赛**：英超 / 德甲 / 意甲 / 西甲 / 法甲
- **时间跨度**：2021-2026 赛季（5 个完整赛季）
- **数据规模**：~9500 场比赛，96 支球队
- **数据来源**：football-data.co.uk + OpenClaw 实时采集

## License

MIT
