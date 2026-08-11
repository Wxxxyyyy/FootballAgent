# ⚽ Football Agent — 基于 OpenClaw 的足球赛事智能 Agent

基于 **LangGraph** 构建的多 Agent 足球赛事分析系统，覆盖赛前预测、数据查询、智能闲聊三大场景。2026 世界杯期间 7×24 实战运行，全自动完成 167 场比赛的预测与推送。

## 系统架构

```
用户输入 → Intent Node (BERT) → 条件路由
                                    │
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
          预测 Agent          信息查询 Agent         闲聊 Agent
          (6 步流水线)         (ReAct + 9 工具)        │
          │                   │                      │
          ├─ OpenClaw 赔率    ├─ Text2SQL (MySQL)    └─ qwen3-coder-next
          ├─ LightGBM 基座    ├─ Text2Cypher (Neo4j)
          ├─ 蒙特卡洛模拟      ├─ 向量检索 (Milvus)
          ├─ Neo4j 交锋       ├─ 历史预测召回
          ├─ 赛前情报采集      └─ 社区工具 (联网/百科/天气/翻译)
          ├─ 6 维爆冷检测
          └─ Kimi 2.5 融合
                 │                  │                  │
                 └──────────────────┼──────────────────┘
                                    ▼
                            Summary Agent
                          (kimi-k2.5 润色 + 安全检查)
                                    ▼
                          Memory Manager
                (L1/L3/L5 上下文压缩 + 动态记忆提取)
                                    ▼
                              最终回复
```

## 核心特性

### 🧠 意图识别
- 微调 **bert-base-chinese** 做三分类，CPU 推理 12ms，MacroF1 = 0.96
- 置信度阈值 0.7 + 兜底机制，低于阈值自动转闲聊引导
- **对话锁（DialogLock）**：多轮槽位填充期间锁定预测节点，连续 3 轮未填满自动熔断释放，防对话死锁

### 🔮 赛前预测（6 步流水线）
- **OpenClaw Docker 容器**：轻量爬虫常驻云端，APScheduler 执行"每日完赛同步 / 每小时赔率快照 / 每 30 分钟赛前扫描"三类定时任务，赛前 3.5h/2.5h/1.5h/0.5h 四档自动触发预测；数据经 **RabbitMQ** 投递（HTTP 直连兜底），消费者按（日期, 主队, 客队）三元组幂等去重入库
- **LightGBM 赔率基座**：19 维赔率特征（初盘/终盘隐含概率、overround、赔率位移、离散度、交互项），验证集早停
- **蒙特卡洛模拟**：Nelder-Mead 从赔率反推泊松 λ，万次模拟输出比分分布
- **Neo4j 历史交锋**：图谱查询两队交锋往绩（Text2Cypher 四道防线：读写隔离 → 关系类型+节点标签校验 → EXPLAIN 语法验证 → Probe Query 实体存在性探测）
- **赛前情报采集器**：伤停 / 首发双档 / 新闻软信号 / 教练风格，结构化解析
- **6 维爆冷信号规则引擎**：近况反差 / 状态断崖 / 交锋克制 / 赛程疲劳 / 火力冲击 / 伤员预警，规则输出结构化信号供 LLM 融合
- **Kimi 2.5 综合分析**：输出结构化 JSON（胜平负 + 大小球 + 比分 + 爆冷预警）；预测结论精简落库 `prediction_records`，支持历史召回

### 📊 数据查询（ReAct + Function Calling）
- 信息查询节点挂载 **9 个工具**，LLM 自主调度：3 个自定义（Text2SQL / Text2Cypher / 向量检索）+ 历史预测召回 + 5 个社区工具（联网搜索 / 维基百科 / 天气 / 日期时间 / 翻译）
- **Text2SQL 四道防线**：读写隔离 → Schema 幻觉校验（表名+列名白名单）→ EXPLAIN 语法验证 → 强制 LIMIT 30；失败自动重试纠错（最多 3 次），准确率 78% → 92%
- **向量检索**：bge-m3 Embedding + **Milvus**（COSINE + HNSW），相似度阈值过滤 + Top-K=3，Top-3 召回率 91%

### 🗄️ 数据存储
- **MySQL**：结构化比赛数据（比分、赔率、技术统计、预测结论、LLM 调用台账）
- **Neo4j**：球队关系图谱（五大联赛 + 国家队，交锋关系）
- **Milvus**：球队底蕴向量知识库（bge-m3 Embedding，COSINE + HNSW）
- **Redis**：会话持久化 + 预测缓存 + 分布式互斥锁
- **RabbitMQ**：采集事件消息队列，采集与业务解耦、削峰填谷

### 🧩 上下文管理（token 预算制，三层压缩）
- **L1 大结果落盘**：单条工具结果超 50KB 自动写磁盘，上下文只留预览 + 路径
- **L3 Micro-Compact**：上下文达 500K token 或距上次压缩超 6h 触发，清掉可重新获取的结果（数据库查询 / 可重跑的预测），保留最近 5 条不可重现结果
- **L5 Auto-Compact**：达 967K token（1M 窗口 − 33K buffer）触发，全量消息交 LLM 重写为结构化摘要，旧消息丢弃

### 🧠 记忆机制（静态规则 + 动态记忆）
- **静态层 `football.md`**：角色边界 / 实体归一化指引 / 数据源分工 / 合规底线 + 动态记忆索引，常驻 system prompt
- **动态层 `data/memory/*.md`**：每条记忆一个文件，只分 4 类（user 用户画像 / feedback 行为偏好 / project 项目动态 / reference 外部指针）
- **写入**：每轮对话后后台线程异步提取，LLM 判断"值得记"才落盘并更新索引
- **召回**：代词/回指条件触发 → 窗口内消解 → 小模型按索引做选择题挑 Top-K 注入，90%+ 请求零检索开销

### 🛡️ 工程化实践
- **全容器化**：Docker Compose 编排全部服务（MySQL / Neo4j / Redis / RabbitMQ / Milvus(etcd+minio) / OpenClaw / 预测 API / MQ 消费者 / 统一调度器），端口全绑 127.0.0.1 零公网暴露
- **高并发治理**：Redis 预测缓存（30min TTL）+ SETNX 分布式互斥锁折叠并发请求，防缓存击穿
- **LLM 多模型调度 + 熔断降级**：5 个远程模型可切换，失败自动降级本地 Ollama
- **全链路可观测**：**Langfuse 自托管**（LLM trace 树 / Token 成本 / 延迟分布，compose `langfuse` profile 一键拉起）+ LLM 调用台账旁路落库 + Prometheus 指标 + 5 条阈值告警规则（成功率/延迟/Token/错误数/零调用假死）+ Bark 推送（指数退避重试 + 双写去重）
- **安全合规**：敏感词拦截 + 赌博风险检测 + 预测免责声明自动注入

### 🏆 世界杯实战（赛事可插拔验证）
- 2026 世界杯期间 7×24 全自动运行，完成 **167 场比赛**预测与 Bark 推送
- 接入新赛事仅需扩充球队映射表 + 教练风格库 + 补采国家队历史交锋，核心 Agent 管线零改动

## 项目结构

```
footballAgent/
├── agents/                     # Agent 核心逻辑
│   ├── graph_builder.py        #   LangGraph 主图（意图路由 → 子 Agent → 总结 → 记忆）
│   ├── states.py               #   全局状态定义 (AgentState)
│   ├── intent_agent/           #   意图识别（BERT 微调）
│   ├── predicted_agent/        #   赛前预测（6 步流水线）
│   │   ├── advance_predictor.py#     主流水线编排 + 6 维爆冷规则引擎
│   │   ├── feature_engineering.py #  19 维赔率特征工程
│   │   ├── prediction_store.py #     预测结论精简落库 + 召回工具
│   │   ├── models/             #     LightGBM 基座 / 蒙特卡洛 / LLM 融合
│   │   └── scouters/           #     赛前情报采集器（伤停/首发/新闻/教练风格）
│   ├── information_agent/      #   数据查询（ReAct + 9 工具自主调度）
│   ├── otherchat_agent/        #   智能闲聊
│   ├── summary_agent/          #   总结输出 + 安全检查
│   ├── memory_manager/         #   上下文压缩 + 记忆机制
│   │   ├── context_manager.py  #     L1/L3/L5 三层上下文压缩引擎
│   │   ├── memory_store.py     #     football.md 静态规则 + 动态记忆引擎
│   │   └── retriever.py        #     条件触发 + 索引召回
│   └── tools/                  #   工具集
│       ├── mysql_tools/        #     Text2SQL + 四道安全防线
│       ├── neo4j_tools/        #     Text2Cypher + 四道安全防线
│       ├── vector_tools/       #     Milvus 向量检索
│       └── community_tools.py  #     社区工具（联网/百科/天气/时间/翻译）
├── api/                        # 服务接口
│   ├── prediction_api.py       #   预测 API（Flask，缓存 + SETNX 锁）
│   └── server_api.py           #   对外服务入口
├── backend/                    # 后端服务
│   ├── core/                   #   基础设施（config / DB / Redis / Milvus / 安全）
│   ├── services/               #   业务服务
│   └── api/                    #   API 路由
├── pipeline/                   # 数据管道
│   ├── mq_consumer.py          #   RabbitMQ 消费者（幂等去重入库）
│   ├── mysql_loader.py         #   MySQL 批量导入
│   ├── neo4j_loader.py         #   Neo4j 批量导入
│   ├── national_team_neo4j_loader.py # 国家队交锋入库（世界杯专题）
│   ├── vector_loader.py        #   Milvus 向量库导入
│   └── openclaw_ingestion.py   #   OpenClaw 增量入库
├── docker/                     # Docker 构建
│   └── openclaw/               #   OpenClaw 爬虫容器（Dockerfile + 定时任务）
├── scripts/                    # 运维脚本
│   ├── scheduler_runner.py     #   容器内统一调度器（准确率追踪 + 告警巡检）
│   ├── accuracy_tracker.py     #   预测准确率追踪
│   ├── alert_rules.py          #   LLM 台账 5 条阈值告警
│   └── health_watchdog.py      #   健康巡检
├── common/                     # 公共模块
│   ├── llm_select.py           #   LLM 统一调度 + 降级
│   ├── redis_cache.py          #   缓存 + 分布式互斥锁
│   ├── tracer.py               #   轻量链路追踪（contextvars + MySQL 落库）
│   ├── llm_call_log.py         #   LLM 调用台账
│   └── team_mapping.py         #   球队名中英文映射
├── evaluation/                 # 评测框架（意图/工具选择/记忆召回三维回归）
├── data/                       # 数据目录
│   ├── memory/                 #   记忆文件（football.md + 动态记忆）
│   ├── predictions/            #   预测结果存档
│   └── processed/              #   清洗后 CSV（五大联赛 5 赛季）
├── docker-compose.yml          # 全栈容器编排
├── Dockerfile                  # 业务服务镜像
├── requirements.txt            # Python 依赖
└── .env.example                # 环境变量模板
```

## 快速开始

### 1. 环境准备

```bash
git clone https://github.com/Wxxxyyyy/FootballAgent.git
cd FootballAgent
cp .env.example .env
# 编辑 .env 填入 API Key、数据库密码等
```

### 2. 一键启动（全容器化）

```bash
docker compose up -d --build
# 启动全部服务：MySQL / Neo4j / Redis / RabbitMQ / Milvus(etcd+minio)
#              / OpenClaw 爬虫 / 预测 API / MQ 消费者 / 统一调度器
```

### 3. 初始化数据

```bash
python pipeline/mysql_loader.py        # 导入 CSV 到 MySQL
python pipeline/neo4j_loader.py        # 导入交锋图谱到 Neo4j
python pipeline/vector_loader.py       # 导入球队简介到 Milvus
```

### 4. 本地开发（可选）

```bash
conda create -n football python=3.12
conda activate football
pip install -r requirements.txt

python agents/graph_builder.py         # 终端交互模式
python -m api.server_api               # 启动 API 服务
```

## 技术栈

| 类别 | 技术 |
|------|------|
| Agent 框架 | LangGraph + LangChain |
| 意图识别 | bert-base-chinese (HuggingFace) |
| ML 模型 | LightGBM（scikit-learn RF 兜底）+ 蒙特卡洛泊松模拟 |
| LLM | Kimi 2.5 / Qwen / GLM (DashScope) + Ollama 本地降级 |
| 数据库 | MySQL 8.0 + Neo4j 5.x + Milvus 2.4 |
| 缓存 / 消息 | Redis 7 + RabbitMQ |
| Embedding | bge-m3 (BAAI) |
| 容器 | Docker Compose 全栈编排 |
| 可观测性 | Langfuse（自托管）+ LLM 调用台账 + Prometheus + Bark 告警 |

## 数据覆盖

- **五大联赛**：英超 / 德甲 / 意甲 / 西甲 / 法甲（2021-2026，~9500 场，96 支球队）
- **国家队**：2026 世界杯 48 支国家队（历史交锋 + 教练风格库）
- **数据来源**：football-data.co.uk + OpenClaw 实时采集 + 11v11 历史补采

## License

MIT
