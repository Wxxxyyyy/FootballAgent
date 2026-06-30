# 🏗️ 长期计划：FootballAgent 工程完善路线图

> **目标**: 将 FootballAgent 打造成面试展示级的完整工程 Agent，覆盖数据管道、模型管理、Agent 能力、可观测性、高可用、工程化全链路。
>
> **定位**: 这不是"能用"的原型，而是"能讲清楚架构决策、能展示工程深度"的面试项目。

---

## 一、现状盘点

### 已有资产

| 模块 | 现状 | 面试可讲度 |
|------|------|-----------|
| LangGraph 多 Agent | 意图路由→预测/查询/闲聊→总结→记忆 | ⭐⭐⭐⭐ |
| BERT 意图识别 | 微调 bert-base-chinese，F1=0.96，12ms | ⭐⭐⭐⭐ |
| RF 赔率模型 | 12维特征，WDL 52.1% | ⭐⭐⭐ |
| OpenClaw 数据采集 | 三级通信链路（容器占位待补） | ⭐⭐⭐⭐ |
| 三层存储 | MySQL + Neo4j + ChromaDB | ⭐⭐⭐⭐ |
| Text2SQL/Cypher | 四道安全防线，92%准确率 | ⭐⭐⭐⭐⭐ |
| LLM 调度 | 5模型+Ollama降级 | ⭐⭐⭐⭐ |
| Redis | 会话持久化+限流+缓存 | ⭐⭐⭐ |
| LangFuse | 全链路追踪 | ⭐⭐⭐ |
| 评测框架 | accuracy/backtest/metrics/profit | ⭐⭐⭐ |
| Docker | MySQL/Neo4j/Redis/OpenClaw 容器化 | ⭐⭐⭐ |

### 缺失/短板（面试会被追问的）

| 短板 | 影响 | 优先级 |
|------|------|--------|
| 测试覆盖率低（tests/ 仅10文件） | 面试必问工程质量 | 🔴 |
| 无 CI/CD | 面试必问工程实践 | 🔴 |
| MQ 未接入（RabbitMQ 注释状态） | 架构完整性 | 🟡 |
| OpenClaw 容器是空壳 | 数据管道不完整 | 🔴 |
| 无模型版本管理/A/B测试 | ML 工程化深度 | 🟡 |
| 实时数据流缺失 | 仅每日批处理 | 🟡 |
| 世界杯/多赛事未扩展 | 业务覆盖面 | 🟡 |
| 文档不够面试展示 | 讲不清楚架构 | 🟡 |
| 无性能压测 | 高可用存疑 | 🟡 |

---

## 二、完善路线图（分5个阶段）

### 📌 阶段 1：数据管道完整化（1-2周）

> 面试核心讲点：实时数据采集 → 消息队列解耦 → 多存储入库

#### 1.1 OpenClaw 容器补全

**现状**：`docker/openclaw/Dockerfile` 是占位文件，无实际爬虫代码。

**目标**：OpenClaw 容器内实现完整的赔率采集服务。

| 任务 | 说明 |
|------|------|
| 编写 OpenClaw `app.py` | FastAPI/Flask 服务，暴露 `/task`、`/health` 接口 |
| 多数据源适配器 | the-odds-api（API）+ OddsPortal（爬虫）+ 澳客（备用），策略模式可切换 |
| 赔率抓取 Loop | 定时抓取，支持动态频率（赛前加密） |
| 赔率变动检测 | 与上次快照对比，大幅变动发事件 |
| 赛后数据抓取 | 比赛结束后自动抓取实际比分 |

#### 1.2 RabbitMQ 接入（事件驱动）

**现状**：`docker-compose.yml` 里 RabbitMQ 注释状态。

**目标**：OpenClaw 采集到数据后发 MQ 消息，消费端异步处理。

```
OpenClaw 采集 → 发 MQ 消息 → 消费者1: 入库 MySQL/Neo4j
                           → 消费者2: 触发预测
                           → 消费者3: 更新 Redis 缓存
```

| 任务 | 说明 |
|------|------|
| 启用 docker-compose RabbitMQ | 取消注释，配置用户密码 |
| 定义消息协议 | `data_type`: `daily_matches` / `odds_update` / `match_result` |
| OpenClaw 生产者 | 采集后发 MQ |
| 消费者服务 | `pipeline/mq_consumer.py`，按 `data_type` 路由处理 |
| 死信队列 | 处理失败的消息进 DLQ，避免丢失 |

**面试讲点**：为什么用 MQ 而非直接 HTTP？→ 解耦采集与处理、削峰填谷、可靠性投递。

#### 1.3 实时数据流

**现状**：`openclaw_sync.py` 每日定时批处理。

**目标**：赔率变动实时驱动预测更新。

| 任务 | 说明 |
|------|------|
| Redis Streams / MQ 事件 | 赔率变动实时推送 |
| WebSocket 推送 | 前端实时展示赔率走势 |
| 赔率时间序列存储 | MySQL 新建 `odds_snapshot` 表，存历史赔率变动 |

---

### 📌 阶段 2：模型工程化（1-2周）

> 面试核心讲点：模型版本管理、自动重训、A/B 测试、迁移学习

#### 2.1 模型版本管理

**现状**：`statistical_model.py` 只有一份 pkl，无版本概念。

| 任务 | 说明 |
|------|------|
| 模型注册表 | `models/registry.py`，记录版本、训练数据、指标、特征列 |
| 模型存储结构 | `models/saved/{version}/` + `model_meta.json` |
| 模型加载策略 | 支持指定版本加载，默认最新 |
| 模型对比 | 不同版本指标对比报告 |

#### 2.2 自动重训管道

| 任务 | 说明 |
|------|------|
| 数据漂移检测 | 监测赔率分布变化，触发重训 |
| 定时重训 | 每周用最新数据重训，发 MQ 通知 |
| 重训验证 | 新模型必须通过回测才上线（对比旧模型 Brier/LogLoss） |
| 灰度发布 | 新模型先跑 10% 流量，指标不退化才全量 |

#### 2.3 A/B 测试框架

| 任务 | 说明 |
|------|------|
| 流量分桶 | 按 `user_id` hash 分桶，支持多组实验 |
| 实验配置 | `configs/experiments.yaml`，定义对照/实验组 |
| 指标收集 | 每组预测准确率、Brier、延迟实时统计 |
| 显著性检验 | 累积足够样本后自动判定实验结论 |

#### 2.4 世界杯迁移学习（承接短期 plan）

| 任务 | 说明 |
|------|------|
| 历史数据入库 | 世界杯/欧洲杯/美洲杯入 `match_master`（`league` 区分） |
| 迁移学习 | 五大联赛预训练 → 世界杯微调（3种方案对比） |
| 隐含概率基线 | 庄家去利润概率作为模型必须超越的门槛 |
| 回测验证 | 用 `evaluation/Backtester` 回测 2022 世界杯 |

**面试讲点**：小样本场景如何做 ML → 迁移学习 + 基线对比 + 兜底策略。

---

### 📌 阶段 3：Agent 能力深化（1-2周）

> 面试核心讲点：Agent 架构设计、工具编排、记忆机制

#### 3.1 实时预测 Agent

**现状**：`realtime_predictor.py` 是空文件（TODO）。

| 任务 | 说明 |
|------|------|
| 实时赔率订阅 | 从 Redis Streams / MQ 接收赔率变动 |
| 动态预测更新 | 赔率变动触发重新预测，输出变化趋势 |
| 预测置信度追踪 | 同一比赛多次预测的置信度变化曲线 |
| 爆冷实时预警 | 赔率异动 + 爆冷信号 → 实时推送 |

#### 3.2 Agent 工具增强

| 任务 | 说明 |
|------|------|
| 赛程查询工具 | 查询某球队近期赛程 |
| 伤病信息工具 | 接入伤病数据源（转移市场/新闻） |
| 天气数据工具 | 比赛城市天气影响分析 |
| FIFA 排名工具 | 国家队实力对比（世界杯用） |

#### 3.3 记忆机制优化

**现状**：20轮触发 Compaction + 条件检索。

| 任务 | 说明 |
|------|------|
| 记忆分层 | 短期（对话窗口）+ 中期（会话摘要）+ 长期（向量库） |
| 实体记忆 | 记住用户偏好球队、常问赛事 |
| 记忆遗忘 | 过期记忆自动清理，避免向量库膨胀 |

---

### 📌 阶段 4：工程化与质量保障（1-2周）

> 面试核心讲点：测试、CI/CD、代码质量、文档

#### 4.1 测试体系

**现状**：`tests/` 仅10个文件，覆盖率不足。

| 任务 | 说明 |
|------|------|
| 单元测试 | 各模块核心函数测试，目标覆盖率 >70% |
| 集成测试 | Agent 端到端流程测试（意图→预测→输出） |
| Mock 测试 | LLM/OpenClaw/MySQL/Neo4j 全部 Mock |
| 回归测试 | 评测脚本作为回归测试，模型更新后自动跑 |
| 测试数据 | 构造固定测试数据集，不依赖线上数据 |

**重点测试模块**：
```
tests/
├── unit/
│   ├── test_feature_engineering.py    # 特征构建正确性
│   ├── test_statistical_model.py      # 模型预测正确性
│   ├── test_odds_to_probs.py          # 隐含概率转换
│   ├── test_text2sql_safety.py        # SQL安全防线
│   ├── test_text2cypher_safety.py     # Cypher安全防线
│   ├── test_upset_signals.py          # 爆冷信号检测
│   ├── test_intent_classifier.py      # 意图识别
│   └── test_memory_compaction.py      # 记忆压缩
├── integration/
│   ├── test_predict_pipeline.py       # 预测全流程
│   ├── test_info_query.py             # 查询全流程
│   └── test_openclaw_ingestion.py     # 数据入库
└── regression/
    └── test_model_accuracy.py         # 模型准确率回归
```

#### 4.2 CI/CD

**现状**：无 CI/CD 配置。

| 任务 | 说明 |
|------|------|
| GitHub Actions | push/PR 自动跑测试 + lint |
| Docker 镜像构建 | CI 自动构建并推送镜像 |
| 自动部署 | 测试通过后自动部署到服务器 |
| 质量门禁 | 测试覆盖率 <70% 阻止合并 |

```yaml
# .github/workflows/ci.yml 示例
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - name: Install
        run: pip install -r requirements.txt && pip install pytest pytest-cov ruff
      - name: Lint
        run: ruff check .
      - name: Test
        run: pytest --cov=. --cov-report=xml --cov-fail-under=70
```

#### 4.3 代码质量

| 任务 | 说明 |
|------|------|
| Ruff / Black | 统一代码风格，CI 强制 |
| 类型检查 | mypy 静态类型检查 |
| 依赖管理 | `requirements.txt` 锁版本，pip-compile |
| 技术债清理 | `llm_predictor.py` 走 `llm_select` 统一调度 |

#### 4.4 文档体系

| 任务 | 说明 |
|------|------|
| 架构设计文档 | 系统架构图 + 模块说明 + 技术选型理由 |
| API 文档 | FastAPI 自动生成 + 补充说明 |
| 部署文档 | Docker 部署 + 环境配置 + 常见问题 |
| 面试讲解文档 | 核心技术决策的"为什么"（A/B 选型、权衡） |

**面试必备文档**：
```
docs/
├── architecture.md          # 系统架构（含架构图）
├── data_pipeline.md         # 数据管道设计
├── model_engineering.md     # 模型工程化
├── agent_design.md          # Agent 架构设计
├── safety_design.md         # Text2SQL/Cypher 安全防线
├── deployment.md            # 部署运维
└── interview_notes.md       # 面试讲解要点（技术决策的 why）
```

---

### 📌 阶段 5：高可用与可观测性（1周）

> 面试核心讲点：熔断降级、监控告警、性能优化

#### 5.1 高可用

| 任务 | 说明 |
|------|------|
| LLM 熔断 | 连续失败 N 次自动降级 Ollama，恢复后自动切回 |
| 数据库连接池 | MySQL/Neo4j 连接池配置 + 超时重试 |
| 限流加固 | Redis 滑动窗口限流，按用户/IP/接口 |
| 优雅降级 | OpenClaw 不可用时用缓存赔率，LLM 不可用时纯 ML 输出 |
| 健康检查 | 所有服务的 healthcheck + 告警 |

#### 5.2 可观测性增强

**现状**：`observability/` 有 LangFuse + 用量统计 + 告警规则。

| 任务 | 说明 |
|------|------|
| 全链路追踪 | LangFuse 追踪 Agent 调用链 + 耗时分布 |
| 指标采集 | Prometheus 指标（QPS/延迟/错误率/缓存命中率） |
| 日志聚合 | 结构化日志 + ELK / Loki 聚合 |
| 告警规则 | 预测延迟 >30s / 模型准确率下降 / 服务宕机 → 告警 |
| Grafana 面板 | 可视化监控（可选） |

#### 5.3 性能优化

| 任务 | 说明 |
|------|------|
| 预测缓存 | 相同赔率24h内复用预测结果（Redis） |
| 模型预加载 | 服务启动时加载模型到内存，避免冷启动 |
| LLM 流式输出 | 流式返回改善用户体验 |
| 数据库索引优化 | 慢查询分析 + 索引调优 |
| 异步化 | FastAPI async + 异步数据库驱动 |

---

## 三、时间线总览

```
阶段1: 数据管道完整化      ████░░░░░░░░░░░░░░░░  Week 1-2
阶段2: 模型工程化          ░░░░████░░░░░░░░░░░░  Week 2-4
阶段3: Agent 能力深化      ░░░░░░░░████░░░░░░░░  Week 4-6
阶段4: 工程化与质量保障    ░░░░░░░░░░░░████░░░░  Week 6-8
阶段5: 高可用与可观测性    ░░░░░░░░░░░░░░░░████  Week 8-9
```

**关键里程碑**：
- Week 2: OpenClaw + MQ 跑通，数据管道完整
- Week 4: 模型版本管理 + A/B 测试上线
- Week 6: 实时预测 Agent 可用
- Week 8: CI/CD + 测试覆盖率 >70%
- Week 9: 全链路监控 + 告警

---

## 四、面试展示矩阵

每个模块的"讲什么"和"亮点":

| 模块 | 面试讲什么 | 技术亮点 |
|------|-----------|---------|
| Agent 架构 | LangGraph 多 Agent 路由设计 | 意图识别→条件路由→子Agent→总结，状态机设计 |
| 数据管道 | OpenClaw → MQ → 多存储 | 三级通信、事件驱动、解耦削峰 |
| ML 工程 | 迁移学习 + 版本管理 + A/B测试 | 小样本迁移、灰度发布、基线对比 |
| 安全防线 | Text2SQL/Cypher 四道防线 | 读写隔离、Schema校验、EXPLAIN验证、Probe Query |
| 记忆机制 | 三层记忆 + 条件检索 | Compaction 压缩、向量检索按需触发 |
| 可观测性 | LangFuse + Prometheus | 全链路追踪、指标采集、告警 |
| 工程质量 | CI/CD + 70%覆盖率 | 自动化测试、质量门禁、回归测试 |
| 高可用 | 熔断降级 + 限流 | LLM 多模型降级、Redis 限流、优雅降级 |

---

## 五、优先级排序（如果时间不够）

如果面试在即，按以下优先级取舍：

| 优先级 | 必做 | 理由 |
|--------|------|------|
| P0 | 测试体系 + CI/CD | 面试必问，没测试等于没工程质量 |
| P0 | OpenClaw 容器补全 | 数据管道不完整讲不动 |
| P0 | 架构文档 | 讲不清楚等于没做 |
| P1 | MQ 接入 | 架构完整性的关键一环 |
| P1 | 模型版本管理 | ML 工程化深度 |
| P2 | A/B 测试 | 锦上添花 |
| P2 | 实时预测 | 锦上添花 |
| P3 | Prometheus/Grafana | 有 LangFuse 够用 |

---

*创建时间: 2026-06-23*
*状态: 规划中，与短期 plan 并行推进*
