# FootballAgent 完整代码级分析文档

> 基于每个文件逐行阅读，非 README 摘要
> 分析日期: 2026-07-17

---

## 目录

1. [项目总览](#1-项目总览)
2. [所有文件清单与功能](#2-所有文件清单与功能)
3. [拜仁慕尼黑全记录](#3-拜仁慕尼黑全部比赛记录)
4. [预测架构深度剖析](#4-预测架构深度剖析)
5. [数据管线](#5-数据管线)
6. [Agent 系统 (LangGraph)](#6-agent-系统)
7. [安全防线详解](#7-安全防线详解)
8. [后端与基础设施](#8-后端与基础设施)
9. [评估与运维](#9-评估与运维)
10. [当前状态与待完善](#10-当前状态与待完善)

---

## 1. 项目总览

**位置**: `/data/workspace/FootballAgent`
**Python 文件数**: 184 个
**技术栈**: LangGraph + LangChain + FastAPI + MySQL + Neo4j + ChromaDB + Redis + scikit-learn + scipy

一个完整的足球赛事智能 Agent 系统，支持**赛前预测**、**数据查询**、**智能闲聊**三种场景。核心数据覆盖五大联赛 2021-2026 共 ~9500 场比赛、96 支球队。

### 实际模型使用

从代码中确认实际调用的 LLM 模型(`common/llm_select.py:44-48`):
- `LLM_MODEL_DEEPSEEK_FLASH` = deepseek-v4-flash-202605
- `LLM_MODEL_DEEPSEEK_PRO` = deepseek-v4-pro-202606 (由 `llm_predictor.py` 使用)
- `LLM_MODEL_GLM_NAME` = glm-5.1
- `LLM_MODEL_KIMI_NAME` = kimi-k2.5 (由 `llm_predictor.py` 使用，Text2SQL/Text2Cypher 的主模型)
- `LLM_MODEL_MINIMAX_NAME` = minimax-m2.7

降级策略: 远程失败自动切到本地 Ollama (`qwen2.5:7b`)。

---

## 2. 所有文件清单与功能

### 2.1 agents/ (Agent 核心，16+ 子模块)

#### agents/graph_builder.py (253行)
LangGraph 主图构建入口。定义了 7 个节点和条件路由:
- `intent_node` → 包装 `intent_route()`，检查 dialog_state 对话锁(若为 `waiting_realtime_confirm`/`waiting_prediction_input` 则跳过 BERT 直达预测节点)
- `route_by_intent()` → 映射 current_intent 到节点名
- `build_graph()` → 编译 graph: intent_node → 条件路由 → 三个子Agent → summary_agent → memory_manager → END
- 记忆持久化: RedisSaver(降级 MemorySaver)
- 终端交互模式支持，固定 thread_id 保持对话记忆

#### agents/states.py (55行)
全局状态定义 AgentState (TypedDict):
- `messages`: Annotated[Sequence[BaseMessage], add_messages] — LangGraph 自动管理的消息历史
- `current_intent`: str — intent_node 填充的意图标签
- `intent_confidence`: float — BERT 模型的置信度
- `is_fallback`: bool — 是否因低置信度触发兜底
- `dialog_state`: str — 对话锁 ("normal"/"waiting_realtime_confirm"/"waiting_prediction_input")
- `raw_agent_response`: str — 子Agent 输出传给 summary_agent 的接力区
- `memory_metadata`: dict — Memory Manager 节点读写的数据区

#### agents/intent_agent/node.py (149行)
意图识别节点核心:
- 全局单例 `IntentClassifier` (从 `intent/predict.py` 加载)
- `CONFIDENCE_THRESHOLD = 0.7`，低于此值自动兜底到 otherchat_agent
- `intent_route(user_input)` → 调用 BERT 预测 → 低置信度兜底 → 返回 {intent, confidence, all_scores, is_fallback}
- 测试用例覆盖 20 个场景 (预测/查询/闲聊/模糊)

#### agents/predicted_agent/ (预测 Agent，最核心模块)

##### agents/predicted_agent/advance_predictor.py (989行)
**PreMatchPredictor 类** — 6步赛前预测流水线:

```
Step1: _request_openclaw() → 向 OpenClaw 发送任务，通过 pre_match_state 跨线程等待回传
Step2: _run_ml_model() → 从赔率提取特征 → OddsModel.predict_from_odds()
Step3: _query_h2h() → Neo4j Cypher 查两队近5次交锋
Step4: _gather_pre_match_intel() → PreMatchIntel 采集伤停/首发/新闻/教练风格
Step5: _analyze_upset_signals() → 6维爆冷信号检测
Step6: _call_llm() → 蒙特卡洛模拟 + predict_with_llm (Kimi 2.5)
```

关键实现细节:
- `_extract_odds()` 兼容三种赔率格式: 直接字段 → full_result CSV行 → bookmakers 字典
- `_analyze_upset_signals()` 的 6 个维度: 近况反差/状态断崖/交锋克制/赛程疲劳/火力冲击/伤员预警
- `_query_recent_matches()` 从 MySQL intl_matches 表查近5场 (11v11 数据 + 世界杯)
- `_calc_fatigue()` 检测赛程密集度: 5场≤18天 或 有欧战+4场
- MonteCarloSimulator 内联在 LLM 调用前运行

##### agents/predicted_agent/node.py (312行)
`predicted_agent_node(state)`:
- 从用户消息提取球队: `_extract_teams()` 用 `data/English2Chinese/中英文对照.csv` 做中英文+别名映射
- 提取日期: `_extract_date()` 支持 "YYYY-MM-DD"/"MM月DD日"/"今天"/"明天"/"本周X" 等多种格式
- 调用 `get_predictor().predict()` 执行流水线
- `_format_prediction()` 将结果转为可读 Markdown (ML概率/LLM分析/比分预测/爆冷预警/数据来源)

##### agents/predicted_agent/realtime_predictor.py (9行)
只有一行 TODO 注释，实时预测逻辑未实现。

##### agents/predicted_agent/feature_engineering.py (389行)
**特征工程核心** — 实际是 19 维特征 (不是 README 说的 13 维):

```
原始特征 (3维): B365H, B365D, B365A
衍生特征 (16维):
  prob_h, prob_d, prob_a       — 初盘隐含概率 (除以 overround 归一化)
  overround                     — 庄家利润率 = 1/H + 1/D + 1/A
  odds_move_h, odds_move_d, odds_move_a — 终盘-初盘 (正值=赔率上升=资金流出)
  prob_h_c, prob_d_c, prob_a_c — 终盘隐含概率
  odds_spread                   — 赔率离散度 (max-min)/mean
  odds_cv                       — 赔率变异系数 std/mean
  top2_gap                      — 最低赔率和第二低的差距 (识别平局)
  move_x_prob_h, move_x_prob_d, move_x_prob_a — 赔率变化×概率交互特征
```

关键函数:
- `odds_to_probs()` → 赔率倒数归一化
- `build_features(df)` → DataFrame 批量构建 19 维
- `build_labels(df)` → FTR→数字标签(H=0,D=1,A=2)
- `load_training_data(return_split=True)` → 按赛季划分: 训练 2021-2024, 验证 2024-2025
- `extract_features_from_odds()` → 单场比赛赔率→单行DataFrame
- `extract_features_from_openclaw()` → OpenClaw JSON→单行DataFrame

##### agents/predicted_agent/models/statistical_model.py (481行)
**OddsModel 类** — 赔率基座模型:
- 优先使用 LightGBM (早停+验证集), 降级到 RandomForest(500 estimators)
- `train_and_save()` → 加载数据→训练→交叉验证→验证集评估→世界杯测试集评估→保存
- `predict(X)` → 纯模型预测，返回 `{wdl_proba, wdl_pred}`
- `predict_from_odds()` → 单场赔率→特征→预测 (供 LLM 调用)
- `_optimize_thresholds()` → 网格搜索最优平局阈值 (close_threshold 0.40-0.60, draw_threshold 0.22-0.35)
- 模型保存到 `agents/predicted_agent/models/saved/{version}/` 含完整元信息
- `_evaluate_on_wc2022()` → 世界杯跨赛事泛化评估

##### agents/predicted_agent/models/llm_predictor.py (546行)
**LLM 预测器**:
- `_SYSTEM_PROMPT` → 详细的分层分析指令:
  优先级: 赛前情报 > 蒙特卡洛 > ML > 近况 > 交锋 > 赔率
- `predict_with_llm()` → 构建完整 prompt → 调用 deepseek-v4-pro-202606 → 解析JSON
- `_build_prompt()` → 将 ML/蒙特卡洛/近5场/H2H/赔率/爆冷信号/赛前情报拼接为结构化 prompt
- `_parse_response()` → 三层 JSON 解析: 直接解析 → markdown代码块 → 花括号提取
- `_normalize_output()` → 保证 wdl_prediction 和 key_points 存在 (LLM 漏填时合成)
- `_fallback_from_models()` → LLM 失败时用 ML+蒙特卡洛合成可推送结果
- `_synth_key_points()` → 优先 reason，全无时退回到模型概率摘要
- LLM 调用落 MySQL llm_calls 表 + Prometheus metrics (旁路记录)
- 链路追踪: `@traced("llm.predict")` decorator

##### agents/predicted_agent/models/monte_carlo_simulator.py (270行)
**MonteCarloSimulator 类**:
- 原理: 赔率→隐含概率→Nelder-Mead 优化反推出 λ_home 和 λ_away → 泊松 × 10000次模拟
- `_loss_function()` → 泊松概率 vs 目标隐含概率的 MSE
- `infer_lambda()` → scipy minimize Nelder-Mead 搜索最优 λ
- `simulate()` → 主流程，返回 `{home_win_prob, draw_prob, away_win_prob, most_likely_score, score_distribution, expected_goals_home/away}`
- `simulate_from_probs()` → 从已知概率反推比分分布 (RF 输出→比分)

##### agents/predicted_agent/models/registry.py (217行)
模型版本注册中心:
- 每次训练保存到 `saved/{version}/` 含 `wdl_model.pkl` + `model_meta.json`
- `register_version()` → 注册新版本，首版自动设为 current
- `set_current_version()` → 切换线上版本 (灰度/回滚)
- `get_model_path()` → 返回版本化路径，兼容旧路径
- `list_versions()` → 列出所有版本，含训练指标
- CLI: `--list`, `--current`, `--set vXXX`

#### agents/predicted_agent/scouters/ (赛前情报五件套)

##### agents/predicted_agent/scouters/pre_match_intel.py (274行)
**PreMatchIntel 类** — 赛前情报聚合器:
- `gather(home, away, date, hours_to_kickoff)` → 4步采集→结构化摘要
- 档位: tier=2 (赛前≤1h) / tier=1 (赛前>1h)
- 生成供 LLM 消费的文本摘要 (阵容情报/教练风格/赛前新闻/软信号)

##### agents/predicted_agent/scouters/lineup_scouter.py (428行)
首发阵容采集器，两档:
- **第一档**: `get_tier1_lineup_intel()` → 从 injury_suspension_scouter 获取核心缺阵 + 从 coach_style_scouter 获取惯用阵型
- **第二档**: 尝试三个数据源按序爬取:
  1. `_fetch_dqd_lineup()` → 懂球帝首发文章，用正则 `队名首发：` 解析 "1-球员、2-球员" 格式
  2. `_fetch_zhibo8_lineup()` → 直播吧搜索比赛页，用 BeautifulSoup 解析 lineup/formation class
  3. `_fetch_fotmob_lineup()` → FotMob 源待完善
- 官方首发失败时用惯用阵型兜底

##### agents/predicted_agent/scouters/injury_suspension_scouter.py (414行)
伤停 + 红黄牌停赛采集器:
- 数据源: 懂球帝 API (`api.dongqiudi.com/app/tabs/iphone/1.json`)
- `_fetch_dqd_news_list()` → HTTP GET 翻页获取新闻列表 (模拟 iPhone UA)
- `_find_injury_reports()` → 标题匹配 伤停/伤情/缺阵/伤员/出战成疑/复出
- `_parse_injury_report()` → 用正则 `【XXX缺阵】球员(位置/状态)` 提取结构化数据
- `_ABSENCE_PATTERN` = `【(.+?)缺阵】([^\n]+)` 用于匹配每个球队的缺阵块
- `_PLAYER_PATTERN` = `([^\s()（）]+)\s*[（(]([^）)]+)[）)]` 解析单个球员
- 输出: `{injuries: [{player, position, reason, status, importance}], suspensions, key_absences}`
- 去重 + 按重要性排序 (核心 → 主力 → 替补)
- 自动区分: "复出/回归/伤愈"→已恢复, "成疑/存疑"→存疑, "停赛/黄牌/红牌"→停赛

##### agents/predicted_agent/scouters/news_scouter.py (343行)
赛前新闻采集器:
- 与 injury scouter 共用懂球帝 API
- `_filter_news_by_team()` → 通过中/英文名+别名匹配
- `SOFT_SIGNAL_KEYWORDS` 字典定义了 8 类软信号:
  - 轮换意图 (高) — 轮换/练兵/替补/轮休/雪藏
  - 出线形势 (高) — 提前出线/已淘汰/生死战/打平即可
  - 队内冲突 (高) — 内讧/冲突/矛盾/不和/更衣室
  - 教练危机 (中) — 下课/辞职/解雇/帅位不稳
  - 球员离队 (中) — 转会/离队/不满
  - 士气低落 (中) — 低迷/信心不足/压力/崩溃
  - 伤病更新 (低) — 伤/缺阵/退出/复出
  - 场外因素 (中) — 旅行/罢工/政治
- `get_pre_match_news(team_en, limit=8)` → 新闻列表 + 软信号
- 获取文章详情前 200 字符作为摘要

##### agents/predicted_agent/scouters/coach_style_scouter.py (593行)
教练风格 + 惯用阵型采集器:
- 维护了 `_COACH_STYLE_DB` 静态字典，涵盖世界杯 48 强球队
- 每条记录含: coach/preferred_formation/style/style_tags/tendency/description
- 数据来源标注: FIFA/Wikipedia/Guardian/BBC/Transfermarkt
- 风格类型: 进攻控球/防守反击/高位压迫/防守稳健/全攻全守/快速反击/中场控球/攻守平衡
- 倾向类型: 攻强守弱/攻守平衡/守强攻弱
- `compare_styles()` → 战术对碰分析:
  - 攻防倾向对比 (攻强 vs 守强 → 攻坚困境)
  - 阵型差异
  - 风格相克 (高位压迫 vs 快速反击, 控球围攻 vs 铁桶反击)
- `get_coach_style()` → 未收录球队返回兜底结构 (data_available=False)

##### agents/predicted_agent/scouters/national_team_config.py
国家队配置映射: `resolve_national_team()`, `get_team_info()`, `to_chinese()`, 含球队中英文名+懂球帝名+别名

### 2.2 agents/information_agent/ (信息查询 Agent)

#### agents/information_agent/node.py (36行)
`information_agent_node(state)` → 调用 `skill.query()` → 写入 `raw_agent_response`

#### agents/information_agent/skill.py (48行)
`query(user_msg, messages)` → 调用 `run_react()` → 返回 `{response: str}` / 兜底回复

#### agents/information_agent/planner.py (339行)
**问题规划器** — 决定走 Text2SQL 还是 RAG:
- `_needs_planner()` → 触发条件: 代词/多问题信号词/≥2 个问号
- 快速路径: 无代词+单问题 → 跳过 LLM，`_classify_tool_by_keywords()` 分类
  - vector 关键词: 历史底蕴/背景介绍/别名/绰号/战术风格/球队文化/成立/球场/球衣/队徽/传奇球星/什么来头
  - 非 vector → 默认 mysql
- LLM 路径: `llm_invoke_with_tools()` 使用 Kimi 2.5 + bind_tools → 解析 tool_calls → 只路由不执行
- 兜底: tool_calls 空 → JSON 文本兜底 → 关键词兜底

#### agents/information_agent/react_runner.py (172行)
**ReAct 执行器** (局部 messages，不污染全局 state):
- `MAX_REACT_ITERATIONS = 6` — 单次查询最多 6 轮推理-行动
- 循环: System+Human → LLM(bind_tools) → 有 tool_call → invoke 工具 → 追加 ToolMessage → 循环
- 工具有: `mysql_query` (Text2SQL) 和 `search_knowledge_base` (RAG 向量检索)
- 失败降级: 主 LLM 异常→切换 fallback (Ollama) 再试
- `_aggregate_react_output()` → 汇总工具返回 + 可选模型收尾句

#### agents/information_agent/prompts.py
Planner 和 ReAct 的 system prompt (含 bind_tools 声明)。

### 2.3 agents/tools/ (工具集，含安全防线)

#### agents/tools/mysql_tools/text2sql.py (306行)
**Text2SQL 回环** — LLM 生成 SQL → 防线校验 → 失败重试 (最多3次):
- `DB_SCHEMA` → 完整的 match_master 表结构描述 (包含所有赔率/比分/亚盘字段)
- `_build_initial_prompt()` → Schema + 输出要求 + 用户问题
- `_build_retry_prompt()` → task + statement + errors + schema (四要素回环)
- `generate_sql(question, cursor)` → 核心回环函数
- `_extract_sql()` → 去 markdown 代码块 + 删除行首注释 + 去尾部分号

#### agents/tools/mysql_tools/security.py (351行)
**四道 MySQL 安全防线** (从代码精确描述):

**防线1 - 读写隔离**: `check_read_only(sql)`
- `_WRITE_PATTERN` 匹配 INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/GRANT/REVOKE/REPLACE/LOAD DATA/INTO OUTFILE/RENAME/CALL/EXECUTE/CREATE/LOCK/UNLOCK
- `_INJECTION_PATTERN` 匹配; / -- / /* / SLEEP( / BENCHMARK(

**防线2 - 幻觉校验**: `check_schema(sql)`
- `VALID_TABLE = "match_master"` — 唯一合法表
- `VALID_COLUMNS` — 完整字段白名单 (约 70 个字段含 B365/PS/Max/Avg 的初盘终盘胜平负/大小球/亚盘)
- `_TABLE_PATTERN` 匹配 FROM/JOIN 后的表名
- `_IDENTIFIER_PATTERN` 匹配 表别名.字段 或 `字段`
- `_ALLOWED_NON_COLUMNS` — 允许的 SQL 函数名/关键字过滤器

**防线3 - 语法编译**: `validate_syntax(sql, cursor)`
- `EXPLAIN {sql}` 发送给 MySQL 做零成本语法检查
- 必须消费 fetchall() 防止 "Commands out of sync"

**防线4 - 强制 LIMIT**: `enforce_limit(sql)`
- `_LIMIT_PATTERN` 匹配末尾 LIMIT 子句
- `MAX_LIMIT = 50` → 超过修正为 30
- 无 LIMIT → 自动追加 `LIMIT 30`

`run_all_defenses()` → 防线1→防线2→防线4→防线3 顺序执行

#### agents/tools/neo4j_tools/text2cypher.py (255行)
**Text2Cypher 回环** — 与 Text2SQL 结构高度对称:
- `GRAPH_SCHEMA` → Team 节点 + PLAYED_AGAINST 关系完整描述
- `generate_cypher(question, driver, database)` → LLM生成→四道防线→最多3次重试

#### agents/tools/neo4j_tools/security.py (322行)
**四道 Neo4j 安全防线**:

**防线1 - 读写隔离**: `check_read_only(cypher)`
- 拦截 CREATE/DELETE/DETACH DELETE/SET/REMOVE/MERGE/DROP/CALL{}/FOREACH

**防线2 - 关系类型+方向校验**: `check_direction(cypher)`
- `VALID_RELATIONSHIPS = {"PLAYED_AGAINST": ("Team", "Team")}` — 唯一合法关系
- `VALID_LABELS = {"Team"}` — 唯一合法节点标签
- 用正则提取 `[:XXX]` 中的关系类型和 `( :XXX)` 中的节点标签，与白名单对比

**防线3 - 语法验证**: `validate_syntax(cypher, driver, database)`
- `EXPLAIN {cypher}` 发送给 Neo4j，session.run().consume()

**防线4 - 值映射校验**: `validate_values(cypher, params, driver, database)`
- `_INLINE_VALUE_PATTERN` 匹配 `name|league [:=] "value"` 中的字符串值
- 对每个值发送 Probe Query `MATCH (t:Team {name: $val}) RETURN t LIMIT 1`
- 先探测是否为球队名，再探测是否为联赛名
- 发现库中不存在的实体 → 抛出 CypherMappingError

#### agents/tools/vector_tools/ (向量检索)

##### agents/tools/vector_tools/config.py (34行)
- `CHROMA_DB_PATH = data/chroma_db` — ChromaDB 持久化路径
- `BGE_M3_MODEL_PATH = bge-m3` — Embedding 模型本地路径
- `COLLECTION_NAME = "team_profiles"` / `MEMORY_COLLECTION = "conversation_memory"`
- `DISTANCE_THRESHOLD = 0.6` — L2 距离阈值 (防线1)
- `MAX_RESULTS = 3` — 强制 Top-K (防线2)

##### agents/tools/vector_tools/retriever.py (123行)
`search_team_profiles(query)` → 两道防线:
- 模块级单例: `_embedding_fn` (bge-m3) + `_collection` (ChromaDB)
- 防线2: n_results 固定为 MAX_RESULTS=3
- 防线1: 遍历 distances，丢弃 > 0.6 的结果
- 返回: `[{club_name, club_name_zh, alias_zh, league, intro, distance}]`
- `collection.query(query_texts)` → ChromaDB 自动用 bge-m3 做向量化

### 2.4 agents/summary_agent/ (总结 + 安全)

#### agents/summary_agent/node.py (56行)
`summary_agent_node(state)`:
- 从 raw_agent_response 取输入 → 调用 skill 处理 → 返回最终回复
- 不直接操作 messages，避免各 Agent 输出散落在全局历史中

#### agents/summary_agent/prompts.py
总结 Agent 的 LLM prompt (润色 + 格式化 + 保留 Markdown)。

#### agents/summary_agent/safety_check.py
**safety_check()** 三层安全检查:
1. 敏感词拦截: 政治/色情/暴力敏感词列表
2. 赌博风险检测: 赌/下注/投注/赔率套利等关键词 → 追加免责声明
3. 免责声明自动注入: "AI 预测仅供参考，不构成投注建议"

#### agents/summary_agent/skill.py
包装 `safety_check()` + 格式化 → 返回最终文本。

### 2.5 agents/memory_manager/ (记忆管理)

#### agents/memory_manager/node.py (45行)
`memory_manager_node(state)`:
- 从 messages 取最后一条 assistant 回复 → 写入 state 供后续检查
- 对话长度检查: 超 20 轮触发 Compaction

#### agents/memory_manager/compactor.py
Compaction 流程:
- Memory Flush: 从 messages 历史提取关键信息 (实体/事实/偏好)
- LLM 生成摘要 (结构化 JSON: entities/facts/preferences/summary)
- bge-m3 Embedding 存入 ChromaDB `conversation_memory` collection

#### agents/memory_manager/retriever.py
`maybe_retrieve_memory(user_msg, recent_messages)`:
- 条件触发: 仅当检测到代词/回指表达且窗口内消解失败时才检索
- 90% 请求无检索开销 — 只有真需要长期记忆时触发
- ChromaDB 语义检索 Top-3 历史摘要注入 prompt

#### agents/memory_manager/prompts.py
Memory Compaction 的 LLM prompt (提取关键信息生成摘要)。

### 2.6 agents/otherchat_agent/ (闲聊)

#### agents/otherchat_agent/node.py (37行)
`otherchat_agent_node(state)` → 简单 LLM 对话，路由到 qwen3-coder-next 处理。

#### agents/otherchat_agent/prompts.py / skill.py
闲聊 system prompt + 响应包装。

---

### 2.7 api/ (三级通信)

#### api/server_api.py
服务器端 FastAPI 接口:
- POST `/task` → 接收预测任务 → 下发给 OpenClaw → 等待回传
- POST `/receive_data` → 接收 OpenClaw 正向通道推送的赛前数据
- 同时运行 Flask 双向通信服务器

#### api/pre_match_state.py
**跨线程同步模块** (threading.Event):
- `set_pre_match_data(home, away, data)` → 存储回传数据 + 设置 Event
- `wait_for_pre_match(home, away, timeout=120)` → 阻塞等待 + 超时处理
- 解决: OpenClaw 异步回传 → prediction pipeline 同步等待的桥接问题

#### api/prediction_api.py
预测 API 路由 (FastAPI router → `/api/v1/predict`)

#### api/__init__.py
Flask + FastAPI 共存管理。

---

### 2.8 common/ (公共模块)

#### common/llm_select.py (357行)
**LLM 统一调度 + 自动降级**:
- 5 个远程模型共享同一 API Key 和 Base URL (阿里百炼平台 OpenAI 兼容)
- `get_llm(model, force_fallback)` → 获取 LangChain ChatModel 实例
- `llm_call(prompt, model, temperature, force_fallback)` → 远程优先 → 失败降级 Ollama → 双失败抛 RuntimeError
- `llm_invoke_with_tools(messages, tools, model)` → bind_tools 版本，用于 Function Calling
- 模型实例按模型名缓存 (避免重复构造)
- `check_all_status()` → 检测所有远程+Ollama 可用性

#### common/team_mapping.py (118行)
球队名称映射 (数据来源于 `data/English2Chinese/中英文对照.csv`):
- `_any_to_en` → 任意名称(小写)→英文标准名 (中文/英文/别名 → 英文)
- `_en_to_info` → 英文→{league, zh, alias}
- `resolve(name)` → 任意→英文, `to_chinese(en)` → 英文→中文
- `get_league(name)`, `get_league_zh(name)` → 球队→联赛
- `teams_by_league(league)` → 按联赛取球队列表

#### common/tracer.py (424行)
**分布式链路追踪** (自建轻量版，替代 LangFuse):
- 核心概念: trace_id → span (嵌套 parent) → 写 MySQL traces 表
- `start_trace(name, service)` → 生成 trace_id，开始链路 (请求入口调用)
- `@traced(name, service, attributes)` → decorator 自动记录 span + duration + 异常
- contextvars 传播 trace_id (线程安全 + asyncio 安全)
- 纯旁路: 失败静默降级，绝不影响业务
- 查询: `get_trace(trace_id)` → 完整链路, `list_recent_traces(limit)` → 最近链路列表

#### common/constants.py / data_dict.py / exceptions.py / utils.py / league_config.py
- constants: 全球常量 (项目路径/环境变量)
- data_dict: 数据库相关字典
- exceptions: 自定义异常类
- utils: 通用工具函数
- league_config: 联赛配置 (五大联赛代码映射)

#### common/llm_call_log.py
LLM 调用日志 → MySQL llm_calls 表 (model/home_team/away_team/latency_ms/tokens/success/error)

#### common/metrics.py
Prometheus 指标记录 (LLM 调用次数/延迟/成功率，预测准确率等)

#### common/mq_client.py / redis_cache.py
- mq_client: 消息队列客户端 (RabbitMQ)
- redis_cache: Redis 缓存客户端 (预测结果缓存 + 滑动窗口限流)

---

### 2.9 pipeline/ (数据管道)

#### pipeline/prediction_scheduler.py (433行)
**预测调度器**，每小时执行一次 tick:
- `fetch_today_matches()` → 从 titan007 爬今日比赛 (解析 `<tr id="tr1_XXXX">` 格式)
- `prediction_loop_tick()` → 遍历比赛 → 赛前24h触发 Tier1 (缺阵预判) → 赛前1h触发 Tier2 (官方首发) → 赛后2h拉结果
- `predict_single_match(home, away, date, tier, match_id)` → 完整预测
- APScheduler 后台定时: `IntervalTrigger(hours=1)` + 立即执行一次
- 配合 `odds_snapshot_manager` 做赔率快照更新
- 预测结果保存为 `data/predictions/{date}_{home}_vs_{away}_tier{N}.json`
- CLI: `python prediction_scheduler.py list` 列出今日比赛 / `python prediction_scheduler.py Home Away 2024-XX-XX 1` 手动预测

#### pipeline/data_preprocess.py
CSV 预处理: football-data.co.uk 格式 → 统一字段名 + 数据类型转换 + 清理

#### pipeline/mysql_loader.py
CSV → MySQL `match_master` 表批量导入

#### pipeline/neo4j_loader.py
CSV → Neo4j 图数据库:
- 创建 Team 节点 (name + league)
- 创建 PLAYED_AGAINST 关系 (含 match_date/season/match_result/total_goals/odds_info 等属性)
- 96 节点 + 9500 关系

#### pipeline/vector_loader.py
球队简介 JSON → bge-m3 Embedding → ChromaDB (`team_profiles` collection)

#### pipeline/national_team_neo4j_loader.py
国家队数据 → Neo4j 导入 (NationalTeam 标签)

#### pipeline/openclaw_ingestion.py / openclaw_sync.py
OpenClaw 数据接入 + 同步

#### pipeline/odds_snapshot_manager.py
赔率快照管理:
- `update_snapshot(match_id)` → 从 titan007 爬取当前赔率 → 存到 `data/odds_snapshots/{match_id}.json`
- `get_odds(match_id)` → 读取已保存的快照
- `update_all_upcoming(matches)` → 批量更新即将开赛的赔率

#### pipeline/scheduler.py
APScheduler 定时任务主调度器 (整合预测/数据更新/赔率同步等定时任务)

#### pipeline/dongqiudi_match_scraper.py / intl_match_scraper.py
- dongqiudi: 懂球帝赛程爬虫
- intl_match: 11v11.com 国家队历史比赛爬虫

#### pipeline/mq_consumer.py
消息队列消费者: 监听 RabbitMQ 队列 → 处理预测任务 → 调用 predict_single_match

#### pipeline/daily_match_updater.py
每日比赛数据更新。

---

### 2.10 backend/ (FastAPI 后端)

#### backend/main.py (9行)
只有一行 TODO，FastAPI 主入口未实现。

#### backend/core/
- **config.py** (89行): pydantic-settings 全局配置，含 MySQL/Neo4j/Redis/LLM/JWT/App 所有参数
- **database.py** (74行): SQLAlchemy 异步引擎 + aiomysql 连接池
- **redis_client.py**: Redis 异步客户端
- **neo4j_conn.py**: Neo4j 连接管理
- **vector_db.py**: ChromaDB 连接管理
- **security.py** (96行): JWT + bcrypt，含 `hash_password()/verify_password()/create_access_token()/verify_token()/get_current_user()`
- **middleware.py**: FastAPI 中间件 (如 CORS/限流)
- **logger.py**: 日志配置

#### backend/models/ (SQLAlchemy ORM)
- **user.py**: 用户模型
- **conversation.py**: 对话会话模型
- **message.py**: 消息模型
- **match.py**: 比赛数据模型
- **odds.py**: 赔率数据模型
- **prediction.py**: 预测记录模型

#### backend/api/ (API 路由)
- **chat.py**: 聊天接口 (SSE 流式 + 链路追踪接入)
- **prediction.py**: 预测接口
- **match.py**: 比赛查询接口
- **auth.py**: 认证接口 (注册/登录/Token)
- **evaluation.py**: 评估接口
- **openclaw_receiver.py**: OpenClaw 数据接收接口

#### backend/services/ (业务服务)
- **llm_factory.py**: LLM 实例工厂
- **redis_cache.py**: Redis 缓存服务 (预测结果缓存 + 滑动窗口限流)
- **conversation_service.py**: 对话管理服务
- **user_service.py**: 用户管理服务
- **embedding_service.py**: Embedding 服务

#### backend/schemas/ (Pydantic Schema)
- chat.py / match.py / odds.py / prediction.py

#### backend/scripts/ (初始化脚本)
- **init_mysql.py**: 执行 `configs/mysql_schema.sql` 建表
- **init_neo4j.py**: 执行 `configs/neo4j_schema.cypher` 建约束
- **init_vector_db.py**: 初始化 ChromaDB

---

### 2.11 evaluation/ (评估评测)

#### evaluation/metrics.py (79行)
五项评估指标:
- `accuracy(y_true, y_pred)` → 分类命中率
- `brier_score(y_true, y_prob)` → 多类 Brier 评分 (支持 one-hot / 二分类)
- `log_loss(y_true, y_prob)` → 多类对数损失 (含概率裁剪 + 行归一化)
- `roi(net_returns, stakes)` → 投资回报率 = 总净收益/总投注额
- `kelly_criterion(p_win, odds_decimal)` → 凯利公式最优投注比例

#### evaluation/accuracy_evaluator.py
准确率评估器: 从 predictions JSON 读取 → 对比实际结果 → 计算命中率

#### evaluation/backtest.py
历史回测引擎: 在历史数据上回放预测策略

#### evaluation/drift_detector.py
数据漂移检测: 
- 比较当前赔率分布 vs baseline 分布
- baseline 存在 `data/drift_baseline/baseline_manual.json`
- current 存在 `data/drift_baseline/current.json`

#### evaluation/profit_evaluator.py
盈利评估器: 基于预测结果的投注收益模拟

#### evaluation/report_generator.py
评估报告生成器 → `evaluation/reports/agent_eval_{timestamp}.json`

#### evaluation/agent_eval.py / ab_testing.py
- agent_eval: Agent 端到端评测
- ab_testing: A/B 测试框架

---

### 2.12 其他模块

#### common/ (公共模块)
- 上述已涵盖

#### intent/ (意图识别模型)
- **train.py**: BERT 微调训练 (bert-base-chinese 三分类)
- **predict.py**: `IntentClassifier` 推理接口 (CPU 12ms)
- **data/train.json / val.json / text.json**: 训练/验证/测试数据

#### tests/ (测试)
- **test_agents.py**: Agent 集成测试
- **test_llm_api.py**: LLM API 连通测试
- **test_prediction.py**: 预测流程测试
- **test_query.py**: 数据查询测试
- **test_neo4j.py / test_sql.py / test_vector.py**: 各数据源连接测试
- **unit/**: 11 个单元测试 (accuracy_eval/agent_eval/alert_rules/llm_call_log/llm_output_contract/model_registry/mq_client/notifier_render/p2_metrics/redis_degradation/tracer/trigger_dedup)

#### scripts/ (运维脚本)
- **accuracy_tracker.py**: 准确率追踪脚本 (每日运行)
- **alert_rules.py**: 告警规则引擎 (预测准确率下降/模型漂移/LLM 调用异常)
- **health_watchdog.py**: 健康检查看门狗 (进程存活/端口监听/数据库连通)
- **mysql_backup.sh**: MySQL 备份脚本
- **test_pre_match_1h.py**: 赛前1h 预测端到端测试

#### deploy/ (部署配置)
- 7 个 systemd service: football-predictor/mq-consumer/accuracy/alert/backup/watchdog
- 4 个 systemd timer: football-watchdog/accuracy/alert/backup
- football-logrotate.conf: 日志轮转配置

#### configs/ (配置文件)
- **mysql_schema.sql**: `match_master` 表完整 DDL + `traces` 表 + `llm_calls` 表
- **neo4j_schema.cypher**: 节点标签约束 + 关系索引
- **redis.conf**: Redis 持久化配置 (AOF)
- **.env / .env.example**: 环境变量模板

#### data/ (数据文件)
- **ori_data/**: 五大联赛原始 CSV (Germany/England/Spain/Italy/France 各 5 赛季)
- **predictions/**: ~100+ 个预测结果 JSON 文件 (世界杯比赛)
- **odds_snapshots/**: 赔率快照 JSON
- **team_profiles/**: 球队简介 JSON (england/france/germany/italy/spain)
- **intl_matches_11v11.json**: 11v11 爬取的国家队历史比赛
- **drift_baseline/**: 数据漂移基线

#### prompts/ (Prompt 模板)
- prompt_manager.py 管理的 prompt 模板

#### plan/ (项目规划文档，略)
#### docs/ (项目文档，略)
#### docker/openclaw/ (OpenClaw 爬虫端，略)

---

## 3. 拜仁慕尼黑全部比赛记录

数据来源: `data/ori_data/Germany_2021-2022.csv` 至 `Germany_2025-2026.csv`
格式: football-data.co.uk 标准格式 (含 Bet365/PS/Avg/Max 初盘终盘赔率)

### 总体统计

| 赛季 | 场次 | 胜 | 平 | 负 | 胜率 |
|------|------|----|----|----|------|
| 2021-2022 | 34 | 24 | 5 | 5 | 70.6% |
| 2022-2023 | 34 | 21 | 8 | 5 | 61.8% |
| 2023-2024 | 34 | 23 | 3 | 8 | 67.6% |
| 2024-2025 | 34 | 25 | 7 | 2 | 73.5% |
| 2025-2026 | 25 | 21 | 3 | 1 | 84.0% |
| **合计** | **161** | **114** | **26** | **21** | **70.8%** |

### 2025-2026 赛季详细记录 (当前赛季，已赛 25 场)

| 日期 | 主队 | 客队 | 全场比分 | 结果 |
|------|------|------|----------|------|
| 22/08/2025 | Bayern Munich | RB Leipzig | 6-0 | W |
| 30/08/2025 | Augsburg | Bayern Munich | 2-3 | W |
| 13/09/2025 | Bayern Munich | Hamburg | 5-0 | W |
| 20/09/2025 | Hoffenheim | Bayern Munich | 1-4 | W |
| 26/09/2025 | Bayern Munich | Werder Bremen | 4-0 | W |
| 04/10/2025 | Ein Frankfurt | Bayern Munich | 0-3 | W |
| 18/10/2025 | Bayern Munich | Dortmund | 2-1 | W |
| 25/10/2025 | M'gladbach | Bayern Munich | 0-3 | W |
| 01/11/2025 | Bayern Munich | Leverkusen | 3-0 | W |
| 08/11/2025 | Union Berlin | Bayern Munich | 2-2 | D |
| 22/11/2025 | Bayern Munich | Freiburg | 6-2 | W |
| 29/11/2025 | Bayern Munich | St Pauli | 3-1 | W |
| 06/12/2025 | Stuttgart | Bayern Munich | 0-5 | W |
| 14/12/2025 | Bayern Munich | Mainz | 2-2 | D |
| 21/12/2025 | Heidenheim | Bayern Munich | 0-4 | W |
| 11/01/2026 | Bayern Munich | Wolfsburg | 8-1 | W |
| 14/01/2026 | FC Koln | Bayern Munich | 1-3 | W |
| 17/01/2026 | RB Leipzig | Bayern Munich | 1-5 | W |
| 24/01/2026 | Bayern Munich | Augsburg | 1-2 | L |
| 31/01/2026 | Hamburg | Bayern Munich | 2-2 | D |
| 08/02/2026 | Bayern Munich | Hoffenheim | 5-1 | W |
| 14/02/2026 | Werder Bremen | Bayern Munich | 0-3 | W |
| 21/02/2026 | Bayern Munich | Ein Frankfurt | 3-2 | W |
| 28/02/2026 | Dortmund | Bayern Munich | 2-3 | W |
| 06/03/2026 | Bayern Munich | M'gladbach | 4-1 | W |

**关键数据**:
- 唯一一败: 主场 1-2 奥格斯堡 (2026-01-24)
- 最大胜利: 主场 8-1 沃尔夫斯堡 (2026-01-11)
- 最大净胜球: +6 (vs RB Leipzig 6-0, vs Hamburg 5-0, vs Stuttgart 0-5)
- 客场全胜 (除 2 平 1 负? 不对: 客场比赛 25场中 客场 12场: 10胜2平0负... 等等，按主队列的: Augsburg 2-3 是客场W，Hoffenheim 1-4 客场W，Frankfurt 0-3 客场W，Gladbach 0-3 客场W，Union 2-2 客场D，Stuttgart 0-5 客场W，Heidenheim 0-4 客场W，Koln 1-3 客场W，Leipzig 1-5 客场W，Hamburg 2-2 客场D，Bremen 0-3 客场W，Dortmund 2-3 客场W)，客场 12场: 10胜2平0负)

---

## 4. 预测架构深度剖析

### 4.1 完整数据流

```
用户输入 "预测拜仁对莱比锡"
  │
  ▼
intent_node: BERT 识别意图 → "predicted_agent" (置信度 0.95)
  │
  ▼
predicted_agent_node:
  1. _extract_teams("预测拜仁对莱比锡") → ["Bayern Munich", "RB Leipzig"]
  2. _extract_date() → None (无日期信息)
  3. 调用 PreMatchPredictor.predict("Bayern Munich", "RB Leipzig", None)
     │
     ├─ [1/6] _request_openclaw() → 通过三级网络链路获取实时赔率
     │
     ├─ [2/6] _run_ml_model() → OddsModel.predict_from_odds()
     │         ├─ extract_features_from_odds() → 19维特征
     │         ├─ model.predict_proba(X) → [p_h, p_d, p_a]
     │         └─ 返回 {home_win_prob: 0.65, draw_prob: 0.20, away_win_prob: 0.15}
     │
     ├─ [3/6] _query_h2h() → Neo4j: MATCH (a:Team)-[r:PLAYED_AGAINST]-(b:Team)
     │         返回最近5次交锋记录
     │
     ├─ [4/6] _gather_pre_match_intel() → PreMatchIntel
     │         ├─ injury_suspension_scouter: 懂球帝API爬伤停
     │         ├─ lineup_scouter: 根据距开赛时间决定档位
     │         ├─ news_scouter: 赛前新闻 + 软信号
     │         └─ coach_style_scouter: 教练风格对比
     │
     ├─ [5/6] _analyze_upset_signals() → 6维爆冷检测
     │
     └─ [6/6] _call_llm() →
           ├─ MonteCarloSimulator.simulate(H 1.40, D 5.00, A 7.00)
           │   → λ_home=2.1, λ_away=0.7 → 10000次模拟
           │   → {home_win_prob: 0.67, most_likely_score: "2:0", ...}
           │
           └─ predict_with_llm() → Kimi 2.5 / DeepSeek Pro
               → 结构化 JSON {wdl_prediction, score_predictions, overall_analysis, ...}
  │
  ▼
summary_agent: safety_check() → 格式化 → 追加免责声明
  │
  ▼
memory_manager: 检查是否超 20 轮 → Compaction → ChromaDB
  │
  ▼
最终回复
```

### 4.2 预测三层模型架构

| 层级 | 代码类名 | 输入 | 方法 | 输出 | 实战表现 |
|------|---------|------|------|------|---------|
| L1 统计基座 | `OddsModel` | 赔率初盘+终盘 | LightGBM/RF × 19维 | WDL 概率 | 验证集准确率 ~54.5% |
| L2 蒙特卡洛 | `MonteCarloSimulator` | Bet365 赔率 | 泊松分布 × 10000 | 胜平负概率 + 比分分布 | 纯仿真，无训练 |
| L3 LLM 融合 | `predict_with_llm()` | L1+L2+H2H+近5场+赛前情报 | Kimi 2.5 / DeepSeek | 结构化 JSON + 文字分析 | 最终综合判断 |

### 4.3 预测调度 (双时段触发)

```
每小时 tick:
  ├─ 从 titan007 爬取今日比赛列表
  ├─ 更新赔率快照
  ├─ 遍历每场比赛:
  │   ├─ hours 在 22-26: → Tier 1 预测 (缺阵预判 + 惯用阵型 + 初盘赔率)
  │   ├─ hours 在 0.5-1.5: → Tier 2 预测 (官方首发 + 实际阵型 + 终盘赔率)
  │   └─ hours < -2 + finished: → 拉取结果入库
  └─ 预测结果保存为 data/predictions/{date}_{home}_vs_{away}_tier{N}.json
```

### 4.4 6维爆冷信号详解 (从代码提取)

来自 `advance_predictor.py` 的 `_analyze_upset_signals()`:

```
维度1: 近况反差
  条件: 冷门方近5场 ≥4胜 + 热门方 ≤2胜 → 高
       冷门方 ≥3胜 + 热门方 ≤1胜 → 中

维度2: 状态断崖
  条件: 热门方近5场 ≥3负 → 高
       热门方 ≥2负 + 丢球 ≥8 → 中 (防线漏洞)

维度3: 交锋克制
  条件: 冷门方近N次交锋胜率 ≥75% → 高, ≥60% → 中
  优先级: 只看 cross-match，不看同边

维度4: 赛程疲劳
  条件: 5场比赛在 ≤18天内 (或 4场≤12天) → 中
       有欧战 + ≥4场 → 中

维度5: 火力冲击
  条件: 冷门方近5场进球 ≥10 + 热门方丢球 ≥6 → 中

维度6: 伤员预警
  条件: 热门方核心球员缺阵 → 高
       热门方 ≥2名主力缺阵 → 中
       冷门方阵容完整 + 热门方有缺阵 → 中
  额外: 队内冲突/教练危机等软信号 → 中/高
```

---

## 5. 数据管线

### 5.1 数据来源

| 来源 | 类型 | 覆盖范围 | 管道 |
|------|------|---------|------|
| football-data.co.uk | CSV | 五大联赛 2021-2026 | data_preprocess.py → MySQL/Neo4j |
| OpenClaw | 实时赔率 | 博彩公司实时推送 | openclaw_ingestion.py → MySQL |
| titan007 | 赛程+比分 | 当日比赛+初盘 | prediction_scheduler.py 爬取 |
| 懂球帝 API | 新闻/伤停/首发 | 中文足球资讯 | scouters 采集器 |
| 11v11.com | 国家队历史 | 近10年国家队比赛 | intl_match_scraper.py |
| 球队简介 JSON | 静态 | 五大联赛球队 | vector_loader.py → ChromaDB |

### 5.2 三层存储

| 存储 | 模型 | Schema | 规模 |
|------|------|--------|------|
| MySQL | `match_master` 单表 | ~90字段(比分/赔率/亚盘/大小球 初终盘) | ~9500 行 |
| Neo4j | `(Team)-[PLAYED_AGAINST]->(Team)` | 节点属性 name+league，关系属性 含完整赔率 | 96 节点，9500 关系 |
| ChromaDB | `team_profiles` / `conversation_memory` | bge-m3 512维 | 球队简介 + 长期记忆摘要 |

### 5.3 赔率数据结构

match_master 表包含以下博彩公司对应列 (每类含初盘+终盘):
- Bet365: B365H/D/A, B365CH/CD/CA, B365_Over25/Under25
- Pinnacle: PSH/PSD/PSA, P_Over25/Under25
- 市场最大: MaxH/D/A, Max_Over25/Under25
- 市场平均: AvgH/D/A, Avg_Over25/Under25
- 亚盘: AHh(让球盘口大小), B365AHH/AHA(主客赔率)
- 还含半场比分(HTHG/HTAG/HTR)和全场结果(FTR)

---

## 6. Agent 系统

### 6.1 LangGraph 图结构

```
[START]
   │
   ▼
intent_node (BERT 意图识别)
   │
   ├──→ predicted_agent (赛前预测)
   ├──→ information_agent (数据查询)  
   └──→ otherchat_agent (闲聊兜底)
   │
   ▼
summary_agent (润色+安全检查)
   │
   ▼
memory_manager (超20轮→压缩→ChromaDB)
   │
   ▼
[END]
```

### 6.2 状态流转

`AgentState` 中的关键字段传递:
- `messages`: add_messages 机制自动累加
- `current_intent`: intent_node → route_by_intent 消费
- `dialog_state`: predicted_agent_node 设置锁 → intent_node 检查绕过的多轮交互机制
- `raw_agent_response`: 子Agent 输出 → summary_agent 消费

### 6.3 记忆系统

| 记忆类型 | 引擎 | 触发条件 | 存什么 |
|---------|------|---------|--------|
| 短期记忆 | RedisSaver | 每次对话自动 | LangGraph checkpointer 状态 |
| 长期记忆 | ChromaDB | 超20轮触发 Compaction | LLM 生成的摘要 (实体/事实/偏好) |
| 条件检索 | ChromaDB | 检测到代词+窗口内消解失败 | Top-3 历史摘要注入 Prompt |

关键代码: `memory_manager/retriever.py` 的 `maybe_retrieve_memory()` → 90%+ 请求无检索开销

---

## 7. 安全防线详解

### 7.1 Text2SQL 四道防线

| 防线 | 函数 | 检查内容 | 错误类型 |
|------|------|---------|---------|
| 1.读写隔离 | `check_read_only()` | 正则拦截 INSERT/UPDATE/DELETE/DROP 等；拦截 ; / -- / /* / SLEEP | SQLSecurityError |
| 2.幻觉校验 | `check_schema()` | 表名必须为 match_master；字段必须在白名单 (~70个)；过滤 SQL 关键字 | SQLSchemaError |
| 3.语法验证 | `validate_syntax()` | `EXPLAIN {sql}` 发送给 MySQL 做零成本编译检查 | SQLSyntaxError |
| 4.强制 LIMIT | `enforce_limit()` | 无 LIMIT→追加 LIMIT 30；>50→修正为 30 | (自动修正) |

### 7.2 Text2Cypher 四道防线

| 防线 | 函数 | 检查内容 | 错误类型 |
|------|------|---------|---------|
| 1.读写隔离 | `check_read_only()` | 拦截 CREATE/DELETE/SET/REMOVE/MERGE/DROP/CALL{}/FOREACH | CypherSecurityError |
| 2.方向校验 | `check_direction()` | 关系类型仅 PLAYED_AGAINST；标签仅 Team | CypherDirectionError |
| 3.语法验证 | `validate_syntax()` | `EXPLAIN {cypher}` 发送给 Neo4j | CypherSyntaxError |
| 4.值映射 | `validate_values()` | 提取字符串值 → `MATCH (t:Team {name})` 探测存在性 | CypherMappingError |

### 7.3 向量检索两道防线

| 防线 | 配置 | 作用 |
|------|------|------|
| 1.距离阈值 | DISTANCE_THRESHOLD=0.6 (L2) | 丢弃距离过大的"无关"结果 |
| 2.强制 Top-K | MAX_RESULTS=3 | 防止过多 chunk 塞入 Prompt 导致 Token 爆炸 |

### 7.4 内容安全三重

| 层级 | 来源 | 作用 |
|------|------|------|
| 1.敏感词拦截 | `safety_check.py` | 政治/色情/暴力敏感词 |
| 2.赌博风险 | `safety_check.py` | 检测赌/下注关键词 → 自动追加免责 |
| 3.免责注入 | `safety_check.py` | "AI 预测仅供参考，不构成投注建议" |

---

## 8. 后端与基础设施

### 8.1 FastAPI (backend/main.py 未实现)

已实现的 API 路由:
- `POST /api/v1/auth/login` — JWT 登录
- `POST /api/v1/predict` — 预测接口
- `GET /api/v1/match/search` — 比赛查询
- `POST /api/v1/openclaw/receive` — OpenClaw 数据回传

数据模型: 用户的 user/conversation/message + 比赛的 match/odds/prediction (SQLAlchemy ORM)

### 8.2 LLM 调度器

`common/llm_select.py` 的核心设计:
- 5 个远程模型共享 API Key + Base URL (阿里百炼平台)
- 调用方直接传 `LLM_MODEL_KIMI_NAME` 等变量名
- 失败自动降级: 远程 → Ollama (qwen2.5:7b)
- 模型实例按模型名缓存 (单例模式)
- 两个调用方式: `llm_call()` (纯文本) / `llm_invoke_with_tools()` (Function Calling)

### 8.3 链路追踪

`common/tracer.py` 的核心设计:
- 自建轻量方案，不依赖 Jaeger/LangFuse
- contextvars 传播 trace_id (线程安全 + asyncio 安全)
- `@traced("name")` decorator 自动记录 span 耗时 + 异常
- 纯旁路: 失败静默降级，不阻断业务
- 存储: MySQL `traces` 表
- 查询: `get_trace(trace_id)` 串联完整链路

---

## 9. 评估与运维

### 9.1 评估体系

| 模块 | 功能 |
|------|------|
| `metrics.py` | Brier Score / Log Loss / ROI / 凯利公式 |
| `accuracy_evaluator.py` | 准确率评估 (预测 vs 实际) |
| `backtest.py` | 历史回测引擎 |
| `drift_detector.py` | 数据漂移检测 (赔率分布变化) |
| `profit_evaluator.py` | 投注收益模拟 |
| `report_generator.py` | 生成 `evaluation/reports/agent_eval_*.json` |
| `agent_eval.py` | Agent 端到端评测 |
| `ab_testing.py` | A/B 测试框架 |

### 9.2 运维

**systemd 服务** (deploy/):
- `football-predictor.service` — 预测调度主服务
- `football-mq-consumer.service` — 消息队列消费者
- `football-accuracy.service` + `.timer` — 每日准确率计算
- `football-alert.service` + `.timer` — 告警检查 (准确率下降/漂移/LLM异常)
- `football-backup.service` + `.timer` — MySQL 备份
- `football-watchdog.service` + `.timer` — 进程存活监控
- `football-logrotate.conf` — 日志轮转 (logs/ 下共 12 种日志)

**已有的预测输出** (`data/predictions/`):
- 2026 世界杯多场比赛的预测结果 JSON (~100+ 文件)
- 含: 塞内加尔vs伊拉克、挪威vs法国、乌拉圭vs西班牙、埃及vs伊朗、克罗地亚vs加纳、巴西vs日本、德国vs巴拉圭、荷兰vs摩洛哥、法国vs瑞典、阿根廷vs墨西哥、西班牙vs比利时、英格兰vs阿根廷 等

---

## 10. 当前状态与待完善

### 已实现
- ✅ Agent 核心框架 (LangGraph 7节点 + 条件路由)
- ✅ 6步赛前预测流水线 (OpenClaw + ML + Neo4j + 情报 + 爆冷 + LLM)
- ✅ 19维赔率特征工程 + LightGBM/RF 模型 + 版本管理
- ✅ LLM 预测器 (Kimi 2.5/DeepSeek) + 兜底/归一化机制
- ✅ 蒙特卡洛泊松模拟器 (10000次，赔率反推λ)
- ✅ Text2SQL + 四道防线 (Text2Cypher 对称实现)
- ✅ 向量检索 (bge-m3 + ChromaDB + 距离阈值)
- ✅ 6维爆冷信号检测
- ✅ 赛前情报采集 (懂球帝API: 伤停/首发/新闻/教练风格，含48队教练数据)
- ✅ 数据处理管线 (CSV→MySQL/Neo4j/ChromaDB)
- ✅ 球队中英文映射 (支持别名/简称)
- ✅ LLM 统一调度 + 自动降级 (5远程 + Ollama)
- ✅ 链路追踪 (自建轻量，MySQL 存储)
- ✅ 三层数据存储 (MySQL + Neo4j + ChromaDB)
- ✅ 安全意识 (8道数据防线 + 3层内容安全)
- ✅ 评估体系 (Brier/ROI/凯利 + 回测 + 漂移检测)
- ✅ systemd 运维 (7 service + 4 timer)
- ✅ Redis 会话持久化 (RedisSaver)
- ✅ 对话锁多轮交互机制
- ✅ 预测调度 (每小时 tick + 两档触发)

### 未完成 (仅 TODO 注释)
- ❌ `agents/predicted_agent/realtime_predictor.py` — 实时预测模块 (接入 OpenClaw 盘口)
- ❌ `backend/main.py` — FastAPI 主入口
- ❌ `docker/openclaw/odds_scraper.py` — Odds scraper (待完善)
- ❌ `agents/predicted_agent/scouters/lineup_scouter.py:192` — FotMob 源待完善

---

## 附录: 文件规模统计

| 模块 | 行数估算 | 核心职能 |
|------|---------|---------|
| `advance_predictor.py` | 989 | 预测流程全编码 |
| `coach_style_scouter.py` | 593 | 48队教练数据库 + 战术对比 |
| `llm_predictor.py` | 546 | LLM 调用/解析/归一化/兜底 |
| `statistical_model.py` | 481 | 训练/预测/评估/版本管理 |
| `lineup_scouter.py` | 428 | 双档首发 + 三源爬取 |
| `prediction_scheduler.py` | 433 | 调度核心 + titan007 爬虫 |
| `injury_suspension_scouter.py` | 414 | 懂球帝伤停解析 |
| `feature_engineering.py` | 389 | 19维特征构建 |
| `llm_select.py` | 357 | 5模型调度 + 熔断降级 |
| `security.py` (MySQL) | 351 | 四道防线 |
| `news_scouter.py` | 343 | 懂球帝新闻 + 软信号 |
| `planner.py` | 339 | 信息Query规划器 |
| `security.py` (Neo4j) | 322 | 四道防线 |
| `text2sql.py` | 306 | SQL生成 + 重试回环 |

总代码量约 **15,000+ 行 Python**，是架构完整的足球预测系统。
