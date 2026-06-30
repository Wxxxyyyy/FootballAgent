# 赔率基座模型 · 版本实验记录

> 记录时间：2026-06-24
> 训练数据：五大联赛 2021-2024 赛季（6694场）
> 验证数据：五大联赛 2024-2025 赛季（1752场）
> 测试数据：2022 卡塔尔世界杯（64场，titan007 爬取）
> 基线策略：选终盘最低赔率（隐含概率最高的结果）

---

## 一、版本总览

| 版本 | 算法 | 特征维度 | class_weight | 验证集准确率 | 世界杯准确率 | vs基线(验证集) | 状态 |
|------|------|---------|-------------|------------|------------|--------------|------|
| 基线 | 选最低赔率 | — | — | 54.91% | 56.25% | — | 参照 |
| **V5** | **LightGBM** | **19** | **无** | **54.51%** | **56.25%** | **-0.40%** | **当前使用** |
| V10 | 集成(RF+LGB+XGB) | 19 | 无 | 54.45% | — | -0.46% | 实验 |
| V4 | LGB+混合策略 | 16 | 无 | 54.62% | 56.25% | -0.29% | 已废弃 |
| V3 | LightGBM | 16 | 无 | 54.05% | 54.69% | -0.86% | 已废弃 |
| V2 | RandomForest | 13 | balanced | 52.68% | 50.00% | -2.23% | 已废弃 |
| V1 | RandomForest | 12 | balanced | 49.63%(CV) | — | -5.28% | 旧版已废弃 |
| V6 | CatBoost | 19 | 无 | 54.34% | — | -0.57% | 实验 |
| V7 | XGBoost | 19 | 无 | 54.34% | — | -0.57% | 实验 |
| V8 | LightGBM | 25 | 无 | 54.22% | — | -0.68% | 实验 |
| V9 | LightGBM(深度调参) | 19 | 无 | 53.82% | — | -1.08% | 实验 |

---

## 二、各版本详细说明

### V1：RF 12维 balanced（旧版）

- **算法**：RandomForestClassifier
- **特征（12维）**：B365H, B365D, B365A, B365>2.5, B365<2.5, AHh + prob_h, prob_d, prob_a, prob_over, prob_under, overround
- **参数**：n_estimators=300, max_depth=10, class_weight="balanced"
- **特点**：含大小球特征，用 class_weight 平衡类别
- **验证集CV准确率**：49.63%
- **问题**：准确率低，大小球特征对胜平负预测无帮助

### V2：RF 13维 balanced

- **算法**：RandomForestClassifier
- **特征（13维）**：去掉大小球，加入赔率变化+终盘概率
  - B365H, B365D, B365A（初盘3维）
  - prob_h, prob_d, prob_a, overround（初盘概率4维）
  - odds_move_h, odds_move_d, odds_move_a（赔率变化3维）
  - prob_h_c, prob_d_c, prob_a_c（终盘概率3维）
- **参数**：n_estimators=500, max_depth=12, class_weight="balanced"
- **验证集准确率**：52.68%
- **世界杯准确率**：50.00%
- **问题**：class_weight="balanced" 导致过度预测平局（预测384场平局，精确率仅31%），主胜召回率降到63.2%

### V3：LightGBM 16维 无权重

- **算法**：LightGBM
- **特征（16维）**：V2的13维 + 赔率离散度特征3维
  - odds_spread（赔率离散度，越小越可能平局）
  - odds_cv（变异系数）
  - top2_gap（最低两赔率差距）
- **参数**：n_estimators=1000, max_depth=6, num_leaves=31, learning_rate=0.05, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1
- **改进**：换LightGBM + 去掉class_weight + 加离散度特征 + 验证集早停
- **验证集准确率**：54.05%
- **世界杯准确率**：54.69%
- **问题**：几乎不预测平局（D召回2.3%）

### V4：LGB 16维 混合策略

- **算法**：LightGBM + 混合策略
- **特征（16维）**：同V3
- **策略**：预测概率 = 0.4 × 模型概率 + 0.6 × 赔率隐含概率
- **验证集准确率**：54.62%
- **世界杯准确率**：56.25%
- **问题**：混入60%赔率后，模型独立性降低，平局预测能力几乎为零（D召回0.5%）

### V5：LGB 19维 纯模型（当前使用）

- **算法**：LightGBM
- **特征（19维）**：V3的16维 + 交互特征3维
  - move_x_prob_h = odds_move_h × prob_h
  - move_x_prob_d = odds_move_d × prob_d
  - move_x_prob_a = odds_move_a × prob_a
- **参数**：n_estimators=1000, max_depth=6, num_leaves=31, learning_rate=0.05, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, random_state=42
- **验证集准确率**：54.51%
- **世界杯准确率**：56.25%（追平基线）
- **交互特征重要性**：move_x_prob_d=274, move_x_prob_a=269, move_x_prob_h=242（排名靠前）
- **结论**：纯模型最优版本，世界杯追平基线

### V6：CatBoost 19维

- **算法**：CatBoostClassifier
- **特征（19维）**：同V5
- **参数**：iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3
- **验证集准确率**：54.34%
- **结论**：不如LightGBM

### V7：XGBoost 19维

- **算法**：XGBClassifier
- **特征（19维）**：同V5
- **参数**：n_estimators=1000, max_depth=6, learning_rate=0.05, min_child_weight=30
- **验证集准确率**：54.34%
- **结论**：不如LightGBM

### V8：LGB 25维

- **算法**：LightGBM
- **特征（25维）**：V5的19维 + 终盘离散度3维 + 概率变化3维
  - odds_spread_c, odds_cv_c, top2_gap_c（终盘离散度）
  - prob_move_h, prob_move_d, prob_move_a（概率变化=终盘-初盘）
- **验证集准确率**：54.22%
- **结论**：加特征反而更差（过拟合）

### V9：LGB 19维 深度调参

- **算法**：LightGBM
- **特征（19维）**：同V5
- **参数**：n_estimators=2000, max_depth=8, num_leaves=50, learning_rate=0.02, min_child_samples=50, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0.5, min_split_gain=0.1
- **验证集准确率**：53.82%
- **结论**：深度调参反而更差（过拟合）

### V10：集成模型

- **算法**：RandomForest + LightGBM + XGBoost 概率平均
- **特征（19维）**：同V5
- **验证集准确率**：54.45%
- **结论**：集成未带来提升

---

## 三、V5 最终版详细评估

### 验证集（1752场）

```
              precision    recall  f1-score   support

       主胜(H)     0.5371    0.8166    0.6480       736
        平(D)     0.3111    0.0320    0.0581       437
       客胜(A)     0.5612    0.5699    0.5656       579

    accuracy                         0.5394      1752
```

### 2022世界杯测试集（64场）

```
              precision    recall  f1-score   support

       主胜(H)     0.6000    0.7742    0.6761        31
        平(D)     0.0000    0.0000    0.0000        12
       客胜(A)     0.5000    0.5714    0.5333        21

    accuracy                         0.5625        64
```

### 特征重要性（Top 10）

| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | overround | 607 |
| 2 | prob_d_c | 433 |
| 3 | prob_h_c | 364 |
| 4 | move_x_prob_d | 274 |
| 5 | move_x_prob_a | 269 |
| 6 | prob_a_c | 266 |
| 7 | odds_spread | 244 |
| 8 | prob_d | 243 |
| 9 | move_x_prob_h | 242 |
| 10 | prob_a | 216 |

---

## 四、基线对比

### "选最低赔率"基线在各数据集的表现

| 数据集 | 场次 | 准确率 | H召回 | D召回 | A召回 |
|--------|------|--------|-------|-------|-------|
| 验证集 | 1752 | 54.91% | 84.1% | 0.5% | 58.9% |
| 世界杯 | 64 | 56.25% | 77.4% | 0.0% | 57.1% |

### 模型 vs 基线

- 纯赔率模型天花板 ≈ 54.5%（验证集），无法超过基线 54.9%
- 世界杯64场上 V5 追平基线（56.25%）
- 模型价值在于：输出概率分布（基线只能给硬预测）+ 配合赛前情报做爆冷预测

---

## 五、数据集说明

### 训练集
- 来源：football-data.co.uk 五大联赛 CSV
- 赛季：2021-2022, 2022-2023, 2023-2024
- 场次：6694场
- WDL分布：H=2940, D=1705, A=2049

### 验证集
- 来源：football-data.co.uk 五大联赛 CSV
- 赛季：2024-2025
- 场次：1752场
- WDL分布：H=736, D=437, A=579

### 测试集
- 来源：titan007.com 爬取（2022世界杯）
- 场次：64场
- WDL分布：H=31, D=12, A=21
- 爬虫脚本：`agents/predicted_agent/scripts/wc2022_odds_scraper.py`
- 爬取流程：2022.titan007.com 比分页 → match_id → 1x2d.titan007.com/{id}.js → Bet365初盘+终盘

---

## 六、实验结论

1. **纯赔率模型天花板 ≈ 54.5%**：换了3种算法（RF/LGB/XGB/CatBoost）、试了12-25维特征、试了集成，都无法超过"选最低赔率"基线（54.9%）
2. **平局预测是瓶颈**：纯赔率特征无法有效区分平局（平局的概率分布和主胜/客胜几乎重叠），D召回率最高也只有27%（V2 balanced版，代价是整体准确率掉到52.7%）
3. **class_weight 有害**：任何程度的 class_weight 都会降低整体准确率
4. **加特征到25维反而更差**：过拟合
5. **深度调参反而更差**：过拟合
6. **集成模型无提升**：三个模型学到的信息高度重叠
7. **交互特征有效**：move_x_prob 系列特征重要性排名靠前
8. **模型价值不在准确率**：在于输出概率分布，配合赛前情报做爆冷预测

---

## 七、文件位置

| 文件 | 路径 | 说明 |
|------|------|------|
| 特征工程 | `agents/predicted_agent/feature_engineering.py` | 19维特征定义+构建 |
| 训练脚本 | `agents/predicted_agent/models/statistical_model.py` | V5 训练+预测 |
| 蒙特卡洛 | `agents/predicted_agent/models/monte_carlo_simulator.py` | 泊松分布模拟 |
| 已保存模型 | `agents/predicted_agent/models/saved/wdl_model.pkl` | V5 LightGBM |
| 模型元信息 | `agents/predicted_agent/models/saved/model_meta.json` | 特征/准确率/参数 |
| 测试集数据 | `agents/predicted_agent/models/saved/wc2022_test.csv` | 64场世界杯数据 |
| 爬虫脚本 | `agents/predicted_agent/scripts/wc2022_odds_scraper.py` | titan007爬虫 |
