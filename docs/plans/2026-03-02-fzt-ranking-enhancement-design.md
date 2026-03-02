# FZT排序增强策略设计文档

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 每日从A股市场选出未来一天涨幅最大的前5只股票，通过FZT初选 + LightGBM LambdaRank排序实现Alpha增强。

**Architecture:** QLib集成框架 + FZT因子封装 + 排序学习模型 + 专业回测系统。采用"市值回归 + 行业标准化"中性化方案，实现风格暴露控制。

**Tech Stack:** Python, QLib, LightGBM (LambdaRank), Pandas, NumPy

---

## 一、项目概述

### 1.1 核心目标
- **业务目标**: 每日选出未来一天涨幅最大的前5只股票
- **技术目标**: 实现FZT公式的机器学习增强，提升排序质量
- **评估目标**: NDCG@5显著优于原始FZT，夏普>1.0，胜率>60%

### 1.2 设计原则
1. **排序优先**: 使用LambdaRank直接优化Top 5排序，而非二分类
2. **信息保留**: 保留FZT所有选股结果，不限制候选池大小
3. **风险控制**: 完整交易成本模拟 + 流动性约束 + 风格中性化
4. **稳健评估**: 三层评估体系，关注成本后真实收益

## 二、系统架构

### 2.1 整体架构
```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 数据层  │───▶│ 特征层   │───▶│ 模型层   │───▶│ 回测层   │───▶│ 评估层   │
│ QLib    │    │ FZT算子  │    │LambdaRank│    │成本模型  │    │NDCG@5    │
│ 数据    │    │ Alpha158 │    │ 排序模型 │    │风险控制  │    │夏普比率  │
└─────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### 2.2 模块设计
| 模块 | 功能 | 关键技术 |
|------|------|----------|
| **QLib FZT算子** | 封装FZT公式为QLib算子 | 无未来函数计算 |
| **特征工程模块** | FZT家族 + 精选Alpha158 | 中性化处理 |
| **排序学习模块** | LightGBM LambdaRank训练 | NDCG@5评估 |
| **回测引擎模块** | QLib回测 + 自定义扩展 | 交易成本模型 |
| **评估分析模块** | 三层评估指标体系 | 特征重要性分析 |

## 三、技术实现细节

### 3.1 QLib FZT算子
```python
class FZTAlpha(Alpha):
    """FZT公式QLib算子"""
    def __init__(self):
        super().__init__()
        # 复用现有FZTBrickFormula
        self.fzt_calculator = FZTBrickFormula()
    
    def __call__(self, df):
        """
        无未来函数计算FZT值
        输入: OHLCV数据
        输出: FZT原始值、选股条件、衍生特征
        """
        # 确保只使用截止当日数据
        fzt_values = self.fzt_calculator.calculate(df)
        return fzt_values
```

### 3.2 特征工程流程
#### 3.2.1 特征集构成
```python
特征集 = {
    # 模块A: FZT因子家族 (核心)
    "fzt_raw": "原始FZT值",
    "fzt_rank": "横截面排名",
    "fzt_slope_5d": "5日斜率",
    "fzt_vs_market": "相对市场强度",
    
    # 模块B: 精选Alpha158 (互补)
    "vwap_related": "成交量加权均价衍生",
    "volume_features": "成交量衍生特征",
    "short_term_reversal": "短期反转因子",
    "volatility_features": "波动率特征",
    "bias_features": "乖离率特征",
    
    # 模块C: 市场状态特征 (全局)
    "market_volatility": "全市场波动率",
    "limit_up_ratio": "涨停股比例",
    "market_breadth": "市场宽度"
}
```

#### 3.2.2 中性化处理流程
```python
def neutralize_features(features, market_cap, industry):
    """
    混合中性化方案: 市值回归 + 行业标准化
    """
    # 1. 预处理: 去极值 + 标准化
    features_clean = mad_winsorize(features, n=3)
    features_std = zscore_normalize(features_clean)
    
    # 2. 市值中性化 (回归法)
    # Factor = α + β * log(MarketCap) + ε
    resid_mkt = features_std - linear_regression(
        np.log(market_cap), features_std
    )
    
    # 3. 行业中性化 (分组标准化)
    # 行业内: 均值为0，标准差为1
    result = resid_mkt.groupby(industry).transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    return result
```

#### 3.2.3 特征筛选标准
```yaml
筛选标准:
  - 单因子有效性: IC > 0.02 且 IR > 0.5
  - 冗余性控制: 与FZT相关性 < 0.6
  - 稳定性要求: 分行业IC标准差 < 0.1
  - 逻辑合理性: 符合金融逻辑，可解释
```

### 3.3 排序模型配置
```python
model_config = {
    # 基础参数
    "objective": "lambdarank",
    "metric": ["ndcg@5", "map@5"],
    "num_leaves": 64,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "min_child_samples": 20,
    "early_stopping_rounds": 50,
    
    # 排序特定参数
    "lambdarank_truncation_level": 5,  # 聚焦Top 5
    "lambdarank_norm": True,
    "label_gain": [0, 1, 3, 7, 15, 31],  # 增益函数
    
    # 训练控制
    "num_boost_round": 1000,
    "verbose_eval": 50,
    "seed": 42
}
```

### 3.4 回测策略实现
```python
class FZTEnhancedStrategy(TopkDropoutStrategy):
    """FZT增强策略 (扩展QLib基础策略)"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 交易成本参数
        self.commission = 0.0003    # 佣金万分之三
        self.stamp_tax = 0.001      # 印花税千分之一 (卖出)
        self.slippage = 0.001       # 基础滑点0.1%
        
        # 流动性控制
        self.min_turnover_ratio = 0.2  # 成交额后20%门槛
        self.min_market_cap_ratio = 0.3  # 市值后30%门槛
        
        # 风险控制
        self.max_industry_weight = 0.3  # 单一行业最大权重30%
    
    def generate_trade_decision(self, execute_info):
        """
        生成交易决策 (扩展版本)
        """
        # 1. 基础交易决策
        trade_decision = super().generate_trade_decision(execute_info)
        
        # 2. 流动性过滤
        trade_decision = self.filter_by_liquidity(trade_decision)
        
        # 3. 可交易性检查 (涨跌停、停牌)
        trade_decision = self.check_tradability(trade_decision)
        
        # 4. 行业中性化约束
        trade_decision = self.apply_industry_neutral(trade_decision)
        
        # 5. 计算交易成本
        cost = self.calculate_trading_cost(trade_decision)
        trade_decision.cost = cost
        
        return trade_decision
    
    def filter_by_liquidity(self, decision):
        """流动性过滤"""
        # 剔除成交额后20%的股票
        # 剔除市值后30%的股票
        # 剔除ST股票
        # 剔除停牌股票
        return filtered_decision
```

### 3.5 评估指标体系
#### 3.5.1 核心指标 (决策依据)
```python
核心指标 = {
    "NDCG@5": "排序质量核心指标 (首要)",
    "年化收益率": "绝对收益表现",
    "夏普比率": "风险调整后收益",
    "胜率": "预测稳定性 (目标>60%)"
}
```

#### 3.5.2 风险指标 (回撤控制)
```python
风险指标 = {
    "最大回撤": "极端风险控制 (目标<20%)",
    "换手率": "交易成本影响",
    "交易成本后收益": "真实收益",
    "单日最大亏损": "日内风险"
}
```

#### 3.5.3 诊断指标 (模型理解)
```python
诊断指标 = {
    "特征重要性": "模型逻辑理解 (FZT vs Alpha158)",
    "IC/ICIR": "预测相关性",
    "行业分布": "风格暴露分析",
    "滚动NDCG@5": "稳定性检验"
}
```

## 四、数据集划分

### 4.1 基础划分方案
```yaml
时间范围:
  训练集: 2005-01-01 到 2017-12-31
  验证集: 2018-01-01 到 2019-12-31 (早停调参)
  测试集: 2020-01-01 到 2020-12-31 (最终评估)
```

### 4.2 优化方案 (推荐)
```python
# 滚动训练验证
滚动配置 = {
    "窗口长度": "3年",
    "滚动步长": "1年",
    "验证方式": "用下一年作为验证集",
    "优势": "更接近实盘，检验模型稳定性"
}
```

### 4.3 数据预处理
```python
预处理流程 = [
    "缺失值处理: 前向填充",
    "异常值处理: MAD去极值 (3倍中位数绝对偏差)",
    "标准化处理: 横截面Z-score",
    "时间对齐: 确保日期一致"
]
```

## 五、实施计划

### 5.1 阶段1: 基础框架 (今天)
**目标**: 建立核心数据流和特征工程
```yaml
任务:
  - 创建QLib FZT算子
  - 实现特征中性化处理
  - 构造排序学习标签
  - 搭建基础训练流水线
```

### 5.2 阶段2: 模型训练 (明天)
**目标**: 实现排序模型训练和评估
```yaml
任务:
  - 实现LambdaRank训练
  - 集成NDCG@5评估
  - 特征重要性分析
  - 参数敏感性测试
```

### 5.3 阶段3: 回测评估 (后天)
**目标**: 完整回测和风险评估
```yaml
任务:
  - 实现成本调整回测
  - 完整评估指标计算
  - 风险控制模块
  - 流动性约束实现
```

### 5.4 阶段4: 优化迭代 (后续)
**目标**: 持续优化和验证
```yaml
任务:
  - 特征工程优化
  - 参数调优
  - 滚动训练验证
  - 样本外测试
```

## 六、成功标准

### 6.1 相对提升标准 (核心)
```yaml
排序提升: "NDCG@5显著优于FZT原始排序"
收益提升: "年化收益率 > FZT原始策略 + 5%"
稳定性: "滚动NDCG@5标准差 < 0.1"
```

### 6.2 绝对门槛标准 (实战)
```yaml
胜率门槛: "胜率 > 60%"
夏普门槛: "夏普比率 > 1.0"
回撤控制: "最大回撤 < 20%"
成本稳健: "扣除0.3%滑点后收益仍为正"
```

### 6.3 模型理解标准
```yaml
特征贡献: "FZT和Alpha158特征都有显著贡献"
逻辑合理: "特征重要性符合金融逻辑"
稳定性: "特征重要性排名相对稳定"
```

## 七、风险与应对

### 7.1 技术风险
| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 过拟合风险 | 样本外表现差 | 早停机制 + 验证集监控 + 滚动训练 |
| 未来函数风险 | 策略不可实盘 | 严格时间序列分割 + 无未来函数计算 |
| 计算复杂度 | 训练时间过长 | 分批处理 + 特征筛选 + 使用QLib优化 |

### 7.2 业务风险
| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 交易成本 | 收益被侵蚀 | 精确成本模拟 + 安全边际设置 |
| 流动性风险 | 无法成交 | 成交量门槛 + 市值门槛 + 顺延机制 |
| 市场风险 | 策略失效 | 分市场状态测试 + 监控策略稳定性 |
| 容量限制 | 资金规模受限 | 换手率分析 + 冲击成本估计 |

### 7.3 监控机制
```python
监控机制 = {
    "实时监控": ["NDCG@5", "特征重要性", "换手率"],
    "定期检查": ["滚动收益", "最大回撤", "胜率变化"],
    "异常警报": ["单日亏损超5%", "连续5日负收益", "特征重要性突变"]
}
```

## 八、文件结构

```
FZT_Quant/
├── src/
│   ├── qlib_operators/          # QLib算子
│   │   ├── fzt_alpha.py         # FZT算子
│   │   └── neutralization.py    # 中性化算子
│   ├── feature_engineering/     # 特征工程
│   │   ├── feature_pipeline.py  # 特征流水线
│   │   ├── alpha158_selector.py # Alpha158精选
│   │   └── neutralization.py    # 中性化处理
│   ├── ranking_model/           # 排序模型
│   │   ├── lambdarank_trainer.py # LambdaRank训练
│   │   ├── ndcg_evaluator.py    # NDCG评估
│   │   └── feature_importance.py # 特征重要性
│   ├── backtest/                # 回测引擎
│   │   ├── enhanced_strategy.py # 增强策略
│   │   ├── cost_model.py        # 成本模型
│   │   └── risk_controller.py   # 风险控制
│   └── evaluation/              # 评估分析
│       ├── metrics_calculator.py # 指标计算
│       ├── report_generator.py  # 报告生成
│       └── visualization.py     # 可视化
├── config/
│   ├── ranking_config.yaml      # 排序模型配置
│   ├── feature_config.yaml      # 特征工程配置
│   ├── backtest_config.yaml     # 回测配置
│   └── evaluation_config.yaml   # 评估配置
├── scripts/
│   ├── train_ranking_model.py   # 训练脚本
│   ├── run_backtest.py          # 回测脚本
│   └── evaluate_strategy.py     # 评估脚本
├── results/
│   ├── models/                  # 训练好的模型
│   ├── features/                # 特征数据
│   ├── backtest_results/        # 回测结果
│   └── evaluation_reports/      # 评估报告
└── docs/
    └── plans/                   # 设计文档
        └── 2026-03-02-fzt-ranking-enhancement-design.md

## 九、下一步行动

### 9.1 立即行动
1. **提交设计文档**到git
2. **使用writing-plans技能**创建详细实施计划
3. **开始阶段1实施** (基础框架)

### 9.2 实施选择
提供两种执行方式:
1. **Subagent-Driven (本会话)**: 我分派新子代理执行每个任务，任务间进行代码审查
2. **并行会话 (独立)**: 在新会话中使用executing-plans，批量执行并设置检查点

**推荐**: Subagent-Driven方式，因为:
- 同一会话，无需上下文切换
- 每个任务使用新子代理 (无上下文污染)
- 两阶段审查: 规范符合性审查 + 代码质量审查
- 更快迭代 (任务间无需人工介入)

---

*设计文档完成时间: 2026年3月2日 13:20*
*设计状态: ✅ 已完成，等待用户确认*