# 风险控制器模块文档

## 概述

风险控制器是FZT排序增强策略实施计划的第十一阶段任务，用于监控和控制投资组合的各种风险。该模块实现了全面的风险管理功能，包括风险监控、指标计算、预警机制和自动调整。

## 功能特性

### 1. 风险监控
- **市场风险监控**: 波动率、相关性、Beta监控
- **信用风险监控**: ST股票识别、财务指标监控、信用评级监控
- **流动性风险监控**: 成交额监控、买卖价差监控
- **操作风险监控**: 仓位集中度、行业集中度、风格暴露监控

### 2. 风险指标计算
- Value at Risk (VaR)
- Conditional VaR (CVaR)
- 最大回撤 (Max Drawdown)
- 夏普比率 (Sharpe Ratio)
- 索提诺比率 (Sortino Ratio)
- 年化波动率
- 偏度和峰度

### 3. 风险控制措施
- 自动减仓
- 止损止盈
- 风险对冲
- 仓位调整
- ST股票过滤

### 4. 预警机制
- 实时监控
- 阈值预警
- 分级预警 (HIGH/MEDIUM/LOW)
- 预警记录和历史查询

## 配置参数

风险控制器从配置文件中加载以下参数：

```yaml
risk:
  market:
    max_volatility: 0.3      # 最大波动率30%
    min_correlation: 0.3     # 最小相关性
    max_beta: 1.5           # 最大Beta
  credit:
    filter_st: true          # 过滤ST股票
    min_current_ratio: 1.0   # 最小流动比率
    max_debt_ratio: 0.7      # 最大资产负债率
  liquidity:
    min_turnover: 10000000   # 最小成交额1000万
    max_bid_ask_spread: 0.02 # 最大买卖价差2%
  concentration:
    max_single_weight: 0.4   # 单一股票最大权重40%
    max_industry_weight: 0.3  # 单一行业最大权重30%
    max_style_exposure: 0.5   # 单一风格最大暴露50%
  stop_loss:
    enabled: true
    threshold: -0.1          # 止损阈值-10%
    trailing: true           # 移动止损
```

## 使用方法

### 1. 基本使用

```python
from backtest.risk_controller import RiskController

# 初始化风险控制器
config = {
    'risk': {
        # ... 风险配置
    }
}
controller = RiskController(config)

# 执行风险评估
risk_report = controller.assess_risk(
    portfolio=portfolio_data,
    market_data=market_data,
    volume_data=volume_data,
    financial_data=financial_data,
    risk_factors=risk_factors
)

# 应用风险控制
adjusted_portfolio = controller.apply_risk_controls(
    portfolio_data,
    risk_report['recommended_actions']
)

# 获取风险摘要
summary = controller.get_risk_summary()
```

### 2. 数据格式要求

#### 投资组合数据
```python
portfolio = {
    'positions': {
        '000001.SZ': {
            'weight': 0.15,      # 权重
            'shares': 1000,      # 股数
            'cost': 10.0,        # 成本价
            'market_value': 150000  # 市值
        },
        # ... 其他股票
    },
    'total_value': 1000000.0,    # 总市值
    'cash': 50000.0,            # 现金
    'pnl': 50000.0              # 盈亏
}
```

#### 市场数据
- `market_data`: pandas DataFrame，索引为日期，列为股票代码，值为价格
- `volume_data`: pandas DataFrame，索引为日期，列为股票代码，值为成交量

#### 财务数据
```python
financial_data = {
    '000001.SZ': {
        'is_st': False,          # 是否为ST股票
        'current_ratio': 1.5,    # 流动比率
        'debt_ratio': 0.4,       # 资产负债率
        'credit_rating': 'AA'    # 信用评级
    },
    # ... 其他股票
}
```

#### 风险因子数据
```python
risk_factors = {
    'industry': {
        '000001.SZ': '金融',
        '000002.SZ': '房地产'
    },
    'style': {
        '000001.SZ': {'value': 0.3, 'growth': 0.7},
        '000002.SZ': {'value': 0.8, 'growth': 0.2}
    }
}
```

### 3. 风险报告

风险评估返回完整的风险报告，包含：

```python
{
    'timestamp': datetime,           # 评估时间
    'market_risk': {},               # 市场风险评估结果
    'credit_risk': {},               # 信用风险评估结果
    'liquidity_risk': {},            # 流动性风险评估结果
    'concentration_risk': {},        # 集中度风险评估结果
    'risk_metrics': {},              # 风险指标计算结果
    'warnings': [],                  # 风险预警列表
    'recommended_actions': []        # 建议的风险控制动作
}
```

## 示例

参见 `examples/risk_controller_demo.py` 获取完整的使用示例。

## 测试

运行测试确保功能正常：

```bash
python -m pytest tests/test_risk_controller.py -v
```

## 与回测系统集成

风险控制器可以与回测引擎集成，在每次调仓时自动执行风险评估和控制：

1. **事前风险控制**: 在调仓前评估风险，过滤高风险股票
2. **事中风险监控**: 在持仓期间持续监控风险指标
3. **事后风险分析**: 生成风险报告，优化风险参数

## 扩展性

风险控制器设计为可扩展的，可以通过以下方式扩展功能：

1. **添加新的风险类型**: 继承RiskController类，实现新的风险评估方法
2. **自定义风险指标**: 添加新的风险指标计算方法
3. **集成外部数据源**: 连接信用评级数据库、风险因子库等
4. **机器学习风险模型**: 集成机器学习模型进行风险预测

## 性能考虑

- 使用向量化计算提高性能
- 支持大数据集的分块处理
- 提供缓存机制减少重复计算
- 支持并行计算加速风险评估

## 注意事项

1. **数据质量**: 风险控制的效果依赖于输入数据的质量
2. **参数调优**: 风险阈值需要根据市场环境和策略特点进行调整
3. **过度控制**: 避免过度风险控制影响策略收益
4. **实时性**: 对于高频交易，需要考虑风险评估的实时性要求