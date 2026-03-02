# 成本模型模块文档

## 概述

成本模型模块 (`src/backtest/cost_model.py`) 提供了A股市场交易成本的精确计算功能。该模块支持多种成本类型的计算，包括佣金、印花税、过户费、滑点成本和冲击成本，并支持动态成本调整和成本优化建议。

## 主要功能

### 1. 成本类型计算
- **佣金计算**: 支持比例佣金和最低佣金
- **印花税计算**: 仅卖出时收取（A股标准）
- **过户费计算**: 沪市和深市不同标准
- **滑点成本计算**: 固定滑点、动态滑点、买卖价差滑点
- **冲击成本计算**: 线性模型、流动性模型、市场冲击模型

### 2. 动态成本调整
- **基于流动性**: 流动性差的股票成本更高
- **基于订单规模**: 大额订单成本更高
- **基于市场状态**: 波动大时成本更高
- **基于交易频率**: 高频交易成本更高

### 3. 成本优化
- 分析交易历史，识别高成本交易
- 提供具体的优化建议
- 估算潜在的成本节省

## 快速开始

### 安装依赖
```bash
pip install numpy pandas pyyaml
```

### 基本使用
```python
from src.backtest.cost_model import CostModel

# 创建成本模型（使用默认配置）
cost_model = CostModel()

# 计算单笔交易成本
total_cost = cost_model.calculate_total_cost(
    trade_amount=10000.0,      # 交易金额
    side='sell',               # 交易方向：'buy' 或 'sell'
    exchange='SH',             # 交易所：'SH' 或 'SZ'
    liquidity_score=0.6,       # 流动性分数 (0-1)
    order_size_ratio=0.01      # 订单规模占日均成交比例
)

print(f"总成本: {total_cost:.2f}元")

# 获取成本分解
breakdown = cost_model.calculate_cost_breakdown(
    trade_amount=10000.0,
    side='sell',
    exchange='SH',
    liquidity_score=0.6,
    order_size_ratio=0.01
)

for cost_type, amount in breakdown.items():
    print(f"{cost_type}: {amount:.2f}元")
```

### 批量计算
```python
import numpy as np

# 批量交易数据
trade_amounts = np.array([10000.0, 20000.0, 30000.0])
sides = np.array(['sell', 'buy', 'sell'])
exchanges = np.array(['SH', 'SZ', 'SH'])

# 批量计算成本
total_costs = cost_model.calculate_total_cost_batch(
    trade_amounts=trade_amounts,
    sides=sides,
    exchanges=exchanges
)

print(f"批量成本: {total_costs}")
```

### 自定义配置
```python
# 自定义成本配置
custom_config = {
    'commission': {
        'rate': 0.00025,  # 0.025% 佣金
        'min': 3.0        # 最低3元
    },
    'stamp_tax': {
        'rate': 0.0008,   # 0.08% 印花税
        'sell_only': True
    },
    'transfer_fee': {
        'rate': 0.000015, # 0.0015% 过户费
        'sh_only': True
    }
}

# 使用自定义配置
custom_model = CostModel(config=custom_config)
```

### 从配置文件加载
```python
# 从YAML配置文件加载
cost_model = CostModel(config='config/backtest_config.yaml')
```

## 配置说明

### 默认配置
```yaml
commission:
  rate: 0.0003    # 0.03% 比例佣金
  min: 5.0        # 最低5元

stamp_tax:
  rate: 0.001     # 0.1% 印花税
  sell_only: true # 仅卖出时收取

transfer_fee:
  rate: 0.00002   # 0.002% 过户费
  sh_only: true   # 仅沪市收取

slippage:
  rate: 0.001     # 0.1% 固定滑点
  dynamic: true   # 启用动态滑点

impact_cost:
  enabled: true   # 启用冲击成本
  liquidity_based: true  # 基于流动性的冲击成本
```

### 支持从backtest_config.yaml加载
成本模型可以自动从项目的 `config/backtest_config.yaml` 文件中提取 `transaction_cost` 部分的配置。

## 高级功能

### 动态成本调整示例
```python
# 测试不同流动性下的成本
high_liquidity_cost = cost_model.calculate_total_cost(
    trade_amount=10000.0,
    side='sell',
    exchange='SH',
    liquidity_score=0.9,  # 高流动性
    order_size_ratio=0.01
)

low_liquidity_cost = cost_model.calculate_total_cost(
    trade_amount=10000.0,
    side='sell',
    exchange='SH',
    liquidity_score=0.2,  # 低流动性
    order_size_ratio=0.01
)

print(f"高流动性成本: {high_liquidity_cost:.2f}元")
print(f"低流动性成本: {low_liquidity_cost:.2f}元")
print(f"成本差异: {low_liquidity_cost - high_liquidity_cost:.2f}元")
```

### 成本优化建议
```python
# 模拟交易历史
trade_history = [
    {
        'trade_amount': 10000.0,
        'side': 'sell',
        'exchange': 'SH',
        'liquidity_score': 0.3,
        'order_size_ratio': 0.02,
        'market_volatility': 0.025,
        'cost': 45.0,
        'cost_ratio': 0.0045
    },
    # ... 更多交易记录
]

# 获取优化建议
suggestions = cost_model.get_cost_optimization_suggestions(trade_history)

print(f"总交易成本: {suggestions['total_cost']:.2f}元")
print(f"平均成本率: {suggestions['avg_cost_ratio']:.4%}")

for suggestion in suggestions['suggestions']:
    print(f"建议: {suggestion['description']}")
    print(f"潜在节省: {suggestion['potential_saving']:.2f}元")
```

### 估算最优订单规模
```python
# 估算最优订单规模
daily_volume = 1000000.0  # 日均成交量100万元
optimal_size = cost_model.estimate_optimal_order_size(
    daily_volume=daily_volume,
    liquidity_score=0.6,
    target_cost_ratio=0.001  # 目标成本率0.1%
)

print(f"最优订单规模: {optimal_size:.2f}元 ({optimal_size/daily_volume:.2%} of daily volume)")
```

## 测试

运行所有测试：
```bash
python -m pytest tests/test_cost_model.py -v
```

运行特定测试：
```bash
python -m pytest tests/test_cost_model.py::TestCostModel::test_total_cost_calculation -v
```

## 示例

查看完整示例：
```bash
python examples/cost_model_example.py
```

示例包含：
1. 单笔交易成本计算
2. 批量交易成本计算
3. 动态成本调整演示
4. 成本优化建议
5. 自定义配置使用

## 集成到回测系统

成本模型可以轻松集成到现有的回测系统中：

```python
from src.backtest.cost_model import CostModel
from src.backtest.backtest_engine import BacktestEngine

class EnhancedBacktestEngine(BacktestEngine):
    def __init__(self, config):
        super().__init__(config)
        self.cost_model = CostModel(config=config)
    
    def calculate_trade_cost(self, trade):
        """计算交易成本"""
        return self.cost_model.calculate_total_cost(
            trade_amount=trade.amount,
            side=trade.side,
            exchange=trade.exchange,
            liquidity_score=trade.liquidity_score,
            order_size_ratio=trade.order_size_ratio
        )
```

## 注意事项

1. **A股市场规则**: 成本模型基于A股市场的实际交易规则实现
2. **流动性分数**: 流动性分数应在0-1之间，1表示流动性最好
3. **订单规模比例**: 订单规模占日均成交量的比例，建议不超过5%
4. **市场波动率**: 使用历史波动率或隐含波动率，单位是百分比（如0.02表示2%）
5. **成本记录**: 成本模型会自动记录交易历史，用于成本优化分析

## 性能优化

- 批量计算使用NumPy向量化操作，性能较高
- 单笔计算适合实时交易场景
- 交易历史记录有长度限制（默认1000笔），避免内存泄漏

## 版本历史

- v1.0.0 (2026-03-02): 初始版本，实现完整的成本模型功能

## 作者

FZT项目组

## 许可证

本项目采用MIT许可证。