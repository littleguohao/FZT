# Task 7: 回测模块实现计划

## 概述
实现完整的回测系统，用于验证FZT选股策略和模型优化策略的性能。

## 目标
1. 实现策略收益率计算和基准对比
2. 计算风险指标（夏普比率、最大回撤等）
3. 实现交易信号生成和持仓管理
4. 创建回测结果可视化和报告系统

## 核心功能

### 1. 回测引擎 (`BacktestEngine`)
- **策略执行**: 模拟交易流程
- **信号生成**: 基于FZT公式或模型预测
- **持仓管理**: 模拟买入、持有、卖出
- **交易成本**: 考虑手续费、滑点等

### 2. 绩效评估 (`PerformanceEvaluator`)
- **收益率计算**: 策略收益率 vs 基准收益率
- **风险指标**:
  - 年化收益率
  - 年化波动率
  - 夏普比率
  - 最大回撤
  - 胜率
  - 盈亏比
- **基准对比**: 与沪深300等指数对比

### 3. 可视化系统 (`VisualizationSystem`)
- **净值曲线**: 策略净值 vs 基准净值
- **回撤曲线**: 最大回撤可视化
- **月度收益**: 月度收益热力图
- **持仓分析**: 持仓分布和换手率

### 4. 报告生成 (`ReportGenerator`)
- **绩效报告**: 完整的策略绩效报告
- **交易记录**: 详细的交易记录
- **风险报告**: 风险指标分析
- **优化建议**: 基于回测结果的优化建议

## 技术实现

### 文件结构
```
FZT_Quant/src/backtest/
├── __init__.py
├── backtest_engine.py      # 回测引擎
├── performance_evaluator.py # 绩效评估
├── visualization.py        # 可视化
├── report_generator.py     # 报告生成
└── utils.py               # 工具函数
```

### 核心类设计

#### 1. `BacktestEngine`
```python
class BacktestEngine:
    def __init__(self, initial_capital=1000000, commission_rate=0.0003):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        
    def run(self, signals, prices, benchmark_prices=None):
        """运行回测"""
        pass
    
    def generate_signals(self, predictions, threshold=0.5):
        """生成交易信号"""
        pass
    
    def execute_trades(self, signals, prices):
        """执行交易"""
        pass
    
    def calculate_portfolio_value(self):
        """计算组合价值"""
        pass
```

#### 2. `PerformanceEvaluator`
```python
class PerformanceEvaluator:
    def __init__(self, portfolio_returns, benchmark_returns=None):
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        
    def calculate_annual_return(self):
        """计算年化收益率"""
        pass
    
    def calculate_annual_volatility(self):
        """计算年化波动率"""
        pass
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.03):
        """计算夏普比率"""
        pass
    
    def calculate_max_drawdown(self):
        """计算最大回撤"""
        pass
    
    def calculate_win_rate(self):
        """计算胜率"""
        pass
    
    def calculate_profit_loss_ratio(self):
        """计算盈亏比"""
        pass
```

#### 3. `StrategyVisualizer`
```python
class StrategyVisualizer:
    def __init__(self, backtest_results):
        self.results = backtest_results
        
    def plot_equity_curve(self):
        """绘制净值曲线"""
        pass
    
    def plot_drawdown_curve(self):
        """绘制回撤曲线"""
        pass
    
    def plot_monthly_returns(self):
        """绘制月度收益热力图"""
        pass
    
    def plot_position_analysis(self):
        """绘制持仓分析"""
        pass
```

## 实施步骤

### 阶段1: 基础回测引擎 (2小时)
1. 创建回测引擎类
2. 实现信号生成逻辑
3. 实现交易执行逻辑
4. 实现组合价值计算

### 阶段2: 绩效评估系统 (2小时)
1. 创建绩效评估类
2. 实现所有风险指标计算
3. 实现基准对比功能
4. 添加交易统计功能

### 阶段3: 可视化系统 (1.5小时)
1. 创建可视化类
2. 实现净值曲线绘制
3. 实现回撤曲线绘制
4. 实现月度收益热力图

### 阶段4: 报告生成系统 (1.5小时)
1. 创建报告生成类
2. 实现HTML/PDF报告生成
3. 实现交易记录导出
4. 实现优化建议生成

### 阶段5: 集成测试 (1小时)
1. 集成所有模块
2. 运行端到端测试
3. 验证回测结果
4. 优化性能

## 预期输出

### 1. 回测结果
- 策略净值曲线
- 风险指标报告
- 交易记录明细

### 2. 可视化图表
- 净值对比图
- 回撤曲线图
- 月度收益热力图
- 持仓分布图

### 3. 分析报告
- 策略绩效报告
- 风险分析报告
- 优化建议报告

## 成功标准

1. ✅ 回测引擎能正确模拟交易流程
2. ✅ 绩效评估指标计算准确
3. ✅ 可视化图表清晰易懂
4. ✅ 报告生成完整详细
5. ✅ 端到端测试通过

## 时间安排
- **总时间**: 8小时
- **开始时间**: 2026年3月1日 20:30
- **预计完成**: 2026年3月1日 23:30

## 依赖关系
- 已完成Task 1-6的所有模块
- 需要Qlib数据和本地CSV数据
- 需要模型训练和评估模块的输出

## 风险与缓解
- **风险1**: 回测逻辑复杂，容易出错
  - **缓解**: 分阶段实现，每个阶段充分测试
- **风险2**: 性能指标计算不准确
  - **缓解**: 使用成熟的金融库作为参考
- **风险3**: 可视化效果不佳
  - **缓解**: 使用专业金融可视化库

## 验收标准
1. 能对原始FZT策略进行回测
2. 能对模型优化策略进行回测
3. 能生成完整的绩效报告
4. 能生成可视化图表
5. 代码质量高，有完整文档

---

**创建时间**: 2026年3月1日 20:25
**创建者**: MC (FZT项目组)