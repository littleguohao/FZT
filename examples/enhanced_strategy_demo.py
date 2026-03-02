#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强回测策略演示

演示FZTEnhancedStrategy的使用方法，包括：
1. 策略初始化
2. 选股逻辑
3. 投资组合构建
4. 交易成本计算
5. 回测运行

作者: FZT项目组
创建日期: 2026-03-02
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.enhanced_strategy import FZTEnhancedStrategy, create_sample_data


def demo_basic_functionality():
    """演示基本功能"""
    print("=" * 60)
    print("FZT增强回测策略演示")
    print("=" * 60)
    
    # 1. 创建策略实例
    print("\n1. 创建策略实例")
    print("-" * 40)
    
    config = {
        'top_k': 5,
        'hold_days': 1,
        'commission': 0.0003,
        'stamp_tax': 0.001,
        'slippage': 0.001,
        'min_turnover_ratio': 0.2,
        'min_market_cap_ratio': 0.3,
        'max_industry_weight': 0.3,
        'max_single_weight': 0.4,
        'max_drawdown_limit': 0.15,
        'volatility_limit': 0.3
    }
    
    strategy = FZTEnhancedStrategy(config)
    
    # 显示策略摘要
    summary = strategy.get_strategy_summary()
    print(f"策略名称: {summary['strategy_name']}")
    print(f"QLib集成: {'可用' if summary['qlib_integration'] else '不可用'}")
    print("\n策略参数:")
    for key, value in summary['parameters'].items():
        print(f"  {key}: {value}")
    
    print("\n策略功能:")
    for feature in summary['features']:
        print(f"  ✓ {feature}")
    
    return strategy, config


def demo_stock_selection(strategy):
    """演示选股逻辑"""
    print("\n\n2. 演示选股逻辑")
    print("-" * 40)
    
    # 创建示例数据
    n_stocks = 20
    stock_codes = [f'{i:06d}.SZ' for i in range(1, n_stocks + 1)]
    
    # 生成股票分数
    np.random.seed(42)
    stock_scores = pd.Series(
        np.random.randn(n_stocks) * 0.1 + 0.5,
        index=stock_codes
    )
    
    # 生成股票数据
    stock_data = pd.DataFrame({
        'score': stock_scores.values,
        'turnover': np.random.lognormal(mean=15, sigma=1.5, size=n_stocks),
        'market_cap': np.random.lognormal(mean=20, sigma=2.0, size=n_stocks),
        'is_st': np.random.rand(n_stocks) < 0.05,  # 5%概率是ST
        'is_suspended': np.random.rand(n_stocks) < 0.02,  # 2%概率停牌
        'limit_up': np.random.rand(n_stocks) < 0.03,  # 3%概率涨停
        'limit_down': np.random.rand(n_stocks) < 0.03,  # 3%概率跌停
        'industry': np.random.choice(['银行', '科技', '医药', '消费', '能源', '制造'], size=n_stocks)
    }, index=stock_codes)
    
    trade_date = datetime.now()
    
    print(f"股票池大小: {len(stock_scores)}")
    print(f"前5只股票分数:")
    for i, (stock, score) in enumerate(stock_scores.head().items()):
        print(f"  {stock}: {score:.4f}")
    
    # 执行选股
    selected_stocks = strategy.select_stocks(stock_scores, stock_data, trade_date)
    
    print(f"\n选中的股票 ({len(selected_stocks)}/{strategy.top_k}):")
    for stock in selected_stocks:
        score = stock_scores[stock]
        industry = stock_data.loc[stock, 'industry']
        turnover = stock_data.loc[stock, 'turnover']
        print(f"  {stock}: 分数={score:.4f}, 行业={industry}, 成交额={turnover:.2e}")
    
    return stock_scores, stock_data, selected_stocks


def demo_portfolio_construction(strategy, selected_stocks, stock_data):
    """演示投资组合构建"""
    print("\n\n3. 演示投资组合构建")
    print("-" * 40)
    
    available_capital = 1000000  # 100万元
    
    # 构建投资组合
    weights = strategy.construct_portfolio(selected_stocks, available_capital, stock_data)
    
    print("投资组合权重分配:")
    total_weight = 0
    for stock, weight in weights.items():
        amount = weight * available_capital
        industry = stock_data.loc[stock, 'industry']
        print(f"  {stock}: 权重={weight:.2%}, 金额={amount:,.2f}元, 行业={industry}")
        total_weight += weight
    
    print(f"\n总权重: {total_weight:.2%}")
    
    # 检查风险控制
    print("\n风险控制检查:")
    print(f"  个股权重限制: ≤{strategy.max_single_weight:.0%}")
    print(f"  行业权重限制: ≤{strategy.max_industry_weight:.0%}")
    
    # 计算行业权重
    industry_weights = {}
    for stock, weight in weights.items():
        industry = stock_data.loc[stock, 'industry']
        industry_weights[industry] = industry_weights.get(industry, 0) + weight
    
    for industry, weight in industry_weights.items():
        status = "✓" if weight <= strategy.max_industry_weight else "✗"
        print(f"  {industry}: {weight:.2%} {status}")
    
    return weights


def demo_trading_cost(strategy):
    """演示交易成本计算"""
    print("\n\n4. 演示交易成本计算")
    print("-" * 40)
    
    # 测试不同金额的交易成本
    test_amounts = [10000, 50000, 100000, 500000, 1000000]  # 1万到100万
    
    print("买入交易成本:")
    print(f"{'金额(元)':>12} {'佣金':>10} {'印花税':>10} {'滑点':>10} {'冲击成本':>12} {'总成本':>12} {'成本率':>10}")
    print("-" * 80)
    
    for amount in test_amounts:
        costs = strategy.calculate_trading_cost(amount, is_buy=True, stock_liquidity=0.8)
        cost_rate = costs['total'] / amount
        print(f"{amount:12,.0f} {costs['commission']:10.2f} {costs['stamp_tax']:10.2f} "
              f"{costs['slippage']:10.2f} {costs['impact_cost']:12.2f} {costs['total']:12.2f} {cost_rate:10.2%}")
    
    print("\n卖出交易成本:")
    print(f"{'金额(元)':>12} {'佣金':>10} {'印花税':>10} {'滑点':>10} {'冲击成本':>12} {'总成本':>12} {'成本率':>10}")
    print("-" * 80)
    
    for amount in test_amounts:
        costs = strategy.calculate_trading_cost(amount, is_buy=False, stock_liquidity=0.8)
        cost_rate = costs['total'] / amount
        print(f"{amount:12,.0f} {costs['commission']:10.2f} {costs['stamp_tax']:10.2f} "
              f"{costs['slippage']:10.2f} {costs['impact_cost']:12.2f} {costs['total']:12.2f} {cost_rate:10.2%}")


def demo_tradability_check(strategy, stock_data):
    """演示可交易性检查"""
    print("\n\n5. 演示可交易性检查")
    print("-" * 40)
    
    # 测试几只股票
    test_stocks = stock_data.index[:5].tolist()
    
    print("股票可交易性检查:")
    print(f"{'股票代码':>12} {'ST':>5} {'停牌':>5} {'涨停':>5} {'跌停':>5} {'可交易':>8} {'原因':>10}")
    print("-" * 80)
    
    trade_date = datetime.now()
    
    for stock in test_stocks:
        stock_info = stock_data.loc[stock]
        tradable, reason = strategy.check_tradability(stock, stock_data, trade_date)
        
        print(f"{stock:>12} "
              f"{str(stock_info['is_st']):>5} "
              f"{str(stock_info['is_suspended']):>5} "
              f"{str(stock_info['limit_up']):>5} "
              f"{str(stock_info['limit_down']):>5} "
              f"{str(tradable):>8} "
              f"{reason:>10}")


def demo_backtest(strategy):
    """演示回测功能"""
    print("\n\n6. 演示回测功能")
    print("-" * 40)
    
    # 创建示例数据
    print("创建示例回测数据...")
    n_stocks = 50
    n_days = 20
    
    stock_scores, stock_data = create_sample_data(n_stocks=n_stocks, n_days=n_days)
    
    print(f"数据规模:")
    print(f"  股票数量: {n_stocks}")
    print(f"  交易日数: {n_days}")
    print(f"  股票分数数据形状: {stock_scores.shape}")
    print(f"  股票数据形状: {stock_data.shape}")
    
    # 运行回测
    print("\n运行回测...")
    start_date = stock_scores.index[0]
    end_date = stock_scores.index[-1]
    initial_capital = 1000000  # 100万元
    
    results = strategy.run_backtest(
        stock_scores, stock_data,
        start_date, end_date,
        initial_capital
    )
    
    # 显示回测结果
    print("\n回测结果:")
    print(f"  初始资金: {results['initial_capital']:,.2f}元")
    print(f"  最终组合价值: {results['final_portfolio_value']:,.2f}元")
    print(f"  总收益率: {results['total_return']:.2%}")
    
    # 绩效指标
    metrics = results['performance_metrics']
    print("\n绩效指标:")
    print(f"  年化收益率: {metrics.get('annual_return', 0):.2%}")
    print(f"  年化波动率: {metrics.get('volatility', 0):.2%}")
    print(f"  最大回撤: {metrics.get('max_drawdown', 0):.2%}")
    print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  胜率: {metrics.get('win_rate', 0):.2%}")
    print(f"  交易次数: {metrics.get('num_trades', 0)}")
    
    # 交易统计
    trade_history = results['trade_history']
    if trade_history:
        executed_trades = [t for t in trade_history if t.get('status') == 'EXECUTED']
        failed_trades = [t for t in trade_history if t.get('status') == 'FAILED']
        
        print(f"\n交易统计:")
        print(f"  总交易次数: {len(trade_history)}")
        print(f"  成功交易: {len(executed_trades)}")
        print(f"  失败交易: {len(failed_trades)}")
        
        if executed_trades:
            total_cost = sum(t.get('costs', {}).get('total', 0) for t in executed_trades)
            print(f"  总交易成本: {total_cost:,.2f}元")
    
    return results


def main():
    """主函数"""
    try:
        # 演示基本功能
        strategy, config = demo_basic_functionality()
        
        # 演示选股逻辑
        stock_scores, stock_data, selected_stocks = demo_stock_selection(strategy)
        
        # 演示投资组合构建
        weights = demo_portfolio_construction(strategy, selected_stocks, stock_data)
        
        # 演示交易成本计算
        demo_trading_cost(strategy)
        
        # 演示可交易性检查
        demo_tradability_check(strategy, stock_data)
        
        # 演示回测功能
        results = demo_backtest(strategy)
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())