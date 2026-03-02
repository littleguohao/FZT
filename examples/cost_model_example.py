#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成本模型使用示例

展示如何使用CostModel类计算交易成本，包括：
1. 单笔交易成本计算
2. 批量交易成本计算
3. 动态成本调整
4. 成本优化建议
"""

import numpy as np
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtest.cost_model import CostModel


def example_single_trade_cost():
    """示例：单笔交易成本计算"""
    print("=" * 60)
    print("示例1: 单笔交易成本计算")
    print("=" * 60)
    
    # 创建成本模型（使用默认配置）
    cost_model = CostModel()
    
    # 计算卖出交易的成本
    trade_amount = 50000.0  # 5万元
    side = 'sell'
    exchange = 'SH'  # 沪市
    liquidity_score = 0.6  # 中等流动性
    order_size_ratio = 0.02  # 订单占日均成交2%
    
    total_cost = cost_model.calculate_total_cost(
        trade_amount=trade_amount,
        side=side,
        exchange=exchange,
        liquidity_score=liquidity_score,
        order_size_ratio=order_size_ratio
    )
    
    cost_breakdown = cost_model.calculate_cost_breakdown(
        trade_amount=trade_amount,
        side=side,
        exchange=exchange,
        liquidity_score=liquidity_score,
        order_size_ratio=order_size_ratio
    )
    
    print(f"交易金额: {trade_amount:,.2f}元")
    print(f"交易方向: {side}")
    print(f"交易所: {exchange}")
    print(f"流动性分数: {liquidity_score}")
    print(f"订单规模比例: {order_size_ratio:.2%}")
    print(f"\n成本分解:")
    for cost_type, amount in cost_breakdown.items():
        if cost_type != 'total':
            ratio = amount / trade_amount
            print(f"  {cost_type:15s}: {amount:8.2f}元 ({ratio:.4%})")
    print(f"  {'-'*30}")
    print(f"  {'total':15s}: {cost_breakdown['total']:8.2f}元 ({cost_breakdown['total']/trade_amount:.4%})")
    
    return cost_model


def example_batch_trade_cost():
    """示例：批量交易成本计算"""
    print("\n" + "=" * 60)
    print("示例2: 批量交易成本计算")
    print("=" * 60)
    
    cost_model = CostModel()
    
    # 创建批量交易数据
    n_trades = 5
    trade_amounts = np.array([10000.0, 20000.0, 30000.0, 15000.0, 25000.0])
    sides = np.array(['sell', 'buy', 'sell', 'buy', 'sell'])
    exchanges = np.array(['SH', 'SZ', 'SH', 'SZ', 'SH'])
    liquidity_scores = np.array([0.8, 0.5, 0.3, 0.7, 0.4])
    order_size_ratios = np.array([0.01, 0.005, 0.03, 0.008, 0.02])
    
    # 批量计算成本
    total_costs = cost_model.calculate_total_cost_batch(
        trade_amounts=trade_amounts,
        sides=sides,
        exchanges=exchanges,
        liquidity_scores=liquidity_scores,
        order_size_ratios=order_size_ratios
    )
    
    print(f"批量交易数量: {n_trades}")
    print("\n交易详情:")
    print("-" * 80)
    print(f"{'序号':4s} {'金额(元)':>12s} {'方向':6s} {'交易所':8s} {'流动性':8s} {'规模比例':10s} {'成本(元)':>12s} {'成本率':>10s}")
    print("-" * 80)
    
    total_amount = 0
    total_cost = 0
    
    for i in range(n_trades):
        amount = trade_amounts[i]
        cost = total_costs[i]
        cost_ratio = cost / amount
        
        total_amount += amount
        total_cost += cost
        
        print(f"{i+1:4d} {amount:12,.2f} {sides[i]:6s} {exchanges[i]:8s} "
              f"{liquidity_scores[i]:8.2f} {order_size_ratios[i]:10.3%} "
              f"{cost:12,.2f} {cost_ratio:10.4%}")
    
    print("-" * 80)
    print(f"{'总计':4s} {total_amount:12,.2f} {'':6s} {'':8s} {'':8s} {'':10s} "
          f"{total_cost:12,.2f} {total_cost/total_amount:10.4%}")
    
    return cost_model


def example_dynamic_cost_adjustment():
    """示例：动态成本调整"""
    print("\n" + "=" * 60)
    print("示例3: 动态成本调整")
    print("=" * 60)
    
    cost_model = CostModel()
    
    # 基础交易参数
    base_trade = {
        'trade_amount': 10000.0,
        'side': 'sell',
        'exchange': 'SH',
        'order_size_ratio': 0.01
    }
    
    # 测试不同流动性下的成本
    print("不同流动性下的成本比较:")
    print("-" * 50)
    print(f"{'流动性分数':12s} {'总成本(元)':>12s} {'成本率':>10s}")
    print("-" * 50)
    
    for liquidity in [0.9, 0.6, 0.3]:
        cost = cost_model.calculate_total_cost(
            **base_trade,
            liquidity_score=liquidity
        )
        cost_ratio = cost / base_trade['trade_amount']
        print(f"{liquidity:12.2f} {cost:12.2f} {cost_ratio:10.4%}")
    
    # 测试不同订单规模下的成本
    print("\n不同订单规模下的成本比较:")
    print("-" * 50)
    print(f"{'订单规模比例':12s} {'总成本(元)':>12s} {'成本率':>10s}")
    print("-" * 50)
    
    for size_ratio in [0.005, 0.01, 0.02, 0.03]:
        cost = cost_model.calculate_total_cost(
            trade_amount=10000.0,
            side='sell',
            exchange='SH',
            liquidity_score=0.6,
            order_size_ratio=size_ratio
        )
        cost_ratio = cost / 10000.0
        print(f"{size_ratio:12.3%} {cost:12.2f} {cost_ratio:10.4%}")
    
    # 测试不同市场波动率下的成本
    print("\n不同市场波动率下的成本比较:")
    print("-" * 50)
    print(f"{'市场波动率':12s} {'总成本(元)':>12s} {'成本率':>10s}")
    print("-" * 50)
    
    for volatility in [0.01, 0.02, 0.03, 0.04]:
        cost = cost_model.calculate_total_cost(
            trade_amount=10000.0,
            side='sell',
            exchange='SH',
            liquidity_score=0.6,
            order_size_ratio=0.01,
            market_volatility=volatility
        )
        cost_ratio = cost / 10000.0
        print(f"{volatility:12.3%} {cost:12.2f} {cost_ratio:10.4%}")


def example_cost_optimization():
    """示例：成本优化建议"""
    print("\n" + "=" * 60)
    print("示例4: 成本优化建议")
    print("=" * 60)
    
    cost_model = CostModel()
    
    # 模拟一些交易历史
    np.random.seed(42)
    n_trades = 20
    
    trade_history = []
    for i in range(n_trades):
        trade_amount = np.random.uniform(5000, 50000)
        side = 'sell' if np.random.random() > 0.5 else 'buy'
        exchange = 'SH' if np.random.random() > 0.5 else 'SZ'
        liquidity_score = np.random.uniform(0.2, 0.9)
        order_size_ratio = np.random.uniform(0.001, 0.05)
        market_volatility = np.random.uniform(0.01, 0.04)
        
        # 计算成本
        cost = cost_model.calculate_total_cost(
            trade_amount=trade_amount,
            side=side,
            exchange=exchange,
            liquidity_score=liquidity_score,
            order_size_ratio=order_size_ratio,
            market_volatility=market_volatility
        )
        
        trade_history.append({
            'trade_amount': trade_amount,
            'side': side,
            'exchange': exchange,
            'liquidity_score': liquidity_score,
            'order_size_ratio': order_size_ratio,
            'market_volatility': market_volatility,
            'cost': cost,
            'cost_ratio': cost / trade_amount
        })
    
    # 获取优化建议
    suggestions = cost_model.get_cost_optimization_suggestions(trade_history)
    
    print(f"交易历史统计:")
    print(f"  交易笔数: {suggestions['num_trades']}")
    print(f"  总交易金额: {suggestions['total_amount']:,.2f}元")
    print(f"  总成本: {suggestions['total_cost']:,.2f}元")
    print(f"  平均成本率: {suggestions['avg_cost_ratio']:.4%}")
    
    print(f"\n优化建议 ({len(suggestions['suggestions'])}条):")
    print("-" * 80)
    
    for i, suggestion in enumerate(suggestions['suggestions'], 1):
        print(f"{i}. [{suggestion['type']}]")
        print(f"   描述: {suggestion['description']}")
        print(f"   潜在节省: {suggestion['potential_saving']:,.2f}元")
        print(f"   建议行动: {suggestion['action']}")
        print()
    
    print(f"预计总节省: {suggestions['estimated_total_saving']:,.2f}元")
    print(f"节省比例: {suggestions['estimated_total_saving']/suggestions['total_cost']:.2%}")


def example_custom_config():
    """示例：自定义配置"""
    print("\n" + "=" * 60)
    print("示例5: 自定义配置")
    print("=" * 60)
    
    # 自定义配置
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
            'rate': 0.000015,  # 0.0015% 过户费
            'sh_only': True
        },
        'slippage': {
            'rate': 0.0005,   # 0.05% 滑点
            'dynamic': True
        },
        'impact_cost': {
            'enabled': True,
            'liquidity_based': True
        }
    }
    
    # 创建自定义成本模型
    custom_model = CostModel(config=custom_config)
    
    # 获取配置摘要
    summary = custom_model.get_cost_summary()
    
    print("自定义配置摘要:")
    print("-" * 50)
    
    for category, params in summary.items():
        if category != 'trade_history_size':
            print(f"\n{category}:")
            for key, value in params.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
    
    print(f"\n交易历史记录数: {summary['trade_history_size']}")
    
    # 使用自定义模型计算成本
    cost = custom_model.calculate_total_cost(
        trade_amount=10000.0,
        side='sell',
        exchange='SH',
        liquidity_score=0.6,
        order_size_ratio=0.01
    )
    
    print(f"\n使用自定义配置计算的成本: {cost:.2f}元 ({cost/10000.0:.4%})")


def main():
    """主函数"""
    print("FZT Quant - 成本模型使用示例")
    print("=" * 60)
    
    try:
        # 运行所有示例
        example_single_trade_cost()
        example_batch_trade_cost()
        example_dynamic_cost_adjustment()
        example_cost_optimization()
        example_custom_config()
        
        print("\n" + "=" * 60)
        print("所有示例执行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()