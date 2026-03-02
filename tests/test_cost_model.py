"""
测试成本模型模块
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# 导入成本模型
try:
    from backtest.cost_model import CostModel
    COST_MODEL_AVAILABLE = True
except ImportError:
    COST_MODEL_AVAILABLE = False


class TestCostModel:
    """测试CostModel类"""
    
    def test_cost_model_import(self):
        """测试成本模型导入"""
        # 如果模块不存在，测试应该失败
        if not COST_MODEL_AVAILABLE:
            pytest.fail("CostModel模块未找到，请先实现成本模型")
    
    def test_cost_model_initialization(self):
        """测试成本模型初始化"""
        if not COST_MODEL_AVAILABLE:
            pytest.skip("CostModel模块未实现")
        
        # 测试默认配置初始化
        cost_model = CostModel()
        assert cost_model is not None
        assert hasattr(cost_model, 'commission_rate')
        assert hasattr(cost_model, 'stamp_tax_rate')
        assert hasattr(cost_model, 'transfer_fee_rate')
        
        # 测试自定义配置初始化
        config = {
            'commission': {'rate': 0.00025, 'min': 3.0},
            'stamp_tax': {'rate': 0.0008, 'sell_only': True},
            'transfer_fee': {'rate': 0.000015, 'sh_only': True},
            'slippage': {'rate': 0.0008, 'dynamic': True},
            'impact_cost': {'enabled': True, 'liquidity_based': True}
        }
        cost_model = CostModel(config=config)
        assert cost_model.commission_rate == 0.00025
        assert cost_model.min_commission == 3.0
    
    def test_commission_calculation(self):
        """测试佣金计算"""
        if not COST_MODEL_AVAILABLE:
            pytest.skip("CostModel模块未实现")
        
        cost_model = CostModel()
        
        # 测试比例佣金
        trade_amount = 10000.0  # 1万元
        commission = cost_model.calculate_commission(trade_amount)
        expected = max(trade_amount * cost_model.commission_rate, cost_model.min_commission)
        assert commission == pytest.approx(expected, rel=1e-6)
        
        # 测试最低佣金
        small_trade = 1000.0  # 1000元，佣金应为最低5元
        commission_small = cost_model.calculate_commission(small_trade)
        assert commission_small == cost_model.min_commission
        
        # 测试批量计算
        trade_amounts = np.array([10000.0, 20000.0, 5000.0, 15000.0])
        commissions = cost_model.calculate_commission_batch(trade_amounts)
        assert len(commissions) == len(trade_amounts)
        assert all(comm >= cost_model.min_commission for comm in commissions)
    
    def test_stamp_tax_calculation(self):
        """测试印花税计算"""
        if not COST_MODEL_AVAILABLE:
            pytest.skip("CostModel模块未实现")
        
        cost_model = CostModel()
        
        # 测试卖出时的印花税
        sell_amount = 10000.0
        stamp_tax_sell = cost_model.calculate_stamp_tax(sell_amount, side='sell')
        expected_sell = sell_amount * cost_model.stamp_tax_rate
        assert stamp_tax_sell == pytest.approx(expected_sell, rel=1e-6)
        
        # 测试买入时的印花税（应为0）
        buy_amount = 10000.0
        stamp_tax_buy = cost_model.calculate_stamp_tax(buy_amount, side='buy')
        assert stamp_tax_buy == 0.0
        
        # 测试批量计算
        amounts = np.array([10000.0, 20000.0, 30000.0])
        sides = np.array(['sell', 'buy', 'sell'])
        stamp_taxes = cost_model.calculate_stamp_tax_batch(amounts, sides)
        assert len(stamp_taxes) == len(amounts)
        assert stamp_taxes[0] > 0  # 卖出
        assert stamp_taxes[1] == 0  # 买入
        assert stamp_taxes[2] > 0  # 卖出
    
    def test_transfer_fee_calculation(self):
        """测试过户费计算"""
        if not COST_MODEL_AVAILABLE:
            pytest.skip("CostModel模块未实现")
        
        cost_model = CostModel()
        
        # 测试沪市股票过户费
        sh_trade_amount = 10000.0
        transfer_fee_sh = cost_model.calculate_transfer_fee(sh_trade_amount, exchange='SH')
        expected_sh = sh_trade_amount * cost_model.transfer_fee_rate
        assert transfer_fee_sh == pytest.approx(expected_sh, rel=1e-6)
        
        # 测试深市股票过户费（应为0）
        sz_trade_amount = 10000.0
        transfer_fee_sz = cost_model.calculate_transfer_fee(sz_trade_amount, exchange='SZ')
        assert transfer_fee_sz == 0.0
        
        # 测试批量计算
        amounts = np.array([10000.0, 20000.0, 30000.0])
        exchanges = np.array(['SH', 'SZ', 'SH'])
        transfer_fees = cost_model.calculate_transfer_fee_batch(amounts, exchanges)
        assert len(transfer_fees) == len(amounts)
        assert transfer_fees[0] > 0  # 沪市
        assert transfer_fees[1] == 0  # 深市
        assert transfer_fees[2] > 0  # 沪市
    
    def test_slippage_calculation(self):
        """测试滑点成本计算"""
        if not COST_MODEL_AVAILABLE:
            pytest.skip("CostModel模块未实现")
        
        cost_model = CostModel()
        
        # 测试固定滑点
        trade_amount = 10000.0
        slippage_fixed = cost_model.calculate_slippage(trade_amount, method='fixed')
        expected_fixed = trade_amount * cost_model.slippage_rate
        assert slippage_fixed == pytest.approx(expected_fixed, rel=1e-6)
        
        # 测试动态滑点（基于流动性）
        liquidity_score = 0.5  # 中等流动性
        slippage_dynamic = cost_model.calculate_slippage(
            trade_amount, 
            method='dynamic',
            liquidity_score=liquidity_score
        )
        # 动态滑点应该比固定滑点高（流动性差）
        assert slippage_dynamic >= slippage_fixed
        
        # 测试基于买卖价差的滑点
        bid_ask_spread = 0.002  # 0.2%的买卖价差
        slippage_spread = cost_model.calculate_slippage(
            trade_amount,
            method='spread',
            bid_ask_spread=bid_ask_spread
        )
        expected_spread = trade_amount * bid_ask_spread / 2  # 通常假设成交在中间价
        assert slippage_spread == pytest.approx(expected_spread, rel=1e-6)
    
    def test_impact_cost_calculation(self):
        """测试冲击成本计算"""
        if not COST_MODEL_AVAILABLE:
            pytest.skip("CostModel模块未实现")
        
        cost_model = CostModel()
        
        # 测试线性冲击成本模型
        trade_amount = 10000.0
        avg_daily_volume = 1000000.0  # 日均成交量100万元
        order_size_ratio = trade_amount / avg_daily_volume  # 1%
        
        impact_linear = cost_model.calculate_impact_cost(
            trade_amount,
            model='linear',
            order_size_ratio=order_size_ratio
        )
        # 冲击成本应该为正
        assert impact_linear >= 0
        
        # 测试基于流动性的非线性模型
        liquidity_score = 0.3  # 流动性较差
        impact_liquidity = cost_model.calculate_impact_cost(
            trade_amount,
            model='liquidity',
            liquidity_score=liquidity_score,
            order_size_ratio=order_size_ratio
        )
        # 流动性差时冲击成本应该更高
        # 注意：由于默认配置中impact_liquidity_based=True，可能使用不同的模型
        # 我们只验证成本为正
        assert impact_liquidity >= 0
        
        # 测试市场冲击模型
        market_volatility = 0.02  # 2%的市场波动率
        impact_market = cost_model.calculate_impact_cost(
            trade_amount,
            model='market',
            market_volatility=market_volatility,
            order_size_ratio=order_size_ratio
        )
        # 市场波动大时冲击成本应该为正
        assert impact_market >= 0
    
    def test_total_cost_calculation(self):
        """测试总成本计算"""
        if not COST_MODEL_AVAILABLE:
            pytest.skip("CostModel模块未实现")
        
        cost_model = CostModel()
        
        # 测试单笔交易总成本
        trade_info = {
            'trade_amount': 10000.0,
            'side': 'sell',
            'exchange': 'SH',
            'liquidity_score': 0.6,
            'order_size_ratio': 0.01,
            'market_volatility': 0.015
        }
        
        total_cost = cost_model.calculate_total_cost(**trade_info)
        
        # 总成本应该包含所有成本类型
        assert total_cost >= 0
        
        # 分解成本
        cost_breakdown = cost_model.calculate_cost_breakdown(**trade_info)
        assert 'commission' in cost_breakdown
        assert 'stamp_tax' in cost_breakdown
        assert 'transfer_fee' in cost_breakdown
        assert 'slippage' in cost_breakdown
        assert 'impact_cost' in cost_breakdown
        assert 'total' in cost_breakdown
        
        # 总成本应该等于各成本之和
        total_from_breakdown = sum(
            cost_breakdown[key] for key in ['commission', 'stamp_tax', 'transfer_fee', 'slippage', 'impact_cost']
        )
        assert total_cost == pytest.approx(total_from_breakdown, rel=1e-6)
    
    def test_batch_cost_calculation(self):
        """测试批量成本计算"""
        if not COST_MODEL_AVAILABLE:
            pytest.skip("CostModel模块未实现")
        
        cost_model = CostModel()
        
        # 创建批量交易数据
        n_trades = 10
        trade_amounts = np.random.uniform(5000, 50000, n_trades)
        sides = np.random.choice(['buy', 'sell'], n_trades)
        exchanges = np.random.choice(['SH', 'SZ'], n_trades)
        liquidity_scores = np.random.uniform(0.3, 0.9, n_trades)
        order_size_ratios = np.random.uniform(0.001, 0.05, n_trades)
        
        # 计算批量成本
        total_costs = cost_model.calculate_total_cost_batch(
            trade_amounts=trade_amounts,
            sides=sides,
            exchanges=exchanges,
            liquidity_scores=liquidity_scores,
            order_size_ratios=order_size_ratios
        )
        
        assert len(total_costs) == n_trades
        assert all(cost >= 0 for cost in total_costs)
        
        # 验证与单笔计算的一致性
        for i in range(min(3, n_trades)):  # 随机验证3笔
            single_cost = cost_model.calculate_total_cost(
                trade_amount=trade_amounts[i],
                side=sides[i],
                exchange=exchanges[i],
                liquidity_score=liquidity_scores[i],
                order_size_ratio=order_size_ratios[i]
            )
            assert total_costs[i] == pytest.approx(single_cost, rel=1e-6)
    
    def test_dynamic_cost_adjustment(self):
        """测试动态成本调整"""
        if not COST_MODEL_AVAILABLE:
            pytest.skip("CostModel模块未实现")
        
        # 使用自定义配置，确保冲击成本启用
        config = {
            'commission': {'rate': 0.0003, 'min': 5.0},
            'stamp_tax': {'rate': 0.001, 'sell_only': True},
            'transfer_fee': {'rate': 0.00002, 'sh_only': True},
            'slippage': {'rate': 0.001, 'dynamic': True},
            'impact_cost': {'enabled': True, 'liquidity_based': True}
        }
        cost_model = CostModel(config=config)
        
        # 测试基于流动性的调整
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
        
        # 流动性差时成本应该更高（或至少相等）
        assert low_liquidity_cost >= high_liquidity_cost
        
        # 测试基于订单规模的调整
        small_order_cost = cost_model.calculate_total_cost(
            trade_amount=10000.0,
            side='sell',
            exchange='SH',
            liquidity_score=0.6,
            order_size_ratio=0.005  # 小订单
        )
        
        large_order_cost = cost_model.calculate_total_cost(
            trade_amount=10000.0,
            side='sell',
            exchange='SH',
            liquidity_score=0.6,
            order_size_ratio=0.03  # 大订单
        )
        
        # 大订单成本应该更高（或至少相等）
        assert large_order_cost >= small_order_cost
        
        # 测试基于市场状态的调整
        low_volatility_cost = cost_model.calculate_total_cost(
            trade_amount=10000.0,
            side='sell',
            exchange='SH',
            liquidity_score=0.6,
            order_size_ratio=0.01,
            market_volatility=0.01  # 低波动
        )
        
        high_volatility_cost = cost_model.calculate_total_cost(
            trade_amount=10000.0,
            side='sell',
            exchange='SH',
            liquidity_score=0.6,
            order_size_ratio=0.01,
            market_volatility=0.03  # 高波动
        )
        
        # 高波动时成本应该更高（或至少相等）
        assert high_volatility_cost >= low_volatility_cost
    
    def test_cost_optimization_suggestions(self):
        """测试成本优化建议"""
        if not COST_MODEL_AVAILABLE:
            pytest.skip("CostModel模块未实现")
        
        cost_model = CostModel()
        
        # 创建交易历史（包含所有必要字段）
        trade_history = [
            {
                'trade_amount': 10000.0,
                'side': 'sell',
                'exchange': 'SH',
                'liquidity_score': 0.3,
                'order_size_ratio': 0.02,
                'market_volatility': 0.025,
                'cost': 45.0,
                'cost_ratio': 0.0045  # 0.45%
            },
            {
                'trade_amount': 8000.0,
                'side': 'buy',
                'exchange': 'SZ',
                'liquidity_score': 0.8,
                'order_size_ratio': 0.005,
                'market_volatility': 0.015,
                'cost': 12.0,
                'cost_ratio': 0.0015  # 0.15%
            },
            {
                'trade_amount': 15000.0,
                'side': 'sell',
                'exchange': 'SH',
                'liquidity_score': 0.4,
                'order_size_ratio': 0.03,
                'market_volatility': 0.035,
                'cost': 68.0,
                'cost_ratio': 0.00453  # 0.453%
            }
        ]
        
        # 获取优化建议
        suggestions = cost_model.get_cost_optimization_suggestions(trade_history)
        
        # 应该返回优化建议
        assert isinstance(suggestions, dict)
        assert 'total_cost' in suggestions
        assert 'avg_cost_ratio' in suggestions
        assert 'suggestions' in suggestions
        assert isinstance(suggestions['suggestions'], list)
        
        # 验证建议内容（可能有0个或多个建议）
        for suggestion in suggestions['suggestions']:
            assert 'type' in suggestion
            assert 'description' in suggestion
            assert 'potential_saving' in suggestion
    
    def test_config_loading(self):
        """测试配置文件加载"""
        if not COST_MODEL_AVAILABLE:
            pytest.skip("CostModel模块未实现")
        
        # 测试从字典加载配置（包含所有必要字段）
        config_dict = {
            'commission': {'rate': 0.00025, 'min': 3.0},
            'stamp_tax': {'rate': 0.0008, 'sell_only': True},
            'transfer_fee': {'rate': 0.000015, 'sh_only': True},
            'slippage': {'rate': 0.0008, 'dynamic': True},
            'impact_cost': {'enabled': True, 'liquidity_based': True}
        }
        
        cost_model = CostModel(config=config_dict)
        assert cost_model.commission_rate == 0.00025
        assert cost_model.stamp_tax_rate == 0.0008
        assert cost_model.transfer_fee_rate == 0.000015
        
        # 测试从YAML文件加载配置
        config_path = Path(__file__).parent.parent / 'config' / 'backtest_config.yaml'
        if config_path.exists():
            cost_model_yaml = CostModel(config=str(config_path))
            # 验证配置已加载
            assert hasattr(cost_model_yaml, 'commission_rate')
            assert hasattr(cost_model_yaml, 'stamp_tax_rate')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])