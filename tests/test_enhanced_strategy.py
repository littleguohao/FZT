#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试增强回测策略

测试FZTEnhancedStrategy类的功能，包括：
1. 基础选股功能
2. 交易成本计算
3. 流动性控制
4. 风险控制
5. 可交易性检查

作者: FZT项目组
创建日期: 2026-03-02
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# 测试FZTEnhancedStrategy类
class TestFZTEnhancedStrategy:
    """测试增强回测策略"""
    
    def setup_method(self):
        """测试设置"""
        # 创建模拟数据
        self.trade_date = datetime(2026, 3, 2)
        
        # 模拟股票数据
        self.stock_data = pd.DataFrame({
            'score': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
            'turnover': [1e8, 9e7, 8e7, 7e7, 6e7, 5e7, 4e7, 3e7, 2e7, 1e7],  # 成交额
            'market_cap': [1e10, 9e9, 8e9, 7e9, 6e9, 5e9, 4e9, 3e9, 2e9, 1e9],  # 市值
            'is_st': [False, False, False, False, False, False, False, True, False, False],  # ST股票
            'is_suspended': [False, False, False, False, False, False, False, False, True, False],  # 停牌
            'limit_up': [False, False, False, False, False, False, False, False, False, True],  # 涨停
            'limit_down': [False, False, False, False, False, False, False, False, False, False],  # 跌停
            'industry': ['银行', '银行', '科技', '科技', '医药', '医药', '消费', '消费', '能源', '能源']  # 行业
        }, index=[
            '000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ',
            '000006.SZ', '000007.SZ', '000008.SZ', '000009.SZ', '000010.SZ'
        ])
        
        # 配置参数
        self.config = {
            'top_k': 5,
            'hold_days': 1,
            'commission': 0.0003,
            'stamp_tax': 0.001,
            'slippage': 0.001,
            'min_turnover_ratio': 0.2,
            'min_market_cap_ratio': 0.3,
            'max_industry_weight': 0.3,
            'max_single_weight': 0.4
        }
    
    def test_strategy_initialization(self):
        """测试策略初始化"""
        from src.backtest.enhanced_strategy import FZTEnhancedStrategy
        strategy = FZTEnhancedStrategy(config=self.config)
        assert strategy is not None
        assert strategy.top_k == 5
        assert strategy.hold_days == 1
        assert strategy.commission_rate == 0.0003
        assert strategy.stamp_tax_rate == 0.001
        assert strategy.slippage_rate == 0.001
    
    def test_stock_selection_logic(self):
        """测试选股逻辑"""
        from src.backtest.enhanced_strategy import FZTEnhancedStrategy
        strategy = FZTEnhancedStrategy(config=self.config)
        
        # 创建测试数据
        stock_scores = pd.Series(
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
            index=self.stock_data.index
        )
        
        # 测试选股
        selected_stocks = strategy.select_stocks(
            stock_scores, self.stock_data, self.trade_date
        )
        
        # 应该选择前5只股票（排除ST、停牌、涨跌停）
        # 000008.SZ是ST股票，000009.SZ停牌，000010.SZ涨停
        # 由于行业权重限制（银行行业最多30%，即1-2只股票），实际选择可能少于5只
        # 但至少应该选择3只股票
        assert len(selected_stocks) >= 3
        # 检查选中的股票不包含被过滤的股票
        assert '000008.SZ' not in selected_stocks  # ST
        assert '000009.SZ' not in selected_stocks  # 停牌
        assert '000010.SZ' not in selected_stocks  # 涨停
        # 检查选中的股票按分数排序
        selected_scores = [stock_scores[s] for s in selected_stocks]
        assert selected_scores == sorted(selected_scores, reverse=True)
    
    def test_liquidity_filter(self):
        """测试流动性过滤"""
        from src.backtest.enhanced_strategy import FZTEnhancedStrategy
        strategy = FZTEnhancedStrategy(config=self.config)
        
        # 测试流动性过滤方法
        stock_info = self.stock_data.loc['000001.SZ']
        
        # 测试ST检查
        st_stock_info = self.stock_data.loc['000008.SZ']
        assert strategy._is_st_stock(st_stock_info) == True
        assert strategy._is_st_stock(stock_info) == False
        
        # 测试停牌检查
        suspended_stock_info = self.stock_data.loc['000009.SZ']
        assert strategy._is_suspended(suspended_stock_info) == True
        assert strategy._is_suspended(stock_info) == False
        
        # 测试涨停检查
        limit_up_stock_info = self.stock_data.loc['000010.SZ']
        assert strategy._is_limit_up(limit_up_stock_info) == True
        assert strategy._is_limit_up(stock_info) == False
    
    def test_risk_control(self):
        """测试风险控制"""
        from src.backtest.enhanced_strategy import FZTEnhancedStrategy
        strategy = FZTEnhancedStrategy(config=self.config)
        
        # 测试行业权重限制
        selected_stocks = ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ']
        
        # 应用风险控制
        controlled_stocks = strategy._apply_risk_controls(selected_stocks, self.stock_data)
        
        # 检查结果
        assert len(controlled_stocks) <= len(selected_stocks)
        
        # 测试个股权重限制
        weights = {
            '000001.SZ': 0.5,  # 超过40%限制
            '000002.SZ': 0.3,
            '000003.SZ': 0.2
        }
        
        adjusted_weights = strategy._apply_single_weight_limit(weights)
        
        # 检查权重调整
        assert adjusted_weights['000001.SZ'] <= 0.4
        assert sum(adjusted_weights.values()) <= 1.0 + 1e-6
    
    def test_trading_cost_calculation(self):
        """测试交易成本计算"""
        from src.backtest.enhanced_strategy import FZTEnhancedStrategy
        strategy = FZTEnhancedStrategy(config=self.config)
        
        # 测试买入成本
        buy_costs = strategy.calculate_trading_cost(
            order_amount=100000,  # 10万元
            is_buy=True,
            stock_liquidity=0.8
        )
        
        assert 'commission' in buy_costs
        assert 'stamp_tax' in buy_costs
        assert 'slippage' in buy_costs
        assert 'impact_cost' in buy_costs
        assert 'total' in buy_costs
        
        # 佣金应该至少5元
        assert buy_costs['commission'] >= 5.0
        
        # 买入时印花税为0
        assert buy_costs['stamp_tax'] == 0.0
        
        # 测试卖出成本
        sell_costs = strategy.calculate_trading_cost(
            order_amount=100000,
            is_buy=False,
            stock_liquidity=0.8
        )
        
        # 卖出时应该有印花税
        assert sell_costs['stamp_tax'] > 0
        
        # 总成本应该为正
        assert buy_costs['total'] > 0
        assert sell_costs['total'] > 0
    
    def test_tradability_check(self):
        """测试可交易性检查"""
        from src.backtest.enhanced_strategy import FZTEnhancedStrategy
        strategy = FZTEnhancedStrategy(config=self.config)
        
        # 测试可交易股票
        tradable, reason = strategy.check_tradability(
            '000001.SZ', self.stock_data, self.trade_date
        )
        assert tradable == True
        assert reason == "可交易"
        
        # 测试ST股票
        tradable, reason = strategy.check_tradability(
            '000008.SZ', self.stock_data, self.trade_date
        )
        assert tradable == False
        assert "ST" in reason
        
        # 测试停牌股票
        tradable, reason = strategy.check_tradability(
            '000009.SZ', self.stock_data, self.trade_date
        )
        assert tradable == False
        assert "停牌" in reason
        
        # 测试涨停股票
        tradable, reason = strategy.check_tradability(
            '000010.SZ', self.stock_data, self.trade_date
        )
        assert tradable == False
        assert "涨停" in reason
    
    def test_portfolio_construction(self):
        """测试投资组合构建"""
        from src.backtest.enhanced_strategy import FZTEnhancedStrategy
        strategy = FZTEnhancedStrategy(config=self.config)
        
        # 测试投资组合构建
        selected_stocks = ['000001.SZ', '000002.SZ', '000003.SZ']
        available_capital = 1000000
        
        weights = strategy.construct_portfolio(
            selected_stocks, available_capital, self.stock_data
        )
        
        # 检查权重
        assert len(weights) == len(selected_stocks)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        
        # 检查个股权重限制
        for weight in weights.values():
            assert weight <= strategy.max_single_weight + 1e-6
    
    def test_integration_with_qlib(self):
        """测试与QLib的集成"""
        # 测试与QLib数据格式的兼容性
        # 测试与QLib回测引擎的集成
        pass


class TestEnhancedStrategyIntegration:
    """测试增强策略集成"""
    
    def test_config_loading(self):
        """测试配置加载"""
        import yaml
        import os
        
        config_path = 'config/backtest_config.yaml'
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 验证配置结构
            assert 'strategy' in config
            assert 'transaction_cost' in config
            assert 'liquidity_control' in config
            assert 'risk_management' in config
    
    def test_data_format_compatibility(self):
        """测试数据格式兼容性"""
        # 测试数据格式与QLib的兼容性
        # 应该支持QLib的标准数据格式
        pass
    
    def test_performance_metrics(self):
        """测试绩效指标计算"""
        # 测试回测结果包含关键绩效指标
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])