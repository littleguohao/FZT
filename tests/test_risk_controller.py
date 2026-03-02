"""
测试风险控制器模块
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from backtest.risk_controller import RiskController
    RISK_CONTROLLER_AVAILABLE = True
except ImportError:
    RISK_CONTROLLER_AVAILABLE = False


class TestRiskController:
    """测试RiskController类"""
    
    def setup_method(self):
        """测试前的准备工作"""
        # 创建测试数据
        self.test_config = {
            'risk': {
                'market': {
                    'max_volatility': 0.3,
                    'min_correlation': 0.3,
                    'max_beta': 1.5
                },
                'credit': {
                    'filter_st': True,
                    'min_current_ratio': 1.0,
                    'max_debt_ratio': 0.7
                },
                'liquidity': {
                    'min_turnover': 10000000,
                    'max_bid_ask_spread': 0.02
                },
                'concentration': {
                    'max_single_weight': 0.4,
                    'max_industry_weight': 0.3,
                    'max_style_exposure': 0.5
                },
                'stop_loss': {
                    'enabled': True,
                    'threshold': -0.1,
                    'trailing': True
                }
            }
        }
        
        # 创建投资组合数据
        self.portfolio_data = {
            'positions': {
                '000001.SZ': {'weight': 0.15, 'shares': 1000, 'cost': 10.0},
                '000002.SZ': {'weight': 0.12, 'shares': 800, 'cost': 12.0},
                '000003.SZ': {'weight': 0.10, 'shares': 600, 'cost': 15.0},
                '000004.SZ': {'weight': 0.08, 'shares': 500, 'cost': 18.0},
                '000005.SZ': {'weight': 0.05, 'shares': 300, 'cost': 20.0}
            },
            'total_value': 1000000.0,
            'cash': 50000.0,
            'pnl': 50000.0
        }
        
        # 创建市场数据
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        stocks = ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ']
        
        # 价格数据
        np.random.seed(42)
        price_data = np.random.randn(len(dates), len(stocks)) * 0.02 + 1.0
        price_data = np.cumprod(price_data, axis=0) * 10
        
        self.market_data = pd.DataFrame(
            price_data,
            index=dates,
            columns=stocks
        )
        
        # 成交量数据
        volume_data = np.random.randint(1000000, 10000000, size=(len(dates), len(stocks)))
        self.volume_data = pd.DataFrame(
            volume_data,
            index=dates,
            columns=stocks
        )
        
        # 财务数据
        self.financial_data = {
            '000001.SZ': {
                'is_st': False,
                'current_ratio': 1.5,
                'debt_ratio': 0.4,
                'credit_rating': 'AA'
            },
            '000002.SZ': {
                'is_st': False,
                'current_ratio': 0.8,
                'debt_ratio': 0.8,
                'credit_rating': 'BB'
            },
            '000003.SZ': {
                'is_st': True,
                'current_ratio': 0.5,
                'debt_ratio': 0.9,
                'credit_rating': 'C'
            },
            '000004.SZ': {
                'is_st': False,
                'current_ratio': 2.0,
                'debt_ratio': 0.3,
                'credit_rating': 'AAA'
            },
            '000005.SZ': {
                'is_st': False,
                'current_ratio': 1.2,
                'debt_ratio': 0.6,
                'credit_rating': 'A'
            }
        }
        
        # 风险因子数据
        self.risk_factors = {
            'industry': {
                '000001.SZ': '金融',
                '000002.SZ': '房地产',
                '000003.SZ': '金融',
                '000004.SZ': '科技',
                '000005.SZ': '消费'
            },
            'style': {
                '000001.SZ': {'value': 0.3, 'growth': 0.7},
                '000002.SZ': {'value': 0.8, 'growth': 0.2},
                '000003.SZ': {'value': 0.5, 'growth': 0.5},
                '000004.SZ': {'value': 0.2, 'growth': 0.8},
                '000005.SZ': {'value': 0.6, 'growth': 0.4}
            },
            'market_beta': {
                '000001.SZ': 1.2,
                '000002.SZ': 1.5,
                '000003.SZ': 0.8,
                '000004.SZ': 1.0,
                '000005.SZ': 1.1
            }
        }
    
    @pytest.mark.skipif(not RISK_CONTROLLER_AVAILABLE, reason="RiskController模块未安装")
    def test_risk_controller_initialization(self):
        """测试风险控制器初始化"""
        controller = RiskController(self.test_config)
        
        # 验证配置加载
        assert controller.config == self.test_config
        assert controller.market_config == self.test_config['risk']['market']
        assert controller.credit_config == self.test_config['risk']['credit']
        assert controller.liquidity_config == self.test_config['risk']['liquidity']
        assert controller.concentration_config == self.test_config['risk']['concentration']
        assert controller.stop_loss_config == self.test_config['risk']['stop_loss']
        
        # 验证预警记录初始化
        assert controller.warnings == []
        assert controller.actions == []
    
    @pytest.mark.skipif(not RISK_CONTROLLER_AVAILABLE, reason="RiskController模块未安装")
    def test_market_risk_monitoring(self):
        """测试市场风险监控"""
        controller = RiskController(self.test_config)
        
        # 计算波动率
        returns = self.market_data.pct_change().dropna()
        volatility = controller._calculate_volatility(returns)
        
        assert isinstance(volatility, dict)
        assert len(volatility) == len(self.market_data.columns)
        
        # 检查波动率是否在合理范围内
        for stock, vol in volatility.items():
            assert vol >= 0  # 波动率应为非负数
        
        # 计算相关性
        correlation = controller._calculate_correlation(returns)
        
        assert isinstance(correlation, pd.DataFrame)
        assert correlation.shape == (len(self.market_data.columns), len(self.market_data.columns))
        
        # 相关性矩阵应是对称的
        assert np.allclose(correlation.values, correlation.values.T)
        
        # 对角线应为1
        for i in range(len(correlation)):
            assert abs(correlation.iloc[i, i] - 1.0) < 1e-10
    
    @pytest.mark.skipif(not RISK_CONTROLLER_AVAILABLE, reason="RiskController模块未安装")
    def test_credit_risk_monitoring(self):
        """测试信用风险监控"""
        controller = RiskController(self.test_config)
        
        # 检查ST股票过滤
        st_stocks = controller._identify_st_stocks(self.financial_data)
        
        assert isinstance(st_stocks, list)
        assert '000003.SZ' in st_stocks  # 这个应该是ST股票
        
        # 检查财务指标
        credit_warnings = controller._check_financial_metrics(self.financial_data)
        
        assert isinstance(credit_warnings, list)
        
        # 000002.SZ的流动比率低于阈值，应该有警告
        has_low_current_ratio_warning = any(
            '000002.SZ' in warning and '流动比率' in warning 
            for warning in credit_warnings
        )
        assert has_low_current_ratio_warning
    
    @pytest.mark.skipif(not RISK_CONTROLLER_AVAILABLE, reason="RiskController模块未安装")
    def test_liquidity_risk_monitoring(self):
        """测试流动性风险监控"""
        controller = RiskController(self.test_config)
        
        # 检查成交额
        latest_volume = self.volume_data.iloc[-1]
        turnover = latest_volume * self.market_data.iloc[-1]
        
        liquidity_warnings = controller._check_liquidity_metrics(
            list(self.market_data.columns),
            turnover.to_dict()
        )
        
        assert isinstance(liquidity_warnings, list)
    
    @pytest.mark.skipif(not RISK_CONTROLLER_AVAILABLE, reason="RiskController模块未安装")
    def test_concentration_risk_monitoring(self):
        """测试集中度风险监控"""
        controller = RiskController(self.test_config)
        
        # 检查仓位集中度
        weights = {stock: pos['weight'] for stock, pos in self.portfolio_data['positions'].items()}
        concentration_warnings = controller._check_concentration_risk(weights)
        
        assert isinstance(concentration_warnings, list)
        
        # 检查行业集中度
        industry_weights = controller._calculate_industry_weights(
            weights,
            self.risk_factors['industry']
        )
        
        assert isinstance(industry_weights, dict)
        
        # 检查行业集中度警告
        industry_warnings = controller._check_industry_concentration(industry_weights)
        assert isinstance(industry_warnings, list)
    
    @pytest.mark.skipif(not RISK_CONTROLLER_AVAILABLE, reason="RiskController模块未安装")
    def test_risk_metrics_calculation(self):
        """测试风险指标计算"""
        controller = RiskController(self.test_config)
        
        # 计算投资组合收益
        portfolio_returns = self.market_data.pct_change().mean(axis=1).dropna()
        
        # 计算VaR
        var_95 = controller._calculate_var(portfolio_returns, confidence_level=0.95)
        var_99 = controller._calculate_var(portfolio_returns, confidence_level=0.99)
        
        assert var_95 is not None
        assert var_99 is not None
        assert var_99 <= var_95  # 99% VaR应该更严格
        
        # 计算CVaR
        cvar_95 = controller._calculate_cvar(portfolio_returns, confidence_level=0.95)
        cvar_99 = controller._calculate_cvar(portfolio_returns, confidence_level=0.99)
        
        assert cvar_95 is not None
        assert cvar_99 is not None
        assert cvar_99 <= cvar_95  # 99% CVaR应该更严格
        
        # 计算最大回撤
        max_drawdown = controller._calculate_max_drawdown(portfolio_returns)
        
        assert max_drawdown is not None
        assert max_drawdown <= 0  # 回撤应为负数或零
        
        # 计算夏普比率
        sharpe_ratio = controller._calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.02)
        
        assert sharpe_ratio is not None
        
        # 计算索提诺比率
        sortino_ratio = controller._calculate_sortino_ratio(portfolio_returns, risk_free_rate=0.02)
        
        assert sortino_ratio is not None
    
    @pytest.mark.skipif(not RISK_CONTROLLER_AVAILABLE, reason="RiskController模块未安装")
    def test_stop_loss_mechanism(self):
        """测试止损机制"""
        controller = RiskController(self.test_config)
        
        # 创建持仓盈亏数据
        position_pnl = {
            '000001.SZ': 0.05,   # 盈利5%
            '000002.SZ': -0.12,  # 亏损12%，应触发止损
            '000003.SZ': -0.08,  # 亏损8%
            '000004.SZ': 0.03,   # 盈利3%
            '000005.SZ': -0.15   # 亏损15%，应触发止损
        }
        
        # 检查止损
        stop_loss_actions = controller._check_stop_loss(
            self.portfolio_data['positions'],
            position_pnl
        )
        
        assert isinstance(stop_loss_actions, list)
        
        # 验证亏损超过阈值的股票有止损动作
        stocks_with_stop_loss = [action['stock'] for action in stop_loss_actions]
        assert '000002.SZ' in stocks_with_stop_loss
        assert '000005.SZ' in stocks_with_stop_loss
        assert '000001.SZ' not in stocks_with_stop_loss  # 盈利股票不应止损
    
    @pytest.mark.skipif(not RISK_CONTROLLER_AVAILABLE, reason="RiskController模块未安装")
    def test_comprehensive_risk_assessment(self):
        """测试综合风险评估"""
        controller = RiskController(self.test_config)
        
        # 执行综合风险评估
        risk_report = controller.assess_risk(
            portfolio=self.portfolio_data,
            market_data=self.market_data,
            volume_data=self.volume_data,
            financial_data=self.financial_data,
            risk_factors=self.risk_factors
        )
        
        # 验证风险报告结构
        assert isinstance(risk_report, dict)
        assert 'market_risk' in risk_report
        assert 'credit_risk' in risk_report
        assert 'liquidity_risk' in risk_report
        assert 'concentration_risk' in risk_report
        assert 'risk_metrics' in risk_report
        assert 'warnings' in risk_report
        assert 'recommended_actions' in risk_report
        
        # 验证风险指标
        assert 'var_95' in risk_report['risk_metrics']
        assert 'cvar_95' in risk_report['risk_metrics']
        assert 'max_drawdown' in risk_report['risk_metrics']
        assert 'sharpe_ratio' in risk_report['risk_metrics']
        assert 'sortino_ratio' in risk_report['risk_metrics']
    
    @pytest.mark.skipif(not RISK_CONTROLLER_AVAILABLE, reason="RiskController模块未安装")
    def test_risk_control_actions(self):
        """测试风险控制动作"""
        controller = RiskController(self.test_config)
        
        # 模拟高风险情况
        high_risk_portfolio = {
            'positions': {
                '000001.SZ': {'weight': 0.45, 'shares': 3000, 'cost': 10.0, 'market_value': 450000},  # 权重超过40%
                '000002.SZ': {'weight': 0.35, 'shares': 2000, 'cost': 12.0, 'market_value': 350000},  # 权重接近阈值
                '000003.SZ': {'weight': 0.20, 'shares': 1000, 'cost': 15.0, 'market_value': 200000}
            },
            'total_value': 1000000.0,
            'cash': 50000.0,
            'pnl': -80000.0  # 亏损8%
        }
        
        # 执行风险评估
        risk_report = controller.assess_risk(
            portfolio=high_risk_portfolio,
            market_data=self.market_data,
            volume_data=self.volume_data,
            financial_data=self.financial_data,
            risk_factors=self.risk_factors
        )
        
        # 验证风险评估生成了正确的建议
        assert 'recommended_actions' in risk_report
        recommendations = risk_report['recommended_actions']
        
        # 检查是否有针对高权重股票的建议
        has_high_weight_recommendation = any(
            rec.get('stock') == '000001.SZ' and rec.get('action') == 'reduce'
            for rec in recommendations
        )
        assert has_high_weight_recommendation
        
        # 检查是否有针对ST股票的建议
        has_st_stock_recommendation = any(
            rec.get('stock') == '000003.SZ' and rec.get('action') == 'sell'
            for rec in recommendations
        )
        assert has_st_stock_recommendation
        
        # 应用风险控制
        adjusted_portfolio = controller.apply_risk_controls(
            high_risk_portfolio,
            recommendations
        )
        
        # 验证调整后的投资组合
        assert adjusted_portfolio is not None
        
        # 检查ST股票是否已被移除
        assert '000003.SZ' not in adjusted_portfolio.get('positions', {})
        
        # 检查现金是否增加（因为卖出了ST股票）
        assert adjusted_portfolio.get('cash', 0) > high_risk_portfolio['cash']
    
    @pytest.mark.skipif(not RISK_CONTROLLER_AVAILABLE, reason="RiskController模块未安装")
    def test_risk_warning_system(self):
        """测试风险预警系统"""
        controller = RiskController(self.test_config)
        
        # 添加一些警告
        controller.add_warning("市场风险", "波动率超过阈值", "HIGH")
        controller.add_warning("信用风险", "ST股票检测", "MEDIUM")
        controller.add_warning("流动性风险", "成交额不足", "LOW")
        
        # 验证警告记录
        assert len(controller.warnings) == 3
        
        # 获取高风险警告
        high_risk_warnings = controller.get_warnings_by_severity("HIGH")
        assert len(high_risk_warnings) == 1
        assert high_risk_warnings[0]['category'] == "市场风险"
        
        # 清除警告
        controller.clear_warnings()
        assert len(controller.warnings) == 0
    
    @pytest.mark.skipif(not RISK_CONTROLLER_AVAILABLE, reason="RiskController模块未安装")
    def test_real_time_monitoring(self):
        """测试实时监控功能"""
        controller = RiskController(self.test_config)
        
        # 模拟实时数据流
        for i in range(5):
            # 创建实时市场数据
            pass  # 这里可以添加实时监控测试逻辑
