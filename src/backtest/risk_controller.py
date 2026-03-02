"""
风险控制器模块
用于监控和控制投资组合的各种风险
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import logging
from scipy import stats
import json


class RiskController:
    """风险控制器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化风险控制器
        
        Args:
            config: 风险配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 加载风险配置
        self.risk_config = config.get('risk', {})
        self.market_config = self.risk_config.get('market', {})
        self.credit_config = self.risk_config.get('credit', {})
        self.liquidity_config = self.risk_config.get('liquidity', {})
        self.concentration_config = self.risk_config.get('concentration', {})
        self.stop_loss_config = self.risk_config.get('stop_loss', {})
        
        # 初始化预警和动作记录
        self.warnings = []  # 风险预警记录
        self.actions = []   # 已执行的风险控制动作
        
        # 风险阈值
        self._setup_thresholds()
        
        # 历史数据缓存
        self.historical_data = {}
        self.risk_metrics_history = []
        
        self.logger.info("风险控制器初始化完成")
    
    def _setup_thresholds(self):
        """设置风险阈值"""
        # 市场风险阈值
        self.max_volatility = self.market_config.get('max_volatility', 0.3)
        self.min_correlation = self.market_config.get('min_correlation', 0.3)
        self.max_beta = self.market_config.get('max_beta', 1.5)
        
        # 信用风险阈值
        self.filter_st = self.credit_config.get('filter_st', True)
        self.min_current_ratio = self.credit_config.get('min_current_ratio', 1.0)
        self.max_debt_ratio = self.credit_config.get('max_debt_ratio', 0.7)
        
        # 流动性风险阈值
        self.min_turnover = self.liquidity_config.get('min_turnover', 10000000)
        self.max_bid_ask_spread = self.liquidity_config.get('max_bid_ask_spread', 0.02)
        
        # 集中度风险阈值
        self.max_single_weight = self.concentration_config.get('max_single_weight', 0.4)
        self.max_industry_weight = self.concentration_config.get('max_industry_weight', 0.3)
        self.max_style_exposure = self.concentration_config.get('max_style_exposure', 0.5)
        
        # 止损阈值
        self.stop_loss_enabled = self.stop_loss_config.get('enabled', True)
        self.stop_loss_threshold = self.stop_loss_config.get('threshold', -0.1)
        self.trailing_stop_loss = self.stop_loss_config.get('trailing', True)
        
        # 移动止损跟踪
        self.trailing_highs = {}
    
    def add_warning(self, category: str, message: str, severity: str = "MEDIUM"):
        """
        添加风险预警
        
        Args:
            category: 风险类别
            message: 预警信息
            severity: 严重程度 (HIGH, MEDIUM, LOW)
        """
        warning = {
            'timestamp': datetime.now(),
            'category': category,
            'message': message,
            'severity': severity
        }
        self.warnings.append(warning)
        self.logger.warning(f"[{severity}] {category}: {message}")
    
    def clear_warnings(self):
        """清除所有预警"""
        self.warnings.clear()
    
    def get_warnings_by_severity(self, severity: str) -> List[Dict]:
        """
        按严重程度获取预警
        
        Args:
            severity: 严重程度
            
        Returns:
            预警列表
        """
        return [w for w in self.warnings if w['severity'] == severity]
    
    def assess_risk(self, 
                   portfolio: Dict[str, Any],
                   market_data: pd.DataFrame,
                   volume_data: Optional[pd.DataFrame] = None,
                   financial_data: Optional[Dict[str, Dict]] = None,
                   risk_factors: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行综合风险评估
        
        Args:
            portfolio: 投资组合数据
            market_data: 市场数据 (价格)
            volume_data: 成交量数据
            financial_data: 财务数据
            risk_factors: 风险因子数据
            
        Returns:
            风险报告
        """
        risk_report = {
            'timestamp': datetime.now(),
            'market_risk': {},
            'credit_risk': {},
            'liquidity_risk': {},
            'concentration_risk': {},
            'risk_metrics': {},
            'warnings': [],
            'recommended_actions': []
        }
        
        try:
            # 1. 市场风险评估
            market_risk_result = self._assess_market_risk(market_data, portfolio)
            risk_report['market_risk'] = market_risk_result
            
            # 2. 信用风险评估
            if financial_data:
                credit_risk_result = self._assess_credit_risk(financial_data, portfolio)
                risk_report['credit_risk'] = credit_risk_result
            
            # 3. 流动性风险评估
            if volume_data is not None:
                liquidity_risk_result = self._assess_liquidity_risk(
                    market_data, volume_data, portfolio
                )
                risk_report['liquidity_risk'] = liquidity_risk_result
            
            # 4. 集中度风险评估
            if 'positions' in portfolio:
                concentration_risk_result = self._assess_concentration_risk(
                    portfolio, risk_factors
                )
                risk_report['concentration_risk'] = concentration_risk_result
            
            # 5. 计算风险指标
            risk_metrics = self._calculate_risk_metrics(market_data, portfolio)
            risk_report['risk_metrics'] = risk_metrics
            
            # 6. 生成预警和建议
            risk_report['warnings'] = self.warnings.copy()
            risk_report['recommended_actions'] = self._generate_recommendations(
                portfolio, risk_report
            )
            
            # 保存历史记录
            self.risk_metrics_history.append({
                'timestamp': risk_report['timestamp'],
                'metrics': risk_metrics
            })
            
        except Exception as e:
            self.logger.error(f"风险评估失败: {e}")
            self.add_warning("系统错误", f"风险评估失败: {str(e)}", "HIGH")
        
        return risk_report
    
    def _assess_market_risk(self, 
                           market_data: pd.DataFrame,
                           portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """评估市场风险"""
        result = {
            'volatility_analysis': {},
            'correlation_analysis': {},
            'beta_analysis': {},
            'warnings': []
        }
        
        try:
            # 计算收益率
            returns = market_data.pct_change().dropna()
            
            if returns.empty:
                return result
            
            # 1. 波动率分析
            volatility = self._calculate_volatility(returns)
            result['volatility_analysis']['individual'] = volatility
            
            # 检查波动率阈值
            volatility_warnings = []
            for stock, vol in volatility.items():
                if vol > self.max_volatility:
                    warning_msg = f"股票{stock}波动率{vol:.2%}超过阈值{self.max_volatility:.2%}"
                    self.add_warning("市场风险", warning_msg, "HIGH")
                    volatility_warnings.append(warning_msg)
            
            if volatility_warnings:
                result['volatility_analysis']['warnings'] = volatility_warnings
            
            # 组合波动率
            if 'positions' in portfolio:
                portfolio_vol = self._calculate_portfolio_volatility(returns, portfolio)
                result['volatility_analysis']['portfolio'] = portfolio_vol
            
            # 2. 相关性分析
            correlation = self._calculate_correlation(returns)
            result['correlation_analysis']['matrix'] = correlation
            
            # 检查相关性
            low_corr_stocks = self._identify_low_correlation_stocks(correlation)
            if low_corr_stocks:
                warning_msg = f"低相关性股票: {', '.join(low_corr_stocks)}"
                self.add_warning("市场风险", warning_msg, "MEDIUM")
                result['correlation_analysis']['warnings'] = low_corr_stocks
            
            # 3. Beta分析
            if len(returns.columns) > 1:
                beta_analysis = self._calculate_beta_analysis(returns)
                result['beta_analysis'] = beta_analysis
                
                # 检查Beta阈值
                for stock, beta in beta_analysis.get('individual_betas', {}).items():
                    if abs(beta) > self.max_beta:
                        warning_msg = f"股票{stock} Beta值{beta:.2f}超过阈值{self.max_beta}"
                        self.add_warning("市场风险", warning_msg, "MEDIUM")
                        
        except Exception as e:
            self.logger.error(f"市场风险评估失败: {e}")
            self.add_warning("市场风险", f"评估失败: {str(e)}", "HIGH")
        
        return result
    
    def _assess_credit_risk(self, 
                           financial_data: Dict[str, Dict],
                           portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """评估信用风险"""
        result = {
            'st_stocks': [],
            'financial_warnings': [],
            'credit_ratings': {}
        }
        
        try:
            # 1. ST股票识别
            st_stocks = self._identify_st_stocks(financial_data)
            result['st_stocks'] = st_stocks
            
            if st_stocks and self.filter_st:
                warning_msg = f"检测到ST股票: {', '.join(st_stocks)}"
                self.add_warning("信用风险", warning_msg, "HIGH")
            
            # 2. 财务指标检查
            financial_warnings = self._check_financial_metrics(financial_data)
            result['financial_warnings'] = financial_warnings
            
            for warning in financial_warnings:
                self.add_warning("信用风险", warning, "MEDIUM")
            
            # 3. 信用评级
            for stock, data in financial_data.items():
                if 'credit_rating' in data:
                    result['credit_ratings'][stock] = data['credit_rating']
                    
                    # 检查低信用评级
                    rating = data['credit_rating']
                    if rating in ['C', 'D', '违约']:
                        warning_msg = f"股票{stock}信用评级较低: {rating}"
                        self.add_warning("信用风险", warning_msg, "HIGH")
                        
        except Exception as e:
            self.logger.error(f"信用风险评估失败: {e}")
            self.add_warning("信用风险", f"评估失败: {str(e)}", "HIGH")
        
        return result
    
    def _assess_liquidity_risk(self,
                              market_data: pd.DataFrame,
                              volume_data: pd.DataFrame,
                              portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """评估流动性风险"""
        result = {
            'turnover_analysis': {},
            'liquidity_warnings': []
        }
        
        try:
            # 获取最新数据
            if market_data.empty or volume_data.empty:
                return result
            
            latest_prices = market_data.iloc[-1]
            latest_volumes = volume_data.iloc[-1]
            
            # 计算成交额
            turnover = latest_volumes * latest_prices
            result['turnover_analysis'] = turnover.to_dict()
            
            # 检查成交额阈值
            low_liquidity_stocks = []
            for stock, turnover_value in turnover.items():
                if turnover_value < self.min_turnover:
                    low_liquidity_stocks.append(stock)
                    warning_msg = f"股票{stock}成交额{turnover_value:,.0f}低于阈值{self.min_turnover:,.0f}"
                    self.add_warning("流动性风险", warning_msg, "MEDIUM")
                    result['liquidity_warnings'].append(warning_msg)
            
            # 检查投资组合中的低流动性股票
            if 'positions' in portfolio:
                portfolio_stocks = list(portfolio['positions'].keys())
                portfolio_low_liquidity = [s for s in portfolio_stocks if s in low_liquidity_stocks]
                
                if portfolio_low_liquidity:
                    warning_msg = f"投资组合中包含低流动性股票: {', '.join(portfolio_low_liquidity)}"
                    self.add_warning("流动性风险", warning_msg, "HIGH")
            
        except Exception as e:
            self.logger.error(f"流动性风险评估失败: {e}")
            self.add_warning("流动性风险", f"评估失败: {str(e)}", "HIGH")
        
        return result
    
    def _assess_concentration_risk(self,
                                  portfolio: Dict[str, Any],
                                  risk_factors: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """评估集中度风险"""
        result = {
            'position_concentration': {},
            'industry_concentration': {},
            'style_exposure': {},
            'warnings': []
        }
        
        try:
            if 'positions' not in portfolio:
                return result
            
            positions = portfolio['positions']
            
            # 1. 仓位集中度
            weights = {stock: pos['weight'] for stock, pos in positions.items()}
            result['position_concentration']['weights'] = weights
            
            # 检查单一股票权重
            concentration_warnings = self._check_concentration_risk(weights)
            result['position_concentration']['warnings'] = concentration_warnings
            
            for warning in concentration_warnings:
                self.add_warning("集中度风险", warning, "MEDIUM")
            
            # 2. 行业集中度
            if risk_factors and 'industry' in risk_factors:
                industry_weights = self._calculate_industry_weights(
                    weights, risk_factors['industry']
                )
                result['industry_concentration']['weights'] = industry_weights
                
                # 检查行业集中度
                industry_warnings = self._check_industry_concentration(industry_weights)
                result['industry_concentration']['warnings'] = industry_warnings
                
                for warning in industry_warnings:
                    self.add_warning("集中度风险", warning, "MEDIUM")
            
            # 3. 风格暴露
            if risk_factors and 'style' in risk_factors:
                style_exposure = self._calculate_style_exposure(
                    weights, risk_factors['style']
                )
                result['style_exposure'] = style_exposure
                
                # 检查风格暴露
                style_warnings = self._check_style_exposure(style_exposure)
                result['style_exposure']['warnings'] = style_warnings
                
                for warning in style_warnings:
                    self.add_warning("集中度风险", warning, "LOW")
                    
        except Exception as e:
            self.logger.error(f"集中度风险评估失败: {e}")
            self.add_warning("集中度风险", f"评估失败: {str(e)}", "HIGH")
        
        return result
    
    def _calculate_risk_metrics(self,
                               market_data: pd.DataFrame,
                               portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """计算风险指标"""
        metrics = {}
        
        try:
            # 计算收益率
            returns = market_data.pct_change().dropna()
            
            if returns.empty:
                return metrics
            
            # 计算投资组合收益率（等权平均）
            portfolio_returns = returns.mean(axis=1)
            
            # 1. VaR (Value at Risk)
            var_95 = self._calculate_var(portfolio_returns, confidence_level=0.95)
            var_99 = self._calculate_var(portfolio_returns, confidence_level=0.99)
            metrics['var_95'] = var_95
            metrics['var_99'] = var_99
            
            # 2. CVaR (Conditional VaR)
            cvar_95 = self._calculate_cvar(portfolio_returns, confidence_level=0.95)
            cvar_99 = self._calculate_cvar(portfolio_returns, confidence_level=0.99)
            metrics['cvar_95'] = cvar_95
            metrics['cvar_99'] = cvar_99
            
            # 3. 最大回撤
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            metrics['max_drawdown'] = max_drawdown
            
            # 4. 夏普比率 (假设无风险利率2%)
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.02)
            metrics['sharpe_ratio'] = sharpe_ratio
            
            # 5. 索提诺比率
            sortino_ratio = self._calculate_sortino_ratio(portfolio_returns, risk_free_rate=0.02)
            metrics['sortino_ratio'] = sortino_ratio
            
            # 6. 波动率
            volatility = portfolio_returns.std() * np.sqrt(252)  # 年化波动率
            metrics['annual_volatility'] = volatility
            
            # 7. 偏度和峰度
            metrics['skewness'] = portfolio_returns.skew()
            metrics['kurtosis'] = portfolio_returns.kurtosis()
            
        except Exception as e:
            self.logger.error(f"风险指标计算失败: {e}")
            self.add_warning("风险指标", f"计算失败: {str(e)}", "MEDIUM")
        
        return metrics
    
    def _generate_recommendations(self,
                                 portfolio: Dict[str, Any],
                                 risk_report: Dict[str, Any]) -> List[Dict]:
        """生成风险控制建议"""
        recommendations = []
        
        try:
            # 1. 处理ST股票
            if self.filter_st and 'credit_risk' in risk_report:
                st_stocks = risk_report['credit_risk'].get('st_stocks', [])
                for stock in st_stocks:
                    if stock in portfolio.get('positions', {}):
                        recommendations.append({
                            'action': 'sell',
                            'stock': stock,
                            'reason': 'ST股票',
                            'priority': 'HIGH'
                        })
            
            # 2. 处理仓位集中度
            if 'positions' in portfolio:
                positions = portfolio['positions']
                weights = {stock: pos['weight'] for stock, pos in positions.items()}
                
                for stock, weight in weights.items():
                    if weight > self.max_single_weight:
                        target_weight = self.max_single_weight * 0.9  # 降低到阈值的90%
                        reduction = weight - target_weight
                        recommendations.append({
                            'action': 'reduce',
                            'stock': stock,
                            'target_weight': target_weight,
                            'reduction': reduction,
                            'reason': f'仓位集中度{weight:.1%}超过阈值{self.max_single_weight:.1%}',
                            'priority': 'MEDIUM'
                        })
            
            # 3. 处理行业集中度
            if 'concentration_risk' in risk_report:
                industry_weights = risk_report['concentration_risk'].get(
                    'industry_concentration', {}
                ).get('weights', {})
                
                for industry, weight in industry_weights.items():
                    if weight > self.max_industry_weight:
                        recommendations.append({
                            'action': 'diversify',
                            'industry': industry,
                            'current_weight': weight,
                            'target_weight': self.max_industry_weight,
                            'reason': f'行业{industry}集中度{weight:.1%}超过阈值{self.max_industry_weight:.1%}',
                            'priority': 'MEDIUM'
                        })
            
            # 4. 处理低流动性股票
            if 'liquidity_risk' in risk_report:
                liquidity_warnings = risk_report['liquidity_risk'].get('liquidity_warnings', [])
                if liquidity_warnings:
                    # 找出投资组合中的低流动性股票
                    portfolio_stocks = list(portfolio.get('positions', {}).keys())
                    for warning in liquidity_warnings:
                        for stock in portfolio_stocks:
                            if stock in warning:
                                recommendations.append({
                                    'action': 'reduce_liquidity',
                                    'stock': stock,
                                    'reason': '流动性不足',
                                    'priority': 'LOW'
                                })
            
            # 5. 处理高波动率股票
            if 'market_risk' in risk_report:
                volatility_warnings = risk_report['market_risk'].get(
                    'volatility_analysis', {}
                ).get('warnings', [])
                
                for warning in volatility_warnings:
                    # 解析警告信息中的股票代码
                    for stock in portfolio.get('positions', {}).keys():
                        if stock in warning:
                            recommendations.append({
                                'action': 'hedge',
                                'stock': stock,
                                'reason': '波动率过高',
                                'priority': 'MEDIUM'
                            })
            
            # 6. 止损建议
            if self.stop_loss_enabled and 'positions' in portfolio:
                # 这里需要实际的盈亏数据，暂时跳过
                pass
            
        except Exception as e:
            self.logger.error(f"生成建议失败: {e}")
        
        return recommendations
    
    def apply_risk_controls(self,
                           portfolio: Dict[str, Any],
                           recommendations: List[Dict]) -> Dict[str, Any]:
        """
        应用风险控制措施
        
        Args:
            portfolio: 原始投资组合
            recommendations: 风险控制建议
            
        Returns:
            调整后的投资组合
        """
        adjusted_portfolio = portfolio.copy()
        
        if 'positions' not in adjusted_portfolio:
            return adjusted_portfolio
        
        # 按优先级排序建议
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        sorted_recommendations = sorted(
            recommendations,
            key=lambda x: priority_order.get(x.get('priority', 'LOW'), 2)
        )
        
        actions_taken = []
        
        for rec in sorted_recommendations:
            action = rec.get('action')
            stock = rec.get('stock')
            reason = rec.get('reason', '')
            
            if not stock or stock not in adjusted_portfolio['positions']:
                continue
            
            position = adjusted_portfolio['positions'][stock]
            
            if action == 'sell':
                # 卖出全部持仓
                adjusted_portfolio['cash'] += position.get('market_value', 0)
                del adjusted_portfolio['positions'][stock]
                actions_taken.append(f"卖出{stock}: {reason}")
                
            elif action == 'reduce':
                # 减少持仓
                target_weight = rec.get('target_weight', 0)
                current_weight = position.get('weight', 0)
                
                if current_weight > target_weight:
                    reduction_ratio = (current_weight - target_weight) / current_weight
                    # 实际应用中需要计算具体的股数和金额
                    actions_taken.append(f"减少{stock}持仓: {reason}")
            
            elif action == 'hedge':
                # 对冲建议（记录但不自动执行）
                actions_taken.append(f"对冲{stock}: {reason}")
            
            elif action == 'diversify':
                # 分散化建议（记录但不自动执行）
                industry = rec.get('industry', '')
                actions_taken.append(f"分散行业{industry}: {reason}")
        
        # 记录执行的动作
        for action in actions_taken:
            self.actions.append({
                'timestamp': datetime.now(),
                'action': action
            })
            self.logger.info(f"执行风险控制: {action}")
        
        return adjusted_portfolio
    
    # ========== 核心计算方法 ==========
    
    def _calculate_volatility(self, returns: pd.DataFrame) -> Dict[str, float]:
        """计算波动率"""
        volatility = {}
        for column in returns.columns:
            # 年化波动率
            vol = returns[column].std() * np.sqrt(252)
            volatility[column] = vol
        return volatility
    
    def _calculate_portfolio_volatility(self, 
                                       returns: pd.DataFrame,
                                       portfolio: Dict[str, Any]) -> float:
        """计算投资组合波动率"""
        if 'positions' not in portfolio:
            return 0.0
        
        positions = portfolio['positions']
        stocks = list(positions.keys())
        
        if not stocks:
            return 0.0
        
        # 获取权重
        weights = np.array([positions[s].get('weight', 0) for s in stocks])
        weights = weights / weights.sum()  # 归一化
        
        # 获取收益率数据
        stock_returns = returns[stocks].values if len(stocks) > 1 else returns[stocks].values.reshape(-1, 1)
        
        # 计算协方差矩阵
        cov_matrix = np.cov(stock_returns.T)
        
        # 计算组合波动率
        if len(stocks) == 1:
            portfolio_vol = np.sqrt(cov_matrix[0, 0]) * np.sqrt(252)
        else:
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
        
        return portfolio_vol
    
    def _calculate_correlation(self, returns: pd.DataFrame) -> pd.DataFrame:
        """计算相关性矩阵"""
        return returns.corr()
    
    def _identify_low_correlation_stocks(self, 
                                        correlation: pd.DataFrame,
                                        threshold: float = None) -> List[str]:
        """识别低相关性股票"""
        if threshold is None:
            threshold = self.min_correlation
        
        low_corr_stocks = []
        stocks = correlation.columns
        
        for i, stock1 in enumerate(stocks):
            for stock2 in stocks[i+1:]:
                corr = correlation.loc[stock1, stock2]
                if abs(corr) < threshold:
                    low_corr_stocks.extend([stock1, stock2])
        
        return list(set(low_corr_stocks))
    
    def _calculate_beta_analysis(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """计算Beta分析"""
        result = {
            'individual_betas': {},
            'portfolio_beta': None
        }
        
        if len(returns.columns) < 2:
            return result
        
        # 使用第一只股票作为市场基准（简化）
        market_returns = returns.iloc[:, 0]
        
        for stock in returns.columns[1:]:
            stock_returns = returns[stock]
            
            # 计算Beta
            covariance = np.cov(stock_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            if market_variance > 0:
                beta = covariance / market_variance
                result['individual_betas'][stock] = beta
        
        return result
    
    def _identify_st_stocks(self, financial_data: Dict[str, Dict]) -> List[str]:
        """识别ST股票"""
        st_stocks = []
        for stock, data in financial_data.items():
            if data.get('is_st', False):
                st_stocks.append(stock)
        return st_stocks
    
    def _check_liquidity_metrics(self,
                                stocks: List[str],
                                turnover: Dict[str, float]) -> List[str]:
        """检查流动性指标"""
        warnings = []
        
        for stock in stocks:
            turnover_value = turnover.get(stock, 0)
            if turnover_value < self.min_turnover:
                warnings.append(f"股票{stock}成交额{turnover_value:,.0f}低于阈值{self.min_turnover:,.0f}")
        
        return warnings
    
    def _check_financial_metrics(self, financial_data: Dict[str, Dict]) -> List[str]:
        """检查财务指标"""
        warnings = []
        
        for stock, data in financial_data.items():
            # 检查流动比率
            current_ratio = data.get('current_ratio')
            if current_ratio is not None and current_ratio < self.min_current_ratio:
                warnings.append(f"股票{stock}流动比率{current_ratio:.2f}低于阈值{self.min_current_ratio}")
            
            # 检查资产负债率
            debt_ratio = data.get('debt_ratio')
            if debt_ratio is not None and debt_ratio > self.max_debt_ratio:
                warnings.append(f"股票{stock}资产负债率{debt_ratio:.1%}超过阈值{self.max_debt_ratio:.1%}")
        
        return warnings
    
    def _check_concentration_risk(self, weights: Dict[str, float]) -> List[str]:
        """检查仓位集中度风险"""
        warnings = []
        
        for stock, weight in weights.items():
            if weight > self.max_single_weight:
                warnings.append(f"股票{stock}权重{weight:.1%}超过阈值{self.max_single_weight:.1%}")
        
        return warnings
    
    def _calculate_industry_weights(self, 
                                   weights: Dict[str, float],
                                   industry_mapping: Dict[str, str]) -> Dict[str, float]:
        """计算行业权重"""
        industry_weights = {}
        
        for stock, weight in weights.items():
            industry = industry_mapping.get(stock)
            if industry:
                industry_weights[industry] = industry_weights.get(industry, 0) + weight
        
        return industry_weights
    
    def _check_industry_concentration(self, industry_weights: Dict[str, float]) -> List[str]:
        """检查行业集中度"""
        warnings = []
        
        for industry, weight in industry_weights.items():
            if weight > self.max_industry_weight:
                warnings.append(f"行业{industry}权重{weight:.1%}超过阈值{self.max_industry_weight:.1%}")
        
        return warnings
    
    def _calculate_style_exposure(self,
                                 weights: Dict[str, float],
                                 style_mapping: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """计算风格暴露"""
        style_exposure = {}
        
        for stock, weight in weights.items():
            styles = style_mapping.get(stock, {})
            for style, exposure in styles.items():
                style_exposure[style] = style_exposure.get(style, 0) + weight * exposure
        
        return style_exposure
    
    def _check_style_exposure(self, style_exposure: Dict[str, float]) -> List[str]:
        """检查风格暴露"""
        warnings = []
        
        for style, exposure in style_exposure.items():
            if exposure > self.max_style_exposure:
                warnings.append(f"风格{style}暴露{exposure:.1%}超过阈值{self.max_style_exposure:.1%}")
        
        return warnings
    
    def _calculate_var(self, 
                      returns: pd.Series, 
                      confidence_level: float = 0.95) -> float:
        """计算Value at Risk (VaR)"""
        if returns.empty:
            return 0.0
        
        # 历史模拟法
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var
    
    def _calculate_cvar(self, 
                       returns: pd.Series, 
                       confidence_level: float = 0.95) -> float:
        """计算Conditional VaR (CVaR)"""
        if returns.empty:
            return 0.0
        
        var = self._calculate_var(returns, confidence_level)
        # CVaR是超过VaR的损失的平均值
        tail_losses = returns[returns <= var]
        
        if len(tail_losses) > 0:
            cvar = tail_losses.mean()
        else:
            cvar = var
        
        return cvar
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        if returns.empty:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self, 
                               returns: pd.Series, 
                               risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if returns.empty or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        
        if volatility > 0:
            sharpe = excess_returns / volatility
        else:
            sharpe = 0.0
        
        return sharpe
    
    def _calculate_sortino_ratio(self, 
                                returns: pd.Series, 
                                risk_free_rate: float = 0.02,
                                target_return: float = 0.0) -> float:
        """计算索提诺比率"""
        if returns.empty:
            return 0.0
        
        # 计算下行偏差
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            downside_deviation = 0.0
        else:
            downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
        
        excess_returns = returns.mean() * 252 - risk_free_rate
        
        if downside_deviation > 0:
            sortino = excess_returns / downside_deviation
        else:
            sortino = 0.0 if excess_returns >= 0 else -np.inf
        
        return sortino
    
    def _check_stop_loss(self,
                        positions: Dict[str, Dict],
                        position_pnl: Dict[str, float]) -> List[Dict]:
        """检查止损"""
        actions = []
        
        if not self.stop_loss_enabled:
            return actions
        
        for stock, position in positions.items():
            pnl = position_pnl.get(stock, 0)
            
            # 检查是否触发止损
            if pnl <= self.stop_loss_threshold:
                actions.append({
                    'action': 'stop_loss',
                    'stock': stock,
                    'pnl': pnl,
                    'threshold': self.stop_loss_threshold,
                    'reason': f'亏损{pnl:.1%}超过止损阈值{self.stop_loss_threshold:.1%}'
                })
            
            # 检查移动止损
            if self.trailing_stop_loss:
                current_price = position.get('current_price', 0)
                cost_price = position.get('cost', 0)
                
                if current_price > 0 and cost_price > 0:
                    # 更新最高价
                    if stock not in self.trailing_highs:
                        self.trailing_highs[stock] = current_price
                    else:
                        self.trailing_highs[stock] = max(self.trailing_highs[stock], current_price)
                    
                    # 检查是否触发移动止损
                    trailing_high = self.trailing_highs[stock]
                    trailing_stop_price = trailing_high * (1 + self.stop_loss_threshold)
                    
                    if current_price <= trailing_stop_price:
                        actions.append({
                            'action': 'trailing_stop_loss',
                            'stock': stock,
                            'current_price': current_price,
                            'trailing_high': trailing_high,
                            'stop_price': trailing_stop_price,
                            'reason': f'价格{current_price:.2f}触发移动止损{trailing_stop_price:.2f}'
                        })
        
        return actions
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        summary = {
            'total_warnings': len(self.warnings),
            'high_risk_warnings': len(self.get_warnings_by_severity('HIGH')),
            'medium_risk_warnings': len(self.get_warnings_by_severity('MEDIUM')),
            'low_risk_warnings': len(self.get_warnings_by_severity('LOW')),
            'total_actions': len(self.actions),
            'latest_metrics': self.risk_metrics_history[-1] if self.risk_metrics_history else {}
        }
        
        return summary
    
    def save_risk_report(self, risk_report: Dict[str, Any], filepath: str):
        """保存风险报告"""
        try:
            # 转换非JSON序列化对象
            report_copy = risk_report.copy()
            
            # 递归处理所有值
            def convert_value(obj):
                if isinstance(obj, dict):
                    return {k: convert_value(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_value(item) for item in obj]
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            report_copy = convert_value(report_copy)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_copy, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"风险报告已保存到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存风险报告失败: {e}")
    
    def load_risk_report(self, filepath: str) -> Dict[str, Any]:
        """加载风险报告"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            self.logger.info(f"风险报告已从{filepath}加载")
            return report
            
        except Exception as e:
            self.logger.error(f"加载风险报告失败: {e}")
            return {}
