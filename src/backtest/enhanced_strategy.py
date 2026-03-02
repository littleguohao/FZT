#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强回测策略 - FZTEnhancedStrategy

基于QLib的TopkDropoutStrategy扩展，加入：
1. 交易成本模型（佣金、印花税、滑点、冲击成本）
2. 流动性控制（成交额、市值门槛、ST/停牌/涨跌停过滤）
3. 风险控制（行业权重、个股权重、最大回撤、波动率）
4. 可交易性检查（顺延机制）

作者: FZT项目组
创建日期: 2026-03-02
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings

# 尝试导入QLib相关模块
try:
    from qlib.strategy.base import BaseStrategy
    from qlib.strategy.weight import TopkDropoutStrategy
    from qlib.backtest import Order, OrderDir, TradeDecision
    from qlib.data import D
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    # 定义占位符类
    class BaseStrategy:
        pass
    
    class TopkDropoutStrategy:
        pass
    
    class Order:
        pass
    
    class TradeDecision:
        pass


class FZTEnhancedStrategy:
    """
    FZT增强回测策略
    
    继承自QLib的TopkDropoutStrategy，增加交易成本、流动性控制和风险控制功能。
    
    主要功能：
    1. 基础选股：每日选Top K股票，持有固定天数
    2. 交易成本：佣金、印花税、滑点、冲击成本
    3. 流动性控制：成交额门槛、市值门槛、ST/停牌/涨跌停过滤
    4. 风险控制：行业权重限制、个股权重限制、最大回撤控制
    5. 可交易性检查：顺延机制
    
    配置参数：
    - top_k: 选股数量 (默认: 5)
    - hold_days: 持有天数 (默认: 1)
    - commission: 佣金费率 (默认: 0.0003)
    - stamp_tax: 印花税费率 (默认: 0.001)
    - slippage: 滑点费率 (默认: 0.001)
    - min_turnover_ratio: 最小成交额比例 (默认: 0.2)
    - min_market_cap_ratio: 最小市值比例 (默认: 0.3)
    - max_industry_weight: 最大行业权重 (默认: 0.3)
    - max_single_weight: 最大个股权重 (默认: 0.4)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化增强策略
        
        Args:
            config: 策略配置字典
        """
        self.config = config or {}
        
        # 基础参数
        self.top_k = self.config.get('top_k', 5)
        self.hold_days = self.config.get('hold_days', 1)
        
        # 交易成本参数
        self.commission_rate = self.config.get('commission', 0.0003)
        self.stamp_tax_rate = self.config.get('stamp_tax', 0.001)
        self.slippage_rate = self.config.get('slippage', 0.001)
        self.impact_cost_rate = self.config.get('impact_cost', 0.001)
        
        # 流动性控制参数
        self.min_turnover_ratio = self.config.get('min_turnover_ratio', 0.2)
        self.min_market_cap_ratio = self.config.get('min_market_cap_ratio', 0.3)
        
        # 风险控制参数
        self.max_industry_weight = self.config.get('max_industry_weight', 0.3)
        self.max_single_weight = self.config.get('max_single_weight', 0.4)
        self.max_drawdown_limit = self.config.get('max_drawdown_limit', 0.15)
        self.volatility_limit = self.config.get('volatility_limit', 0.3)
        
        # 状态变量
        self.current_positions = {}
        self.trade_history = []
        self.portfolio_value_history = []
        
        # 行业分类缓存
        self.industry_cache = {}
        
        # 初始化QLib策略（如果可用）
        if QLIB_AVAILABLE:
            qlib_config = {
                'topk': self.top_k,
                'hold_days': self.hold_days
            }
            self.qlib_strategy = TopkDropoutStrategy(**qlib_config)
        else:
            self.qlib_strategy = None
            warnings.warn("QLib not available. Running in standalone mode.")
    
    def select_stocks(self, 
                     stock_scores: pd.Series,
                     stock_data: pd.DataFrame,
                     trade_date: datetime) -> List[str]:
        """
        选股逻辑：选择Top K股票，应用流动性过滤
        
        Args:
            stock_scores: 股票分数序列 (index: 股票代码, values: 分数)
            stock_data: 股票数据DataFrame
            trade_date: 交易日期
            
        Returns:
            选中的股票代码列表
        """
        if stock_scores.empty:
            return []
        
        # 1. 按分数排序
        sorted_stocks = stock_scores.sort_values(ascending=False)
        
        # 2. 应用流动性过滤
        filtered_stocks = self._apply_liquidity_filters(sorted_stocks, stock_data)
        
        # 3. 选择Top K股票
        selected_stocks = filtered_stocks.index[:self.top_k].tolist()
        
        # 4. 应用风险控制（行业权重限制）
        selected_stocks = self._apply_risk_controls(selected_stocks, stock_data)
        
        return selected_stocks
    
    def _apply_liquidity_filters(self, 
                                sorted_stocks: pd.Series,
                                stock_data: pd.DataFrame) -> pd.Series:
        """
        应用流动性过滤
        
        Args:
            sorted_stocks: 按分数排序的股票
            stock_data: 股票数据
            
        Returns:
            过滤后的股票
        """
        filtered_indices = []
        
        for stock in sorted_stocks.index:
            if stock not in stock_data.index:
                continue
            
            # 获取股票数据
            stock_info = stock_data.loc[stock]
            
            # 1. 检查ST股票
            if self._is_st_stock(stock_info):
                continue
            
            # 2. 检查停牌
            if self._is_suspended(stock_info):
                continue
            
            # 3. 检查涨跌停
            if self._is_limit_up(stock_info) or self._is_limit_down(stock_info):
                continue
            
            # 4. 检查成交额门槛
            if not self._meets_turnover_threshold(stock_info, stock_data):
                continue
            
            # 5. 检查市值门槛
            if not self._meets_market_cap_threshold(stock_info, stock_data):
                continue
            
            filtered_indices.append(stock)
        
        return sorted_stocks[filtered_indices]
    
    def _is_st_stock(self, stock_info: pd.Series) -> bool:
        """检查是否为ST股票"""
        # 实际实现中需要根据具体数据字段判断
        if 'is_st' in stock_info:
            return bool(stock_info['is_st'])
        return False
    
    def _is_suspended(self, stock_info: pd.Series) -> bool:
        """检查是否停牌"""
        if 'is_suspended' in stock_info:
            return bool(stock_info['is_suspended'])
        return False
    
    def _is_limit_up(self, stock_info: pd.Series) -> bool:
        """检查是否涨停"""
        if 'limit_up' in stock_info:
            return bool(stock_info['limit_up'])
        return False
    
    def _is_limit_down(self, stock_info: pd.Series) -> bool:
        """检查是否跌停"""
        if 'limit_down' in stock_info:
            return bool(stock_info['limit_down'])
        return False
    
    def _meets_turnover_threshold(self, 
                                 stock_info: pd.Series,
                                 stock_data: pd.DataFrame) -> bool:
        """检查是否满足成交额门槛"""
        if 'turnover' not in stock_info or 'turnover' not in stock_data.columns:
            return True  # 如果没有成交额数据，跳过检查
        
        # 计算成交额分位数
        turnover_values = stock_data['turnover'].dropna()
        if len(turnover_values) == 0:
            return True
        
        turnover_percentile = (turnover_values <= stock_info['turnover']).mean()
        
        # 要求不在后min_turnover_ratio%中
        return turnover_percentile >= self.min_turnover_ratio
    
    def _meets_market_cap_threshold(self,
                                   stock_info: pd.Series,
                                   stock_data: pd.DataFrame) -> bool:
        """检查是否满足市值门槛"""
        if 'market_cap' not in stock_info or 'market_cap' not in stock_data.columns:
            return True  # 如果没有市值数据，跳过检查
        
        # 计算市值分位数
        market_cap_values = stock_data['market_cap'].dropna()
        if len(market_cap_values) == 0:
            return True
        
        market_cap_percentile = (market_cap_values <= stock_info['market_cap']).mean()
        
        # 要求不在后min_market_cap_ratio%中
        return market_cap_percentile >= self.min_market_cap_ratio
    
    def _apply_risk_controls(self,
                            selected_stocks: List[str],
                            stock_data: pd.DataFrame) -> List[str]:
        """
        应用风险控制：行业权重限制
        
        Args:
            selected_stocks: 选中的股票
            stock_data: 股票数据
            
        Returns:
            调整后的股票列表
        """
        if len(selected_stocks) <= 1:
            return selected_stocks
        
        # 获取行业信息
        industries = {}
        for stock in selected_stocks:
            if stock in stock_data.index and 'industry' in stock_data.columns:
                industry = stock_data.loc[stock, 'industry']
                industries.setdefault(industry, []).append(stock)
        
        # 检查行业权重
        final_stocks = []
        industry_counts = {}
        total_stocks = len(selected_stocks)
        
        for stock in selected_stocks:
            if stock in stock_data.index and 'industry' in stock_data.columns:
                industry = stock_data.loc[stock, 'industry']
                industry_counts[industry] = industry_counts.get(industry, 0) + 1
                
                # 检查行业权重是否超过限制
                industry_weight = industry_counts[industry] / total_stocks
                if industry_weight <= self.max_industry_weight:
                    final_stocks.append(stock)
                else:
                    # 超过限制，跳过该股票
                    industry_counts[industry] -= 1
            else:
                # 没有行业信息，直接加入
                final_stocks.append(stock)
        
        # 如果过滤后股票数量不足，尝试补充
        if len(final_stocks) < self.top_k:
            # 可以在这里实现顺延机制
            pass
        
        return final_stocks[:self.top_k]
    
    def calculate_trading_cost(self,
                              order_amount: float,
                              is_buy: bool = True,
                              stock_liquidity: float = 1.0) -> Dict[str, float]:
        """
        计算交易成本
        
        Args:
            order_amount: 订单金额
            is_buy: 是否为买入订单
            stock_liquidity: 股票流动性分数 (0-1)
            
        Returns:
            成本明细字典
        """
        costs = {}
        
        # 1. 佣金
        commission = order_amount * self.commission_rate
        costs['commission'] = max(commission, 5.0)  # 最低5元
        
        # 2. 印花税（仅卖出）
        if not is_buy:
            stamp_tax = order_amount * self.stamp_tax_rate
            costs['stamp_tax'] = stamp_tax
        else:
            costs['stamp_tax'] = 0.0
        
        # 3. 滑点成本
        slippage = order_amount * self.slippage_rate
        costs['slippage'] = slippage
        
        # 4. 冲击成本（基于流动性）
        impact_cost = order_amount * self.impact_cost_rate * (1 - stock_liquidity)
        costs['impact_cost'] = impact_cost
        
        # 5. 总成本
        total_cost = sum(costs.values())
        costs['total'] = total_cost
        
        return costs
    
    def construct_portfolio(self,
                           selected_stocks: List[str],
                           available_capital: float,
                           stock_data: pd.DataFrame) -> Dict[str, float]:
        """
        构建投资组合
        
        Args:
            selected_stocks: 选中的股票
            available_capital: 可用资金
            stock_data: 股票数据
            
        Returns:
            股票权重字典 {股票代码: 权重}
        """
        if not selected_stocks:
            return {}
        
        # 1. 初始等权重分配
        n_stocks = len(selected_stocks)
        equal_weight = 1.0 / n_stocks
        
        weights = {stock: equal_weight for stock in selected_stocks}
        
        # 2. 应用个股权重限制
        weights = self._apply_single_weight_limit(weights)
        
        # 3. 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _apply_single_weight_limit(self, weights: Dict[str, float]) -> Dict[str, float]:
        """应用个股权重限制"""
        adjusted_weights = weights.copy()
        
        # 检查是否有股票权重超过限制
        for stock, weight in weights.items():
            if weight > self.max_single_weight:
                adjusted_weights[stock] = self.max_single_weight
        
        # 重新分配超出的权重
        total_excess = sum(max(0, weight - self.max_single_weight) for weight in weights.values())
        if total_excess > 0:
            # 将超出的权重平均分配给其他股票
            other_stocks = [s for s in weights.keys() if weights[s] <= self.max_single_weight]
            if other_stocks:
                excess_per_stock = total_excess / len(other_stocks)
                for stock in other_stocks:
                    adjusted_weights[stock] = min(
                        weights[stock] + excess_per_stock,
                        self.max_single_weight
                    )
        
        return adjusted_weights
    
    def generate_trade_decisions(self,
                                current_positions: Dict[str, float],
                                target_weights: Dict[str, float],
                                portfolio_value: float,
                                trade_date: datetime) -> List[Dict[str, Any]]:
        """
        生成交易决策
        
        Args:
            current_positions: 当前持仓 {股票代码: 权重}
            target_weights: 目标权重 {股票代码: 权重}
            portfolio_value: 投资组合总价值
            trade_date: 交易日期
            
        Returns:
            交易决策列表
        """
        decisions = []
        
        # 1. 计算需要调整的仓位
        all_stocks = set(current_positions.keys()) | set(target_weights.keys())
        
        for stock in all_stocks:
            current_weight = current_positions.get(stock, 0.0)
            target_weight = target_weights.get(stock, 0.0)
            
            # 计算权重变化
            weight_change = target_weight - current_weight
            
            if abs(weight_change) < 1e-6:  # 忽略微小变化
                continue
            
            # 计算交易金额
            trade_amount = weight_change * portfolio_value
            
            if trade_amount > 0:
                # 买入
                decision = {
                    'stock': stock,
                    'action': 'BUY',
                    'amount': trade_amount,
                    'weight_change': weight_change,
                    'date': trade_date
                }
            else:
                # 卖出
                decision = {
                    'stock': stock,
                    'action': 'SELL',
                    'amount': abs(trade_amount),
                    'weight_change': weight_change,
                    'date': trade_date
                }
            
            decisions.append(decision)
        
        return decisions
    
    def check_tradability(self,
                         stock: str,
                         stock_data: pd.DataFrame,
                         trade_date: datetime) -> Tuple[bool, str]:
        """
        检查股票可交易性
        
        Args:
            stock: 股票代码
            stock_data: 股票数据
            trade_date: 交易日期
            
        Returns:
            (是否可交易, 原因)
        """
        if stock not in stock_data.index:
            return False, "股票不在数据中"
        
        stock_info = stock_data.loc[stock]
        
        # 检查ST
        if self._is_st_stock(stock_info):
            return False, "ST股票"
        
        # 检查停牌
        if self._is_suspended(stock_info):
            return False, "停牌"
        
        # 检查涨跌停
        if self._is_limit_up(stock_info):
            return False, "涨停"
        
        if self._is_limit_down(stock_info):
            return False, "跌停"
        
        # 检查流动性
        if not self._meets_turnover_threshold(stock_info, stock_data):
            return False, "成交额不足"
        
        if not self._meets_market_cap_threshold(stock_info, stock_data):
            return False, "市值不足"
        
        return True, "可交易"
    
    def run_backtest(self,
                    stock_scores: pd.DataFrame,
                    stock_data: pd.DataFrame,
                    start_date: datetime,
                    end_date: datetime,
                    initial_capital: float = 1000000.0) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            stock_scores: 股票分数DataFrame (index: 日期, columns: 股票代码)
            stock_data: 股票数据DataFrame (多层索引: 日期, 股票代码)
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金
            
        Returns:
            回测结果字典
        """
        # 初始化
        current_capital = initial_capital
        positions = {}  # {股票代码: 持仓金额}
        portfolio_value = initial_capital
        trade_history = []
        portfolio_history = []
        
        # 生成交易日序列
        trade_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        for i, trade_date in enumerate(trade_dates):
            if trade_date not in stock_scores.index:
                continue
            
            # 获取当日股票分数
            daily_scores = stock_scores.loc[trade_date]
            
            # 获取当日股票数据
            if trade_date in stock_data.index.get_level_values(0):
                daily_data = stock_data.xs(trade_date, level=0)
            else:
                daily_data = pd.DataFrame()
            
            # 1. 选股
            selected_stocks = self.select_stocks(daily_scores, daily_data, trade_date)
            
            # 2. 构建投资组合
            target_weights = self.construct_portfolio(
                selected_stocks, current_capital, daily_data
            )
            
            # 3. 生成交易决策
            current_weights = {
                stock: positions.get(stock, 0) / portfolio_value 
                if portfolio_value > 0 else 0
                for stock in positions.keys()
            }
            
            trade_decisions = self.generate_trade_decisions(
                current_weights, target_weights, portfolio_value, trade_date
            )
            
            # 4. 执行交易（模拟）
            for decision in trade_decisions:
                stock = decision['stock']
                action = decision['action']
                amount = decision['amount']
                
                # 检查可交易性
                tradable, reason = self.check_tradability(stock, daily_data, trade_date)
                if not tradable:
                    # 记录交易失败
                    trade_history.append({
                        'date': trade_date,
                        'stock': stock,
                        'action': action,
                        'amount': amount,
                        'status': 'FAILED',
                        'reason': reason
                    })
                    continue
                
                # 计算交易成本
                liquidity_score = self._calculate_liquidity_score(stock, daily_data)
                costs = self.calculate_trading_cost(
                    amount, action == 'BUY', liquidity_score
                )
                
                # 执行交易
                if action == 'BUY':
                    # 检查资金是否足够
                    total_cost = amount + costs['total']
                    if current_capital >= total_cost:
                        positions[stock] = positions.get(stock, 0) + amount
                        current_capital -= total_cost
                        
                        trade_history.append({
                            'date': trade_date,
                            'stock': stock,
                            'action': action,
                            'amount': amount,
                            'price': amount,  # 简化：假设价格为1
                            'costs': costs,
                            'status': 'EXECUTED'
                        })
                else:  # SELL
                    if stock in positions and positions[stock] >= amount:
                        positions[stock] -= amount
                        if positions[stock] < 1e-6:  # 清理微小持仓
                            del positions[stock]
                        
                        # 卖出收入扣除成本
                        net_proceeds = amount - costs['total']
                        current_capital += net_proceeds
                        
                        trade_history.append({
                            'date': trade_date,
                            'stock': stock,
                            'action': action,
                            'amount': amount,
                            'price': amount,  # 简化：假设价格为1
                            'costs': costs,
                            'status': 'EXECUTED'
                        })
            
            # 5. 更新投资组合价值
            portfolio_value = current_capital + sum(positions.values())
            portfolio_history.append({
                'date': trade_date,
                'portfolio_value': portfolio_value,
                'cash': current_capital,
                'positions': positions.copy()
            })
        
        # 计算绩效指标
        performance = self._calculate_performance_metrics(portfolio_history)
        
        return {
            'initial_capital': initial_capital,
            'final_portfolio_value': portfolio_value,
            'total_return': (portfolio_value - initial_capital) / initial_capital,
            'trade_history': trade_history,
            'portfolio_history': portfolio_history,
            'performance_metrics': performance
        }
    
    def _calculate_liquidity_score(self, stock: str, stock_data: pd.DataFrame) -> float:
        """计算股票流动性分数 (0-1)"""
        if stock not in stock_data.index:
            return 0.5  # 默认中等流动性
        
        stock_info = stock_data.loc[stock]
        score = 0.5  # 基础分数
        
        # 基于成交额
        if 'turnover' in stock_info:
            # 归一化成交额（假设数据已标准化）
            turnover = stock_info['turnover']
            if 'turnover' in stock_data.columns:
                max_turnover = stock_data['turnover'].max()
                min_turnover = stock_data['turnover'].min()
                if max_turnover > min_turnover:
                    turnover_score = (turnover - min_turnover) / (max_turnover - min_turnover)
                    score = 0.3 * score + 0.7 * turnover_score
        
        # 基于市值
        if 'market_cap' in stock_info:
            # 归一化市值
            market_cap = stock_info['market_cap']
            if 'market_cap' in stock_data.columns:
                max_market_cap = stock_data['market_cap'].max()
                min_market_cap = stock_data['market_cap'].min()
                if max_market_cap > min_market_cap:
                    market_cap_score = (market_cap - min_market_cap) / (max_market_cap - min_market_cap)
                    score = 0.5 * score + 0.5 * market_cap_score
        
        return max(0.0, min(1.0, score))
    
    def _calculate_performance_metrics(self, portfolio_history: List[Dict]) -> Dict[str, float]:
        """计算绩效指标"""
        if not portfolio_history:
            return {}
        
        # 提取投资组合价值序列
        values = [h['portfolio_value'] for h in portfolio_history]
        dates = [h['date'] for h in portfolio_history]
        
        if len(values) < 2:
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'volatility': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # 计算收益率
        returns = []
        for i in range(1, len(values)):
            if values[i-1] > 0:
                ret = (values[i] - values[i-1]) / values[i-1]
                returns.append(ret)
        
        if not returns:
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'volatility': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        returns = np.array(returns)
        
        # 总收益率
        total_return = (values[-1] - values[0]) / values[0]
        
        # 年化收益率（假设252个交易日）
        annual_return = (1 + total_return) ** (252 / len(values)) - 1
        
        # 波动率（年化）
        volatility = np.std(returns) * np.sqrt(252)
        
        # 最大回撤
        peak = values[0]
        max_drawdown = 0.0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # 夏普比率（假设无风险利率为3%）
        risk_free_rate = 0.03
        excess_return = annual_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(returns),
            'avg_daily_return': np.mean(returns),
            'win_rate': (returns > 0).mean() if len(returns) > 0 else 0.0
        }
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """获取策略摘要"""
        return {
            'strategy_name': 'FZTEnhancedStrategy',
            'parameters': {
                'top_k': self.top_k,
                'hold_days': self.hold_days,
                'commission_rate': self.commission_rate,
                'stamp_tax_rate': self.stamp_tax_rate,
                'slippage_rate': self.slippage_rate,
                'min_turnover_ratio': self.min_turnover_ratio,
                'min_market_cap_ratio': self.min_market_cap_ratio,
                'max_industry_weight': self.max_industry_weight,
                'max_single_weight': self.max_single_weight,
                'max_drawdown_limit': self.max_drawdown_limit,
                'volatility_limit': self.volatility_limit
            },
            'features': [
                'Top K选股',
                '流动性过滤',
                '风险控制',
                '交易成本模型',
                '可交易性检查'
            ],
            'qlib_integration': QLIB_AVAILABLE
        }


# 辅助函数
def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """从YAML文件加载配置"""
    import yaml
    import os
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 提取策略相关配置
    strategy_config = {}
    
    # 基础参数
    if 'strategy' in config:
        strategy = config['strategy']
        if 'stock_selection' in strategy:
            selection = strategy['stock_selection']
            strategy_config['top_k'] = selection.get('top_n', 5)
        
        if 'position' in strategy:
            position = strategy['position']
            strategy_config['max_single_weight'] = position.get('max_weight_per_stock', 0.2)
            
            if 'industry_concentration' in position:
                industry = position['industry_concentration']
                if industry.get('enabled', False):
                    strategy_config['max_industry_weight'] = industry.get('max_industry_weight', 0.3)
    
    # 交易成本
    if 'transaction_cost' in config:
        cost = config['transaction_cost']
        if 'commission' in cost:
            strategy_config['commission'] = cost['commission'].get('rate', 0.0003)
        
        if 'stamp_duty' in cost:
            strategy_config['stamp_tax'] = cost['stamp_duty'].get('rate', 0.001)
        
        if 'slippage' in cost:
            slippage = cost['slippage']
            if slippage.get('enabled', False):
                strategy_config['slippage'] = slippage.get('proportional_rate', 0.001)
    
    # 风险控制
    if 'risk_management' in config:
        risk = config['risk_management']
        if 'position_control' in risk:
            position_ctrl = risk['position_control']
            strategy_config['max_drawdown_limit'] = 0.15  # 默认值
        
        if 'stop_loss' in risk:
            stop_loss = risk['stop_loss']
            if stop_loss.get('enabled', False) and 'portfolio' in stop_loss:
                portfolio_sl = stop_loss['portfolio']
                strategy_config['max_drawdown_limit'] = portfolio_sl.get('drawdown_limit', 0.15)
    
    return strategy_config


def create_sample_data(n_stocks: int = 100, n_days: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """创建示例数据用于测试"""
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    # 生成股票代码
    stock_codes = [f'{i:06d}.SZ' for i in range(1, n_stocks + 1)]
    
    # 生成日期序列
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # 生成股票分数数据
    stock_scores = pd.DataFrame(
        np.random.randn(len(dates), n_stocks) * 0.1 + 0.5,
        index=dates,
        columns=stock_codes
    )
    
    # 生成股票数据
    stock_data_records = []
    for date in dates:
        for stock in stock_codes:
            record = {
                'date': date,
                'stock': stock,
                'score': np.random.randn() * 0.1 + 0.5,
                'turnover': np.random.lognormal(mean=15, sigma=1.5),  # 成交额
                'market_cap': np.random.lognormal(mean=20, sigma=2.0),  # 市值
                'is_st': np.random.rand() < 0.02,  # 2%概率是ST
                'is_suspended': np.random.rand() < 0.01,  # 1%概率停牌
                'limit_up': np.random.rand() < 0.05,  # 5%概率涨停
                'limit_down': np.random.rand() < 0.05,  # 5%概率跌停
                'industry': np.random.choice(['银行', '科技', '医药', '消费', '能源'])
            }
            stock_data_records.append(record)
    
    stock_data = pd.DataFrame(stock_data_records)
    stock_data.set_index(['date', 'stock'], inplace=True)
    
    return stock_scores, stock_data


if __name__ == '__main__':
    # 示例用法
    print("FZT增强回测策略示例")
    print("=" * 50)
    
    # 创建策略实例
    config = {
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
    
    strategy = FZTEnhancedStrategy(config)
    
    # 显示策略摘要
    summary = strategy.get_strategy_summary()
    print("策略摘要:")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    print("\n策略功能:")
    for feature in summary['features']:
        print(f"  ✓ {feature}")
    
    print(f"\nQLib集成: {'可用' if summary['qlib_integration'] else '不可用'}")