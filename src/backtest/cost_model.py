"""
成本模型模块 - 用于精确计算交易成本

支持A股市场的交易成本计算，包括：
1. 佣金计算
2. 印花税计算
3. 过户费计算
4. 滑点成本计算
5. 冲击成本计算
6. 动态成本调整
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import yaml
from pathlib import Path
import warnings


class CostModel:
    """
    交易成本模型类
    
    用于计算A股市场的交易成本，支持动态成本调整和成本优化建议。
    """
    
    # 默认成本配置（A股市场标准）
    DEFAULT_CONFIG = {
        'commission': {
            'rate': 0.0003,  # 0.03% 比例佣金
            'min': 5.0       # 最低5元
        },
        'stamp_tax': {
            'rate': 0.001,   # 0.1% 印花税
            'sell_only': True  # 仅卖出时收取
        },
        'transfer_fee': {
            'rate': 0.00002,  # 0.002% 过户费
            'sh_only': True   # 仅沪市收取
        },
        'slippage': {
            'rate': 0.001,    # 0.1% 固定滑点
            'dynamic': True,  # 启用动态滑点
            'spread_factor': 0.5  # 买卖价差因子
        },
        'impact_cost': {
            'enabled': True,      # 启用冲击成本
            'liquidity_based': True,  # 基于流动性的冲击成本
            'linear_factor': 0.0005,  # 线性冲击因子
            'nonlinear_factor': 0.001,  # 非线性冲击因子
            'market_factor': 0.0003   # 市场冲击因子
        },
        'dynamic_adjustment': {
            'liquidity_weight': 0.3,   # 流动性调整权重
            'order_size_weight': 0.3,  # 订单规模调整权重
            'volatility_weight': 0.2,  # 波动率调整权重
            'frequency_weight': 0.2    # 交易频率调整权重
        }
    }
    
    def __init__(self, config: Optional[Union[Dict, str]] = None):
        """
        初始化成本模型
        
        参数:
            config: 成本配置，可以是字典或YAML文件路径
                  如果为None，使用默认配置
        """
        # 加载配置
        self.config = self._load_config(config)
        
        # 解析配置参数
        self._parse_config()
        
        # 初始化动态调整参数
        self._init_dynamic_params()
        
        # 交易历史记录（用于成本优化）
        self.trade_history = []
        
    def _load_config(self, config: Optional[Union[Dict, str]]) -> Dict:
        """加载配置"""
        if config is None:
            return self.DEFAULT_CONFIG.copy()
        
        if isinstance(config, str):
            # 从YAML文件加载
            config_path = Path(config)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                # 提取成本配置部分
                if 'transaction_cost' in loaded_config:
                    return self._extract_cost_config(loaded_config['transaction_cost'])
                else:
                    warnings.warn("配置文件中未找到transaction_cost部分，使用默认配置")
                    return self.DEFAULT_CONFIG.copy()
            else:
                warnings.warn(f"配置文件不存在: {config}，使用默认配置")
                return self.DEFAULT_CONFIG.copy()
        
        elif isinstance(config, dict):
            # 使用提供的字典配置
            return config
        
        else:
            raise ValueError("config参数必须是字典、文件路径或None")
    
    def _extract_cost_config(self, transaction_cost: Dict) -> Dict:
        """从transaction_cost配置中提取成本配置"""
        cost_config = self.DEFAULT_CONFIG.copy()
        
        # 提取佣金配置
        if 'commission' in transaction_cost:
            comm = transaction_cost['commission']
            if 'rate' in comm:
                cost_config['commission']['rate'] = comm['rate']
            if 'min_commission' in comm:
                cost_config['commission']['min'] = comm['min_commission']
        
        # 提取印花税配置
        if 'stamp_duty' in transaction_cost:
            stamp = transaction_cost['stamp_duty']
            if 'rate' in stamp:
                cost_config['stamp_tax']['rate'] = stamp['rate']
            if 'side' in stamp:
                cost_config['stamp_tax']['sell_only'] = (stamp['side'] == 'sell')
        
        # 提取过户费配置
        if 'transfer_fee' in transaction_cost:
            transfer = transaction_cost['transfer_fee']
            if 'rate' in transfer:
                cost_config['transfer_fee']['rate'] = transfer['rate']
            # 注意：配置中可能没有sh_only字段
        
        # 提取滑点配置
        if 'slippage' in transaction_cost:
            slip = transaction_cost['slippage']
            if 'enabled' in slip:
                cost_config['slippage']['dynamic'] = slip['enabled']
            if 'fixed_rate' in slip:
                cost_config['slippage']['rate'] = slip['fixed_rate']
            if 'proportional_rate' in slip:
                cost_config['slippage']['rate'] = slip['proportional_rate']
        
        # 提取冲击成本配置
        if 'market_impact' in transaction_cost:
            impact = transaction_cost['market_impact']
            if 'enabled' in impact:
                cost_config['impact_cost']['enabled'] = impact['enabled']
            if 'impact_per_trade' in impact:
                cost_config['impact_cost']['linear_factor'] = impact['impact_per_trade']
        
        return cost_config
    
    def _parse_config(self):
        """解析配置参数"""
        # 佣金参数
        self.commission_rate = self.config['commission']['rate']
        self.min_commission = self.config['commission']['min']
        
        # 印花税参数
        self.stamp_tax_rate = self.config['stamp_tax']['rate']
        self.stamp_tax_sell_only = self.config['stamp_tax']['sell_only']
        
        # 过户费参数
        self.transfer_fee_rate = self.config['transfer_fee']['rate']
        self.transfer_fee_sh_only = self.config['transfer_fee'].get('sh_only', True)
        
        # 滑点参数
        self.slippage_rate = self.config['slippage']['rate']
        self.slippage_dynamic = self.config['slippage']['dynamic']
        self.slippage_spread_factor = self.config['slippage'].get('spread_factor', 0.5)
        
        # 冲击成本参数
        self.impact_cost_enabled = self.config['impact_cost']['enabled']
        self.impact_liquidity_based = self.config['impact_cost']['liquidity_based']
        self.impact_linear_factor = self.config['impact_cost'].get('linear_factor', 0.0005)
        self.impact_nonlinear_factor = self.config['impact_cost'].get('nonlinear_factor', 0.001)
        self.impact_market_factor = self.config['impact_cost'].get('market_factor', 0.0003)
        
        # 动态调整参数
        dynamic_adj = self.config.get('dynamic_adjustment', {})
        self.liquidity_weight = dynamic_adj.get('liquidity_weight', 0.3)
        self.order_size_weight = dynamic_adj.get('order_size_weight', 0.3)
        self.volatility_weight = dynamic_adj.get('volatility_weight', 0.2)
        self.frequency_weight = dynamic_adj.get('frequency_weight', 0.2)
    
    def _init_dynamic_params(self):
        """初始化动态调整参数"""
        # 流动性调整系数（流动性越差，系数越高）
        self.liquidity_adjustment = lambda x: 1.0 + (1.0 - x) * 2.0
        
        # 订单规模调整系数（订单越大，系数越高）
        self.order_size_adjustment = lambda x: 1.0 + x * 3.0
        
        # 波动率调整系数（波动率越高，系数越高）
        self.volatility_adjustment = lambda x: 1.0 + x * 2.0
        
        # 交易频率调整系数（频率越高，系数越高）
        self.frequency_adjustment = lambda x: 1.0 + x * 1.5
    
    def calculate_commission(self, trade_amount: float) -> float:
        """
        计算佣金
        
        参数:
            trade_amount: 交易金额
            
        返回:
            佣金金额
        """
        commission = trade_amount * self.commission_rate
        return max(commission, self.min_commission)
    
    def calculate_commission_batch(self, trade_amounts: np.ndarray) -> np.ndarray:
        """
        批量计算佣金
        
        参数:
            trade_amounts: 交易金额数组
            
        返回:
            佣金金额数组
        """
        commissions = trade_amounts * self.commission_rate
        commissions = np.maximum(commissions, self.min_commission)
        return commissions
    
    def calculate_stamp_tax(self, trade_amount: float, side: str = 'sell') -> float:
        """
        计算印花税
        
        参数:
            trade_amount: 交易金额
            side: 交易方向 ('buy' 或 'sell')
            
        返回:
            印花税金额
        """
        if self.stamp_tax_sell_only and side != 'sell':
            return 0.0
        return trade_amount * self.stamp_tax_rate
    
    def calculate_stamp_tax_batch(self, trade_amounts: np.ndarray, sides: np.ndarray) -> np.ndarray:
        """
        批量计算印花税
        
        参数:
            trade_amounts: 交易金额数组
            sides: 交易方向数组 ('buy' 或 'sell')
            
        返回:
            印花税金额数组
        """
        stamp_taxes = trade_amounts * self.stamp_tax_rate
        
        if self.stamp_tax_sell_only:
            # 仅卖出时收取
            mask = (sides == 'sell')
            stamp_taxes = np.where(mask, stamp_taxes, 0.0)
        
        return stamp_taxes
    
    def calculate_transfer_fee(self, trade_amount: float, exchange: str = 'SH') -> float:
        """
        计算过户费
        
        参数:
            trade_amount: 交易金额
            exchange: 交易所 ('SH' 或 'SZ')
            
        返回:
            过户费金额
        """
        if self.transfer_fee_sh_only and exchange != 'SH':
            return 0.0
        return trade_amount * self.transfer_fee_rate
    
    def calculate_transfer_fee_batch(self, trade_amounts: np.ndarray, exchanges: np.ndarray) -> np.ndarray:
        """
        批量计算过户费
        
        参数:
            trade_amounts: 交易金额数组
            exchanges: 交易所数组 ('SH' 或 'SZ')
            
        返回:
            过户费金额数组
        """
        transfer_fees = trade_amounts * self.transfer_fee_rate
        
        if self.transfer_fee_sh_only:
            # 仅沪市收取
            mask = (exchanges == 'SH')
            transfer_fees = np.where(mask, transfer_fees, 0.0)
        
        return transfer_fees
    
    def calculate_slippage(self, 
                          trade_amount: float, 
                          method: str = 'fixed',
                          **kwargs) -> float:
        """
        计算滑点成本
        
        参数:
            trade_amount: 交易金额
            method: 滑点计算方法 ('fixed', 'dynamic', 'spread')
            **kwargs: 额外参数
                - liquidity_score: 流动性分数 (0-1)
                - bid_ask_spread: 买卖价差
                
        返回:
            滑点成本金额
        """
        if method == 'fixed':
            # 固定滑点
            return trade_amount * self.slippage_rate
        
        elif method == 'dynamic':
            # 动态滑点（基于流动性）
            liquidity_score = kwargs.get('liquidity_score', 0.5)
            adjustment = self.liquidity_adjustment(liquidity_score)
            return trade_amount * self.slippage_rate * adjustment
        
        elif method == 'spread':
            # 基于买卖价差的滑点
            bid_ask_spread = kwargs.get('bid_ask_spread', 0.002)  # 默认0.2%
            # 假设成交在中间价，滑点为价差的一半
            return trade_amount * bid_ask_spread * self.slippage_spread_factor
        
        else:
            raise ValueError(f"不支持的滑点计算方法: {method}")
    
    def calculate_impact_cost(self,
                            trade_amount: float,
                            model: str = 'linear',
                            **kwargs) -> float:
        """
        计算冲击成本
        
        参数:
            trade_amount: 交易金额
            model: 冲击成本模型 ('linear', 'liquidity', 'market')
            **kwargs: 额外参数
                - order_size_ratio: 订单规模占日均成交比例
                - liquidity_score: 流动性分数 (0-1)
                - market_volatility: 市场波动率
                
        返回:
            冲击成本金额
        """
        if not self.impact_cost_enabled:
            return 0.0
        
        if model == 'linear':
            # 线性冲击成本模型
            order_size_ratio = kwargs.get('order_size_ratio', 0.01)
            return trade_amount * self.impact_linear_factor * order_size_ratio
        
        elif model == 'liquidity':
            # 基于流动性的非线性模型
            liquidity_score = kwargs.get('liquidity_score', 0.5)
            order_size_ratio = kwargs.get('order_size_ratio', 0.01)
            
            # 流动性越差，冲击成本越高
            liquidity_factor = 1.0 + (1.0 - liquidity_score) * 2.0
            return trade_amount * self.impact_nonlinear_factor * order_size_ratio * liquidity_factor
        
        elif model == 'market':
            # 市场冲击模型
            order_size_ratio = kwargs.get('order_size_ratio', 0.01)
            market_volatility = kwargs.get('market_volatility', 0.02)
            
            # 考虑市场波动率和订单规模
            volatility_factor = 1.0 + market_volatility * 10.0  # 放大波动率影响
            return trade_amount * self.impact_market_factor * order_size_ratio * volatility_factor
        
        else:
            raise ValueError(f"不支持的冲击成本模型: {model}")
    
    def calculate_total_cost(self,
                           trade_amount: float,
                           side: str = 'sell',
                           exchange: str = 'SH',
                           liquidity_score: float = 0.5,
                           order_size_ratio: float = 0.01,
                           market_volatility: float = 0.02,
                           bid_ask_spread: float = 0.002,
                           include_impact: bool = True) -> float:
        """
        计算单笔交易总成本
        
        参数:
            trade_amount: 交易金额
            side: 交易方向 ('buy' 或 'sell')
            exchange: 交易所 ('SH' 或 'SZ')
            liquidity_score: 流动性分数 (0-1)
            order_size_ratio: 订单规模占日均成交比例
            market_volatility: 市场波动率
            bid_ask_spread: 买卖价差
            include_impact: 是否包含冲击成本
            
        返回:
            总成本金额
        """
        # 计算各项成本
        commission = self.calculate_commission(trade_amount)
        stamp_tax = self.calculate_stamp_tax(trade_amount, side)
        transfer_fee = self.calculate_transfer_fee(trade_amount, exchange)
        
        # 动态滑点
        slippage = self.calculate_slippage(
            trade_amount, 
            method='dynamic' if self.slippage_dynamic else 'fixed',
            liquidity_score=liquidity_score,
            bid_ask_spread=bid_ask_spread
        )
        
        # 冲击成本
        impact_cost = 0.0
        if include_impact and self.impact_cost_enabled:
            if self.impact_liquidity_based:
                impact_cost = self.calculate_impact_cost(
                    trade_amount,
                    model='liquidity',
                    order_size_ratio=order_size_ratio,
                    liquidity_score=liquidity_score
                )
            else:
                impact_cost = self.calculate_impact_cost(
                    trade_amount,
                    model='market',
                    order_size_ratio=order_size_ratio,
                    market_volatility=market_volatility
                )
        
        # 总成本
        total_cost = commission + stamp_tax + transfer_fee + slippage + impact_cost
        
        # 记录交易历史（用于成本优化）
        self._record_trade_history(
            trade_amount, side, exchange, liquidity_score, 
            order_size_ratio, market_volatility, total_cost
        )
        
        return total_cost
    
    def calculate_cost_breakdown(self,
                               trade_amount: float,
                               side: str = 'sell',
                               exchange: str = 'SH',
                               liquidity_score: float = 0.5,
                               order_size_ratio: float = 0.01,
                               market_volatility: float = 0.02,
                               bid_ask_spread: float = 0.002,
                               include_impact: bool = True) -> Dict[str, float]:
        """
        计算成本分解
        
        参数:
            同 calculate_total_cost
            
        返回:
            成本分解字典
        """
        # 计算各项成本
        commission = self.calculate_commission(trade_amount)
        stamp_tax = self.calculate_stamp_tax(trade_amount, side)
        transfer_fee = self.calculate_transfer_fee(trade_amount, exchange)
        
        # 动态滑点
        slippage = self.calculate_slippage(
            trade_amount, 
            method='dynamic' if self.slippage_dynamic else 'fixed',
            liquidity_score=liquidity_score,
            bid_ask_spread=bid_ask_spread
        )
        
        # 冲击成本
        impact_cost = 0.0
        if include_impact and self.impact_cost_enabled:
            if self.impact_liquidity_based:
                impact_cost = self.calculate_impact_cost(
                    trade_amount,
                    model='liquidity',
                    order_size_ratio=order_size_ratio,
                    liquidity_score=liquidity_score
                )
            else:
                impact_cost = self.calculate_impact_cost(
                    trade_amount,
                    model='market',
                    order_size_ratio=order_size_ratio,
                    market_volatility=market_volatility
                )
        
        # 总成本
        total_cost = commission + stamp_tax + transfer_fee + slippage + impact_cost
        
        return {
            'commission': commission,
            'stamp_tax': stamp_tax,
            'transfer_fee': transfer_fee,
            'slippage': slippage,
            'impact_cost': impact_cost,
            'total': total_cost
        }
    
    def calculate_total_cost_batch(self,
                                 trade_amounts: np.ndarray,
                                 sides: np.ndarray,
                                 exchanges: np.ndarray,
                                 liquidity_scores: Optional[np.ndarray] = None,
                                 order_size_ratios: Optional[np.ndarray] = None,
                                 market_volatilities: Optional[np.ndarray] = None,
                                 bid_ask_spreads: Optional[np.ndarray] = None,
                                 include_impact: bool = True) -> np.ndarray:
        """
        批量计算总成本
        
        参数:
            trade_amounts: 交易金额数组
            sides: 交易方向数组 ('buy' 或 'sell')
            exchanges: 交易所数组 ('SH' 或 'SZ')
            liquidity_scores: 流动性分数数组 (0-1)
            order_size_ratios: 订单规模比例数组
            market_volatilities: 市场波动率数组
            bid_ask_spreads: 买卖价差数组
            include_impact: 是否包含冲击成本
            
        返回:
            总成本数组
        """
        n_trades = len(trade_amounts)
        
        # 设置默认值
        if liquidity_scores is None:
            liquidity_scores = np.full(n_trades, 0.5)
        if order_size_ratios is None:
            order_size_ratios = np.full(n_trades, 0.01)
        if market_volatilities is None:
            market_volatilities = np.full(n_trades, 0.02)
        if bid_ask_spreads is None:
            bid_ask_spreads = np.full(n_trades, 0.002)
        
        # 批量计算各项成本
        commissions = self.calculate_commission_batch(trade_amounts)
        stamp_taxes = self.calculate_stamp_tax_batch(trade_amounts, sides)
        transfer_fees = self.calculate_transfer_fee_batch(trade_amounts, exchanges)
        
        # 批量计算滑点成本
        slippages = np.zeros(n_trades)
        for i in range(n_trades):
            slippages[i] = self.calculate_slippage(
                trade_amounts[i],
                method='dynamic' if self.slippage_dynamic else 'fixed',
                liquidity_score=liquidity_scores[i],
                bid_ask_spread=bid_ask_spreads[i]
            )
        
        # 批量计算冲击成本
        impact_costs = np.zeros(n_trades)
        if include_impact and self.impact_cost_enabled:
            for i in range(n_trades):
                if self.impact_liquidity_based:
                    impact_costs[i] = self.calculate_impact_cost(
                        trade_amounts[i],
                        model='liquidity',
                        order_size_ratio=order_size_ratios[i],
                        liquidity_score=liquidity_scores[i]
                    )
                else:
                    impact_costs[i] = self.calculate_impact_cost(
                        trade_amounts[i],
                        model='market',
                        order_size_ratio=order_size_ratios[i],
                        market_volatility=market_volatilities[i]
                    )
        
        # 总成本
        total_costs = commissions + stamp_taxes + transfer_fees + slippages + impact_costs
        
        # 记录交易历史
        for i in range(n_trades):
            self._record_trade_history(
                trade_amounts[i], sides[i], exchanges[i], liquidity_scores[i],
                order_size_ratios[i], market_volatilities[i], total_costs[i]
            )
        
        return total_costs
    
    def _record_trade_history(self,
                            trade_amount: float,
                            side: str,
                            exchange: str,
                            liquidity_score: float,
                            order_size_ratio: float,
                            market_volatility: float,
                            total_cost: float):
        """
        记录交易历史
        
        参数:
            trade_amount: 交易金额
            side: 交易方向
            exchange: 交易所
            liquidity_score: 流动性分数
            order_size_ratio: 订单规模比例
            market_volatility: 市场波动率
            total_cost: 总成本
        """
        trade_record = {
            'trade_amount': trade_amount,
            'side': side,
            'exchange': exchange,
            'liquidity_score': liquidity_score,
            'order_size_ratio': order_size_ratio,
            'market_volatility': market_volatility,
            'cost': total_cost,
            'cost_ratio': total_cost / trade_amount if trade_amount > 0 else 0.0
        }
        self.trade_history.append(trade_record)
        
        # 限制历史记录长度（保留最近1000笔交易）
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
    
    def get_cost_optimization_suggestions(self, 
                                        trade_history: Optional[List[Dict]] = None) -> Dict:
        """
        获取成本优化建议
        
        参数:
            trade_history: 交易历史记录，如果为None则使用内部记录
            
        返回:
            优化建议字典
        """
        if trade_history is None:
            trade_history = self.trade_history
        
        if not trade_history:
            return {
                'total_cost': 0.0,
                'avg_cost_ratio': 0.0,
                'suggestions': []
            }
        
        # 计算统计指标
        total_cost = sum(trade['cost'] for trade in trade_history)
        total_amount = sum(trade['trade_amount'] for trade in trade_history)
        avg_cost_ratio = total_cost / total_amount if total_amount > 0 else 0.0
        
        # 分析成本结构
        suggestions = []
        
        # 1. 检查高成本交易
        high_cost_trades = [t for t in trade_history if t['cost_ratio'] > avg_cost_ratio * 1.5]
        if high_cost_trades:
            suggestions.append({
                'type': 'high_cost_trades',
                'description': f'发现{len(high_cost_trades)}笔高成本交易（成本率>{avg_cost_ratio*1.5:.4%}）',
                'potential_saving': sum(t['cost'] for t in high_cost_trades) * 0.3,  # 假设可节省30%
                'action': '考虑优化这些交易的执行时机或拆分大额订单'
            })
        
        # 2. 检查流动性差的交易
        low_liquidity_trades = [t for t in trade_history if t['liquidity_score'] < 0.3]
        if low_liquidity_trades:
            suggestions.append({
                'type': 'low_liquidity',
                'description': f'发现{len(low_liquidity_trades)}笔低流动性交易（流动性分数<0.3）',
                'potential_saving': sum(t['cost'] for t in low_liquidity_trades) * 0.2,
                'action': '避免在流动性差的时段交易，或使用限价单代替市价单'
            })
        
        # 3. 检查大额订单
        large_order_trades = [t for t in trade_history if t['order_size_ratio'] > 0.02]
        if large_order_trades:
            suggestions.append({
                'type': 'large_orders',
                'description': f'发现{len(large_order_trades)}笔大额订单（规模>日均成交2%）',
                'potential_saving': sum(t['cost'] for t in large_order_trades) * 0.25,
                'action': '考虑拆分大额订单，使用TWAP/VWAP算法执行'
            })
        
        # 4. 检查高波动市场中的交易
        high_volatility_trades = [t for t in trade_history if t['market_volatility'] > 0.03]
        if high_volatility_trades:
            suggestions.append({
                'type': 'high_volatility',
                'description': f'发现{len(high_volatility_trades)}笔高波动市场中的交易（波动率>3%）',
                'potential_saving': sum(t['cost'] for t in high_volatility_trades) * 0.15,
                'action': '避免在市场剧烈波动时交易，或设置更宽松的滑点容忍度'
            })
        
        # 5. 检查交易频率
        if len(trade_history) > 50:  # 如果有足够多的交易记录
            # 计算平均交易间隔（假设交易历史按时间顺序）
            avg_trade_size = total_amount / len(trade_history)
            suggestions.append({
                'type': 'trading_frequency',
                'description': f'平均交易规模：{avg_trade_size:,.2f}元，共{len(trade_history)}笔交易',
                'potential_saving': total_cost * 0.1,  # 假设通过优化频率可节省10%
                'action': '考虑合并小额交易，降低交易频率以减少固定成本'
            })
        
        return {
            'total_cost': total_cost,
            'total_amount': total_amount,
            'avg_cost_ratio': avg_cost_ratio,
            'num_trades': len(trade_history),
            'suggestions': suggestions,
            'estimated_total_saving': sum(s['potential_saving'] for s in suggestions)
        }
    
    def clear_trade_history(self):
        """清空交易历史记录"""
        self.trade_history = []
    
    def get_cost_summary(self) -> Dict:
        """
        获取成本模型摘要
        
        返回:
            成本模型配置摘要
        """
        return {
            'commission': {
                'rate': self.commission_rate,
                'min': self.min_commission
            },
            'stamp_tax': {
                'rate': self.stamp_tax_rate,
                'sell_only': self.stamp_tax_sell_only
            },
            'transfer_fee': {
                'rate': self.transfer_fee_rate,
                'sh_only': self.transfer_fee_sh_only
            },
            'slippage': {
                'rate': self.slippage_rate,
                'dynamic': self.slippage_dynamic
            },
            'impact_cost': {
                'enabled': self.impact_cost_enabled,
                'liquidity_based': self.impact_liquidity_based
            },
            'trade_history_size': len(self.trade_history)
        }
    
    def estimate_optimal_order_size(self,
                                  daily_volume: float,
                                  liquidity_score: float = 0.5,
                                  target_cost_ratio: float = 0.001) -> float:
        """
        估算最优订单规模
        
        参数:
            daily_volume: 日均成交量
            liquidity_score: 流动性分数
            target_cost_ratio: 目标成本率
            
        返回:
            最优订单规模
        """
        # 简化模型：基于目标成本率反推订单规模
        # 假设成本主要由冲击成本和滑点构成
        
        # 基础成本率（固定成本部分）
        base_cost_ratio = self.commission_rate + self.transfer_fee_rate
        
        # 可变的成本率（与订单规模相关）
        variable_cost_ratio = target_cost_ratio - base_cost_ratio
        
        if variable_cost_ratio <= 0:
            # 如果目标成本率太低，返回保守估计
            return daily_volume * 0.01  # 1%的日均成交量
        
        # 估算最大订单规模（基于可变成本率）
        # 简化假设：可变成本与订单规模比例成正比
        max_order_size_ratio = variable_cost_ratio / (self.impact_linear_factor * 2)
        
        # 考虑流动性调整
        liquidity_factor = 1.0 / self.liquidity_adjustment(liquidity_score)
        optimal_ratio = max_order_size_ratio * liquidity_factor
        
        # 限制在合理范围内
        optimal_ratio = max(0.001, min(optimal_ratio, 0.05))  # 0.1%到5%
        
        return daily_volume * optimal_ratio