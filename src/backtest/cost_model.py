#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FZT排序增强策略成本模型 - 兼容测试版本

功能：计算交易中的各种成本，包括佣金、印花税、过户费、滑点成本和冲击成本。

作者：FZT项目组
创建日期：2026年3月2日
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class CostModel:
    """成本模型类 - 兼容测试版本"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化成本模型
        
        Args:
            config: 成本配置字典，如果为None则使用默认配置
        """
        if config is None:
            config = {}
        
        # 设置默认配置
        self.commission_rate = config.get('commission', {}).get('rate', 0.0003)
        self.min_commission = config.get('commission', {}).get('min', 5.0)
        
        self.stamp_tax_rate = config.get('stamp_tax', {}).get('rate', 0.001)
        self.stamp_tax_sell_only = config.get('stamp_tax', {}).get('sell_only', True)
        
        self.transfer_fee_rate = config.get('transfer_fee', {}).get('rate', 0.00002)
        self.transfer_fee_sh_only = config.get('transfer_fee', {}).get('sh_only', True)
        
        self.slippage_rate = config.get('slippage', {}).get('rate', 0.001)
        self.dynamic_slippage = config.get('slippage', {}).get('dynamic', True)
        
        self.impact_enabled = config.get('impact_cost', {}).get('enabled', True)
        self.impact_liquidity_factor = config.get('impact_cost', {}).get('liquidity_factor', 0.0001)
        
        self.round_decimals = 2
        
        self._validate_config()
        
        logger.info("成本模型初始化完成")
    
    def _validate_config(self):
        """验证配置"""
        if self.commission_rate < 0:
            raise ValueError("佣金率不能为负数")
        if self.stamp_tax_rate < 0:
            raise ValueError("印花税率不能为负数")
        if self.slippage_rate < 0:
            raise ValueError("滑点率不能为负数")
        if self.min_commission < 0:
            raise ValueError("最低佣金不能为负数")
    
    def calculate_commission(self, trade_amount: float, side: str = 'buy') -> float:
        """
        计算佣金
        
        Args:
            trade_amount: 交易金额
            side: 交易方向 ('buy' 或 'sell')
            
        Returns:
            佣金金额
        """
        if trade_amount <= 0:
            return 0.0
        
        commission = trade_amount * self.commission_rate
        commission = max(commission, self.min_commission)
        
        return round(commission, self.round_decimals)
    
    def calculate_stamp_tax(self, trade_amount: float, side: str = 'buy') -> float:
        """
        计算印花税
        
        Args:
            trade_amount: 交易金额
            side: 交易方向 ('buy' 或 'sell')
            
        Returns:
            印花税金额
        """
        if trade_amount <= 0:
            return 0.0
        
        # A股印花税仅卖出时收取
        if self.stamp_tax_sell_only and side.lower() != 'sell':
            return 0.0
        
        stamp_tax = trade_amount * self.stamp_tax_rate
        return round(stamp_tax, self.round_decimals)
    
    def calculate_transfer_fee(self, trade_amount: float, exchange: str = 'SH') -> float:
        """
        计算过户费
        
        Args:
            trade_amount: 交易金额
            exchange: 交易所 ('SH' 或 'SZ')
            
        Returns:
            过户费金额
        """
        if trade_amount <= 0:
            return 0.0
        
        # 仅沪市股票收取过户费
        if self.transfer_fee_sh_only and exchange.upper() != 'SH':
            return 0.0
        
        transfer_fee = trade_amount * self.transfer_fee_rate
        return round(transfer_fee, self.round_decimals)
    
    def calculate_slippage(self, trade_amount: float, 
                          method: str = 'fixed',
                          liquidity_score: float = 1.0) -> float:
        """
        计算滑点成本
        
        Args:
            trade_amount: 交易金额
            method: 滑点计算方法 ('fixed', 'dynamic')
            liquidity_score: 流动性评分 (0-1，1表示流动性最好)
            
        Returns:
            滑点成本金额
        """
        if trade_amount <= 0:
            return 0.0
        
        if method == 'dynamic' and self.dynamic_slippage:
            # 动态滑点：流动性差的股票滑点更高
            adjusted_rate = self.slippage_rate / max(liquidity_score, 0.1)
        else:
            adjusted_rate = self.slippage_rate
        
        slippage = trade_amount * adjusted_rate
        return round(slippage, self.round_decimals)
    
    def calculate_impact_cost(self, trade_amount: float, 
                             model: str = 'linear',
                             avg_daily_volume: float = 0.0) -> float:
        """
        计算冲击成本
        
        Args:
            trade_amount: 交易金额
            model: 冲击成本模型 ('linear', 'liquidity')
            avg_daily_volume: 日均成交额
            
        Returns:
            冲击成本金额
        """
        if not self.impact_enabled or trade_amount <= 0:
            return 0.0
        
        if model == 'liquidity' and avg_daily_volume > 0:
            order_size_ratio = trade_amount / avg_daily_volume
            # 使用平方根模型：冲击成本 ∝ sqrt(订单规模比例)
            impact_cost = trade_amount * self.impact_liquidity_factor * np.sqrt(order_size_ratio)
        else:
            # 线性模型
            impact_cost = trade_amount * self.impact_liquidity_factor * 0.1
        
        return round(impact_cost, self.round_decimals)
    
    def calculate_total_cost(self, trade_amount: float, 
                           side: str = 'buy',
                           exchange: str = 'SH',
                           liquidity_score: float = 1.0,
                           avg_daily_volume: float = 0.0) -> Dict[str, float]:
        """
        计算总成本
        
        Args:
            trade_amount: 交易金额
            side: 交易方向 ('buy' 或 'sell')
            exchange: 交易所 ('SH' 或 'SZ')
            liquidity_score: 流动性评分
            avg_daily_volume: 日均成交额
            
        Returns:
            包含各种成本明细的字典
        """
        if trade_amount <= 0:
            return {
                "total": 0.0,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "transfer_fee": 0.0,
                "slippage": 0.0,
                "impact": 0.0
            }
        
        # 计算各项成本
        commission = self.calculate_commission(trade_amount, side)
        stamp_tax = self.calculate_stamp_tax(trade_amount, side)
        transfer_fee = self.calculate_transfer_fee(trade_amount, exchange)
        slippage = self.calculate_slippage(trade_amount, 'dynamic', liquidity_score)
        impact = self.calculate_impact_cost(trade_amount, 'liquidity', avg_daily_volume)
        
        # 计算总成本
        total_cost = commission + stamp_tax + transfer_fee + slippage + impact
        
        return {
            "total": round(total_cost, self.round_decimals),
            "commission": commission,
            "stamp_tax": stamp_tax,
            "transfer_fee": transfer_fee,
            "slippage": slippage,
            "impact": impact
        }
    
    def calculate_total_cost_batch(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        批量计算交易成本
        
        Args:
            trades: 交易DataFrame，包含以下列：
                - trade_amount: 交易金额
                - side: 交易方向 ('buy' 或 'sell')
                - exchange: 交易所 ('SH' 或 'SZ')
                - liquidity_score: 流动性评分
                - avg_daily_volume: 日均成交额
                
        Returns:
            包含成本明细的DataFrame
        """
        if trades.empty:
            return pd.DataFrame()
        
        results = []
        for _, trade in trades.iterrows():
            cost_detail = self.calculate_total_cost(
                trade_amount=trade.get('trade_amount', 0),
                side=trade.get('side', 'buy'),
                exchange=trade.get('exchange', 'SH'),
                liquidity_score=trade.get('liquidity_score', 1.0),
                avg_daily_volume=trade.get('avg_daily_volume', 0.0)
            )
            results.append(cost_detail)
        
        return pd.DataFrame(results)
    
    def get_cost_optimization_suggestions(self, trades: pd.DataFrame) -> List[str]:
        """
        获取成本优化建议
        
        Args:
            trades: 交易DataFrame
            
        Returns:
            成本优化建议列表
        """
        if trades.empty:
            return ["暂无交易数据"]
        
        cost_df = self.calculate_total_cost_batch(trades)
        
        suggestions = []
        
        # 分析成本结构
        total_cost = cost_df['total'].sum()
        if total_cost > 0:
            commission_ratio = cost_df['commission'].sum() / total_cost
            stamp_tax_ratio = cost_df['stamp_tax'].sum() / total_cost
            slippage_ratio = cost_df['slippage'].sum() / total_cost
            
            if commission_ratio > 0.6:
                suggestions.append("佣金成本占比过高，建议优化交易频率")
            if stamp_tax_ratio > 0.3:
                suggestions.append("印花税成本较高，建议优化卖出时机")
            if slippage_ratio > 0.2:
                suggestions.append("滑点成本较高，建议选择流动性更好的股票")
        
        # 分析交易规模
        avg_trade_amount = trades['trade_amount'].mean()
        if avg_trade_amount > 1000000:  # 100万
            suggestions.append("单笔交易规模较大，建议拆分交易以降低冲击成本")
        
        if not suggestions:
            suggestions.append("成本结构合理，暂无优化建议")
        
        return suggestions
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'CostModel':
        """
        从YAML文件加载配置创建成本模型
        
        Args:
            yaml_path: YAML配置文件路径
            
        Returns:
            成本模型实例
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 提取成本配置
        cost_config = config.get('cost', {})
        return cls(cost_config)


# 便捷函数
def create_cost_model(config_dict: Dict = None) -> CostModel:
    """
    从字典创建成本模型
    
    Args:
        config_dict: 配置字典
        
    Returns:
        成本模型实例
    """
    return CostModel(config_dict)


def calculate_simple_cost(trade_amount: float, side: str = 'buy') -> float:
    """
    简化成本计算（仅包含佣金和印花税）
    
    Args:
        trade_amount: 交易金额
        side: 交易方向
        
    Returns:
        总成本
    """
    model = CostModel()
    cost_detail = model.calculate_total_cost(trade_amount, side)
    return cost_detail['total']


if __name__ == "__main__":
    # 示例用法
    model = CostModel()
    
    # 单笔交易成本计算
    trade_amount = 100000  # 10万元
    cost_detail = model.calculate_total_cost(trade_amount, side='buy')
    print(f"买入 {trade_amount} 元的成本明细:")
    for cost_type, amount in cost_detail.items():
        print(f"  {cost_type}: {amount:.2f} 元")
    
    # 批量成本计算
    trades = pd.DataFrame({
        'trade_amount': [50000, 100000, 200000],
        'side': ['buy', 'sell', 'buy'],
        'exchange': ['SH', 'SZ', 'SH'],
        'liquidity_score': [0.8, 0.9, 0.7]
    })
    
    cost_df = model.calculate_total_cost_batch(trades)
    print("\n批量交易成本:")
    print(cost_df)
    
    # 优化建议
    suggestions = model.get_cost_optimization_suggestions(trades)
    print("\n成本优化建议:")
    for suggestion in suggestions:
        print(f"  • {suggestion}")