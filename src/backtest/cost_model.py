#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FZT排序增强策略成本模型

功能：计算交易中的各种成本，包括佣金、印花税、过户费、滑点成本和冲击成本。

作者：FZT项目组
创建日期：2026年3月2日
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CostType(Enum):
    """成本类型枚举"""
    COMMISSION = "commission"  # 佣金
    STAMP_TAX = "stamp_tax"    # 印花税
    TRANSFER_FEE = "transfer_fee"  # 过户费
    SLIPPAGE = "slippage"      # 滑点成本
    IMPACT = "impact"          # 冲击成本


@dataclass
class CostConfig:
    """成本配置"""
    # 佣金配置
    commission_rate: float = 0.0003  # 0.03%
    min_commission: float = 5.0      # 最低5元
    
    # 印花税配置
    stamp_tax_rate: float = 0.001    # 0.1%
    stamp_tax_sell_only: bool = True  # 仅卖出时收取
    
    # 过户费配置
    transfer_fee_rate: float = 0.00002  # 0.002%
    transfer_fee_sh_only: bool = True   # 仅沪市收取
    
    # 滑点配置
    slippage_rate: float = 0.001        # 0.1%
    dynamic_slippage: bool = True       # 动态滑点
    
    # 冲击成本配置
    impact_enabled: bool = True
    impact_liquidity_factor: float = 0.0001  # 流动性因子
    
    # 其他配置
    round_decimals: int = 2  # 金额舍入小数位


class CostModel:
    """成本模型类"""
    
    def __init__(self, config: Optional[CostConfig] = None):
        """
        初始化成本模型
        
        Args:
            config: 成本配置，如果为None则使用默认配置
        """
        self.config = config or CostConfig()
        self._validate_config()
        
        logger.info("成本模型初始化完成")
        logger.info(f"佣金率: {self.config.commission_rate:.4%}")
        logger.info(f"印花税率: {self.config.stamp_tax_rate:.3%}")
        logger.info(f"滑点率: {self.config.slippage_rate:.3%}")
    
    def _validate_config(self):
        """验证配置"""
        if self.config.commission_rate < 0:
            raise ValueError("佣金率不能为负数")
        if self.config.stamp_tax_rate < 0:
            raise ValueError("印花税率不能为负数")
        if self.config.slippage_rate < 0:
            raise ValueError("滑点率不能为负数")
        if self.config.min_commission < 0:
            raise ValueError("最低佣金不能为负数")
    
    def calculate_commission(self, trade_amount: float, is_buy: bool = True) -> float:
        """
        计算佣金
        
        Args:
            trade_amount: 交易金额
            is_buy: 是否为买入交易
            
        Returns:
            佣金金额
        """
        if trade_amount <= 0:
            return 0.0
        
        commission = trade_amount * self.config.commission_rate
        commission = max(commission, self.config.min_commission)
        
        # 买入和卖出都收取佣金
        return round(commission, self.config.round_decimals)
    
    def calculate_stamp_tax(self, trade_amount: float, is_buy: bool = True) -> float:
        """
        计算印花税
        
        Args:
            trade_amount: 交易金额
            is_buy: 是否为买入交易
            
        Returns:
            印花税金额
        """
        if trade_amount <= 0:
            return 0.0
        
        # A股印花税仅卖出时收取
        if self.config.stamp_tax_sell_only and is_buy:
            return 0.0
        
        stamp_tax = trade_amount * self.config.stamp_tax_rate
        return round(stamp_tax, self.config.round_decimals)
    
    def calculate_transfer_fee(self, trade_amount: float, stock_code: str = "") -> float:
        """
        计算过户费
        
        Args:
            trade_amount: 交易金额
            stock_code: 股票代码，用于判断是否为沪市股票
            
        Returns:
            过户费金额
        """
        if trade_amount <= 0:
            return 0.0
        
        # 仅沪市股票收取过户费
        if self.config.transfer_fee_sh_only:
            if not stock_code.startswith('SH'):
                return 0.0
        
        transfer_fee = trade_amount * self.config.transfer_fee_rate
        return round(transfer_fee, self.config.round_decimals)
    
    def calculate_slippage(self, trade_amount: float, 
                          liquidity_score: float = 1.0) -> float:
        """
        计算滑点成本
        
        Args:
            trade_amount: 交易金额
            liquidity_score: 流动性评分 (0-1，1表示流动性最好)
            
        Returns:
            滑点成本金额
        """
        if trade_amount <= 0:
            return 0.0
        
        if self.config.dynamic_slippage:
            # 动态滑点：流动性差的股票滑点更高
            adjusted_rate = self.config.slippage_rate / max(liquidity_score, 0.1)
        else:
            adjusted_rate = self.config.slippage_rate
        
        slippage = trade_amount * adjusted_rate
        return round(slippage, self.config.round_decimals)
    
    def calculate_impact_cost(self, trade_amount: float, 
                             avg_daily_volume: float,
                             order_size_ratio: float = None) -> float:
        """
        计算冲击成本
        
        Args:
            trade_amount: 交易金额
            avg_daily_volume: 日均成交额
            order_size_ratio: 订单规模占日均成交额比例
            
        Returns:
            冲击成本金额
        """
        if not self.config.impact_enabled or trade_amount <= 0:
            return 0.0
        
        if order_size_ratio is None:
            order_size_ratio = trade_amount / avg_daily_volume if avg_daily_volume > 0 else 0
        
        # 简单的冲击成本模型：订单规模越大，冲击成本越高
        # 使用平方根模型：冲击成本 ∝ sqrt(订单规模比例)
        impact_factor = self.config.impact_liquidity_factor
        impact_cost = trade_amount * impact_factor * np.sqrt(order_size_ratio)
        
        return round(impact_cost, self.config.round_decimals)
    
    def calculate_total_cost(self, trade_amount: float, 
                           is_buy: bool = True,
                           stock_code: str = "",
                           liquidity_score: float = 1.0,
                           avg_daily_volume: float = 0.0,
                           order_size_ratio: float = None) -> Dict[str, float]:
        """
        计算总成本
        
        Args:
            trade_amount: 交易金额
            is_buy: 是否为买入交易
            stock_code: 股票代码
            liquidity_score: 流动性评分
            avg_daily_volume: 日均成交额
            order_size_ratio: 订单规模占日均成交额比例
            
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
        commission = self.calculate_commission(trade_amount, is_buy)
        stamp_tax = self.calculate_stamp_tax(trade_amount, is_buy)
        transfer_fee = self.calculate_transfer_fee(trade_amount, stock_code)
        slippage = self.calculate_slippage(trade_amount, liquidity_score)
        impact = self.calculate_impact_cost(trade_amount, avg_daily_volume, order_size_ratio)
        
        # 计算总成本
        total_cost = commission + stamp_tax + transfer_fee + slippage + impact
        
        return {
            "total": round(total_cost, self.config.round_decimals),
            "commission": commission,
            "stamp_tax": stamp_tax,
            "transfer_fee": transfer_fee,
            "slippage": slippage,
            "impact": impact
        }
    
    def calculate_batch_costs(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        批量计算交易成本
        
        Args:
            trades: 交易DataFrame，包含以下列：
                - trade_amount: 交易金额
                - is_buy: 是否为买入
                - stock_code: 股票代码
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
                is_buy=trade.get('is_buy', True),
                stock_code=trade.get('stock_code', ''),
                liquidity_score=trade.get('liquidity_score', 1.0),
                avg_daily_volume=trade.get('avg_daily_volume', 0.0),
                order_size_ratio=trade.get('order_size_ratio')
            )
            results.append(cost_detail)
        
        return pd.DataFrame(results)
    
    def estimate_cost_ratio(self, trade_amount: float, 
                          cost_detail: Dict[str, float] = None) -> float:
        """
        估算成本比率（成本占交易金额的比例）
        
        Args:
            trade_amount: 交易金额
            cost_detail: 成本明细（如果为None则计算）
            
        Returns:
            成本比率
        """
        if trade_amount <= 0:
            return 0.0
        
        if cost_detail is None:
            cost_detail = self.calculate_total_cost(trade_amount)
        
        total_cost = cost_detail.get('total', 0)
        cost_ratio = total_cost / trade_amount if trade_amount > 0 else 0
        
        return cost_ratio
    
    def optimize_trade_size(self, target_amount: float,
                          liquidity_score: float = 1.0,
                          max_cost_ratio: float = 0.01) -> Dict[str, float]:
        """
        优化交易规模，使成本比率不超过阈值
        
        Args:
            target_amount: 目标交易金额
            liquidity_score: 流动性评分
            max_cost_ratio: 最大允许的成本比率
            
        Returns:
            优化后的交易建议
        """
        if target_amount <= 0:
            return {
                "optimized_amount": 0.0,
                "original_amount": 0.0,
                "cost_ratio": 0.0,
                "is_optimized": False
            }
        
        # 计算原始成本比率
        original_cost = self.calculate_total_cost(target_amount, liquidity_score=liquidity_score)
        original_ratio = self.estimate_cost_ratio(target_amount, original_cost)
        
        if original_ratio <= max_cost_ratio:
            # 原始成本比率已满足要求
            return {
                "optimized_amount": target_amount,
                "original_amount": target_amount,
                "cost_ratio": original_ratio,
                "is_optimized": False
            }
        
        # 二分查找优化交易规模
        low, high = 0.0, target_amount
        optimized_amount = target_amount
        
        for _ in range(20):  # 最多迭代20次
            mid = (low + high) / 2
            cost = self.calculate_total_cost(mid, liquidity_score=liquidity_score)
            ratio = self.estimate_cost_ratio(mid, cost)
            
            if ratio <= max_cost_ratio:
                low = mid
                optimized_amount = mid
            else:
                high = mid
        
        optimized_cost = self.calculate_total_cost(optimized_amount, liquidity_score=liquidity_score)
        optimized_ratio = self.estimate_cost_ratio(optimized_amount, optimized_cost)
        
        return {
            "optimized_amount": round(optimized_amount, self.config.round_decimals),
            "original_amount": target_amount,
            "cost_ratio": optimized_ratio,
            "original_cost_ratio": original_ratio,
            "cost_reduction": original_cost['total'] - optimized_cost['total'],
            "is_optimized": True
        }
    
    def generate_cost_report(self, trades: pd.DataFrame) -> Dict[str, any]:
        """
        生成成本分析报告
        
        Args:
            trades: 交易DataFrame
            
        Returns:
            成本分析报告
        """
        if trades.empty:
            return {
                "summary": {
                    "total_trades": 0,
                    "total_amount": 0.0,
                    "total_cost": 0.0,
                    "avg_cost_ratio": 0.0
                },
                "breakdown": {},
                "recommendations": []
            }
        
        # 计算成本明细
        cost_df = self.calculate_batch_costs(trades)
        
        # 汇总统计
        total_trades = len(trades)
        total_amount = trades['trade_amount'].sum()
        total_cost = cost_df['total'].sum()
        avg_cost_ratio = total_cost / total_amount if total_amount > 0 else 0
        
        # 成本细分
        cost_breakdown = {
            "commission": cost_df['commission'].sum(),
            "stamp_tax": cost_df['stamp_tax'].sum(),
            "transfer_fee": cost_df['transfer_fee'].sum(),
            "slippage": cost_df['slippage'].sum(),
            "impact": cost_df['impact'].sum()
        }
        
        # 成本占比
        cost_percentage = {
            cost_type: amount / total_cost if total_cost > 0 else 0
            for cost_type, amount in cost_breakdown.items()
        }
        
        # 生成建议
        recommendations = []
        
        if cost_percentage.get('slippage', 0) > 0.3:
            recommendations.append("滑点成本占比过高，建议优化交易时机或拆分大额订单")
        
        if cost_percentage.get('impact', 0) > 0.2:
            recommendations.append("冲击成本较高，建议选择流动性更好的股票或减小单笔交易规模")
        
        if avg_cost_ratio > 0.005:  # 0.5%
            recommendations.append(f"平均成本比率较高({avg_cost_ratio:.3%})，建议优化交易策略")
        
        return {
            "summary": {
                "total_trades": total_trades,
                "total_amount": round(total_amount, 2),
                "total_cost": round(total_cost, 2),
                "avg_cost_ratio": round(avg_cost_ratio, 5),
                "avg_cost_per_trade": round(total_cost / total_trades, 2) if total_trades > 0 else 0
            },
            "breakdown": {
                "amounts": cost_breakdown,
                "percentages": cost_percentage
            },
            "recommendations": recommendations
        }


# 便捷函数
def create_cost_model(config_dict: Dict = None) -> CostModel:
    """
    从字典创建成本模型
    
    Args:
        config_dict: 配置字典
        
    Returns:
        成本模型实例
    """
    if config_dict is None:
        config_dict = {}
    
    config = CostConfig(
        commission_rate=config_dict.get('commission_rate', 0.0003),
        min_commission=config_dict.get('min_commission', 5.0),
        stamp_tax_rate=config_dict.get('stamp_tax_rate', 0.001),
        stamp_tax_sell_only=config_dict.get('stamp_tax_sell_only', True),
        transfer_fee_rate=config_dict.get('transfer_fee_rate', 0.00002),
        transfer_fee_sh_only=config_dict.get('transfer_fee_sh_only', True),
        slippage_rate=config_dict.get('slippage_rate', 0.001),
        dynamic_slippage=config_dict.get('dynamic_slippage', True),
        impact_enabled=config_dict.get('impact_enabled', True),
        impact_liquidity_factor=config_dict.get('impact_liquidity_factor', 0.0001)
    )
    
    return CostModel(config)


def calculate_simple_cost(trade_amount: float, is_buy: bool = True) -> float:
    """
    简化成本计算（仅包含佣金和印花税）
    
    Args:
        trade_amount: 交易金额
        is_buy: 是否为买入
        
    Returns:
        总成本
    """
    model = CostModel()
    cost_detail = model.calculate_total_cost(trade_amount, is_buy)
    return cost_detail['total']


if __name__ == "__main__":
    # 示例用法
    model = CostModel()
    
    # 单笔交易成本计算
    trade_amount = 100000  # 10万元
    cost_detail = model.calculate_total_cost(trade_amount, is_buy=True)
    print(f"买入 {trade_amount} 元的成本明细:")
    for cost_type, amount in cost_detail.items():
        print(f"  {cost_type}: {amount:.2f} 元")
    
    # 批量成本计算
    trades = pd.DataFrame({
        'trade_amount': [50000, 100000, 200000],
        'is_buy': [True, False, True],
        'stock_code': ['SH600000', 'SZ000001', 'SH600036'],
        'liquidity_score': [0.8, 0.9, 0.7]
    })
    
    cost_df = model.c