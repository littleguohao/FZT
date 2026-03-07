#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子模块 - 包含各种技术因子计算函数

包含以下因子：
1. 成交量比和乖离率因子
2. RSI和OBV因子

作者: MC
创建日期: 2026-03-07
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


# ===================== 成交量比和乖离率因子 =====================
def add_volume_and_bias_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    为已有数据添加成交量比和乖离率列
    
    假设df已包含 instrument, datetime, close, volume 列
    
    因子定义：
    1. 成交量比 = volume / MA(volume, 5)
    2. 乖离率 = (close - MA(close, 60)) / MA(close, 60)
    
    :param df: 包含股票数据的DataFrame
    :return: 添加了成交量比和乖离率因子的DataFrame
    """
    df = df.copy()
    
    # 按股票分组计算
    grouped = df.groupby('instrument')
    
    # 1. 成交量比（5日均量）
    df['volume_ma5'] = grouped['volume'].transform(lambda x: x.rolling(5, min_periods=5).mean())
    df['成交量比'] = df['volume'] / df['volume_ma5']
    df['成交量比'] = df['成交量比'].replace([np.inf, -np.inf], np.nan)
    
    # 2. 乖离率（60日均线）
    df['ma60'] = grouped['close'].transform(lambda x: x.rolling(60, min_periods=60).mean())
    df['乖离率'] = (df['close'] - df['ma60']) / df['ma60']
    
    # 清理中间列
    df.drop(columns=['volume_ma5', 'ma60'], inplace=True)
    
    return df


def filter_by_volume_bias_factors(
    df: pd.DataFrame,
    volume_ratio_threshold: float = 1.2,
    bias_lower: float = -0.05,
    bias_upper: float = 0.2
) -> pd.DataFrame:
    """
    根据成交量比和乖离率因子筛选信号（新条件）
    
    筛选条件：
    1. 成交量比 > volume_ratio_threshold
    2. 乖离率在[bias_lower, bias_upper]之间
    
    :param df: 包含成交量比和乖离率因子的DataFrame
    :param volume_ratio_threshold: 成交量比阈值（默认1.2）
    :param bias_lower: 乖离率下限（默认-0.05）
    :param bias_upper: 乖离率上限（默认0.2）
    :return: 添加了筛选信号的DataFrame
    """
    df = df.copy()
    
    # 确保包含必要的因子列
    if '成交量比' not in df.columns or '乖离率' not in df.columns:
        raise ValueError("DataFrame必须包含'成交量比'和'乖离率'列")
    
    # 筛选条件（新条件）
    df['volume_ratio_condition'] = df['成交量比'] > volume_ratio_threshold
    df['bias_condition'] = df['乖离率'].between(bias_lower, bias_upper)
    
    # 组合条件
    df['volume_bias_signal'] = df['volume_ratio_condition'] & df['bias_condition']
    
    return df


# ===================== RSI和OBV因子 =====================
def add_rsi_obv_factors(
    df: pd.DataFrame,
    rsi_period: int = 14,
    rsi_low: float = 40,
    rsi_high: float = 70,
    obv_lookback: int = 20
) -> pd.DataFrame:
    """
    为数据添加RSI和OBV因子，以及组合条件。
    
    输入df必须包含：instrument, datetime, close, volume（已复权）
    
    返回的df新增列：
    rsi: RSI(14)值
    rsi_in_range: RSI在[40,70]内为True
    obv: OBV值
    obv_new_high: OBV创20日新高为True（大于前20日最高）
    rsi_obv_condition: 同时满足RSI区间和OBV创新高
    
    :param df: 包含股票数据的DataFrame
    :param rsi_period: RSI计算周期（默认14）
    :param rsi_low: RSI下限（默认40）
    :param rsi_high: RSI上限（默认70）
    :param obv_lookback: OBV创新高回看周期（默认20）
    :return: 添加了RSI和OBV因子的DataFrame
    """
    df = df.copy()
    
    # 确保按股票和时间排序
    df = df.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    grouped = df.groupby('instrument')
    
    # ---------- 1. RSI计算 ----------
    def calc_rsi(group_close):
        delta = group_close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        
        # 用滚动平均（通达信常用SMA平滑，这里用简单移动平均近似）
        avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
        
        # 处理除零：若avg_loss为0，RSI应为100
        rsi = rsi.fillna(100)
        return rsi
    
    df['rsi'] = grouped['close'].transform(calc_rsi)
    df['rsi_in_range'] = (df['rsi'] >= rsi_low) & (df['rsi'] <= rsi_high)
    
    # ---------- 2. OBV计算 ----------
    # 计算每日收盘价变动符号
    df['close_diff'] = grouped['close'].diff().fillna(0)
    df['sign'] = np.sign(df['close_diff'])
    
    # OBV增量 = 符号 * 成交量
    df['obv_delta'] = df['sign'] * df['volume']
    
    # 累积得到OBV
    df['obv'] = grouped['obv_delta'].cumsum()
    
    # 判断OBV创新高：当日OBV > 过去20日最高（不含当日）
    df['obv_past_max'] = grouped['obv'].transform(
        lambda x: x.rolling(obv_lookback, min_periods=obv_lookback).max().shift(1)
    )
    df['obv_new_high'] = df['obv'] > df['obv_past_max']
    df['obv_new_high'] = df['obv_new_high'].fillna(False)
    
    # ---------- 3. 组合条件 ----------
    df['rsi_obv_condition'] = df['rsi_in_range'] & df['obv_new_high']
    
    # 删除临时列
    df.drop(columns=['close_diff', 'sign', 'obv_delta', 'obv_past_max'], inplace=True)
    
    return df


def filter_by_rsi_obv_factors(
    df: pd.DataFrame,
    rsi_period: int = 14,
    rsi_low: float = 40,
    rsi_high: float = 70,
    obv_lookback: int = 20
) -> pd.DataFrame:
    """
    根据RSI和OBV因子筛选信号
    
    筛选条件：
    1. RSI在[rsi_low, rsi_high]之间
    2. OBV创obv_lookback日新高
    
    :param df: 包含股票数据的DataFrame
    :param rsi_period: RSI计算周期（默认14）
    :param rsi_low: RSI下限（默认40）
    :param rsi_high: RSI上限（默认70）
    :param obv_lookback: OBV创新高回看周期（默认20）
    :return: 添加了RSI和OBV筛选信号的DataFrame
    """
    df = df.copy()
    
    # 添加RSI和OBV因子
    df = add_rsi_obv_factors(
        df,
        rsi_period=rsi_period,
        rsi_low=rsi_low,
        rsi_high=rsi_high,
        obv_lookback=obv_lookback
    )
    
    # 筛选信号
    df['rsi_obv_signal'] = df['rsi_obv_condition']
    
    return df


# ===================== 测试函数 =====================
def test_factors():
    """测试因子计算函数"""
    print("🧪 测试因子模块...")
    
    # 创建测试数据
    test_data = {
        'instrument': ['TEST'] * 100,
        'datetime': pd.date_range('2023-01-01', periods=100, freq='D'),
        'close': np.random.randn(100) * 10 + 100,
        'volume': np.random.randint(10000, 100000, 100)
    }
    df_test = pd.DataFrame(test_data)
    
    # 测试成交量比和乖离率因子
    print("  测试成交量比和乖离率因子...")
    df_vb = add_volume_and_bias_factors(df_test)
    print(f"    新增列: {[col for col in df_vb.columns if col not in df_test.columns]}")
    
    df_vb_filtered = filter_by_volume_bias_factors(df_vb)
    print(f"    筛选信号列: 'volume_bias_signal' in df: {'volume_bias_signal' in df_vb_filtered.columns}")
    
    # 测试RSI和OBV因子
    print("  测试RSI和OBV因子...")
    df_ro = add_rsi_obv_factors(df_test)
    print(f"    新增列: {[col for col in df_ro.columns if col not in df_test.columns]}")
    
    df_ro_filtered = filter_by_rsi_obv_factors(df_ro)
    print(f"    筛选信号列: 'rsi_obv_signal' in df: {'rsi_obv_signal' in df_ro_filtered.columns}")
    
    print("✅ 因子模块测试完成")
    return True


if __name__ == "__main__":
    test_factors()