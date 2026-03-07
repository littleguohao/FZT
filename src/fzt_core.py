#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FZT核心计算模块 - 通用FZT公式实现

包含两个回测脚本共用的函数：
1. 通达信SMA递归算法
2. 向量化FZT公式计算

作者: MC
创建日期: 2026-03-06
参考文件：
1. FZT参考实现---5134fc49-465f-427a-898e-9fea5d032908.txt
2. 优化参考---9f1f0f32-521a-423b-8027-090c283c9de7.txt
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


# ===================== 精准复刻通达信SMA =====================
def tdx_sma_series(series_vals: np.ndarray, N: int, M: int = 1) -> np.ndarray:
    """
    精准复刻通达信SMA(X,N,M)递归算法
    :param series_vals: 单只股票的数值数组（时间升序）
    :param N: SMA周期
    :param M: 权重（默认1）
    :return: SMA计算结果（前N-1位为NaN）
    """
    len_series = len(series_vals)
    sma = np.full(len_series, np.nan)
    
    # 仅当数据长度≥N时计算
    if len_series >= N:
        # 初始值：前N天简单平均（仅保留有效值，无有效值则保持NaN）
        valid_mask = ~np.isnan(series_vals[:N])
        if valid_mask.any():
            sma[N-1] = np.nanmean(series_vals[:N])
        
        # 递归计算：sma[i] = (sma[i-1] * (N - M) + series_vals[i] * M) / N
        for i in range(N, len_series):
            if not np.isnan(sma[i-1]) and not np.isnan(series_vals[i]):
                sma[i] = (sma[i-1] * (N - M) + series_vals[i] * M) / N
    
    return sma


def tdx_sma(group_series: pd.Series, N: int, M: int = 1) -> pd.Series:
    """
    pandas Series版的通达信SMA（用于groupby.transform）
    :param group_series: 单只股票的时间序列（已按时间排序）
    :param N: SMA周期
    :param M: 权重（默认1）
    :return: SMA计算结果Series
    """
    # 确保是数值类型
    series_vals = group_series.values.astype(float)
    
    # 计算SMA
    sma_vals = tdx_sma_series(series_vals, N, M)
    
    # 返回与输入相同索引的Series
    return pd.Series(sma_vals, index=group_series.index)


# ===================== 最终版FZT计算函数 =====================
def calc_brick_pattern_final(
    df_raw: pd.DataFrame,
    target_start_date: Optional[str] = None,
    target_end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    最终版砖型图选股公式计算（基于用户提供的两个参考文件）
    
    公式逻辑：
    VAR1A:=(HHV(HIGH,4)-CLOSE)/(HHV(HIGH,4)-LLV(LOW,4))*100-90;
    VAR2A:=SMA(VAR1A,4,1)+100;
    VAR3A:=(CLOSE-LLV(LOW,4))/(HHV(HIGH,4)-LLV(LOW,4))*100;
    VAR4A:=SMA(VAR3A,6,1);
    VAR5A:=SMA(VAR4A,6,1)+100;
    VAR6A:=VAR5A-VAR2A;
    砖型图:=IF(VAR6A>4,VAR6A-4,0);
    砖型图面积:=ABS(砖型图 - REF(砖型图,1));
    AA:=(REF(砖型图,1)<砖型图);
    首次多头增强:=(REF(AA,1)=0) AND (AA=1);
    砖型图面积增幅:=砖型图面积 > REF(砖型图面积,1) * 2/3;
    选股条件:=首次多头增强 AND 砖型图面积增幅;
    
    :param df_raw: 原始数据DataFrame，必须包含列：instrument, datetime, open, high, low, close, volume
    :param target_start_date: 目标开始日期（可选）
    :param target_end_date: 目标结束日期（可选）
    :return: 包含FZT计算结果的DataFrame
    """
    # 1. 数据预处理（基于FZT参考实现）
    df = df_raw.copy()
    
    # 强制类型转换（避免整数精度丢失）
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[col] = df[col].astype(float)
    
    # 确保datetime是datetime类型
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 按股票和时间排序
    df = df.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    grouped = df.groupby('instrument')
    
    # 2. 核心指标计算（使用通达信SMA）
    # 2.1 4日高低极值
    df['HHV4'] = grouped['high'].transform(lambda x: x.rolling(4, min_periods=4).max())
    df['LLV4'] = grouped['low'].transform(lambda x: x.rolling(4, min_periods=4).min())
    df['diff_HL4'] = df['HHV4'] - df['LLV4']
    
    # 2.2 VAR1A：分母为0时填0，避免NaN扩散（基于FZT参考实现）
    df['VAR1A'] = np.where(
        df['diff_HL4'] == 0,
        0,
        (df['HHV4'] - df['close']) / df['diff_HL4'] * 100 - 90
    )
    
    # 2.3 VAR2A = SMA(VAR1A,4,1) + 100（使用通达信SMA）
    df['SMA_VAR1A_4'] = grouped['VAR1A'].transform(lambda x: tdx_sma(x, N=4, M=1))
    df['VAR2A'] = df['SMA_VAR1A_4'] + 100
    
    # 2.4 VAR3A：分母为0时填0
    df['VAR3A'] = np.where(
        df['diff_HL4'] == 0,
        0,
        (df['close'] - df['LLV4']) / df['diff_HL4'] * 100
    )
    
    # 2.5 VAR4A = SMA(VAR3A,6,1)（使用通达信SMA）
    df['SMA_VAR3A_6'] = grouped['VAR3A'].transform(lambda x: tdx_sma(x, N=6, M=1))
    df['VAR4A'] = df['SMA_VAR3A_6']
    
    # 2.6 VAR5A = SMA(VAR4A,6,1) + 100（使用通达信SMA）
    df['SMA_VAR4A_6'] = grouped['VAR4A'].transform(lambda x: tdx_sma(x, N=6, M=1))
    df['VAR5A'] = df['SMA_VAR4A_6'] + 100
    
    # 2.7 VAR6A = VAR5A - VAR2A
    df['VAR6A'] = df['VAR5A'] - df['VAR2A']
    
    # 3. 砖型图相关计算
    df['砖型图'] = np.where(df['VAR6A'] > 4, df['VAR6A'] - 4, 0)
    df['砖型图'] = df['砖型图'].fillna(0)
    df['砖型图_prev'] = grouped['砖型图'].transform(lambda x: x.shift(1))
    df['砖型图_prev'] = df['砖型图_prev'].fillna(0)
    df['砖型图面积'] = (df['砖型图'] - df['砖型图_prev']).abs()
    
    # 4. 多头信号计算
    df['AA'] = (df['砖型图_prev'] < df['砖型图']).astype(int)
    df['AA_prev'] = grouped['AA'].transform(lambda x: x.shift(1))
    df['AA_prev'] = df['AA_prev'].fillna(0)
    df['首次多头增强'] = (df['AA_prev'] == 0) & (df['AA'] == 1)
    
    # 5. 面积增幅 + 最终选股条件
    df['砖型图面积_prev'] = grouped['砖型图面积'].transform(lambda x: x.shift(1))
    df['砖型图面积_prev'] = df['砖型图面积_prev'].fillna(0)
    df['砖型图面积增幅'] = df['砖型图面积'] > (df['砖型图面积_prev'] * 2 / 3)
    df['选股条件'] = df['首次多头增强'] & df['砖型图面积增幅']
    
    # 6. 筛选目标时间段
    if target_start_date:
        target_start_dt = pd.Timestamp(target_start_date)
        df = df[df['datetime'] >= target_start_dt]
    if target_end_date:
        target_end_dt = pd.Timestamp(target_end_date)
        df = df[df['datetime'] <= target_end_dt]
    
    return df


# ===================== 向量化FZT计算（批量处理） =====================
def calculate_fzt_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    向量化计算FZT特征（全市场一次性计算）
    基于优化参考：从'分批逐股票循环'升级为'全市场向量化一次性计算'
    
    :param df: 包含所有股票数据的DataFrame
    :return: 包含FZT特征的DataFrame
    """
    # 使用calc_brick_pattern_final函数（已经是向量化实现）
    return calc_brick_pattern_final(df)


# ===================== 测试函数 =====================
def test_fzt_core():
    """测试FZT核心函数"""
    print("🧪 测试FZT核心函数...")
    
    # 创建测试数据
    test_data = {
        'instrument': ['TEST'] * 20,
        'datetime': pd.date_range('2023-01-01', periods=20),
        'open': np.random.randn(20) * 10 + 100,
        'high': np.random.randn(20) * 10 + 105,
        'low': np.random.randn(20) * 10 + 95,
        'close': np.random.randn(20) * 10 + 100,
        'volume': np.random.randint(10000, 100000, 20)
    }
    df_test = pd.DataFrame(test_data)
    
    # 测试SMA函数
    print("  测试通达信SMA函数...")
    test_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sma_result = tdx_sma(test_series, N=3, M=1)
    print(f"    SMA(3,1) of [1..10]: {sma_result.values}")
    
    # 测试FZT计算
    print("  测试FZT计算函数...")
    df_result = calc_brick_pattern_final(df_test)
    print(f"    输入行数: {len(df_test)}")
    print(f"    输出行数: {len(df_result)}")
    print(f"    新增列数: {len(df_result.columns) - len(df_test.columns)}")
    
    if '选股条件' in df_result.columns:
        signals = df_result['选股条件'].sum()
        print(f"    FZT信号数量: {signals}")
    
    print("✅ FZT核心函数测试完成")
    return True


if __name__ == "__main__":
    test_fzt_core()


# ===================== 新增：成交量比和乖离率因子 =====================
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