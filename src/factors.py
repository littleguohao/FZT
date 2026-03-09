#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子模块 - 包含各种技术因子计算函数

包含以下因子：
1. 成交量比和乖离率因子
2. RSI和OBV因子
3. 短期暴涨过滤因子
4. 新增可配置因子：最大涨幅限制和上涨日累计换手率限制

作者: MC
创建日期: 2026-03-07
更新日期: 2026-03-09
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


# ===================== 成交量比因子 =====================
def add_volume_ratio_factor(df: pd.DataFrame) -> pd.DataFrame:
    """
    为已有数据添加成交量比因子
    
    因子定义：成交量比 = volume / MA(volume, 5)
    
    :param df: 包含股票数据的DataFrame
    :return: 添加了成交量比因子的DataFrame
    """
    df = df.copy()
    
    # 按股票分组计算
    grouped = df.groupby('instrument')
    
    # 成交量比（5日均量）
    df['volume_ma5'] = grouped['volume'].transform(lambda x: x.rolling(5, min_periods=5).mean())
    df['成交量比'] = df['volume'] / df['volume_ma5']
    df['成交量比'] = df['成交量比'].replace([np.inf, -np.inf], np.nan)
    
    # 清理中间列
    df.drop(columns=['volume_ma5'], inplace=True)
    
    return df


def filter_by_volume_ratio_factor(
    df: pd.DataFrame,
    volume_ratio_threshold: float = 1.2
) -> pd.DataFrame:
    """
    根据成交量比因子筛选信号
    
    筛选条件：成交量比 >= 阈值
    
    :param df: 包含股票数据的DataFrame
    :param volume_ratio_threshold: 成交量比阈值 (默认: 1.2)
    :return: 添加了成交量比筛选信号的DataFrame
    """
    df = df.copy()
    
    # 添加成交量比因子
    df = add_volume_ratio_factor(df)
    
    # 筛选信号
    df['volume_ratio_signal'] = df['成交量比'] >= volume_ratio_threshold
    
    return df


# ===================== 乖离率因子 =====================
def add_bias_factor(df: pd.DataFrame) -> pd.DataFrame:
    """
    为已有数据添加乖离率因子
    
    因子定义：乖离率 = (close - MA(close, 20)) / MA(close, 20)
    
    :param df: 包含股票数据的DataFrame
    :return: 添加了乖离率因子的DataFrame
    """
    df = df.copy()
    
    # 按股票分组计算
    grouped = df.groupby('instrument')
    
    # 乖离率（20日均线）
    df['close_ma20'] = grouped['close'].transform(lambda x: x.rolling(20, min_periods=20).mean())
    df['乖离率'] = (df['close'] - df['close_ma20']) / df['close_ma20']
    
    # 清理中间列
    df.drop(columns=['close_ma20'], inplace=True)
    
    return df


def filter_by_bias_factor(
    df: pd.DataFrame,
    bias_threshold: float = 0.05
) -> pd.DataFrame:
    """
    根据乖离率因子筛选信号
    
    筛选条件：乖离率 <= 阈值（避免涨幅过大）
    
    :param df: 包含股票数据的DataFrame
    :param bias_threshold: 乖离率阈值 (默认: 0.05)
    :return: 添加了乖离率筛选信号的DataFrame
    """
    df = df.copy()
    
    # 添加乖离率因子
    df = add_bias_factor(df)
    
    # 筛选信号
    df['bias_signal'] = df['乖离率'] <= bias_threshold
    
    return df


# ===================== 成交量比 + 乖离率组合因子 =====================
def add_volume_and_bias_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    同时添加成交量比和乖离率因子
    
    :param df: 包含股票数据的DataFrame
    :return: 添加了成交量比和乖离率因子的DataFrame
    """
    df = df.copy()
    
    # 添加成交量比因子
    df = add_volume_ratio_factor(df)
    
    # 添加乖离率因子
    df = add_bias_factor(df)
    
    return df


def filter_by_volume_bias_factors(
    df: pd.DataFrame,
    volume_ratio_threshold: float = 1.2,
    bias_threshold: float = 0.05
) -> pd.DataFrame:
    """
    根据成交量比和乖离率因子组合筛选信号
    
    筛选条件：成交量比 >= 阈值 且 乖离率 <= 阈值
    
    :param df: 包含股票数据的DataFrame
    :param volume_ratio_threshold: 成交量比阈值 (默认: 1.2)
    :param bias_threshold: 乖离率阈值 (默认: 0.05)
    :return: 添加了成交量比+乖离率筛选信号的DataFrame
    """
    df = df.copy()
    
    # 添加两个因子
    df = add_volume_and_bias_factors(df)
    
    # 组合筛选条件
    df['volume_bias_condition'] = (df['成交量比'] >= volume_ratio_threshold) & (df['乖离率'] <= bias_threshold)
    
    # 筛选信号
    df['volume_bias_signal'] = df['volume_bias_condition']
    
    return df


# ===================== RSI因子 =====================
def add_rsi_factor(
    df: pd.DataFrame,
    period: int = 14
) -> pd.DataFrame:
    """
    为已有数据添加RSI因子
    
    因子定义：RSI = 100 - 100 / (1 + RS)
    RS = 平均涨幅 / 平均跌幅
    
    :param df: 包含股票数据的DataFrame
    :param period: RSI计算周期 (默认: 14)
    :return: 添加了RSI因子的DataFrame
    """
    df = df.copy()
    
    # 确保按股票和时间排序
    df = df.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    grouped = df.groupby('instrument')
    
    # 计算价格变化
    df['price_change'] = grouped['close'].diff()
    
    # 计算涨幅和跌幅
    df['gain'] = np.where(df['price_change'] > 0, df['price_change'], 0)
    df['loss'] = np.where(df['price_change'] < 0, -df['price_change'], 0)
    
    # 计算平均涨幅和平均跌幅
    df['avg_gain'] = grouped['gain'].transform(
        lambda x: x.rolling(period, min_periods=period).mean()
    )
    df['avg_loss'] = grouped['loss'].transform(
        lambda x: x.rolling(period, min_periods=period).mean()
    )
    
    # 计算RS和RSI
    df['rs'] = df['avg_gain'] / df['avg_loss'].replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + df['rs']))
    
    # 清理中间列
    df.drop(columns=['price_change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'], inplace=True)
    
    return df


def filter_by_rsi_factor(
    df: pd.DataFrame,
    rsi_threshold: float = 70
) -> pd.DataFrame:
    """
    根据RSI因子筛选信号
    
    筛选条件：RSI <= 阈值（避免超买）
    
    :param df: 包含股票数据的DataFrame
    :param rsi_threshold: RSI阈值 (默认: 70)
    :return: 添加了RSI筛选信号的DataFrame
    """
    df = df.copy()
    
    # 添加RSI因子
    df = add_rsi_factor(df)
    
    # 筛选信号
    df['rsi_signal'] = df['rsi'] <= rsi_threshold
    
    return df


# ===================== OBV因子 =====================
def add_obv_factor(df: pd.DataFrame) -> pd.DataFrame:
    """
    为已有数据添加OBV因子
    
    因子定义：OBV = 前一日OBV + 当日成交量 * 符号
    符号: 当日收盘价 > 前一日收盘价 ? 1 : -1
    
    :param df: 包含股票数据的DataFrame
    :return: 添加了OBV因子的DataFrame
    """
    df = df.copy()
    
    # 确保按股票和时间排序
    df = df.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    grouped = df.groupby('instrument')
    
    # 计算价格变化方向
    df['price_dir'] = np.where(
        df['close'] > grouped['close'].shift(1),
        1,
        np.where(df['close'] < grouped['close'].shift(1), -1, 0)
    )
    
    # 计算OBV
    df['obv'] = (df['volume'] * df['price_dir']).groupby(df['instrument']).cumsum()
    
    # 清理中间列
    df.drop(columns=['price_dir'], inplace=True)
    
    return df


def filter_by_obv_factor(
    df: pd.DataFrame,
    obv_threshold: float = 0
) -> pd.DataFrame:
    """
    根据OBV因子筛选信号
    
    筛选条件：OBV > 阈值（资金流入）
    
    :param df: 包含股票数据的DataFrame
    :param obv_threshold: OBV阈值 (默认: 0)
    :return: 添加了OBV筛选信号的DataFrame
    """
    df = df.copy()
    
    # 添加OBV因子
    df = add_obv_factor(df)
    
    # 筛选信号
    df['obv_signal'] = df['obv'] > obv_threshold
    
    return df


# ===================== RSI + OBV组合因子 =====================
def add_rsi_obv_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    同时添加RSI和OBV因子
    
    :param df: 包含股票数据的DataFrame
    :return: 添加了RSI和OBV因子的DataFrame
    """
    df = df.copy()
    
    # 添加RSI因子
    df = add_rsi_factor(df)
    
    # 添加OBV因子
    df = add_obv_factor(df)
    
    return df


def filter_by_rsi_obv_factors(
    df: pd.DataFrame,
    rsi_threshold: float = 70,
    obv_threshold: float = 0
) -> pd.DataFrame:
    """
    根据RSI和OBV因子组合筛选信号
    
    筛选条件：RSI <= 阈值 且 OBV > 阈值
    
    :param df: 包含股票数据的DataFrame
    :param rsi_threshold: RSI阈值 (默认: 70)
    :param obv_threshold: OBV阈值 (默认: 0)
    :return: 添加了RSI+OBV筛选信号的DataFrame
    """
    df = df.copy()
    
    # 添加两个因子
    df = add_rsi_obv_factors(df)
    
    # 组合筛选条件
    df['rsi_obv_condition'] = (df['rsi'] <= rsi_threshold) & (df['obv'] > obv_threshold)
    
    # 筛选信号
    df['rsi_obv_signal'] = df['rsi_obv_condition']
    
    return df


# ===================== 短期暴涨过滤因子 =====================
def add_max_gain_condition(
    df: pd.DataFrame,
    lookback: int = 35,
    threshold: float = 0.7
) -> pd.DataFrame:
    """
    添加"过去N日最大涨幅超过阈值"的过滤条件
    
    因子定义：过去N个交易日内，最高价相对于N日前收盘价的涨幅超过阈值
    
    :param df: 包含 instrument, datetime, close, high 的DataFrame
    :param lookback: 回溯周期N (默认: 35)
    :param threshold: 涨幅阈值，如0.7表示70% (默认: 0.7)
    :return: 添加了短期暴涨因子的DataFrame
    """
    df = df.copy()
    
    # 确保按股票和时间排序
    df = df.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    grouped = df.groupby('instrument')
    
    # 计算N日前的收盘价
    df['close_shift'] = grouped['close'].transform(lambda x: x.shift(lookback))
    
    # 计算过去N日的最高价（滚动最大值）
    df['high_rolling_max'] = grouped['high'].transform(
        lambda x: x.rolling(lookback, min_periods=lookback).max()
    )
    
    # 计算最大涨幅
    df['max_gain'] = (df['high_rolling_max'] / df['close_shift'] - 1)
    
    # 标记是否超过阈值（短期暴涨）
    df['短期暴涨'] = df['max_gain'] > threshold
    
    # 删除临时列
    df.drop(columns=['close_shift', 'high_rolling_max'], inplace=True)
    
    return df


def filter_by_max_gain_condition(
    df: pd.DataFrame,
    lookback: int = 35,
    threshold: float = 0.7
) -> pd.DataFrame:
    """
    根据短期暴涨因子筛选信号
    
    筛选条件：过滤掉最近N日涨幅超过阈值的股票（短期暴涨为False）
    
    :param df: 包含股票数据的DataFrame
    :param lookback: 回溯周期N (默认: 35)
    :param threshold: 涨幅阈值，如0.7表示70% (默认: 0.7)
    :return: 添加了短期暴涨筛选信号的DataFrame
    """
    df = df.copy()
    
    # 添加短期暴涨因子
    df = add_max_gain_condition(df, lookback=lookback, threshold=threshold)
    
    # 筛选信号：过滤掉短期暴涨的股票（短期暴涨为False）
    df['max_gain_signal'] = ~df['短期暴涨']
    
    return df


# ===================== 新增可配置因子 =====================
def calc_custom_factors(
    df: pd.DataFrame,
    N: int = 20,
    M: float = 0.45,  # 默认最大涨幅45%
    Y: int = 20,
    X: float = 0.30,  # 默认换手率30%
    exclude_today: bool = True
) -> pd.DataFrame:
    """
    计算两个可配置的因子：
    1. 前N个交易日的最大涨幅不超过 M（M为小数，如0.35）
    2. 前Y个交易日中上涨日的累计换手率小于 X（X为小数，如0.35）

    参数:
    df: DataFrame，必须包含列 ['instrument', 'datetime', 'close', 'turnover']
    N: 最大涨幅的回溯窗口 (默认: 20)
    M: 最大涨幅阈值（小数，默认: 0.35，即35%）
    Y: 累计换手率的回溯窗口 (默认: 20)
    X: 累计换手率阈值（小数，默认: 0.35，即35%）
    exclude_today: 是否排除当日数据（True 表示使用前N/Y个交易日，不含当日；False 表示包含当日）

    返回:
    原DataFrame新增以下列：
    - max_gain_{N}: 过去N日最大涨幅（小数）
    - up_turnover_sum_{Y}: 过去Y日上涨日累计换手率（小数）
    - cond_gain: 是否满足涨幅条件
    - cond_turnover: 是否满足换手率条件
    """
    # 复制数据避免修改原df
    df = df.copy()
    
    # 确保按股票和时间排序
    df = df.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    grouped = df.groupby('instrument')

    # 计算日收益率
    df['pct_chg'] = grouped['close'].pct_change()  # 保持为小数

    # 如果需要排除当日，将收益率和换手率整体向后平移一天，使当日因子值基于历史数据
    if exclude_today:
        df['pct_chg'] = grouped['pct_chg'].shift(1)
        df['turnover_shifted'] = grouped['turnover'].shift(1)
    else:
        df['turnover_shifted'] = df['turnover']

    # ---------- 因子1：过去N日最大涨幅 ----------
    # 滚动窗口计算最大涨幅（忽略NaN）
    df[f'max_gain_{N}'] = grouped['pct_chg'].transform(
        lambda x: x.rolling(window=N, min_periods=1).max()
    )
    # 条件1：最大涨幅 <= M
    df['cond_gain'] = df[f'max_gain_{N}'] <= M

    # ---------- 因子2：过去Y日上涨日累计换手率 ----------
    # 标记上涨日（pct_chg > 0）
    df['up'] = df['pct_chg'] > 0
    # 上涨日的换手率，非上涨日记为0
    df['up_turnover'] = np.where(df['up'], df['turnover_shifted'], 0.0)
    # 滚动求和
    df[f'up_turnover_sum_{Y}'] = grouped['up_turnover'].transform(
        lambda x: x.rolling(window=Y, min_periods=1).sum()
    )
    # 条件2：累计换手率 < X
    df['cond_turnover'] = df[f'up_turnover_sum_{Y}'] < X

    # 删除临时列（可选）
    # df.drop(['pct_chg', 'turnover_shifted', 'up', 'up_turnover'], axis=1, inplace=True)

    return df


def add_custom_factors(
    df: pd.DataFrame,
    N: int = 20,
    M: float = 0.45,  # 默认最大涨幅45%
    Y: int = 20,
    X: float = 0.30,  # 默认换手率30%
    exclude_today: bool = True
) -> pd.DataFrame:
    """
    添加两个可配置因子到DataFrame
    
    参数:
    df: DataFrame，必须包含列 ['instrument', 'datetime', 'close', 'turnover']
    N: 最大涨幅的回溯窗口 (默认: 20)
    M: 最大涨幅阈值（小数，默认: 0.35，即35%）
    Y: 累计换手率的回溯窗口 (默认: 20)
    X: 累计换手率阈值（小数，默认: 0.35，即35%）
    exclude_today: 是否排除当日数据（True 表示使用前N/Y个交易日，不含当日；False 表示包含当日）
    
    返回:
    添加了自定义因子的DataFrame
    """
    return calc_custom_factors(df, N, M, Y, X, exclude_today)


def filter_by_custom_factors(
    df: pd.DataFrame,
    N: int = 20,
    M: float = 0.45,  # 默认最大涨幅45%
    Y: int = 20,
    X: float = 0.30,  # 默认换手率30%
    exclude_today: bool = True
) -> pd.DataFrame:
    """
    使用两个可配置因子筛选股票
    
    筛选条件：同时满足两个条件
    1. 前N个交易日的最大涨幅不超过 M
    2. 前Y个交易日中上涨日的累计换手率小于 X
    
    参数:
    df: DataFrame，必须包含列 ['instrument', 'datetime', 'close', 'turnover']
    N: 最大涨幅的回溯窗口 (默认: 20)
    M: 最大涨幅阈值（小数，默认: 0.35，即35%）
    Y: 累计换手率的回溯窗口 (默认: 20)
    X: 累计换手率阈值（小数，默认: 0.35，即35%）
    exclude_today: 是否排除当日数据（True 表示使用前N/Y个交易日，不含当日；False 表示包含当日）
    
    返回:
    筛选后的DataFrame
    """
    df_with_factors = calc_custom_factors(df, N, M, Y, X, exclude_today)
    
    # 筛选条件：两个条件都满足
    df_with_factors['custom_factors_signal'] = df_with_factors['cond_gain'] & df_with_factors['cond_turnover']
    
    return df_with_factors[df_with_factors['custom_factors_signal'] == True]


# ===================== 测试函数 =====================
def test_factors():
    """测试因子计算函数"""
    print("🧪 测试因子模块...")
    
    # 创建测试数据
    test_data = {
        'instrument': ['TEST'] * 100,
        'datetime': pd.date_range('2023-01-01', periods=100, freq='D'),
        'close': np.random.randn(100) * 10 + 100,
        'volume': np.random.randint(10000, 100000, 100),
        'turnover': np.random.rand(100) * 0.1  # 换手率数据
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
    
    # 测试短期暴涨因子
    print("  测试短期暴涨因子...")
    df_mg = add_max_gain_condition(df_test)
    print(f"    新增列: {[col for col in df_mg.columns if col not in df_test.columns]}")
    
    # 测试新增可配置因子
    print("  测试新增可配置因子...")
    df_custom = add_custom_factors(df_test, N=20, M=0.45, Y=20, X=0.30)
    print(f"    新增列: {[col for col in df_custom.columns if col not in df_test.columns]}")
    
    df_custom_filtered = filter_by_custom_factors(df_test, N=20, M=0.45, Y=20, X=0.30)
    print(f"    筛选信号列: 'custom_factors_signal' in df: {'custom_factors_signal' in df_custom_filtered.columns}")
    
    print("✅ 因子模块测试完成")
    return True


if __name__ == "__main__":
    test_factors()
