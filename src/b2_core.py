#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B2因子核心模块

B2因子条件：
1. 前一日的KDJ的J值 < 13
2. 当日的KDJ的J值 < 55
3. 当日的涨幅 > 4%
4. 当日的成交量 > 前一日成交量
5. 当日上影线很小

作者: MC
创建日期: 2026-03-07
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def compute_kdj(
    df: pd.DataFrame,
    n: int = 9,
    m1: int = 3,
    m2: int = 3
) -> pd.DataFrame:
    """
    计算KDJ指标（K, D, J）
    
    输入df需包含：instrument, datetime, high, low, close
    返回的df新增列：kdj_k, kdj_d, kdj_j
    
    :param df: 包含股票数据的DataFrame
    :param n: KDJ计算周期 (默认: 9)
    :param m1: K值平滑周期 (默认: 3)
    :param m2: D值平滑周期 (默认: 3)
    :return: 添加了KDJ指标的DataFrame
    """
    df = df.copy()
    
    # 确保按股票和时间排序
    df = df.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    grouped = df.groupby('instrument')
    
    # 计算最近n日的最高价和最低价
    df['high_n'] = grouped['high'].transform(lambda x: x.rolling(n, min_periods=n).max())
    df['low_n'] = grouped['low'].transform(lambda x: x.rolling(n, min_periods=n).min())
    
    # 计算RSV
    df['rsv'] = (df['close'] - df['low_n']) / (df['high_n'] - df['low_n']) * 100
    df['rsv'] = df['rsv'].fillna(50)  # 处理分母为0或NaN的情况，用50填充
    
    # 计算K值（RSV的m1日移动平均）
    df['kdj_k'] = grouped['rsv'].transform(lambda x: x.rolling(m1, min_periods=m1).mean())
    # 计算D值（K的m2日移动平均）
    df['kdj_d'] = grouped['kdj_k'].transform(lambda x: x.rolling(m2, min_periods=m2).mean())
    # 计算J值
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    
    # 删除临时列
    df.drop(columns=['high_n', 'low_n', 'rsv'], inplace=True)
    
    return df


def add_B2_factor(
    df: pd.DataFrame,
    j_prev_thresh: float = 13,          # 前一日J值上限
    j_today_thresh: float = 55,         # 当日J值上限
    gain_thresh: float = 0.04,          # 涨幅阈值（4%）
    upper_shadow_pct: float = 0.005,    # 上影线占收盘价比例上限（0.5%）
    upper_shadow_vs_body: float = 0.5   # 上影线相对于实体长度的比例上限
) -> pd.DataFrame:
    """
    添加B2因子组合条件
    
    条件：
    1. 前一日的KDJ的J值 < 13
    2. 当日的KDJ的J值 < 55
    3. 当日的涨幅 > 4%
    4. 当日的成交量 > 前一日成交量
    5. 当日上影线很小
    
    输入df必须包含：instrument, datetime, open, high, low, close, volume
    返回的df新增列：B2 (布尔)
    
    :param df: 包含股票数据的DataFrame
    :param j_prev_thresh: 前一日J值上限 (默认: 13)
    :param j_today_thresh: 当日J值上限 (默认: 55)
    :param gain_thresh: 涨幅阈值，如0.04表示4% (默认: 0.04)
    :param upper_shadow_pct: 上影线占收盘价比例上限 (默认: 0.005)
    :param upper_shadow_vs_body: 上影线相对于实体长度的比例上限 (默认: 0.5)
    :return: 添加了B2因子的DataFrame
    """
    df = df.copy()
    
    # 确保按股票和时间排序
    df = df.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    
    # 1. 计算KDJ
    df = compute_kdj(df)
    
    # 重新分组（因为df已经更新）
    grouped = df.groupby('instrument')
    
    # 2. 前一日J值
    df['kdj_j_prev'] = grouped['kdj_j'].transform(lambda x: x.shift(1))
    
    # 3. 当日涨幅
    df['close_prev'] = grouped['close'].transform(lambda x: x.shift(1))
    df['gain'] = (df['close'] - df['close_prev']) / df['close_prev']
    
    # 4. 成交量比较
    df['volume_prev'] = grouped['volume'].transform(lambda x: x.shift(1))
    
    # 5. 上影线计算
    # 上影线 = 最高价 - max(开盘价, 收盘价)
    df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
    # K线实体长度
    df['body'] = np.abs(df['close'] - df['open'])
    
    # 条件1: 前一日J值 < 13
    cond1 = df['kdj_j_prev'] < j_prev_thresh
    
    # 条件2: 当日J值 < 55
    cond2 = df['kdj_j'] < j_today_thresh
    
    # 条件3: 当日涨幅 > 4%
    cond3 = df['gain'] > gain_thresh
    
    # 条件4: 当日成交量 > 前一日成交量
    cond4 = df['volume'] > df['volume_prev']
    
    # 条件5: 上影线很小（满足两个子条件）
    cond5 = (df['upper_shadow'] < upper_shadow_pct * df['close']) & \
            (df['upper_shadow'] < upper_shadow_vs_body * df['body'])
    
    # 合并所有条件
    df['B2'] = cond1 & cond2 & cond3 & cond4 & cond5
    
    # 删除临时列
    df.drop(columns=[
        'kdj_j_prev', 'close_prev', 'gain', 'volume_prev',
        'upper_shadow', 'body', 'kdj_k', 'kdj_d', 'kdj_j'
    ], inplace=True)
    
    return df


def filter_by_B2_factor(
    df: pd.DataFrame,
    j_prev_thresh: float = 13,
    j_today_thresh: float = 55,
    gain_thresh: float = 0.04,
    upper_shadow_pct: float = 0.005,
    upper_shadow_vs_body: float = 0.5
) -> pd.DataFrame:
    """
    根据B2因子筛选信号
    
    筛选条件：B2因子为True
    
    :param df: 包含股票数据的DataFrame
    :param j_prev_thresh: 前一日J值上限 (默认: 13)
    :param j_today_thresh: 当日J值上限 (默认: 55)
    :param gain_thresh: 涨幅阈值，如0.04表示4% (默认: 0.04)
    :param upper_shadow_pct: 上影线占收盘价比例上限 (默认: 0.005)
    :param upper_shadow_vs_body: 上影线相对于实体长度的比例上限 (默认: 0.5)
    :return: 添加了B2筛选信号的DataFrame
    """
    df = df.copy()
    
    # 添加B2因子
    df = add_B2_factor(
        df,
        j_prev_thresh=j_prev_thresh,
        j_today_thresh=j_today_thresh,
        gain_thresh=gain_thresh,
        upper_shadow_pct=upper_shadow_pct,
        upper_shadow_vs_body=upper_shadow_vs_body
    )
    
    # 筛选信号
    df['B2_signal'] = df['B2']
    
    return df


def calculate_b2_success_rate(
    df: pd.DataFrame,
    j_prev_thresh: float = 13,
    j_today_thresh: float = 55,
    gain_thresh: float = 0.04,
    upper_shadow_pct: float = 0.005,
    upper_shadow_vs_body: float = 0.5
) -> Dict[str, Any]:
    """
    计算B2因子的成功率
    
    :param df: 包含股票数据的DataFrame
    :param j_prev_thresh: 前一日J值上限 (默认: 13)
    :param j_today_thresh: 当日J值上限 (默认: 55)
    :param gain_thresh: 涨幅阈值，如0.04表示4% (默认: 0.04)
    :param upper_shadow_pct: 上影线占收盘价比例上限 (默认: 0.005)
    :param upper_shadow_vs_body: 上影线相对于实体长度的比例上限 (默认: 0.5)
    :return: 包含成功率统计的字典
    """
    df = df.copy()
    
    # 添加B2因子
    df = add_B2_factor(
        df,
        j_prev_thresh=j_prev_thresh,
        j_today_thresh=j_today_thresh,
        gain_thresh=gain_thresh,
        upper_shadow_pct=upper_shadow_pct,
        upper_shadow_vs_body=upper_shadow_vs_body
    )
    
    # 计算成功条件（次日收盘价 > 当日收盘价）
    df = df.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    grouped = df.groupby('instrument')
    df['next_close'] = grouped['close'].transform(lambda x: x.shift(-1))
    df['success'] = df['next_close'] > df['close']
    
    # 筛选B2信号
    b2_signals = df[df['B2'] == True]
    
    # 计算统计结果
    total_signals = len(b2_signals)
    success_signals = b2_signals['success'].sum() if total_signals > 0 else 0
    success_rate = success_signals / total_signals if total_signals > 0 else 0
    
    # 年度分析
    b2_signals['year'] = b2_signals['datetime'].dt.year
    yearly_stats = []
    
    for year in sorted(b2_signals['year'].unique()):
        year_data = b2_signals[b2_signals['year'] == year]
        year_total = len(year_data)
        year_success = year_data['success'].sum() if year_total > 0 else 0
        year_rate = year_success / year_total if year_total > 0 else 0
        
        yearly_stats.append({
            'year': year,
            'total': year_total,
            'success': year_success,
            'rate': year_rate
        })
    
    # 准备结果
    results = {
        'total_signals': total_signals,
        'success_signals': success_signals,
        'success_rate': success_rate,
        'yearly_stats': yearly_stats,
        'parameters': {
            'j_prev_thresh': j_prev_thresh,
            'j_today_thresh': j_today_thresh,
            'gain_thresh': gain_thresh,
            'upper_shadow_pct': upper_shadow_pct,
            'upper_shadow_vs_body': upper_shadow_vs_body
        }
    }
    
    return results


def test_b2_core():
    """
    测试B2核心模块
    """
    print("🧪 测试B2核心模块...")
    
    # 创建测试数据
    test_data = {
        'instrument': ['SH600000'] * 20,
        'datetime': pd.date_range('2020-01-01', periods=20, freq='D'),
        'open': np.random.uniform(10, 20, 20),
        'high': np.random.uniform(12, 22, 20),
        'low': np.random.uniform(8, 18, 20),
        'close': np.random.uniform(10, 20, 20),
        'volume': np.random.randint(10000, 100000, 20)
    }
    
    df = pd.DataFrame(test_data)
    print(f"✅ 测试数据创建成功，形状: {df.shape}")
    
    # 测试KDJ计算
    df_kdj = compute_kdj(df)
    print(f"✅ KDJ计算成功，新增列: {[col for col in df_kdj.columns if 'kdj' in col]}")
    
    # 测试B2因子
    df_b2 = add_B2_factor(df)
    print(f"✅ B2因子计算成功，B2信号数量: {df_b2['B2'].sum()}")
    
    # 测试成功率计算
    results = calculate_b2_success_rate(df)
    print(f"✅ B2成功率计算成功:")
    print(f"   总信号数: {results['total_signals']}")
    print(f"   成功率: {results['success_rate']:.2%}")
    
    print("🎉 B2核心模块测试完成！")


if __name__ == "__main__":
    test_b2_core()