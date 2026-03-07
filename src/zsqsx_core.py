#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZSQSX公式核心计算模块

包含ZSDKX公式的向量化实现，支持参数自定义。
严格对齐通达信逻辑：
- QSX = EMA(EMA(CLOSE,10),10)
- MA1 = MA(CLOSE,60)
- MA2 = EMA(CLOSE,13)
- DKS = (MA(CLOSE,M1)+MA(CLOSE,M2)+MA(CLOSE,M3)+MA(CLOSE,M4))/4

作者: MC
创建日期: 2026-03-07
参考: 用户提供的向量化实现
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def calc_zsdkx(
    df_raw: pd.DataFrame,
    M1: int = 14,
    M2: int = 28,
    M3: int = 57,
    M4: int = 114,
    target_start_date: Optional[str] = None,
    target_end_date: Optional[str] = None,
    output_stats: bool = True
) -> pd.DataFrame:
    """
    计算ZSDKX公式指标（QSX, MA1, MA2, DKS），支持参数自定义。
    严格对齐通达信逻辑：
    - QSX = EMA(EMA(CLOSE,10),10)
    - MA1 = MA(CLOSE,60)
    - MA2 = EMA(CLOSE,13)
    - DKS = (MA(CLOSE,M1)+MA(CLOSE,M2)+MA(CLOSE,M3)+MA(CLOSE,M4))/4
    仅当四个MA均有效时（非NaN）才有值，否则为NaN。

    :param df_raw: 原始数据，必须包含 instrument, datetime, close 列
    :param M1, M2, M3, M4: 四个均线周期参数，默认值与通达信缺省值一致
    :param target_start_date: 目标时间段起始，格式 'YYYY-MM-DD'，仅返回该时段结果
    :param target_end_date: 目标时间段结束，格式 'YYYY-MM-DD'
    :param output_stats: 是否输出统计信息
    :return: 包含所有新指标的结果表，列包括 instrument, datetime, QSX, MA1, MA2, DKS
    """
    # 1. 数据校验与预处理
    required_cols = ['instrument', 'datetime', 'close']
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"缺少必要列：{missing}")

    df = df_raw.copy().reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.drop_duplicates(subset=['instrument', 'datetime'])
    df = df.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    grouped = df.groupby('instrument')

    # 2. 计算 QSX：EMA(EMA(close,10),10)
    # 通达信 EMA 使用递归公式 α=2/(N+1)，与 pandas ewm(adjust=False) 一致
    df['ema10_1'] = grouped['close'].transform(lambda x: x.ewm(span=10, adjust=False).mean())
    df['QSX'] = grouped['ema10_1'].transform(lambda x: x.ewm(span=10, adjust=False).mean())

    # 3. 计算 MA1：60日简单移动平均（前59天为NaN）
    df['MA1'] = grouped['close'].transform(lambda x: x.rolling(60, min_periods=60).mean())

    # 4. 计算 MA2：13日指数移动平均
    df['MA2'] = grouped['close'].transform(lambda x: x.ewm(span=13, adjust=False).mean())

    # 5. 计算四个周期的简单移动平均（用于 DKS）
    ma_cols = []
    for period, name in zip([M1, M2, M3, M4], ['M1', 'M2', 'M3', 'M4']):
        col_name = f'MA_{period}'
        df[col_name] = grouped['close'].transform(
            lambda x, p=period: x.rolling(p, min_periods=p).mean()
        )
        ma_cols.append(col_name)

    # 6. 计算 DKS：四个均值的平均，但必须全部非 NaN
    # 先检查每行是否有缺失值
    has_nan = df[ma_cols].isnull().any(axis=1)
    # 计算均值时，只对没有缺失的行赋值，其余保持NaN
    df['DKS'] = np.where(has_nan, np.nan, df[ma_cols].mean(axis=1))

    # 7. 删除临时中间列（可选，保留QSX等核心指标）
    df.drop(columns=['ema10_1'] + ma_cols, inplace=True, errors='ignore')

    # 8. 筛选目标时间段
    if target_start_date:
        df = df[df['datetime'] >= pd.to_datetime(target_start_date)]
    if target_end_date:
        df = df[df['datetime'] <= pd.to_datetime(target_end_date)]

    # 9. 保留核心列
    core_cols = ['instrument', 'datetime', 'QSX', 'MA1', 'MA2', 'DKS']
    df_result = df[core_cols].copy()

    # 10. 输出统计信息
    if output_stats:
        print("=" * 50)
        print("📊 ZSDKX指标计算统计")
        print("=" * 50)
        print(f"总样本数：{len(df_result):,}")
        print(f"股票数：{df_result['instrument'].nunique():,}")
        print(f"日期范围：{df_result['datetime'].min()} 至 {df_result['datetime'].max()}")
        print("\n指标非空统计：")
        for col in ['QSX', 'MA1', 'MA2', 'DKS']:
            non_null = df_result[col].notna().sum()
            print(f" {col}: {non_null:,} 个有效值 ({non_null/len(df_result):.1%})")
        print("=" * 50)

    return df_result


def calculate_zsdkx_features_vectorized(
    df: pd.DataFrame,
    M1: int = 14,
    M2: int = 28,
    M3: int = 57,
    M4: int = 114
) -> pd.DataFrame:
    """
    向量化计算ZSDKX特征（全市场一次性计算）
    
    :param df: 包含所有股票数据的DataFrame，必须有instrument, datetime, close列
    :param M1, M2, M3, M4: 四个均线周期参数
    :return: 添加了ZSDKX指标的DataFrame
    """
    return calc_zsdkx(
        df_raw=df,
        M1=M1,
        M2=M2,
        M3=M3,
        M4=M4,
        target_start_date=None,
        target_end_date=None,
        output_stats=False
    )


def get_zsdkx_signal_conditions(
    df: pd.DataFrame,
    qsx_threshold: float = 0.0,
    ma1_threshold: float = 0.0,
    ma2_threshold: float = 0.0,
    dks_threshold: float = 0.0
) -> pd.DataFrame:
    """
    根据ZSDKX指标生成信号条件
    
    :param df: 包含ZSDKX指标的DataFrame
    :param qsx_threshold: QSX阈值
    :param ma1_threshold: MA1阈值
    :param ma2_threshold: MA2阈值
    :param dks_threshold: DKS阈值
    :return: 添加了信号条件的DataFrame
    """
    df_signal = df.copy()
    
    # 基础信号条件
    df_signal['QSX_above'] = df_signal['QSX'] > qsx_threshold
    df_signal['MA1_above'] = df_signal['MA1'] > ma1_threshold
    df_signal['MA2_above'] = df_signal['MA2'] > ma2_threshold
    df_signal['DKS_above'] = df_signal['DKS'] > dks_threshold
    
    # 组合信号
    df_signal['ZSQSX_signal'] = (
        df_signal['QSX_above'] &
        df_signal['MA1_above'] &
        df_signal['MA2_above'] &
        df_signal['DKS_above']
    )
    
    return df_signal


def analyze_zsdkx_performance(
    df: pd.DataFrame,
    success_col: str = 'success'
) -> Dict[str, Any]:
    """
    分析ZSDKX信号的性能
    
    :param df: 包含ZSQSX_signal和成功列的DataFrame
    :param success_col: 成功列名
    :return: 性能统计字典
    """
    if 'ZSQSX_signal' not in df.columns:
        raise ValueError("DataFrame必须包含'ZSQSX_signal'列")
    
    if success_col not in df.columns:
        raise ValueError(f"DataFrame必须包含'{success_col}'列")
    
    signals = df[df['ZSQSX_signal'] == True]
    
    if signals.empty:
        return {
            'total_signals': 0,
            'successful_signals': 0,
            'success_rate': 0.0,
            'signal_dates': [],
            'instruments': []
        }
    
    total_signals = len(signals)
    successful_signals = signals[success_col].sum()
    success_rate = successful_signals / total_signals if total_signals > 0 else 0.0
    
    return {
        'total_signals': total_signals,
        'successful_signals': successful_signals,
        'success_rate': success_rate,
        'signal_dates': signals['datetime'].unique().tolist(),
        'instruments': signals['instrument'].unique().tolist()
    }


def test_zsdkx_core():
    """测试ZSDKX核心函数"""
    print("🧪 测试ZSDKX核心函数...")
    
    # 生成模拟数据（3只股票，120天）
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    instruments = ["600519", "000001", "000858"]
    test_data = []
    for inst in instruments:
        price = 100 + np.cumsum(np.random.randn(120) * 2)
        for i, dt in enumerate(dates):
            test_data.append({
                "instrument": inst,
                "datetime": dt,
                "close": price[i]
            })
    test_df = pd.DataFrame(test_data)
    
    print("  测试calc_zsdkx函数...")
    try:
        result = calc_zsdkx(
            test_df,
            M1=14, M2=28, M3=57, M4=114,
            target_start_date="2023-03-01",
            output_stats=True
        )
        print(f"   计算成功，结果形状: {result.shape}")
    except Exception as e:
        print(f"   测试失败: {e}")
    
    print("  测试calculate_zsdkx_features_vectorized函数...")
    try:
        df_vectorized = calculate_zsdkx_features_vectorized(test_df)
        print(f"   向量化计算成功，结果形状: {df_vectorized.shape}")
    except Exception as e:
        print(f"   测试失败: {e}")
    
    print("  测试get_zsdkx_signal_conditions函数...")
    try:
        df_with_signals = get_zsdkx_signal_conditions(result)
        signal_count = df_with_signals['ZSQSX_signal'].sum()
        print(f"   信号条件生成成功，信号数量: {signal_count}")
    except Exception as e:
        print(f"   测试失败: {e}")
    
    print("✅ ZSDKX核心函数测试完成")
    return True


if __name__ == "__main__":
    test_zsdkx_core()