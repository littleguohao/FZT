#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FZT选股公式统一回测脚本

支持2006-2020年和2021-2026年两个时期，以及所有技术因子参数化控制。

作者: MC
创建日期: 2026-03-07
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.fzt_core import calc_brick_pattern_final, calculate_fzt_features_vectorized
from src.zsqsx_core import calc_zsdkx, get_zsdkx_signal_conditions
from src.factors import (
    add_volume_ratio_factor, filter_by_volume_ratio_factor,
    add_bias_factor, filter_by_bias_factor,
    add_rsi_factor, filter_by_rsi_factor,
    add_obv_factor, filter_by_obv_factor,
    add_max_gain_condition, filter_by_max_gain_condition,
    add_volume_and_bias_factors, filter_by_volume_bias_factors,
    filter_by_rsi_obv_factors,
    add_custom_factors, filter_by_custom_factors  # 新增可配置因子
)
from src.b2_core import compute_kdj  # 导入KDJ计算函数
from src.data_loader import load_stock_data_qlib, get_instruments_from_file


def load_data_by_period(project_root: Path, period: str) -> pd.DataFrame:
    """
    根据时期加载数据
    
    :param project_root: 项目根目录
    :param period: 时期 ('2006_2020' 或 '2021_2026')
    :return: 包含股票数据的DataFrame
    """
    if period == '2006_2020':
        # 数据目录
        data_dir = str(project_root / 'data' / '2006_2020')
        
        # 读取股票列表
        instruments_file = data_dir + '/instruments/all.txt'
        instruments = get_instruments_from_file(instruments_file)
        
        # 时间范围
        calc_start = '2006-01-01'
        calc_end = '2020-12-31'
        
        print(f"📊 加载2006-2020年数据:")
        print(f"   数据目录: {data_dir}")
        print(f"   股票数量: {len(instruments):,}")
        print(f"   时间范围: {calc_start} 到 {calc_end}")
        
    elif period == '2021_2026':
        # 数据目录
        data_dir = str(project_root / 'data' / '2021_2026')
        
        # 读取股票列表
        instruments_file = data_dir + '/instruments/all.txt'
        instruments = get_instruments_from_file(instruments_file)
        
        # 时间范围
        calc_start = '2021-08-02'
        calc_end = '2026-02-06'
        
        print(f"📊 加载2021-2026年数据:")
        print(f"   数据目录: {data_dir}")
        print(f"   股票数量: {len(instruments):,}")
        print(f"   时间范围: {calc_start} 到 {calc_end}")
    
    else:
        raise ValueError(f"不支持的时期: {period}")
    
    # 加载数据
    df = load_stock_data_qlib(
        data_dir=data_dir,
        instruments=instruments,
        calc_start=calc_start,
        calc_end=calc_end,
        fields=['$open', '$high', '$low', '$close', '$volume']
    )
    
    print(f"✅ 数据加载完成:")
    print(f"   总行数: {len(df):,}")
    print(f"   股票数量: {df['instrument'].nunique():,}")
    print(f"   时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    
    return df


def run_fzt_backtest(
    df: pd.DataFrame,
    use_fzt: bool = True,
    use_zsqsx: bool = True,
    verify_fzt_only: bool = False,
    top_n: int = 0,
    top_k: int = 4,
    # 成交量比因子参数
    volume_ratio_enable: bool = False,
    volume_ratio_threshold: float = 1.2,
    # 乖离率因子参数
    bias_enable: bool = False,
    bias_lower: float = -0.05,
    bias_upper: float = 0.2,
    # RSI因子参数
    rsi_enable: bool = False,
    rsi_period: int = 14,
    rsi_low: float = 40,
    rsi_high: float = 70,
    # OBV因子参数
    obv_enable: bool = False,
    obv_lookback: int = 20,
    # 短期暴涨过滤因子参数
    max_gain_enable: bool = False,
    max_gain_lookback: int = 35,
    max_gain_threshold: float = 0.7,
    # 组合因子参数（向后兼容）
    volume_bias: bool = False,
    rsi_obv: bool = False,
    # KDJ J值判断参数
    kdj_j_enable: bool = False,
    kdj_j_threshold: float = 13
) -> Dict[str, Any]:
    """
    运行FZT公式回测
    
    :param df: 包含股票数据的DataFrame
    :param use_fzt: 是否使用FZT公式 (默认: True)
    :param use_zsqsx: 是否使用ZSQSX公式 (默认: True)
    :param verify_fzt_only: 是否只验证FZT公式 (默认: False)
    :param top_n: TOP排序的N值 (默认: 0，表示不启用)
    :param top_k: TOP排序的K值 (默认: 4)
    :param volume_ratio_enable: 启用成交量比因子筛选 (默认: False)
    :param volume_ratio_threshold: 成交量比阈值 (默认: 1.2)
    :param bias_enable: 启用乖离率因子筛选 (默认: False)
    :param bias_lower: 乖离率下限 (默认: -0.05)
    :param bias_upper: 乖离率上限 (默认: 0.2)
    :param rsi_enable: 启用RSI因子筛选 (默认: False)
    :param rsi_period: RSI计算周期 (默认: 14)
    :param rsi_low: RSI下限 (默认: 40)
    :param rsi_high: RSI上限 (默认: 70)
    :param obv_enable: 启用OBV因子筛选 (默认: False)
    :param obv_lookback: OBV创新高回看周期 (默认: 20)
    :param max_gain_enable: 启用短期暴涨过滤因子 (默认: False)
    :param max_gain_lookback: 短期暴涨回看周期 (默认: 35)
    :param max_gain_threshold: 短期暴涨阈值，如0.7表示70% (默认: 0.7)
    :param volume_bias: 启用成交量比和乖离率因子筛选 (默认: False)
    :param rsi_obv: 启用RSI和OBV因子筛选 (默认: False)
    :param kdj_j_enable: 启用KDJ J值判断筛选 (默认: False)
    :param kdj_j_threshold: KDJ J值阈值 (默认: 13)
    :return: 回测结果字典
    """
    start_time = time.time()
    results = {}
    
    print(f"\n🧮 开始回测计算...")
    
    # 1. 准备数据
    df = df.copy()
    df = df.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    
    # 2. 计算成功条件（次日收盘价 > 当日收盘价）
    grouped = df.groupby('instrument')
    df['next_close'] = grouped['close'].transform(lambda x: x.shift(-1))
    df['success'] = df['next_close'] > df['close']
    
    # 3. 计算FZT公式
    df_fzt = None
    if use_fzt:
        print(f"📈 计算FZT公式...")
        df_fzt = calc_brick_pattern_final(df)
        
        # 筛选FZT信号
        fzt_signals = df_fzt[df_fzt['选股条件'] == True]
        
        # 合并成功数据
        df_combined = pd.merge(
            fzt_signals[['instrument', 'datetime', '选股条件', '砖型图面积']],
            df[['instrument', 'datetime', 'success']],
            on=['instrument', 'datetime'],
            how='inner'
        )
        
        # 重命名列
        df_combined = df_combined.rename(columns={'选股条件': 'FZT_signal'})
        
        # 计算FZT成功率
        fzt_total = len(df_combined)
        fzt_success = df_combined['success'].sum() if fzt_total > 0 else 0
        fzt_rate = fzt_success / fzt_total if fzt_total > 0 else 0
        
        print(f"   FZT总信号数: {fzt_total:,}")
        print(f"   FZT成功信号数: {fzt_success:,}")
        print(f"   FZT成功率: {fzt_rate:.2%}")
        
        results['fzt'] = {
            'total': fzt_total,
            'success': fzt_success,
            'rate': fzt_rate
        }
        
        # 年度分析
        print(f"\n📅 FZT年度成功率分析:")
        df_combined['year'] = df_combined['datetime'].dt.year
        
        yearly_stats = []
        for year in sorted(df_combined['year'].unique()):
            year_data = df_combined[df_combined['year'] == year]
            year_total = len(year_data)
            year_success = year_data['success'].sum() if year_total > 0 else 0
            year_rate = year_success / year_total if year_total > 0 else 0
            
            yearly_stats.append({
                'year': year,
                'total': year_total,
                'success': year_success,
                'rate': year_rate
            })
            
            print(f"   {year}: {year_rate:.2%} ({year_total:,} 个信号)")
        
        results['yearly_fzt_stats'] = yearly_stats
        
        # 4. TOP排序筛选
        if top_n > 0 and top_k > 0:
            print(f"\n🏆 TOP{top_k}排序筛选 (N={top_n}, K={top_k}):")
            
            # 按砖型图面积排序
            df_combined = df_combined.sort_values(['datetime', '砖型图面积'], ascending=[True, False])
            
            # 分组取TOP K
            top_signals = df_combined.groupby('datetime').head(top_k)
            
            top_total = len(top_signals)
            top_success = top_signals['success'].sum() if top_total > 0 else 0
            top_rate = top_success / top_total if top_total > 0 else 0
            
            print(f"   TOP{top_k}总信号数: {top_total:,}")
            print(f"   TOP{top_k}成功信号数: {top_success:,}")
            print(f"   TOP{top_k}成功率: {top_rate:.2%}")
            
            results['fzt_top'] = {
                'total': top_total,
                'success': top_success,
                'rate': top_rate
            }
        
        # 5. 技术因子筛选
        # 获取包含close和volume的完整数据
        if df_fzt is not None and 'close' in df_fzt.columns and 'volume' in df_fzt.columns:
            # 准备基础数据
            df_with_factors = df_fzt[['instrument', 'datetime', 'close', 'volume', 'high', '选股条件']].copy()
            df_with_factors = df_with_factors.rename(columns={'选股条件': 'FZT_signal'})
            
            # 合并成功数据
            df_merged_base = pd.merge(
                df_with_factors,
                df[['instrument', 'datetime', 'success']],
                on=['instrument', 'datetime'],
                how='inner'
            )
            
            # 5.1 成交量比因子筛选
            if volume_ratio_enable:
                print(f"\n📊 新增：成交量比因子筛选:")
                print(f"   成交量比 > {volume_ratio_threshold}")
                
                df_vr = add_volume_ratio_factor(df_merged_base.copy())
                df_vr = filter_by_volume_ratio_factor(df_vr, volume_ratio_threshold=volume_ratio_threshold)
                
                vr_signals = df_vr[
                    (df_vr['FZT_signal'] == True) & 
                    (df_vr['volume_ratio_signal'] == True)
                ]
                
                if not vr_signals.empty:
                    vr_total = len(vr_signals)
                    vr_success = vr_signals['success'].sum() if vr_total > 0 else 0
                    vr_rate = vr_success / vr_total if vr_total > 0 else 0
                    
                    print(f"   FZT+成交量比总信号数: {vr_total:,}")
                    print(f"   FZT+成交量比成功信号数: {vr_success:,}")
                    print(f"   FZT+成交量比成功率: {vr_rate:.2%}")
                    
                    results['fzt_volume_ratio'] = {
                        'total': vr_total,
                        'success': vr_success,
                        'rate': vr_rate
                    }
                else:
                    print(f"   ⚠️ 没有符合条件的FZT+成交量比信号")
            
            # 5.2 乖离率因子筛选
            if bias_enable:
                print(f"\n📊 新增：乖离率因子筛选:")
                print(f"   乖离率在 [{bias_lower:.2%}, {bias_upper:.2%}] 之间")
                
                df_bias = add_bias_factor(df_merged_base.copy())
                df_bias = filter_by_bias_factor(df_bias, bias_lower=bias_lower, bias_upper=bias_upper)
                
                bias_signals = df_bias[
                    (df_bias['FZT_signal'] == True) & 
                    (df_bias['bias_signal'] == True)
                ]
                
                if not bias_signals.empty:
                    bias_total = len(bias_signals)
                    bias_success = bias_signals['success'].sum() if bias_total > 0 else 0
                    bias_rate = bias_success / bias_total if bias_total > 0 else 0
                    
                    print(f"   FZT+乖离率总信号数: {bias_total:,}")
                    print(f"   FZT+乖离率成功信号数: {bias_success:,}")
                    print(f"   FZT+乖离率成功率: {bias_rate:.2%}")
                    
                    results['fzt_bias'] = {
                        'total': bias_total,
                        'success': bias_success,
                        'rate': bias_rate
                    }
                else:
                    print(f"   ⚠️ 没有符合条件的FZT+乖离率信号")
            
            # 5.3 RSI因子筛选
            if rsi_enable:
                print(f"\n📊 新增：RSI因子筛选:")
                print(f"   RSI({rsi_period})在 [{rsi_low}, {rsi_high}] 之间")
                
                df_rsi = add_rsi_factor(df_merged_base.copy(), rsi_period=rsi_period)
                df_rsi = filter_by_rsi_factor(df_rsi, rsi_low=rsi_low, rsi_high=rsi_high)
                
                rsi_signals = df_rsi[
                    (df_rsi['FZT_signal'] == True) & 
                    (df_rsi['rsi_signal'] == True)
                ]
                
                if not rsi_signals.empty:
                    rsi_total = len(rsi_signals)
                    rsi_success = rsi_signals['success'].sum() if rsi_total > 0 else 0
                    rsi_rate = rsi_success / rsi_total if rsi_total > 0 else 0
                    
                    print(f"   FZT+RSI总信号数: {rsi_total:,}")
                    print(f"   FZT+RSI成功信号数: {rsi_success:,}")
                    print(f"   FZT+RSI成功率: {rsi_rate:.2%}")
                    
                    results['fzt_rsi'] = {
                        'total': rsi_total,
                        'success': rsi_success,
                        'rate': rsi_rate
                    }
                else:
                    print(f"   ⚠️ 没有符合条件的FZT+RSI信号")
            
            # 5.4 OBV因子筛选
            if obv_enable:
                print(f"\n📊 新增：OBV因子筛选:")
                print(f"   OBV创{obv_lookback}日新高")
                
                df_obv = add_obv_factor(df_merged_base.copy())
                df_obv = filter_by_obv_factor(df_obv, obv_lookback=obv_lookback)
                
                obv_signals = df_obv[
                    (df_obv['FZT_signal'] == True) & 
                    (df_obv['obv_signal'] == True)
                ]
                
                if not obv_signals.empty:
                    obv_total = len(obv_signals)
                    obv_success = obv_signals['success'].sum() if obv_total > 0 else 0
                    obv_rate = obv_success / obv_total if obv_total > 0 else 0
                    
                    print(f"   FZT+OBV总信号数: {obv_total:,}")
                    print(f"   FZT+OBV成功信号数: {obv_success:,}")
                    print(f"   FZT+OBV成功率: {obv_rate:.2%}")
                    
                    results['fzt_obv'] = {
                        'total': obv_total,
                        'success': obv_success,
                        'rate': obv_rate
                    }
                else:
                    print(f"   ⚠️ 没有符合条件的FZT+OBV信号")
            
            # 5.5 短期暴涨过滤因子筛选
            if max_gain_enable:
                print(f"\n📊 新增：短期暴涨过滤因子筛选:")
                print(f"   过滤掉最近{max_gain_lookback}日涨幅超过{max_gain_threshold:.0%}的股票")
                
                df_mg = filter_by_max_gain_condition(
                    df_merged_base.copy(),
                    lookback=max_gain_lookback,
                    threshold=max_gain_threshold
                )
                
                mg_signals = df_mg[
                    (df_mg['FZT_signal'] == True) & 
                    (df_mg['max_gain_signal'] == True)
                ]
                
                if not mg_signals.empty:
                    mg_total = len(mg_signals)
                    mg_success = mg_signals['success'].sum() if mg_total > 0 else 0
                    mg_rate = mg_success / mg_total if mg_total > 0 else 0
                    
                    print(f"   FZT+过滤短期暴涨总信号数: {mg_total:,}")
                    print(f"   FZT+过滤短期暴涨成功信号数: {mg_success:,}")
                    print(f"   FZT+过滤短期暴涨成功率: {mg_rate:.2%}")
                    
                    # 计算过滤掉的股票数量
                    fzt_total = len(df_mg[df_mg['FZT_signal'] == True])
                    filtered_out = fzt_total - mg_total
                    print(f"   过滤掉的短期暴涨股票数: {filtered_out:,}")
                    print(f"   过滤比例: {filtered_out/fzt_total:.2%}")
                    
                    results['fzt_filter_max_gain'] = {
                        'total': mg_total,
                        'success': mg_success,
                        'rate': mg_rate,
                        'filtered_out': filtered_out,
                        'filter_ratio': filtered_out/fzt_total if fzt_total > 0 else 0
                    }
                else:
                    print(f"   ⚠️ 没有符合条件的FZT+过滤短期暴涨信号")
            
            # 5.6 组合因子筛选（向后兼容）
            if volume_bias:
                print(f"\n📊 新增：成交量比和乖离率组合因子筛选（向后兼容）:")
                print(f"   成交量比 > {volume_ratio_threshold}")
                print(f"   乖离率在 [{bias_lower:.2%}, {bias_upper:.2%}] 之间")
                
                df_vb = add_volume_and_bias_factors(df_merged_base.copy())
                df_vb = filter_by_volume_bias_factors(
                    df_vb,
                    volume_ratio_threshold=volume_ratio_threshold,
                    bias_lower=bias_lower,
                    bias_upper=bias_upper
                )
                
                vb_signals = df_vb[
                    (df_vb['FZT_signal'] == True) & 
                    (df_vb['volume_bias_signal'] == True)
                ]
                
                if not vb_signals.empty:
                    vb_total = len(vb_signals)
                    vb_success = vb_signals['success'].sum() if vb_total > 0 else 0
                    vb_rate = vb_success / vb_total if vb_total > 0 else 0
                    
                    print(f"   FZT+成交量比乖离率总信号数: {vb_total:,}")
                    print(f"   FZT+成交量比乖离率成功信号数: {vb_success:,}")
                    print(f"   FZT+成交量比乖离率成功率: {vb_rate:.2%}")
                    
                    results['fzt_volume_bias'] = {
                        'total': vb_total,
                        'success': vb_success,
                        'rate': vb_rate
                    }
                else:
                    print(f"   ⚠️ 没有符合条件的FZT+成交量比乖离率信号")
            
            if rsi_obv:
                print(f"\n📊 新增：RSI和OBV组合因子筛选（向后兼容）:")
                print(f"   RSI({rsi_period})在 [{rsi_low}, {rsi_high}] 之间")
                print(f"   OBV创{obv_lookback}日新高")
                
                df_ro = filter_by_rsi_obv_factors(
                    df_merged_base.copy(),
                    rsi_period=rsi_period,
                    rsi_low=rsi_low,
                    rsi_high=rsi_high,
                    obv_lookback=obv_lookback
                )
                
                ro_signals = df_ro[
                    (df_ro['FZT_signal'] == True) & 
                    (df_ro['rsi_obv_signal'] == True)
                ]
                
                if not ro_signals.empty:
                    ro_total = len(ro_signals)
                    ro_success = ro_signals['success'].sum() if ro_total > 0 else 0
                    ro_rate = ro_success / ro_total if ro_total > 0 else 0
                    
                    print(f"   FZT+RSI-OBV总信号数: {ro_total:,}")
                    print(f"   FZT+RSI-OBV成功信号数: {ro_success:,}")
                    print(f"   FZT+RSI-OBV成功率: {ro_rate:.2%}")
                    
                    results['fzt_rsi_obv'] = {
                        'total': ro_total,
                        'success': ro_success,
                        'rate': ro_rate
                    }
                else:
                    print(f"   ⚠️ 没有符合条件的FZT+RSI-OBV信号")
            
            # 5.7 KDJ J值判断筛选
            if kdj_j_enable:
                print(f"\n📊 新增：KDJ J值判断筛选:")
                print(f"   前一日KDJ J值 < {kdj_j_threshold}")
                
                # 需要从原始数据中获取完整的OHLC数据来计算KDJ
                # 首先获取所有FZT信号的股票和时间
                fzt_signals_info = df_merged_base[['instrument', 'datetime', 'FZT_signal', 'success']].copy()
                
                # 从原始数据中获取这些股票和时间的完整OHLC数据
                df_fzt_ohlc = df[['instrument', 'datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
                
                # 合并FZT信号信息和OHLC数据
                df_kdj_base = pd.merge(
                    fzt_signals_info,
                    df_fzt_ohlc,
                    on=['instrument', 'datetime'],
                    how='inner'
                )
                
                # 计算KDJ指标
                df_kdj = compute_kdj(df_kdj_base)
                
                # 添加前一日J值
                df_kdj = df_kdj.sort_values(['instrument', 'datetime']).reset_index(drop=True)
                grouped = df_kdj.groupby('instrument')
                df_kdj['prev_kdj_j'] = grouped['kdj_j'].transform(lambda x: x.shift(1))
                
                # 筛选条件：前一日J值 < 阈值
                df_kdj['kdj_j_signal'] = df_kdj['prev_kdj_j'] < kdj_j_threshold
                
                # 筛选同时满足FZT和KDJ J值条件的信号
                kdj_signals = df_kdj[
                    (df_kdj['FZT_signal'] == True) & 
                    (df_kdj['kdj_j_signal'] == True)
                ]
                
                if not kdj_signals.empty:
                    kdj_total = len(kdj_signals)
                    kdj_success = kdj_signals['success'].sum() if kdj_total > 0 else 0
                    kdj_rate = kdj_success / kdj_total if kdj_total > 0 else 0
                    
                    print(f"   FZT+KDJ J值判断总信号数: {kdj_total:,}")
                    print(f"   FZT+KDJ J值判断成功信号数: {kdj_success:,}")
                    print(f"   FZT+KDJ J值判断成功率: {kdj_rate:.2%}")
                    
                    # 计算过滤掉的股票数量
                    fzt_total = len(df_kdj[df_kdj['FZT_signal'] == True])
                    filtered_out = fzt_total - kdj_total
                    print(f"   过滤掉的股票数: {filtered_out:,}")
                    print(f"   过滤比例: {filtered_out/fzt_total:.2%}")
                    
                    results['fzt_kdj_j'] = {
                        'total': kdj_total,
                        'success': kdj_success,
                        'rate': kdj_rate,
                        'filtered_out': filtered_out,
                        'filter_ratio': filtered_out/fzt_total if fzt_total > 0 else 0
                    }
                else:
                    print(f"   ⚠️ 没有符合条件的FZT+KDJ J值判断信号")
        
        elif (volume_ratio_enable or bias_enable or rsi_enable or obv_enable or max_gain_enable or volume_bias or rsi_obv or kdj_j_enable) and not use_fzt:
            print(f"\nℹ️  技术因子需要FZT公式，但FZT未启用")
    
    # 6. 单独ZSQSX公式
    if use_zsqsx and not verify_fzt_only:
        print(f"\n📈 计算ZSQSX公式...")
        
        # 计算ZSQSX
        df_zsdkx = calc_zsdkx(df)
        df_zsqsx = get_zsdkx_signal_conditions(df_zsdkx)
        
        # 筛选ZSQSX信号
        zsqsx_signals = df_zsqsx[df_zsqsx['ZSQSX_signal'] == True]
        
        # 合并成功数据
        df_zsqsx_combined = pd.merge(
            zsqsx_signals[['instrument', 'datetime', 'ZSQSX_signal']],
            df[['instrument', 'datetime', 'success']],
            on=['instrument', 'datetime'],
            how='inner'
        )
        
        # 计算ZSQSX成功率
        zsqsx_total = len(df_zsqsx_combined)
        zsqsx_success = df_zsqsx_combined['success'].sum() if zsqsx_total > 0 else 0
        zsqsx_rate = zsqsx_success / zsqsx_total if zsqsx_total > 0 else 0
        
        print(f"   ZSQSX总信号数: {zsqsx_total:,}")
        print(f"   ZSQSX成功信号数: {zsqsx_success:,}")
        print(f"   ZSQSX成功率: {zsqsx_rate:.2%}")
        
        results['zsqsx'] = {
            'total': zsqsx_total,
            'success': zsqsx_success,
            'rate': zsqsx_rate
        }
        
        # 年度分析
        print(f"\n📅 ZSQSX年度成功率分析:")
        df_zsqsx_combined['year'] = df_zsqsx_combined['datetime'].dt.year
        
        yearly_zsqsx_stats = []
        for year in sorted(df_zsqsx_combined['year'].unique()):
            year_data = df_zsqsx_combined[df_zsqsx_combined['year'] == year]
            year_total = len(year_data)
            year_success = year_data['success'].sum() if year_total > 0 else 0
            year_rate = year_success / year_total if year_total > 0 else 0
            
            yearly_zsqsx_stats.append({
                'year': year,
                'total': year_total,
                'success': year_success,
                'rate': year_rate
            })
            
            print(f"   {year}: {year_rate:.2%} ({year_total:,} 个信号)")
        
        results['yearly_zsqsx_stats'] = yearly_zsqsx_stats
    
    # 7. FZT+ZSQSX组合
    if use_fzt and use_zsqsx and not verify_fzt_only:
        print(f"\n📈 计算FZT+ZSQSX组合...")
        
        # 确保有FZT和ZSQSX数据
        if df_fzt is not None and 'ZSQSX_signal' in locals():
            # 合并FZT和ZSQSX信号
            df_fzt_zsqsx = pd.merge(
                df_fzt[['instrument', 'datetime', '选股条件']],
                df_zsqsx[['instrument', 'datetime', 'ZSQSX_signal']],
                on=['instrument', 'datetime'],
                how='inner'
            )
            
            # 筛选同时满足FZT和ZSQSX的信号
            fzt_zsqsx_signals = df_fzt_zsqsx[
                (df_fzt_zsqsx['选股条件'] == True) & 
                (df_fzt_zsqsx['ZSQSX_signal'] == True)
            ]
            
            # 合并成功数据
            df_fzt_zsqsx_combined = pd.merge(
                fzt_zsqsx_signals[['instrument', 'datetime']],
                df[['instrument', 'datetime', 'success']],
                on=['instrument', 'datetime'],
                how='inner'
            )
            
            # 计算组合成功率
            fzt_zsqsx_total = len(df_fzt_zsqsx_combined)
            fzt_zsqsx_success = df_fzt_zsqsx_combined['success'].sum() if fzt_zsqsx_total > 0 else 0
            fzt_zsqsx_rate = fzt_zsqsx_success / fzt_zsqsx_total if fzt_zsqsx_total > 0 else 0
            
            print(f"   FZT+ZSQSX总信号数: {fzt_zsqsx_total:,}")
            print(f"   FZT+ZSQSX成功信号数: {fzt_zsqsx_success:,}")
            print(f"   FZT+ZSQSX成功率: {fzt_zsqsx_rate:.2%}")
            
            results['fzt_zsqsx'] = {
                'total': fzt_zsqsx_total,
                'success': fzt_zsqsx_success,
                'rate': fzt_zsqsx_rate
            }
    
    # 计算执行时间
    elapsed_time = time.time() - start_time
    results['elapsed_time'] = elapsed_time
    
    print(f"\n⏱️  总执行时间: {elapsed_time:.2f} 秒")
    
    return results


def print_summary(results: Dict[str, Any], period_name: str):
    """
    打印回测结果摘要
    
    :param results: 回测结果字典
    :param period_name: 时期名称
    """
    print(f"\n{'='*60}")
    print(f"📊 FZT回测结果摘要 ({period_name})")
    print(f"{'='*60}")
    
    if 'fzt' in results:
        fzt = results['fzt']
        print(f"📈 FZT公式:")
        print(f"   总信号数: {fzt['total']:,}")
        print(f"   成功信号数: {fzt['success']:,}")
        print(f"   成功率: {fzt['rate']:.2%}")
    
    if 'fzt_top' in results:
        fzt_top = results['fzt_top']
        print(f"\n🏆 FZT TOP{results.get('top_k', 4)}排序:")
        print(f"   总信号数: {fzt_top['total']:,}")
        print(f"   成功信号数: {fzt_top['success']:,}")
        print(f"   成功率: {fzt_top['rate']:.2%}")
    
    if 'zsqsx' in results:
        zsqsx = results['zsqsx']
        print(f"\n📈 ZSQSX公式:")
        print(f"   总信号数: {zsqsx['total']:,}")
        print(f"   成功信号数: {zsqsx['success']:,}")
        print(f"   成功率: {zsqsx['rate']:.2%}")
    
    if 'fzt_zsqsx' in results:
        fzt_zsqsx = results['fzt_zsqsx']
        print(f"\n📈 FZT+ZSQSX组合:")
        print(f"   总信号数: {fzt_zsqsx['total']:,}")
        print(f"   成功信号数: {fzt_zsqsx['success']:,}")
        print(f"   成功率: {fzt_zsqsx['rate']:.2%}")
    
    # 技术因子结果
    tech_factors = [
        ('fzt_volume_ratio', '成交量比'),
        ('fzt_bias', '乖离率'),
        ('fzt_rsi', 'RSI'),
        ('fzt_obv', 'OBV'),
        ('fzt_filter_max_gain', '过滤短期暴涨'),
        ('fzt_volume_bias', '成交量比+乖离率'),
        ('fzt_rsi_obv', 'RSI+OBV'),
        ('fzt_kdj_j', 'KDJ J值判断')
    ]
    
    tech_results = []
    for key, name in tech_factors:
        if key in results:
            tech_results.append((name, results[key]))
    
    if tech_results:
        print(f"\n🔧 技术因子筛选结果:")
        for name, result in tech_results:
            print(f"   {name}: {result['rate']:.2%} ({result['total']:,} 个信号)")
    
    print(f"\n⏱️  执行时间: {results['elapsed_time']:.2f} 秒")
    print(f"{'='*60}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FZT选股公式统一回测脚本')
    
    # 时期选择
    parser.add_argument('--period', type=str, default='2006_2020',
                       choices=['2006_2020', '2021_2026', 'both'],
                       help='回测时期 (默认: 2006_2020)')
    
    # 公式选择
    parser.add_argument('--fzt', action='store_true', default=True,
                       help='启用FZT公式 (默认: True)')
    parser.add_argument('--no-fzt', action='store_false', dest='fzt',
                       help='禁用FZT公式')
    parser.add_argument('--zsqsx', action='store_true', default=True,
                       help='启用ZSQSX公式 (默认: True)')
    parser.add_argument('--no-zsqsx', action='store_false', dest='zsqsx',
                       help='禁用ZSQSX公式')
    parser.add_argument('--verify-fzt', action='store_true', default=False,
                       help='只验证FZT公式 (默认: False)')
    
    # TOP排序参数
    parser.add_argument('--top-n', type=int, default=0,
                       help='TOP排序的N值 (默认: 0，表示不启用)')
    parser.add_argument('--top-k', type=int, default=4,
                       help='TOP排序的K值 (默认: 4)')
    
    # 成交量比因子参数
    parser.add_argument('--volume-ratio-enable', action='store_true', default=False,
                       help='启用成交量比因子筛选 (默认: False)')
    parser.add_argument('--volume-ratio-threshold', type=float, default=1.2,
                       help='成交量比阈值 (默认: 1.2)')
    
    # 乖离率因子参数
    parser.add_argument('--bias-enable', action='store_true', default=False,
                       help='启用乖离率因子筛选 (默认: False)')
    parser.add_argument('--bias-lower', type=float, default=-0.05,
                       help='乖离率下限 (默认: -0.05)')
    parser.add_argument('--bias-upper', type=float, default=0.2,
                       help='乖离率上限 (默认: 0.2)')
    
    # RSI因子参数
    parser.add_argument('--rsi-enable', action='store_true', default=False,
                       help='启用RSI因子筛选 (默认: False)')
    parser.add_argument('--rsi-period', type=int, default=14,
                       help='RSI计算周期 (默认: 14)')
    parser.add_argument('--rsi-low', type=float, default=40,
                       help='RSI下限 (默认: 40)')
    parser.add_argument('--rsi-high', type=float, default=70,
                       help='RSI上限 (默认: 70)')
    
    # OBV因子参数
    parser.add_argument('--obv-enable', action='store_true', default=False,
                       help='启用OBV因子筛选 (默认: False)')
    parser.add_argument('--obv-lookback', type=int, default=20,
                       help='OBV创新高回看周期 (默认: 20)')
    
    # 短期暴涨过滤因子参数
    parser.add_argument('--max-gain-enable', action='store_true', default=False,
                       help='启用短期暴涨过滤因子 (默认: False)')
    parser.add_argument('--max-gain-lookback', type=int, default=35,
                       help='短期暴涨回看周期 (默认: 35)')
    parser.add_argument('--max-gain-threshold', type=float, default=0.7,
                       help='短期暴涨阈值，如0.7表示70%% (默认: 0.7)')
    
    # 组合因子参数（向后兼容）
    parser.add_argument('--volume-bias', action='store_true', default=False,
                       help='启用成交量比和乖离率因子筛选 (默认: False)')
    parser.add_argument('--rsi-obv', action='store_true', default=False,
                       help='启用RSI和OBV因子筛选 (默认: False)')
    
    # KDJ J值判断参数
    parser.add_argument('--kdj-j-enable', action='store_true', default=False,
                       help='启用KDJ J值判断筛选 (默认: False)')
    parser.add_argument('--kdj-j-threshold', type=float, default=13,
                       help='KDJ J值阈值 (默认: 13)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 FZT选股公式统一回测脚本")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    
    all_results = {}
    
    # 运行2006-2020年回测
    if args.period in ['2006_2020', 'both']:
        print(f"\n{'='*60}")
        print("📅 2006-2020年回测")
        print(f"{'='*60}")
        
        try:
            df_2006_2020 = load_data_by_period(project_root, '2006_2020')
            results_2006_2020 = run_fzt_backtest(
                df_2006_2020,
                use_fzt=args.fzt,
                use_zsqsx=args.zsqsx,
                verify_fzt_only=args.verify_fzt,
                top_n=args.top_n,
                top_k=args.top_k,
                # 成交量比因子参数
                volume_ratio_enable=args.volume_ratio_enable,
                volume_ratio_threshold=args.volume_ratio_threshold,
                # 乖离率因子参数
                bias_enable=args.bias_enable,
                bias_lower=args.bias_lower,
                bias_upper=args.bias_upper,
                # RSI因子参数
                rsi_enable=args.rsi_enable,
                rsi_period=args.rsi_period,
                rsi_low=args.rsi_low,
                rsi_high=args.rsi_high,
                # OBV因子参数
                obv_enable=args.obv_enable,
                obv_lookback=args.obv_lookback,
                # 短期暴涨过滤因子参数
                max_gain_enable=args.max_gain_enable,
                max_gain_lookback=args.max_gain_lookback,
                max_gain_threshold=args.max_gain_threshold,
                # 组合因子参数（向后兼容）
                volume_bias=args.volume_bias,
                rsi_obv=args.rsi_obv,
                # KDJ J值判断参数
                kdj_j_enable=args.kdj_j_enable,
                kdj_j_threshold=args.kdj_j_threshold
            )
            print_summary(results_2006_2020, "2006-2020年")
            all_results['2006_2020'] = results_2006_2020
        except Exception as e:
            print(f"❌ 2006-2020年回测失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 运行2021-2026年回测
    if args.period in ['2021_2026', 'both']:
        print(f"\n{'='*60}")
        print("📅 2021-2026年回测")
        print(f"{'='*60}")
        
        try:
            df_2021_2026 = load_data_by_period(project_root, '2021_2026')
            results_2021_2026 = run_fzt_backtest(
                df_2021_2026,
                use_fzt=args.fzt,
                use_zsqsx=args.zsqsx,
                verify_fzt_only=args.verify_fzt,
                top_n=args.top_n,
                top_k=args.top_k,
                # 成交量比因子参数
                volume_ratio_enable=args.volume_ratio_enable,
                volume_ratio_threshold=args.volume_ratio_threshold,
                # 乖离率因子参数
                bias_enable=args.bias_enable,
                bias_lower=args.bias_lower,
                bias_upper=args.bias_upper,
                # RSI因子参数
                rsi_enable=args.rsi_enable,
                rsi_period=args.rsi_period,
                rsi_low=args.rsi_low,
                rsi_high=args.rsi_high,
                # OBV因子参数
                obv_enable=args.obv_enable,
                obv_lookback=args.obv_lookback,
                # 短期暴涨过滤因子参数
                max_gain_enable=args.max_gain_enable,
                max_gain_lookback=args.max_gain_lookback,
                max_gain_threshold=args.max_gain_threshold,
                # 组合因子参数（向后兼容）
                volume_bias=args.volume_bias,
                rsi_obv=args.rsi_obv,
                # KDJ J值判断参数
                kdj_j_enable=args.kdj_j_enable,
                kdj_j_threshold=args.kdj_j_threshold
            )
            print_summary(results_2021_2026, "2021-2026年")
            all_results['2021_2026'] = results_2021_2026
        except Exception as e:
            print(f"❌ 2021-2026年回测失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总结果
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("📈 汇总结果")
        print(f"{'='*60}")
        
        # 计算FZT汇总
        fzt_total = sum(r.get('fzt', {}).get('total', 0) for r in all_results.values())
        fzt_success = sum(r.get('fzt', {}).get('success', 0) for r in all_results.values())
        fzt_rate = fzt_success / fzt_total if fzt_total > 0 else 0
        
        print(f"📈 FZT公式汇总:")
        print(f"   总信号数: {fzt_total:,}")
        print(f"   总成功信号数: {fzt_success:,}")
        print(f"   总成功率: {fzt_rate:.2%}")
        
        # 计算ZSQSX汇总
        zsqsx_total = sum(r.get('zsqsx', {}).get('total', 0) for r in all_results.values())
        zsqsx_success = sum(r.get('zsqsx', {}).get('success', 0) for r in all_results.values())
        zsqsx_rate = zsqsx_success / zsqsx_total if zsqsx_total > 0 else 0
        
        if zsqsx_total > 0:
            print(f"\n📈 ZSQSX公式汇总:")
            print(f"   总信号数: {zsqsx_total:,}")
            print(f"   总成功信号数: {zsqsx_success:,}")
            print(f"   总成功率: {zsqsx_rate:.2%}")
        
        # 计算FZT+ZSQSX汇总
        fzt_zsqsx_total = sum(r.get('fzt_zsqsx', {}).get('total', 0) for r in all_results.values())
        fzt_zsqsx_success = sum(r.get('fzt_zsqsx', {}).get('success', 0) for r in all_results.values())
        fzt_zsqsx_rate = fzt_zsqsx_success / fzt_zsqsx_total if fzt_zsqsx_total > 0 else 0
        
        if fzt_zsqsx_total > 0:
            print(f"\n📈 FZT+ZSQSX组合汇总:")
            print(f"   总信号数: {fzt_zsqsx_total:,}")
            print(f"   总成功信号数: {fzt_zsqsx_success:,}")
            print(f"   总成功率: {fzt_zsqsx_rate:.2%}")
    
    print(f"\n{'='*60}")
    print("✅ FZT统一回测完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
