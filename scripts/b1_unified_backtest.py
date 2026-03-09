#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1因子统一回测脚本

包含纯B1因子和B1+新增因子两种模式，通过参数开关控制。

B1因子条件：
1. 当天KDJ的J小于阈值（默认: 13）
2. 当天收盘价大于DKS
3. 当天QSX大于DKS

新增可配置因子（可选）：
1. 前N个交易日的最大涨幅不超过 M（默认: 前20日最大涨幅≤45%）
2. 前Y个交易日中上涨日的累计换手率小于 X（默认: 前20日上涨日累计换手率<30%）

成功条件：
T+10日内的最高收盘价 > T日收盘价

作者: MC
创建日期: 2026-03-09
更新日期: 2026-03-09 (统一版本)
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.b2_core import compute_kdj
from src.zsqsx_core import calc_zsdkx
from src.factors import add_custom_factors
from src.data_loader import load_stock_data_qlib, get_instruments_from_file


def load_data(project_root: Path, period: str) -> pd.DataFrame:
    """加载数据"""
    if period == '2006_2020':
        data_dir = str(project_root / 'data' / '2006_2020')
        calc_start = '2006-01-01'
        calc_end = '2020-12-31'
    else:  # '2021_2026'
        data_dir = str(project_root / 'data' / '2021_2026')
        calc_start = '2021-08-02'
        calc_end = '2026-02-06'
    
    instruments_file = data_dir + '/instruments/all.txt'
    instruments = get_instruments_from_file(instruments_file)
    
    print(f"📊 加载数据 ({period}):")
    print(f"   数据目录: {data_dir}")
    print(f"   股票数量: {len(instruments):,}")
    print(f"   时间范围: {calc_start} 到 {calc_end}")
    
    load_start_time = time.time()
    df = load_stock_data_qlib(
        data_dir=data_dir,
        instruments=instruments,
        calc_start=calc_start,
        calc_end=calc_end,
        fields=['$open', '$high', '$low', '$close', '$volume', '$turnover_rate']
    )
    
    # 重命名字段以匹配因子函数
    if '$turnover_rate' in df.columns:
        df = df.rename(columns={'$turnover_rate': 'turnover'})
    
    print(f"✅ 数据加载完成，耗时: {time.time() - load_start_time:.2f} 秒")
    print(f"   总数据行数: {len(df):,}")
    print(f"   时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    
    return df


def calculate_b1_success_rate(
    df: pd.DataFrame,
    lookforward_days: int = 10,
    kdj_j_threshold: float = 13,
    M1: int = 14,
    M2: int = 28,
    M3: int = 57,
    M4: int = 114,
    # 新增因子参数
    enable_custom_factors: bool = False,
    N: int = 20,
    M: float = 0.45,
    Y: int = 20,
    X: float = 0.30
) -> Dict[str, Any]:
    """
    计算B1因子成功率，可选是否启用新增因子
    
    :param df: 包含股票数据的DataFrame
    :param lookforward_days: 向前看的天数
    :param kdj_j_threshold: KDJ J值阈值
    :param M1-M4: ZSQSX参数
    :param enable_custom_factors: 是否启用新增因子
    :param N: 最大涨幅回溯窗口
    :param M: 最大涨幅阈值
    :param Y: 累计换手率回溯窗口
    :param X: 累计换手率阈值
    :return: 回测结果字典
    """
    start_time = time.time()
    
    # 准备数据
    df = df.copy()
    df = df.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    
    # 计算成功条件
    grouped = df.groupby('instrument')
    df['future_max_close'] = grouped['close'].transform(
        lambda x: x.shift(-lookforward_days).rolling(window=lookforward_days, min_periods=1).max()
    )
    df['success'] = df['future_max_close'] > df['close']
    
    # 计算KDJ指标
    df_kdj = compute_kdj(df)
    
    # 计算ZSQSX指标
    df_zsdkx = calc_zsdkx(df, M1=M1, M2=M2, M3=M3, M4=M4)
    
    # 合并数据
    df_combined = pd.merge(
        df_kdj[['instrument', 'datetime', 'kdj_j']],
        df_zsdkx[['instrument', 'datetime', 'QSX', 'DKS']],
        on=['instrument', 'datetime'],
        how='inner'
    )
    
    # 添加收盘价和换手率
    df_combined = pd.merge(
        df_combined,
        df[['instrument', 'datetime', 'close', 'turnover', 'success']],
        on=['instrument', 'datetime'],
        how='inner'
    )
    
    # 计算B1因子条件
    df_combined['condition1'] = df_combined['kdj_j'] < kdj_j_threshold
    df_combined['condition2'] = df_combined['close'] > df_combined['DKS']
    df_combined['condition3'] = df_combined['QSX'] > df_combined['DKS']
    df_combined['B1'] = df_combined['condition1'] & df_combined['condition2'] & df_combined['condition3']
    
    # 筛选B1信号
    b1_signals = df_combined[df_combined['B1'] == True]
    
    # 计算纯B1因子成功率
    b1_total = len(b1_signals)
    b1_success = b1_signals['success'].sum() if b1_total > 0 else 0
    b1_rate = b1_success / b1_total if b1_total > 0 else 0
    
    # 如果启用新增因子
    if enable_custom_factors and b1_total > 0:
        b1_signals_with_factors = add_custom_factors(b1_signals, N=N, M=M, Y=Y, X=X)
        
        # 筛选同时满足B1和自定义因子的信号
        combined_signals = b1_signals_with_factors[
            b1_signals_with_factors['cond_gain'] & b1_signals_with_factors['cond_turnover']
        ]
        
        combined_total = len(combined_signals)
        combined_success = combined_signals['success'].sum() if combined_total > 0 else 0
        combined_rate = combined_success / combined_total if combined_total > 0 else 0
    else:
        combined_total = 0
        combined_success = 0
        combined_rate = 0
    
    # 准备结果
    results = {
        'b1_total': b1_total,
        'b1_success': b1_success,
        'b1_rate': b1_rate,
        'enable_custom_factors': enable_custom_factors,
        'combined_total': combined_total,
        'combined_success': combined_success,
        'combined_rate': combined_rate,
        'filter_ratio': combined_total / b1_total if b1_total > 0 else 0,
        'elapsed_time': time.time() - start_time,
        'parameters': {
            'lookforward_days': lookforward_days,
            'kdj_j_threshold': kdj_j_threshold,
            'M1': M1, 'M2': M2, 'M3': M3, 'M4': M4,
            'enable_custom_factors': enable_custom_factors,
            'N': N, 'M': M, 'Y': Y, 'X': X
        }
    }
    
    return results


def print_results(results: Dict[str, Any], period_name: str):
    """打印结果"""
    print(f"\n📊 B1因子回测结果 ({period_name}):")
    print(f"   B1因子总信号数: {results['b1_total']:,}")
    print(f"   B1因子成功信号数: {results['b1_success']:,}")
    print(f"   B1因子成功率: {results['b1_rate']:.2%}")
    
    if results['enable_custom_factors']:
        print(f"   B1+新因子总信号数: {results['combined_total']:,}")
        print(f"   B1+新因子成功信号数: {results['combined_success']:,}")
        print(f"   B1+新因子成功率: {results['combined_rate']:.2%}")
        print(f"   过滤比例: {results['filter_ratio']:.2%}")
    
    print(f"   执行时间: {results['elapsed_time']:.2f} 秒")
    
    # 参数配置
    params = results['parameters']
    print(f"\n⚙️  参数配置:")
    print(f"   成功条件: T+{params['lookforward_days']}日内的最高收盘价 > T日收盘价")
    print(f"   KDJ J值阈值: {params['kdj_j_threshold']}")
    print(f"   ZSQSX参数: M1={params['M1']}, M2={params['M2']}, M3={params['M3']}, M4={params['M4']}")
    
    if params['enable_custom_factors']:
        print(f"   新增因子: 启用")
        print(f"     最大涨幅: 前{params['N']}日 ≤ {params['M']:.2f}")
        print(f"     累计换手率: 前{params['Y']}日上涨日 < {params['X']:.2f}")
    else:
        print(f"   新增因子: 禁用")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='B1因子统一回测脚本 - 支持纯B1和B1+新增因子两种模式')
    
    # 时期选择
    parser.add_argument('--period', type=str, default='both',
                       choices=['2006_2020', '2021_2026', 'both'],
                       help='回测时期 (默认: both)')
    
    # 股票数量控制（快速测试用）
    parser.add_argument('--max-stocks', type=int, default=0,
                       help='最大股票数量，0表示全部 (默认: 0，全部股票)')
    
    # B1因子参数
    parser.add_argument('--kdj-j-threshold', type=float, default=13,
                       help='KDJ J值阈值 (默认: 13)')
    
    # 成功条件参数
    parser.add_argument('--lookforward-days', type=int, default=10,
                       help='向前看的天数 (默认: 10)')
    
    # ZSQSX参数
    parser.add_argument('--M1', type=int, default=14,
                       help='ZSQSX公式参数M1 (默认: 14)')
    parser.add_argument('--M2', type=int, default=28,
                       help='ZSQSX公式参数M2 (默认: 28)')
    parser.add_argument('--M3', type=int, default=57,
                       help='ZSQSX公式参数M3 (默认: 57)')
    parser.add_argument('--M4', type=int, default=114,
                       help='ZSQSX公式参数M4 (默认: 114)')
    
    # 新增因子开关
    parser.add_argument('--enable-custom-factors', action='store_true', default=False,
                       help='启用新增可配置因子 (默认: False)')
    
    # 新增因子参数
    parser.add_argument('--N', type=int, default=20,
                       help='最大涨幅回溯窗口 (默认: 20)')
    parser.add_argument('--M', type=float, default=0.45,
                       help='最大涨幅阈值 (默认: 0.45)')
    parser.add_argument('--Y', type=int, default=20,
                       help='累计换手率回溯窗口 (默认: 20)')
    parser.add_argument('--X', type=float, default=0.30,
                       help='累计换手率阈值 (默认: 0.30)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 B1因子统一回测脚本")
    print("=" * 60)
    print("B1因子条件:")
    print("1. 当天KDJ的J小于阈值")
    print("2. 当天收盘价大于DKS")
    print("3. 当天QSX大于DKS")
    
    if args.enable_custom_factors:
        print("\n新增因子条件 (已启用):")
        print(f"1. 前{args.N}个交易日的最大涨幅不超过 {args.M:.2f}")
        print(f"2. 前{args.Y}个交易日中上涨日的累计换手率小于 {args.X:.2f}")
    else:
        print("\n新增因子条件: 禁用 (使用纯B1因子)")
    
    print(f"\n成功条件: T+{args.lookforward_days}日内的最高收盘价 > T日收盘价")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    all_results = {}
    
    # 运行2006-2020年回测
    if args.period in ['2006_2020', 'both']:
        print(f"\n{'='*60}")
        print("📅 2006-2020年回测")
        print(f"{'='*60}")
        
        try:
            df_2006_2020 = load_data(project_root, '2006_2020')
            
            # 如果限制股票数量
            if args.max_stocks > 0:
                instruments = df_2006_2020['instrument'].unique()[:args.max_stocks]
                df_2006_2020 = df_2006_2020[df_2006_2020['instrument'].isin(instruments)]
                print(f"   限制股票数量: {args.max_stocks}")
            
            results_2006_2020 = calculate_b1_success_rate(
                df_2006_2020,
                lookforward_days=args.lookforward_days,
                kdj_j_threshold=args.kdj_j_threshold,
                M1=args.M1, M2=args.M2, M3=args.M3, M4=args.M4,
                enable_custom_factors=args.enable_custom_factors,
                N=args.N, M=args.M, Y=args.Y, X=args.X
            )
            print_results(results_2006_2020, "2006-2020年")
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
            df_2021_2026 = load_data(project_root, '2021_2026')
            
            # 如果限制股票数量
            if args.max_stocks > 0:
                instruments = df_2021_2026['instrument'].unique()[:args.max_stocks]
                df_2021_2026 = df_2021_2026[df_2021_2026['instrument'].isin(instruments)]
                print(f"   限制股票数量: {args.max_stocks}")
            
            results_2021_2026 = calculate_b1_success_rate(
                df_2021_2026,
                lookforward_days=args.lookforward_days,
                kdj_j_threshold=args.kdj_j_threshold,
                M1=args.M1, M2=args.M2, M3=args.M3, M4=args.M4,
                enable_custom_factors=args.enable_custom_factors,
                N=args.N, M=args.M, Y=args.Y, X=args.X
            )
            print_results(results_2021_2026, "2021-2026年")
            all_results['2021_2026'] = results_2021_2026
        except Exception as e:
            print(f"❌ 2021-2026年回测失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总结果
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("📈 汇总结果 (2006-2026)")
        print(f"{'='*60}")
        
        # 计算汇总
        b1_total = sum(r['b1_total'] for r in all_results.values())
        b1_success = sum(r['b1_success'] for r in all_results.values())
        b1_rate = b1_success / b1_total if b1_total > 0 else 0
        
        if args.enable_custom_factors:
            combined_total = sum(r['combined_total'] for r in all_results.values())
            combined_success = sum(r['combined_success'] for r in all_results.values())
            combined_rate = combined_success / combined_total if combined_total > 0 else 0
        
        print(f"📊 汇总结果:")
        print(f"   B1因子总信号数: {b1_total:,}")
        print(f"   B1因子成功信号数: {b1_success:,}")
        print(f"   B1因子成功率: {b1_rate:.2%}")
        
        if args.enable_custom_factors:
            print(f"   B1+新因子总信号数: {combined_total:,}")
            print(f"   B1+新因子成功信号数: {combined_success:,}")
            print(f"   B1+新因子成功率: {combined_rate:.2%}")
            print(f"   过滤比例: {combined_total/b1_total:.2%} (保留 {combined_total/b1_total*100:.1f}%)")
        
        # 各时期结果
        print(f"\n📅 各时期结果:")
        for period_name, results in all_results.items():
            print(f"   {period_name}:")
            print(f"     B1成功率: {results['b1_rate']:.2%} ({results['b1_success']:,}/{results['b1_total']:,})")
            if args.enable_custom_factors:
                print(f"     B1+新因子成功率: {results['combined_rate']:.2%} ({results['combined_success']:,}/{results['combined_total']:,})")
                print(f"     过滤比例: {results['filter_ratio']:.2%}")
    
    print(f"\n{'='*60}")
    print("✅ 回测完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()