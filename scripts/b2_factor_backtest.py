#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B2因子独立回测脚本

B2因子条件：
1. 前一日的KDJ的J值 < 13
2. 当日的KDJ的J值 < 55
3. 当日的涨幅 > 4%
4. 当日的成交量 > 前一日成交量
5. 当日上影线很小

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
from src.b2_core import add_B2_factor, filter_by_B2_factor, calculate_b2_success_rate
from src.data_loader import load_stock_data_qlib, get_instruments_from_file


def load_2006_2020_data(project_root: Path, max_stocks: int = 0) -> pd.DataFrame:
    """
    加载2006-2020年数据（私有函数）
    
    :param project_root: 项目根目录
    :param max_stocks: 最大股票数量，0表示全部 (默认: 0)
    :return: 包含股票数据的DataFrame
    """
    # 数据目录
    data_dir = str(project_root / 'data' / '2006_2020')
    
    # 读取股票列表
    instruments_file = data_dir + '/instruments/all.txt'
    instruments = get_instruments_from_file(instruments_file)
    
    # 限制股票数量（如果指定）
    if max_stocks > 0:
        instruments = instruments[:max_stocks]
        print(f"   限制股票数量: {max_stocks}")
    
    # 时间范围
    calc_start = '2006-01-01'
    calc_end = '2020-12-31'
    
    print(f"📊 加载数据:")
    print(f"   数据目录: {data_dir}")
    print(f"   股票数量: {len(instruments):,}")
    print(f"   时间范围: {calc_start} 到 {calc_end}")
    
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


def load_2021_2026_data(project_root: Path, max_stocks: int = 0) -> pd.DataFrame:
    """
    加载2021-2026年数据（私有函数）
    
    :param project_root: 项目根目录
    :param max_stocks: 最大股票数量，0表示全部 (默认: 0)
    :return: 包含股票数据的DataFrame
    """
    # 数据目录
    data_dir = str(project_root / 'data' / '2021_2026')
    
    # 读取股票列表
    instruments_file = data_dir + '/instruments/all.txt'
    instruments = get_instruments_from_file(instruments_file)
    
    # 限制股票数量（如果指定）
    if max_stocks > 0:
        instruments = instruments[:max_stocks]
        print(f"   限制股票数量: {max_stocks}")
    
    # 时间范围
    calc_start = '2021-08-02'
    calc_end = '2026-02-06'
    
    print(f"📊 加载数据:")
    print(f"   数据目录: {data_dir}")
    print(f"   股票数量: {len(instruments):,}")
    print(f"   时间范围: {calc_start} 到 {calc_end}")
    
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


def run_b2_backtest(
    df: pd.DataFrame,
    j_prev_thresh: float = 13,
    j_today_thresh: float = 55,
    gain_thresh: float = 0.04,
    upper_shadow_pct: float = 0.005,
    upper_shadow_vs_body: float = 0.5
) -> Dict[str, Any]:
    """
    运行B2因子回测
    
    :param df: 包含股票数据的DataFrame
    :param j_prev_thresh: 前一日J值上限 (默认: 13)
    :param j_today_thresh: 当日J值上限 (默认: 55)
    :param gain_thresh: 涨幅阈值，如0.04表示4% (默认: 0.04)
    :param upper_shadow_pct: 上影线占收盘价比例上限 (默认: 0.005)
    :param upper_shadow_vs_body: 上影线相对于实体长度的比例上限 (默认: 0.5)
    :return: 回测结果字典
    """
    start_time = time.time()
    
    print(f"\n🧮 计算B2因子...")
    print(f"   参数配置:")
    print(f"     前一日J值上限: {j_prev_thresh}")
    print(f"     当日J值上限: {j_today_thresh}")
    print(f"     涨幅阈值: {gain_thresh:.1%}")
    print(f"     上影线占收盘价比例上限: {upper_shadow_pct:.3%}")
    print(f"     上影线相对于实体长度比例上限: {upper_shadow_vs_body:.1%}")
    
    # 使用b2_core模块计算成功率
    results = calculate_b2_success_rate(
        df,
        j_prev_thresh=j_prev_thresh,
        j_today_thresh=j_today_thresh,
        gain_thresh=gain_thresh,
        upper_shadow_pct=upper_shadow_pct,
        upper_shadow_vs_body=upper_shadow_vs_body
    )
    
    # 添加执行时间
    results['elapsed_time'] = time.time() - start_time
    
    return results


def print_results(results: Dict[str, Any], period_name: str):
    """
    打印回测结果
    
    :param results: 回测结果字典
    :param period_name: 时期名称
    """
    print(f"\n📊 B2因子回测结果 ({period_name}):")
    print(f"   总信号数: {results['total_signals']:,}")
    print(f"   成功信号数: {results['success_signals']:,}")
    print(f"   成功率: {results['success_rate']:.2%}")
    print(f"   执行时间: {results['elapsed_time']:.2f} 秒")
    
    if results['yearly_stats']:
        print(f"\n📅 年度成功率分析:")
        for stat in results['yearly_stats']:
            print(f"   {stat['year']}: {stat['rate']:.2%} ({stat['total']:,} 个信号)")
    
    print(f"\n⚙️  参数配置:")
    params = results['parameters']
    print(f"   前一日J值上限: {params['j_prev_thresh']}")
    print(f"   当日J值上限: {params['j_today_thresh']}")
    print(f"   涨幅阈值: {params['gain_thresh']:.1%}")
    print(f"   上影线占收盘价比例上限: {params['upper_shadow_pct']:.3%}")
    print(f"   上影线相对于实体长度比例上限: {params['upper_shadow_vs_body']:.1%}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='B2因子独立回测脚本')
    parser.add_argument('--period', type=str, default='2006_2020',
                       choices=['2006_2020', '2021_2026', 'both'],
                       help='回测时期 (默认: 2006_2020)')
    
    # 股票数量控制
    parser.add_argument('--max-stocks', type=int, default=0,
                       help='最大股票数量，0表示全部 (默认: 0，全部股票)')
    
    # B2因子参数
    parser.add_argument('--j-prev-thresh', type=float, default=13,
                       help='前一日J值上限 (默认: 13)')
    parser.add_argument('--j-today-thresh', type=float, default=55,
                       help='当日J值上限 (默认: 55)')
    parser.add_argument('--gain-thresh', type=float, default=0.04,
                       help='涨幅阈值，如0.04表示4%% (默认: 0.04)')
    parser.add_argument('--upper-shadow-pct', type=float, default=0.005,
                       help='上影线占收盘价比例上限 (默认: 0.005)')
    parser.add_argument('--upper-shadow-vs-body', type=float, default=0.5,
                       help='上影线相对于实体长度的比例上限 (默认: 0.5)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 B2因子独立回测脚本")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    
    all_results = {}
    
    # 运行2006-2020年回测
    if args.period in ['2006_2020', 'both']:
        print(f"\n{'='*60}")
        print("📅 2006-2020年回测")
        print(f"{'='*60}")
        
        try:
            df_2006_2020 = load_2006_2020_data(project_root, max_stocks=args.max_stocks)
            results_2006_2020 = run_b2_backtest(
                df_2006_2020,
                j_prev_thresh=args.j_prev_thresh,
                j_today_thresh=args.j_today_thresh,
                gain_thresh=args.gain_thresh,
                upper_shadow_pct=args.upper_shadow_pct,
                upper_shadow_vs_body=args.upper_shadow_vs_body
            )
            print_results(results_2006_2020, "2006-2020年")
            all_results['2006_2020'] = results_2006_2020
        except Exception as e:
            print(f"❌ 2006-2020年回测失败: {e}")
    
    # 运行2021-2026年回测
    if args.period in ['2021_2026', 'both']:
        print(f"\n{'='*60}")
        print("📅 2021-2026年回测")
        print(f"{'='*60}")
        
        try:
            df_2021_2026 = load_2021_2026_data(project_root, max_stocks=args.max_stocks)
            results_2021_2026 = run_b2_backtest(
                df_2021_2026,
                j_prev_thresh=args.j_prev_thresh,
                j_today_thresh=args.j_today_thresh,
                gain_thresh=args.gain_thresh,
                upper_shadow_pct=args.upper_shadow_pct,
                upper_shadow_vs_body=args.upper_shadow_vs_body
            )
            print_results(results_2021_2026, "2021-2026年")
            all_results['2021_2026'] = results_2021_2026
        except Exception as e:
            print(f"❌ 2021-2026年回测失败: {e}")
    
    # 汇总结果
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("📈 汇总结果")
        print(f"{'='*60}")
        
        total_signals = sum(r['total_signals'] for r in all_results.values())
        total_success = sum(r['success_signals'] for r in all_results.values())
        total_rate = total_success / total_signals if total_signals > 0 else 0
        
        print(f"   总信号数: {total_signals:,}")
        print(f"   总成功信号数: {total_success:,}")
        print(f"   总成功率: {total_rate:.2%}")
    
    print(f"\n{'='*60}")
    print("✅ B2因子回测完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()