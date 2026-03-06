#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终版2021-2026年FZT公式回测（使用通用模块）

结合用户提供的两个参考文件：
1. 优化参考：全市场向量化一次性计算
2. FZT参考实现：通达信SMA递归算法 + 复权处理

作者: MC
创建日期: 2026-03-06
参考文件：
1. 优化参考---9f1f0f32-521a-423b-8027-090c283c9de7.txt
2. FZT参考实现---5134fc49-465f-427a-898e-9fea5d032908.txt
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# 导入通用模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.fzt_core import calc_brick_pattern_final
from src.data_loader import load_all_stock_data_bin

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===================== 最终版2021-2026年回测 =====================
def final_2021_2026_backtest():
    """最终版2021-2026年回测（向量化 + 通达信SMA）"""
    try:
        print("=" * 80)
        print("🚀 最终版2021-2026年FZT公式回测")
        print("=" * 80)
        
        start_time = time.time()
        
        # 数据目录（使用项目内数据）
        project_root = Path(__file__).parent.parent
        data_dir = str(project_root / 'data' / '2021_2026')
        
        # 设置时间范围
        target_start = '2021-08-02'
        target_end = '2026-02-06'
        
        print(f"🎯 回测时间范围: {target_start} 到 {target_end}")
        
        # 1. 一次性加载所有股票数据（使用数据加载模块）
        print("\n📥 一次性加载所有股票数据...")
        data_load_start = time.time()
        df, stock_codes = load_all_stock_data_bin(data_dir)
        data_load_time = time.time() - data_load_start
        
        if df is None or df.empty:
            print("❌ 数据加载失败，无法继续")
            return 1
        
        print(f"   数据加载耗时: {data_load_time:.2f} 秒")
        
        # 2. 计算FZT指标（使用通用FZT核心模块）
        print("\n🧮 计算FZT指标（向量化计算）...")
        calc_start_time = time.time()
        
        df = calc_brick_pattern_final(
            df_raw=df,
            target_start_date=target_start,
            target_end_date=target_end
        )
        
        calc_time = time.time() - calc_start_time
        print(f"✅ FZT计算完成，耗时: {calc_time:.2f} 秒")
        
        # 3. 筛选FZT信号
        print("\n🎯 筛选FZT信号...")
        signals = df[df['选股条件'] == True].copy()
        
        if signals.empty:
            print("❌ 没有找到任何FZT信号")
            return 1
        
        # 4. 计算次日收益率（成功条件）
        print("📈 计算成功率...")
        signals['next_close'] = signals.groupby('instrument')['close'].transform(lambda x: x.shift(-1))
        signals['success'] = signals['next_close'] > signals['close']
        
        # 5. 统计结果
        total_signals = len(signals)
        successful_signals = signals['success'].sum()
        success_rate = successful_signals / total_signals if total_signals > 0 else 0
        
        print(f"\n📊 回测结果统计:")
        print(f"   总FZT信号数: {total_signals:,}")
        print(f"   成功信号数: {successful_signals:,}")
        print(f"   成功率: {success_rate:.2%}")
        
        # 6. 年度分析
        print("\n📅 年度成功率分析:")
        signals['year'] = signals['datetime'].dt.year
        yearly_stats = signals.groupby('year').agg({
            'success': ['count', 'sum']
        }).round(2)
        
        yearly_stats.columns = ['total_signals', 'successful_signals']
        yearly_stats['success_rate'] = yearly_stats['successful_signals'] / yearly_stats['total_signals']
        
        for year, row in yearly_stats.iterrows():
            print(f"   {year}: {row['total_signals']:,} 信号, {row['success_rate']:.2%} 成功率")
        
        # 7. 股票数量统计
        print("\n📈 股票数量统计:")
        yearly_stock_counts = signals.groupby('year')['instrument'].nunique()
        for year, count in yearly_stock_counts.items():
            print(f"   {year}: {count} 只股票产生信号")
        
        # 8. 保存结果
        print("\n💾 保存结果...")
        results_dir = project_root / 'results' / 'fzt_final_2021_2026'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细报告
        report_path = results_dir / 'full_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("FZT公式回测报告 (2021-2026年)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"回测时间范围: {target_start} 到 {target_end}\n")
            f.write(f"股票数量: {len(stock_codes)}\n")
            f.write(f"总FZT信号数: {total_signals:,}\n")
            f.write(f"成功信号数: {successful_signals:,}\n")
            f.write(f"整体成功率: {success_rate:.2%}\n\n")
            
            f.write("年度成功率:\n")
            for year, row in yearly_stats.iterrows():
                f.write(f"  {year}: {row['total_signals']:,} 信号, {row['success_rate']:.2%} 成功率\n")
            
            f.write("\n年度股票数量:\n")
            for year, count in yearly_stock_counts.items():
                f.write(f"  {year}: {count} 只股票\n")
        
        # 保存年度统计数据
        yearly_stats.to_csv(results_dir / 'yearly_stats.csv')
        
        total_time = time.time() - start_time
        print(f"\n✅ 回测完成！总耗时: {total_time:.2f} 秒")
        print(f"   报告已保存至: {report_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 回测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # 运行回测
    exit_code = final_2021_2026_backtest()
    sys.exit(exit_code)