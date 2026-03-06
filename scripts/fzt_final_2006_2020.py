#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终版2006-2020年FZT公式回测（使用通用模块）

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
from src.data_loader import load_qlib_data_all_instruments

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===================== 最终版2006-2020年回测 =====================
def final_2006_2020_backtest():
    """最终版2006-2020年回测（向量化 + 通达信SMA + 复权）"""
    try:
        print("=" * 80)
        print("🚀 最终版2006-2020年FZT公式回测")
        print("=" * 80)
        
        start_time = time.time()
        
        # 导入QLIB
        import qlib
        from qlib.data import D
        from qlib.config import REG_CN
        
        # 初始化QLIB（使用项目内数据）
        project_root = Path(__file__).parent.parent
        data_path = str(project_root / 'data' / '2006_2020')
        qlib.init(provider_uri=data_path, region=REG_CN)
        print(f"✅ QLIB初始化成功，数据路径: {data_path}")
        
        # 设置时间范围
        calc_start = '2005-10-01'
        calc_end = '2020-09-25'
        target_start = '2006-01-01'
        target_end = '2020-09-25'
        
        print(f"📅 计算时间范围: {calc_start} 到 {calc_end}")
        print(f"🎯 回测时间范围: {target_start} 到 {target_end}")
        
        # 1. 获取全市场股票列表（基于优化参考：一次性获取）
        print("📊 获取全市场股票列表...")
        load_start = time.time()
        
        instruments_file = project_root / 'data' / '2006_2020' / 'instruments' / 'all.txt'
        instruments = []
        
        with open(instruments_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 1:
                        instruments.append(parts[0])
        
        instruments = instruments[:3875]  # 使用所有3875只股票
        print(f"✅ 获取到 {len(instruments)} 只股票")
        print(f"   股票示例: {instruments[:5]}...")
        
        # 2. 一次性加载所有股票数据（使用数据加载模块）
        print("\n📥 一次性加载所有股票数据...")
        df = load_qlib_data_all_instruments(
            instruments=instruments,
            calc_start=calc_start,
            calc_end=calc_end
        )
        
        if df is None or df.empty:
            print("❌ 数据加载失败，无法继续")
            return 1
        
        # 3. 计算FZT指标（使用通用FZT核心模块）
        print("\n🧮 计算FZT指标（向量化计算）...")
        calc_start_time = time.time()
        
        df = calc_brick_pattern_final(
            df_raw=df,
            target_start_date=target_start,
            target_end_date=target_end
        )
        
        calc_time = time.time() - calc_start_time
        print(f"✅ FZT计算完成，耗时: {calc_time:.2f} 秒")
        
        # 4. 筛选FZT信号
        print("\n🎯 筛选FZT信号...")
        signals = df[df['选股条件'] == True].copy()
        
        if signals.empty:
            print("❌ 没有找到任何FZT信号")
            return 1
        
        # 5. 计算次日收益率（成功条件）
        print("📈 计算成功率...")
        signals['next_close'] = signals.groupby('instrument')['close'].transform(lambda x: x.shift(-1))
        signals['success'] = signals['next_close'] > signals['close']
        
        # 6. 统计结果
        total_signals = len(signals)
        successful_signals = signals['success'].sum()
        success_rate = successful_signals / total_signals if total_signals > 0 else 0
        
        print(f"\n📊 回测结果统计:")
        print(f"   总FZT信号数: {total_signals:,}")
        print(f"   成功信号数: {successful_signals:,}")
        print(f"   成功率: {success_rate:.2%}")
        
        # 7. 年度分析
        print("\n📅 年度成功率分析:")
        signals['year'] = signals['datetime'].dt.year
        yearly_stats = signals.groupby('year').agg({
            'success': ['count', 'sum']
        }).round(2)
        
        yearly_stats.columns = ['total_signals', 'successful_signals']
        yearly_stats['success_rate'] = yearly_stats['successful_signals'] / yearly_stats['total_signals']
        
        for year, row in yearly_stats.iterrows():
            print(f"   {year}: {row['total_signals']:,} 信号, {row['success_rate']:.2%} 成功率")
        
        # 8. 保存结果
        print("\n💾 保存结果...")
        results_dir = project_root / 'results' / 'fzt_final_2006_2020'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细报告
        report_path = results_dir / 'full_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("FZT公式回测报告 (2006-2020年)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"回测时间范围: {target_start} 到 {target_end}\n")
            f.write(f"股票数量: {len(instruments)}\n")
            f.write(f"总FZT信号数: {total_signals:,}\n")
            f.write(f"成功信号数: {successful_signals:,}\n")
            f.write(f"整体成功率: {success_rate:.2%}\n\n")
            
            f.write("年度成功率:\n")
            for year, row in yearly_stats.iterrows():
                f.write(f"  {year}: {row['total_signals']:,} 信号, {row['success_rate']:.2%} 成功率\n")
        
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
    exit_code = final_2006_2020_backtest()
    sys.exit(exit_code)