#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FZT + ZSQSX 双公式回测脚本 (2021-2026年)

同时计算FZT和ZSQSX公式，分析：
1. 单独FZT公式的成功率
2. 单独ZSQSX公式的成功率  
3. 同时满足FZT和ZSQSX的标的成功率

作者: MC
创建日期: 2026-03-07
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
from typing import Optional
warnings.filterwarnings('ignore')

# 导入核心模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.fzt_core import calc_brick_pattern_final
from src.zsqsx_core import calc_zsdkx, get_zsdkx_signal_conditions
from src.data_loader import load_stock_data_qlib, get_instruments_from_file

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===================== 私有数据加载函数 =====================
def load_2021_2026_data(project_root: Path) -> Optional[pd.DataFrame]:
    """
    加载2021-2026年数据（私有函数）
    """
    data_dir = str(project_root / 'data' / '2021_2026')
    instruments_file = project_root / 'data' / '2021_2026' / 'instruments' / 'all.txt'
    
    instruments = get_instruments_from_file(instruments_file)
    if not instruments:
        return None
    
    # 使用所有5484只股票
    instruments = instruments[:5484]
    
    return load_stock_data_qlib(
        data_dir=data_dir,
        instruments=instruments,
        calc_start='2021-08-02',
        calc_end='2026-02-06',
        target_start='2021-08-02',
        target_end='2026-02-06'
    )


# ===================== 双公式回测 =====================
def fzt_zsqsx_2021_2026_backtest():
    """FZT + ZSQSX 双公式回测 (2021-2026年)"""
    try:
        print("=" * 80)
        print("🚀 FZT + ZSQSX 双公式回测 (2021-2026年)")
        print("=" * 80)
        
        start_time = time.time()
        project_root = Path(__file__).parent.parent
        
        # 1. 加载数据
        print("\n📥 加载2021-2026年数据...")
        df = load_2021_2026_data(project_root)
        
        if df is None or df.empty:
            print("❌ 数据加载失败，无法继续")
            return 1
        
        # 2. 计算FZT指标
        print("\n🧮 计算FZT指标...")
        fzt_start_time = time.time()
        df_fzt = calc_brick_pattern_final(df)
        fzt_time = time.time() - fzt_start_time
        print(f"✅ FZT计算完成，耗时: {fzt_time:.2f} 秒")
        
        # 3. 计算ZSQSX指标
        print("\n🧮 计算ZSQSX指标...")
        zsqsx_start_time = time.time()
        df_zsdkx = calc_zsdkx(df)
        df_zsqsx = get_zsdkx_signal_conditions(df_zsdkx)
        zsqsx_time = time.time() - zsqsx_start_time
        print(f"✅ ZSQSX计算完成，耗时: {zsqsx_time:.2f} 秒")
        
        # 4. 合并两个公式的结果
        print("\n🔗 合并FZT和ZSQSX结果...")
        # 确保两个DataFrame有相同的索引结构
        df_fzt = df_fzt.set_index(['instrument', 'datetime']).sort_index()
        df_zsqsx = df_zsqsx.set_index(['instrument', 'datetime']).sort_index()
        
        # 合并两个公式的信号
        df_combined = pd.concat([
            df_fzt[['close', '选股条件']].rename(columns={'选股条件': 'FZT_signal'}),
            df_zsqsx[['ZSQSX_signal']]
        ], axis=1).reset_index()
        
        # 5. 计算次日收益率（成功条件）
        print("📈 计算成功率...")
        df_combined['next_close'] = df_combined.groupby('instrument')['close'].transform(lambda x: x.shift(-1))
        df_combined['success'] = df_combined['next_close'] > df_combined['close']
        
        # 6. 统计结果
        print("\n📊 回测结果统计:")
        
        # 6.1 单独FZT公式
        fzt_signals = df_combined[df_combined['FZT_signal'] == True]
        fzt_total = len(fzt_signals)
        fzt_success = fzt_signals['success'].sum() if fzt_total > 0 else 0
        fzt_rate = fzt_success / fzt_total if fzt_total > 0 else 0
        
        # 6.2 单独ZSQSX公式
        zsqsx_signals = df_combined[df_combined['ZSQSX_signal'] == True]
        zsqsx_total = len(zsqsx_signals)
        zsqsx_success = zsqsx_signals['success'].sum() if zsqsx_total > 0 else 0
        zsqsx_rate = zsqsx_success / zsqsx_total if zsqsx_total > 0 else 0
        
        # 6.3 同时满足FZT和ZSQSX
        combined_signals = df_combined[
            (df_combined['FZT_signal'] == True) & 
            (df_combined['ZSQSX_signal'] == True)
        ]
        combined_total = len(combined_signals)
        combined_success = combined_signals['success'].sum() if combined_total > 0 else 0
        combined_rate = combined_success / combined_total if combined_total > 0 else 0
        
        print(f"\n📈 单独FZT公式:")
        print(f"   总信号数: {fzt_total:,}")
        print(f"   成功信号数: {fzt_success:,}")
        print(f"   成功率: {fzt_rate:.2%}")
        
        print(f"\n📈 单独ZSQSX公式:")
        print(f"   总信号数: {zsqsx_total:,}")
        print(f"   成功信号数: {zsqsx_success:,}")
        print(f"   成功率: {zsqsx_rate:.2%}")
        
        print(f"\n🎯 同时满足FZT和ZSQSX:")
        print(f"   总信号数: {combined_total:,}")
        print(f"   成功信号数: {combined_success:,}")
        print(f"   成功率: {combined_rate:.2%}")
        
        # 7. 年度分析
        print("\n📅 年度成功率分析:")
        df_combined['year'] = df_combined['datetime'].dt.year
        
        yearly_stats = []
        for year in sorted(df_combined['year'].unique()):
            year_data = df_combined[df_combined['year'] == year]
            
            # 单独FZT
            fzt_year = year_data[year_data['FZT_signal'] == True]
            fzt_year_total = len(fzt_year)
            fzt_year_success = fzt_year['success'].sum() if fzt_year_total > 0 else 0
            fzt_year_rate = fzt_year_success / fzt_year_total if fzt_year_total > 0 else 0
            
            # 单独ZSQSX
            zsqsx_year = year_data[year_data['ZSQSX_signal'] == True]
            zsqsx_year_total = len(zsqsx_year)
            zsqsx_year_success = zsqsx_year['success'].sum() if zsqsx_year_total > 0 else 0
            zsqsx_year_rate = zsqsx_year_success / zsqsx_year_total if zsqsx_year_total > 0 else 0
            
            # 同时满足
            combined_year = year_data[
                (year_data['FZT_signal'] == True) & 
                (year_data['ZSQSX_signal'] == True)
            ]
            combined_year_total = len(combined_year)
            combined_year_success = combined_year['success'].sum() if combined_year_total > 0 else 0
            combined_year_rate = combined_year_success / combined_year_total if combined_year_total > 0 else 0
            
            yearly_stats.append({
                'year': year,
                'fzt_total': fzt_year_total,
                'fzt_rate': fzt_year_rate,
                'zsqsx_total': zsqsx_year_total,
                'zsqsx_rate': zsqsx_year_rate,
                'combined_total': combined_year_total,
                'combined_rate': combined_year_rate
            })
            
            print(f"   {year}: FZT({fzt_year_rate:.2%}), ZSQSX({zsqsx_year_rate:.2%}), 组合({combined_year_rate:.2%})")
        
        # 8. 股票数量统计
        print("\n📈 股票数量统计:")
        yearly_stock_counts = df_combined.groupby('year')['instrument'].nunique()
        for year, count in yearly_stock_counts.items():
            print(f"   {year}: {count} 只股票")
        
        # 9. 保存结果
        print("\n💾 保存结果...")
        results_dir = project_root / 'results' / 'fzt_zsqsx_2021_2026'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细报告
        report_path = results_dir / 'full_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("FZT + ZSQSX 双公式回测报告 (2021-2026年)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"回测时间范围: 2021-08-02 到 2026-02-06\n")
            f.write(f"总数据行数: {len(df_combined):,}\n")
            f.write(f"股票数量: {df_combined['instrument'].nunique():,}\n\n")
            
            f.write("单独FZT公式:\n")
            f.write(f"  总信号数: {fzt_total:,}\n")
            f.write(f"  成功信号数: {fzt_success:,}\n")
            f.write(f"  成功率: {fzt_rate:.2%}\n\n")
            
            f.write("单独ZSQSX公式:\n")
            f.write(f"  总信号数: {zsqsx_total:,}\n")
            f.write(f"  成功信号数: {zsqsx_success:,}\n")
            f.write(f"  成功率: {zsqsx_rate:.2%}\n\n")
            
            f.write("同时满足FZT和ZSQSX:\n")
            f.write(f"  总信号数: {combined_total:,}\n")
            f.write(f"  成功信号数: {combined_success:,}\n")
            f.write(f"  成功率: {combined_rate:.2%}\n\n")
            
            f.write("年度成功率:\n")
            for stat in yearly_stats:
                f.write(f"  {stat['year']}: FZT({stat['fzt_rate']:.2%}), ZSQSX({stat['zsqsx_rate']:.2%}), 组合({stat['combined_rate']:.2%})\n")
            
            f.write("\n年度股票数量:\n")
            for year, count in yearly_stock_counts.items():
                f.write(f"  {year}: {count} 只股票\n")
        
        # 保存年度统计数据
        yearly_df = pd.DataFrame(yearly_stats)
        yearly_df.to_csv(results_dir / 'yearly_stats.csv', index=False)
        
        # 保存详细数据
        df_combined.to_csv(results_dir / 'detailed_data.csv', index=False)
        
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
    exit_code = fzt_zsqsx_2021_2026_backtest()
    sys.exit(exit_code)