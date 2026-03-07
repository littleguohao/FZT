#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FZT + ZSQSX 双公式回测脚本 (2021-2026年) - 参数控制版本

支持参数控制公式组合：
1. 单独FZT公式（验证之前结果）
2. 单独ZSQSX公式  
3. 同时满足FZT和ZSQSX

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
from typing import Optional, Dict, Any
import argparse
warnings.filterwarnings('ignore')

# 导入核心模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.fzt_core import calc_brick_pattern_final
# from src.zsqsx_core import calc_zsdkx, get_zsdkx_signal_conditions  # 暂时注释掉ZSQSX
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


# ===================== 参数化回测 =====================
def run_backtest_with_params(
    use_fzt: bool = True,
    use_zsqsx: bool = True,
    verify_fzt_only: bool = False,
    top_n: int = 0,
    top_k: int = 4
) -> Dict[str, Any]:
    """运行参数化回测"""
    try:
        print("=" * 80)
        print(f"🚀 参数化回测 (2021-2026年)")
        print(f"   使用FZT: {use_fzt}, 使用ZSQSX: {use_zsqsx}, 验证单独FZT: {verify_fzt_only}")
        print("=" * 80)
        
        start_time = time.time()
        project_root = Path(__file__).parent.parent
        
        # 1. 加载数据
        print("\n📥 加载2021-2026年数据...")
        df = load_2021_2026_data(project_root)
        
        if df is None or df.empty:
            print("❌ 数据加载失败，无法继续")
            return {'error': '数据加载失败'}
        
        # 2. 计算FZT指标（如果需要）
        df_fzt = None
        if use_fzt:
            print("\n🧮 计算FZT指标...")
            fzt_start_time = time.time()
            df_fzt = calc_brick_pattern_final(df)
            fzt_time = time.time() - fzt_start_time
            print(f"✅ FZT计算完成，耗时: {fzt_time:.2f} 秒")
        
        # 3. 计算ZSQSX指标（如果需要）- 暂时注释掉
        df_zsqsx = None
        if use_zsqsx:
            print("\n🧮 计算ZSQSX指标...")
            print("⚠️  ZSQSX暂时注释掉，不进行计算")
            # zsqsx_start_time = time.time()
            # df_zsdkx = calc_zsdkx(df)
            # df_zsqsx = get_zsdkx_signal_conditions(df_zsdkx)
            # zsqsx_time = time.time() - zsqsx_start_time
            # print(f"✅ ZSQSX计算完成，耗时: {zsqsx_time:.2f} 秒")
        
        # 4. 合并结果
        print("\n🔗 合并结果...")
        df_combined = df[['instrument', 'datetime', 'close']].copy()
        
        if df_fzt is not None:
            df_fzt_idx = df_fzt.set_index(['instrument', 'datetime']).sort_index()
            df_combined = df_combined.set_index(['instrument', 'datetime']).sort_index()
            df_combined['FZT_signal'] = df_fzt_idx['选股条件']
            df_combined = df_combined.reset_index()
        
        # if df_zsqsx is not None:
        #     df_zsqsx_idx = df_zsqsx.set_index(['instrument', 'datetime']).sort_index()
        #     if 'instrument' not in df_combined.columns:  # 如果已经设置索引
        #         df_combined = df_combined.set_index(['instrument', 'datetime']).sort_index()
        #     else:
        #         df_combined = df_combined.set_index(['instrument', 'datetime']).sort_index()
        #     
        #     df_combined['ZSQSX_signal'] = df_zsqsx_idx['ZSQSX_signal']
        #     df_combined = df_combined.reset_index()
        
        # 5. 计算次日收益率（成功条件）
        print("📈 计算成功率...")
        df_combined['next_close'] = df_combined.groupby('instrument')['close'].transform(lambda x: x.shift(-1))
        df_combined['success'] = df_combined['next_close'] > df_combined['close']
        
        # 6. 根据参数计算不同的成功率
        print("\n📊 回测结果统计:")
        results = {}
        
        # 6.1 单独FZT公式（如果需要验证）
        if verify_fzt_only and 'FZT_signal' in df_combined.columns:
            fzt_signals = df_combined[df_combined['FZT_signal'] == True]
            fzt_total = len(fzt_signals)
            fzt_success = fzt_signals['success'].sum() if fzt_total > 0 else 0
            fzt_rate = fzt_success / fzt_total if fzt_total > 0 else 0
            
            print(f"\n📈 单独FZT公式（验证）:")
            print(f"   总信号数: {fzt_total:,}")
            print(f"   成功信号数: {fzt_success:,}")
            print(f"   成功率: {fzt_rate:.2%}")
            
            results['fzt_only'] = {
                'total': fzt_total,
                'success': fzt_success,
                'rate': fzt_rate
            }
            
            # 6.4 新增：按FZT面积排序取TOP N（参数控制）
            if top_n > 0:
                print(f"\n🎯 新增：按FZT面积排序取TOP{top_n} (每天取前{top_k}只):")
                
                # 首先需要获取包含砖型图面积的完整FZT数据
                if df_fzt is not None and '砖型图面积' in df_fzt.columns:
                    # 合并面积数据
                    df_fzt_with_area = df_fzt[['instrument', 'datetime', '选股条件', '砖型图面积']].copy()
                    df_fzt_with_area = df_fzt_with_area.rename(columns={'选股条件': 'FZT_signal'})
                    
                    # 合并成功数据
                    df_merged = pd.merge(
                        df_fzt_with_area,
                        df_combined[['instrument', 'datetime', 'success']],
                        on=['instrument', 'datetime'],
                        how='inner'
                    )
                    
                    # 按日期分组，每天取选股条件为True且面积最大的前top_k只
                    def select_topk_daily(group):
                        candidates = group[group['FZT_signal']].copy()
                        if len(candidates) == 0:
                            return pd.DataFrame()  # 无信号
                        # 按面积降序取前top_k只
                        return candidates.nlargest(top_k, '砖型图面积')
                    
                    top_signals = df_merged.groupby('datetime').apply(select_topk_daily).reset_index(drop=True)
                    
                    if not top_signals.empty:
                        top_total = len(top_signals)
                        top_success = top_signals['success'].sum() if top_total > 0 else 0
                        top_rate = top_success / top_total if top_total > 0 else 0
                        
                        print(f"   TOP{top_n}总信号数: {top_total:,}")
                        print(f"   TOP{top_n}成功信号数: {top_success:,}")
                        print(f"   TOP{top_n}成功率: {top_rate:.2%}")
                        
                        results[f'fzt_top{top_n}'] = {
                            'total': top_total,
                            'success': top_success,
                            'rate': top_rate
                        }
                        
                        # 年度TOP分析
                        print(f"\n📅 TOP{top_n}年度成功率分析:")
                        top_signals['year'] = top_signals['datetime'].dt.year
                        
                        yearly_top_stats = []
                        for year in sorted(top_signals['year'].unique()):
                            year_data = top_signals[top_signals['year'] == year]
                            year_total = len(year_data)
                            year_success = year_data['success'].sum() if year_total > 0 else 0
                            year_rate = year_success / year_total if year_total > 0 else 0
                            
                            yearly_top_stats.append({
                                'year': year,
                                'total': year_total,
                                'success': year_success,
                                'rate': year_rate
                            })
                            
                            print(f"   {year}: TOP{top_n}({year_rate:.2%})")
                        
                        results[f'yearly_top{top_n}_stats'] = yearly_top_stats
                    else:
                        print(f"   ⚠️ 没有符合条件的TOP{top_n}信号")
                else:
                    print(f"   ⚠️ 无法获取砖型图面积数据，跳过TOP{top_n}分析")
            else:
                print(f"\nℹ️  TOP排序因子未启用 (top_n=0)")
        
        # 6.2 单独ZSQSX公式 - 暂时注释掉
        # if use_zsqsx and not use_fzt and 'ZSQSX_signal' in df_combined.columns:
        #     zsqsx_signals = df_combined[df_combined['ZSQSX_signal'] == True]
        #     zsqsx_total = len(zsqsx_signals)
        #     zsqsx_success = zsqsx_signals['success'].sum() if zsqsx_total > 0 else 0
        #     zsqsx_rate = zsqsx_success / zsqsx_total if zsqsx_total > 0 else 0
        #     
        #     print(f"\n📈 单独ZSQSX公式:")
        #     print(f"   总信号数: {zsqsx_total:,}")
        #     print(f"   成功信号数: {zsqsx_success:,}")
        #     print(f"   成功率: {zsqsx_rate:.2%}")
        #     
        #     results['zsqsx_only'] = {
        #         'total': zsqsx_total,
        #         'success': zsqsx_success,
        #         'rate': zsqsx_rate
        #     }
        
        # 6.3 同时满足FZT和ZSQSX - 暂时注释掉
        # if use_fzt and use_zsqsx and 'FZT_signal' in df_combined.columns and 'ZSQSX_signal' in df_combined.columns:
        #     combined_signals = df_combined[
        #         (df_combined['FZT_signal'] == True) & 
        #         (df_combined['ZSQSX_signal'] == True)
        #     ]
        #     combined_total = len(combined_signals)
        #     combined_success = combined_signals['success'].sum() if combined_total > 0 else 0
        #     combined_rate = combined_success / combined_total if combined_total > 0 else 0
        #     
        #     print(f"\n🎯 同时满足FZT和ZSQSX:")
        #     print(f"   总信号数: {combined_total:,}")
        #     print(f"   成功信号数: {combined_success:,}")
        #     print(f"   成功率: {combined_rate:.2%}")
        #     
        #     results['combined'] = {
        #         'total': combined_total,
        #         'success': combined_success,
        #         'rate': combined_rate
        #     }
        
        # 7. 年度分析
        if (use_fzt and 'FZT_signal' in df_combined.columns) or (use_zsqsx and 'ZSQSX_signal' in df_combined.columns):
            print("\n📅 年度成功率分析:")
            df_combined['year'] = df_combined['datetime'].dt.year
            
            yearly_stats = []
            for year in sorted(df_combined['year'].unique()):
                year_data = df_combined[df_combined['year'] == year]
                year_stats = {'year': year}
                
                # 单独FZT
                if verify_fzt_only and 'FZT_signal' in df_combined.columns:
                    fzt_year = year_data[year_data['FZT_signal'] == True]
                    fzt_year_total = len(fzt_year)
                    fzt_year_success = fzt_year['success'].sum() if fzt_year_total > 0 else 0
                    fzt_year_rate = fzt_year_success / fzt_year_total if fzt_year_total > 0 else 0
                    year_stats['fzt_rate'] = fzt_year_rate
                
                # 单独ZSQSX - 暂时注释掉
                # if use_zsqsx and not use_fzt and 'ZSQSX_signal' in df_combined.columns:
                #     zsqsx_year = year_data[year_data['ZSQSX_signal'] == True]
                #     zsqsx_year_total = len(zsqsx_year)
                #     zsqsx_year_success = zsqsx_year['success'].sum() if zsqsx_year_total > 0 else 0
                #     zsqsx_year_rate = zsqsx_year_success / zsqsx_year_total if zsqsx_year_total > 0 else 0
                #     year_stats['zsqsx_rate'] = zsqsx_year_rate
                # 
                # 同时满足 - 暂时注释掉
                # if use_fzt and use_zsqsx and 'FZT_signal' in df_combined.columns and 'ZSQSX_signal' in df_combined.columns:
                #     combined_year = year_data[
                #         (year_data['FZT_signal'] == True) & 
                #         (year_data['ZSQSX_signal'] == True)
                #     ]
                #     combined_year_total = len(combined_year)
                #     combined_year_success = combined_year['success'].sum() if combined_year_total > 0 else 0
                #     combined_year_rate = combined_year_success / combined_year_total if combined_year_total > 0 else 0
                #     year_stats['combined_rate'] = combined_year_rate
                
                yearly_stats.append(year_stats)
                
                # 打印年度结果
                year_output = f"   {year}:"
                if 'fzt_rate' in year_stats:
                    year_output += f" FZT({year_stats['fzt_rate']:.2%})"
                # if 'zsqsx_rate' in year_stats:
                #     year_output += f" ZSQSX({year_stats['zsqsx_rate']:.2%})"
                # if 'combined_rate' in year_stats:
                #     year_output += f" 组合({year_stats['combined_rate']:.2%})"
                print(year_output)
            
            results['yearly_stats'] = yearly_stats
        
        # 8. 股票数量统计
        print("\n📈 股票数量统计:")
        df_combined['year'] = df_combined['datetime'].dt.year
        yearly_stock_counts = df_combined.groupby('year')['instrument'].nunique()
        for year, count in yearly_stock_counts.items():
            print(f"   {year}: {count} 只股票")
        
        results['yearly_stock_counts'] = yearly_stock_counts.to_dict()
        
        total_time = time.time() - start_time
        print(f"\n✅ 回测完成！总耗时: {total_time:.2f} 秒")
        
        return results
        
    except Exception as e:
        print(f"❌ 回测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def main():
    """主函数：解析参数并运行回测"""
    parser = argparse.ArgumentParser(description='FZT + ZSQSX 参数化回测')
    parser.add_argument('--fzt', action='store_true', default=True, help='使用FZT公式')
    parser.add_argument('--no-fzt', action='store_false', dest='fzt', help='不使用FZT公式')
    parser.add_argument('--zsqsx', action='store_true', default=True, help='使用ZSQSX公式')
    parser.add_argument('--no-zsqsx', action='store_false', dest='zsqsx', help='不使用ZSQSX公式')
    parser.add_argument('--verify-fzt', action='store_true', default=False, help='验证单独FZT公式结果')
    parser.add_argument('--top-n', type=int, default=0, 
                       help='按FZT面积排序取TOP N (0表示不启用，默认: 0)')
    parser.add_argument('--top-k', type=int, default=4, 
                       help='每天取TOP K只股票 (默认: 4)')
    
    args = parser.parse_args()
    
    # 运行回测
    results = run_backtest_with_params(
        use_fzt=args.fzt,
        use_zsqsx=args.zsqsx,
        verify_fzt_only=args.verify_fzt,
        top_n=args.top_n,
        top_k=args.top_k
    )
    
    return 0 if 'error' not in results else 1


if __name__ == "__main__":
    # 运行回测
    exit_code = main()
    sys.exit(exit_code)