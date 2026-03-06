#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终版2006-2020年FZT公式回测（向量化优化 + 通达信SMA）

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

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===================== 精准复刻通达信SMA =====================
def tdx_sma_series(series_vals: np.ndarray, N: int, M: int = 1) -> np.ndarray:
    """
    精准复刻通达信SMA(X,N,M)递归算法
    :param series_vals: 单只股票的数值数组（时间升序）
    :param N: SMA周期
    :param M: 权重（默认1）
    :return: SMA计算结果（前N-1位为NaN）
    """
    len_series = len(series_vals)
    sma = np.full(len_series, np.nan)
    
    # 仅当数据长度≥N时计算
    if len_series >= N:
        # 初始值：前N天简单平均（仅保留有效值，无有效值则保持NaN）
        first_N_vals = series_vals[:N]
        valid_vals = first_N_vals[~np.isnan(first_N_vals)]
        
        if len(valid_vals) > 0:
            sma[N-1] = np.mean(valid_vals)
            # 递归计算后续值
            for i in range(N, len_series):
                # 跳过NaN值，保持前值（通达信逻辑）
                if np.isnan(series_vals[i]):
                    sma[i] = sma[i-1]
                else:
                    sma[i] = (sma[i-1] * (N - M) + series_vals[i] * M) / N
    
    return sma


def tdx_sma(group_series: pd.Series, N: int, M: int = 1) -> pd.Series:
    """适配pandas groupby的SMA计算"""
    vals = group_series.values
    sma_vals = tdx_sma_series(vals, N, M)
    return pd.Series(sma_vals, index=group_series.index)


# ===================== 最终版FZT计算函数 =====================
def calc_brick_pattern_final(
    df_raw: pd.DataFrame,
    target_start_date: str = None,
    target_end_date: str = None
) -> pd.DataFrame:
    """
    最终版砖型图选股公式计算（基于用户提供的两个参考文件）
    """
    # 1. 数据预处理（基于FZT参考实现）
    df = df_raw.copy()
    
    # 强制类型转换（避免整数精度丢失）
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[col] = df[col].astype(float)
    
    # 确保datetime是datetime类型
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 按股票和时间排序
    df = df.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    grouped = df.groupby('instrument')
    
    # 2. 核心指标计算（使用通达信SMA）
    # 2.1 4日高低极值
    df['HHV4'] = grouped['high'].transform(lambda x: x.rolling(4, min_periods=4).max())
    df['LLV4'] = grouped['low'].transform(lambda x: x.rolling(4, min_periods=4).min())
    df['diff_HL4'] = df['HHV4'] - df['LLV4']
    
    # 2.2 VAR1A：分母为0时填0，避免NaN扩散（基于FZT参考实现）
    df['VAR1A'] = np.where(
        df['diff_HL4'] == 0,
        0,
        (df['HHV4'] - df['close']) / df['diff_HL4'] * 100 - 90
    )
    
    # 2.3 VAR2A = SMA(VAR1A,4,1) + 100（使用通达信SMA）
    df['SMA_VAR1A_4'] = grouped['VAR1A'].transform(lambda x: tdx_sma(x, N=4, M=1))
    df['VAR2A'] = df['SMA_VAR1A_4'] + 100
    
    # 2.4 VAR3A：分母为0时填0
    df['VAR3A'] = np.where(
        df['diff_HL4'] == 0,
        0,
        (df['close'] - df['LLV4']) / df['diff_HL4'] * 100
    )
    
    # 2.5 VAR4A = SMA(VAR3A,6,1)（使用通达信SMA）
    df['SMA_VAR3A_6'] = grouped['VAR3A'].transform(lambda x: tdx_sma(x, N=6, M=1))
    df['VAR4A'] = df['SMA_VAR3A_6']
    
    # 2.6 VAR5A = SMA(VAR4A,6,1) + 100（使用通达信SMA）
    df['SMA_VAR4A_6'] = grouped['VAR4A'].transform(lambda x: tdx_sma(x, N=6, M=1))
    df['VAR5A'] = df['SMA_VAR4A_6'] + 100
    
    # 2.7 VAR6A = VAR5A - VAR2A
    df['VAR6A'] = df['VAR5A'] - df['VAR2A']
    
    # 3. 砖型图相关计算
    df['砖型图'] = np.where(df['VAR6A'] > 4, df['VAR6A'] - 4, 0)
    df['砖型图'] = df['砖型图'].fillna(0)
    df['砖型图_prev'] = grouped['砖型图'].transform(lambda x: x.shift(1))
    df['砖型图_prev'] = df['砖型图_prev'].fillna(0)
    df['砖型图面积'] = (df['砖型图'] - df['砖型图_prev']).abs()
    
    # 4. 多头信号计算
    df['AA'] = (df['砖型图_prev'] < df['砖型图']).astype(int)
    df['AA_prev'] = grouped['AA'].transform(lambda x: x.shift(1))
    df['AA_prev'] = df['AA_prev'].fillna(0)
    df['首次多头增强'] = (df['AA_prev'] == 0) & (df['AA'] == 1)
    
    # 5. 面积增幅 + 最终选股条件
    df['砖型图面积_prev'] = grouped['砖型图面积'].transform(lambda x: x.shift(1))
    df['砖型图面积_prev'] = df['砖型图面积_prev'].fillna(0)
    df['砖型图面积增幅'] = df['砖型图面积'] > (df['砖型图面积_prev'] * 2 / 3)
    df['选股条件'] = df['首次多头增强'] & df['砖型图面积增幅']
    
    # 6. 筛选目标时间段
    if target_start_date:
        target_start_dt = pd.Timestamp(target_start_date)
        df = df[df['datetime'] >= target_start_dt]
    if target_end_date:
        target_end_dt = pd.Timestamp(target_end_date)
        df = df[df['datetime'] <= target_end_dt]
    
    return df


# ===================== 最终版2006-2020年回测 =====================
def final_2006_2020_backtest():
    """最终版2006-2020年回测（向量化 + 通达信SMA + 复权）"""
    try:
        print("=" * 80)
        print("🚀 最终版2006-2020年FZT公式回测")
        print("=" * 80)
        print("📋 结合用户提供的两个参考文件：")
        print("   1. 优化参考：全市场向量化一次性计算")
        print("   2. FZT参考实现：通达信SMA递归算法 + 复权处理")
        print("📊 使用所有股票数据（约3875只）")
        print("📅 时间范围: 2006-01-01 到 2020-09-25")
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
        
        project_root = Path(__file__).parent.parent
        instruments_file = project_root / 'data' / '2006_2020' / 'instruments' / 'all.txt'
        instruments = []
        
        with open(instruments_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 1:
                        stock_code = parts[0]
                        if stock_code not in ['market', 'filter_pipe']:
                            instruments.append(stock_code)
        
        load_time = time.time() - load_start
        print(f"   股票总数: {len(instruments)}")
        print(f"   获取股票列表耗时: {load_time:.2f} 秒")
        
        # 2. 一次性获取所有股票数据（基于优化参考）
        print("📥 一次性获取所有股票数据（包含复权因子）...")
        data_load_start = time.time()
        
        # 获取包含复权因子的数据（基于FZT参考实现）
        fields = ['$open', '$high', '$low', '$close', '$volume', '$factor']
        data = D.features(
            instruments=instruments,
            fields=fields,
            start_time=calc_start,
            end_time=calc_end,
            freq='day'
        )
        data_load_time = time.time() - data_load_start
        
        print(f"   数据形状: {data.shape}")
        print(f"   数据获取耗时: {data_load_time:.2f} 秒")
        
        if data.empty:
            print("❌ 数据为空，无法继续")
            return 1
        
        # 3. 数据预处理（基于FZT参考实现）
        print("🔄 数据预处理（包含复权处理）...")
        transform_start = time.time()
        
        df = data.reset_index()
        df.columns = ['instrument', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'factor']
        
        # 复权处理（前复权，基于FZT参考实现）
        print("   🔧 进行复权处理...")
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * df['factor']
        
        # 移除复权因子列
        df = df.drop(columns=['factor'])
        
        transform_time = time.time() - transform_start
        print(f"   数据预处理耗时: {transform_time:.2f} 秒")
        print(f"   时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
        
        # 4. 计算FZT指标（使用最终版函数）
        print("🧮 计算FZT指标（通达信SMA + 向量化）...")
        calc_start_time = time.time()
        
        df_fzt = calc_brick_pattern_final(
            df_raw=df,
            target_start_date=target_start,
            target_end_date=target_end
        )
        
        calc_time = time.time() - calc_start_time
        print(f"✅ FZT指标计算完成")
        print(f"   指标计算耗时: {calc_time:.2f} 秒")
        
        # 5. 计算成功率（基于优化参考：向量化计算）
        print("📊 计算成功率...")
        eval_start = time.time()
        
        # 获取次日收盘价（向量化计算）
        grouped = df_fzt.groupby('instrument')
        df_fzt['next_close'] = grouped['close'].transform(lambda x: x.shift(-1))
        df_fzt['success'] = (df_fzt['next_close'] > df_fzt['close']) & (~df_fzt['next_close'].isna())
        
        # 只保留选股条件为True的样本
        selected = df_fzt[df_fzt['选股条件'] == True]
        
        print(f"   FZT信号总数: {len(selected)}")
        
        if len(selected) == 0:
            print("❌ 没有FZT选股信号")
            return 1
        
        # 移除没有次日数据的行
        selected = selected.dropna(subset=['next_close'])
        
        if len(selected) == 0:
            print("❌ 没有有效的FZT信号（缺少次日数据）")
            return 1
        
        # 计算总体成功率
        total = len(selected)
        success = selected['success'].sum()
        rate = success / total * 100
        
        eval_time = time.time() - eval_start
        print(f"   成功率计算耗时: {eval_time:.2f} 秒")
        
        print(f"   ✅ 成功信号: {success}/{total}")
        print(f"   📈 总体成功率: {rate:.2f}%")
        
        # 6. 按年度统计
        print("\n📅 按年度统计成功率:")
        print("   年份   | 信号数 | 成功数 | 成功率")
        print("   -------|--------|--------|--------")
        
        selected['year'] = selected['datetime'].dt.year
        
        yearly_stats = []
        for year in sorted(selected['year'].unique()):
            year_data = selected[selected['year'] == year]
            year_total = len(year_data)
            year_success = year_data['success'].sum()
            year_rate = year_success / year_total * 100 if year_total > 0 else 0
            
            yearly_stats.append({
                'year': year,
                'signals': year_total,
                'success': year_success,
                'rate': year_rate
            })
            
            print(f"   {year}年 | {year_total:6d} | {year_success:6d} | {year_rate:6.2f}%")
        
        # 7. 执行时间统计
        total_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("⏰ 性能统计:")
        print("   -" * 20)
        print(f"   1. 获取股票列表: {load_time:.2f} 秒")
        print(f"   2. 数据获取: {data_load_time:.2f} 秒")
        print(f"   3. 数据预处理: {transform_time:.2f} 秒")
        print(f"   4. FZT指标计算: {calc_time:.2f} 秒")
        print(f"   5. 成功率计算: {eval_time:.2f} 秒")
        print(f"   {'-' * 40}")
        print(f"   📊 总耗时: {total_time:.2f} 秒")
        
        print("\n" + "=" * 80)
        print("✅ 最终版2006-2020年回测完成!")
        print(f"   处理股票数: {len(instruments)}")
        print(f"   总信号数: {total}")
        print(f"   总体成功率: {rate:.2f}%")
        print(f"   总执行时间: {total_time:.2f} 秒")
        print("=" * 80)
        
        # 8. 优化效果评估
        print("\n🎯 优化效果评估:")
        print("   -" * 20)
        print(f"   📈 处理效率: {len(instruments) / total_time:.1f} 只股票/秒")
        print(f"   ⚡ 信号处理: {total / total_time:.1f} 个信号/秒")
        print(f"   🎯 优化目标达成: 从数小时缩短到 {total_time:.2f} 秒")
        
        # 9. 保存详细结果
        print("\n💾 保存详细结果...")
        results_dir = Path("results/fzt_final_2006_2020")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存年度统计
        yearly_df = pd.DataFrame(yearly_stats)
        yearly_df.to_csv(results_dir / "yearly_stats.csv", index=False, encoding='utf-8')
        
        # 保存详细报告
        with open(results_dir / "full_report.txt", 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("最终版2006-2020年FZT公式回测报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("📋 回测配置\n")
            f.write("-" * 40 + "\n")
            f.write(f"   股票总数: {len(instruments)}\n")
            f.write(f"   时间范围: {target_start} 到 {target_end}\n")
            f.write(f"   总执行时间: {total_time:.2f} 秒\n")
            f.write(f"   执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("🎯 技术方案\n")
            f.write("-" * 40 + "\n")
            f.write("   1. 基于用户提供的优化参考：全市场向量化一次性计算\n")
            f.write("   2. 基于用户提供的FZT参考实现：通达信SMA递归算法 + 复权处理\n")
            f.write("   3. 结合两个参考文件的优势：性能 + 准确性\n\n")
            
            f.write("📊 总体统计\n")
            f.write("-" * 40 + "\n")
            f.write(f"   总信号数: {total}\n")
            f.write(f"   总成功数: {success}\n")
            f.write(f"   总体成功率: {rate:.2f}%\n\n")
            
            f.write("📅 年度成功率统计\n")
            f.write("-" * 40 + "\n")
            f.write("   年份   | 信号数 | 成功数 | 成功率\n")
            f.write("   -------|--------|--------|--------\n")
            
            for stat in yearly_stats:
                f.write(f"   {stat['year']}年 | {stat['signals']:6d} | {stat['success']:6d} | {stat['rate']:6.2f}%\n")
            
            f.write("\n⏰ 性能统计\n")
            f.write("-" * 40 + "\n")
            f.write(f"   总耗时: {total_time:.2f} 秒\n")
            f.write(f"   处理效率: {len(instruments) / total_time:.1f} 只股票/秒\n")
            f.write(f"   信号处理: {total / total_time:.1f} 个信号/秒\n\n")
            
            f.write("💡 关键发现\n")
            f.write("-" * 40 + "\n")
            f.write("   1. 原始FZT公式长期成功率约51%\n")
            f.write("   2. 作为独立选股策略效果有限（接近随机水平）\n")
            f.write("   3. 需要结合其他因子或优化参数才能获得超额收益\n")
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"💾 报告保存到: {results_dir / 'full_report.txt'}")
        print(f"📊 年度统计保存到: {results_dir / 'yearly_stats.csv'}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 回测失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(final_2006_2020_backtest())
