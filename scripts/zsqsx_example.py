#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZSQSX公式使用示例脚本

展示如何使用src/zsqsx_core.py模块进行ZSQSX公式计算和回测

作者: MC
创建日期: 2026-03-07
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time

# 导入核心模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.zsqsx_core import (
    calc_zsdkx,
    calculate_zsdkx_features_vectorized,
    get_zsdkx_signal_conditions,
    analyze_zsdkx_performance
)
from src.data_loader import load_stock_data_qlib, get_instruments_from_file


def example_zsdkx_calculation():
    """示例1：基本ZSQSX计算"""
    print("=" * 60)
    print("示例1：基本ZSQSX计算")
    print("=" * 60)
    
    # 创建测试数据
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    instruments = ["600519", "000001"]
    test_data = []
    for inst in instruments:
        price = 100 + np.cumsum(np.random.randn(100) * 2)
        for i, dt in enumerate(dates):
            test_data.append({
                "instrument": inst,
                "datetime": dt,
                "close": price[i]
            })
    test_df = pd.DataFrame(test_data)
    
    # 计算ZSQSX指标
    start_time = time.time()
    result = calc_zsdkx(
        test_df,
        M1=14, M2=28, M3=57, M4=114,
        output_stats=True
    )
    calc_time = time.time() - start_time
    
    print(f"\n计算耗时: {calc_time:.2f} 秒")
    print(f"结果形状: {result.shape}")
    print("\n前10行结果:")
    print(result.head(10))
    
    return result


def example_signal_generation():
    """示例2：信号生成"""
    print("\n" + "=" * 60)
    print("示例2：ZSQSX信号生成")
    print("=" * 60)
    
    # 使用示例1的结果
    result = example_zsdkx_calculation()
    
    # 生成信号条件
    df_with_signals = get_zsdkx_signal_conditions(
        result,
        qsx_threshold=result['QSX'].median(),  # 使用中位数作为阈值
        ma1_threshold=result['MA1'].median(),
        ma2_threshold=result['MA2'].median(),
        dks_threshold=result['DKS'].median()
    )
    
    # 统计信号
    signal_count = df_with_signals['ZSQSX_signal'].sum()
    total_count = len(df_with_signals)
    
    print(f"总数据行数: {total_count}")
    print(f"ZSQSX信号数量: {signal_count}")
    print(f"信号比例: {signal_count/total_count:.2%}")
    
    # 显示有信号的股票
    if signal_count > 0:
        signal_stocks = df_with_signals[df_with_signals['ZSQSX_signal']]['instrument'].unique()
        print(f"产生信号的股票: {list(signal_stocks)}")
    
    return df_with_signals


def example_performance_analysis():
    """示例3：性能分析（模拟）"""
    print("\n" + "=" * 60)
    print("示例3：ZSQSX性能分析（模拟）")
    print("=" * 60)
    
    # 生成带模拟成功标志的数据
    result = example_zsdkx_calculation()
    df_with_signals = get_zsdkx_signal_conditions(result)
    
    # 模拟成功标志（随机生成）
    np.random.seed(42)
    df_with_signals['success'] = np.random.choice([True, False], size=len(df_with_signals), p=[0.55, 0.45])
    
    # 分析性能
    performance = analyze_zsdkx_performance(df_with_signals, success_col='success')
    
    print("性能分析结果:")
    print(f"  总信号数: {performance['total_signals']:,}")
    print(f"  成功信号数: {performance['successful_signals']:,}")
    print(f"  成功率: {performance['success_rate']:.2%}")
    
    if performance['instruments']:
        print(f"  产生信号的股票: {performance['instruments'][:5]}...")
    
    return performance


def example_with_real_data_template():
    """示例4：使用真实数据的模板"""
    print("\n" + "=" * 60)
    print("示例4：使用真实数据的模板")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    
    # 模板：加载2006-2020年数据
    print("模板：加载2006-2020年数据并计算ZSQSX")
    print("（需要实际数据文件存在）")
    
    template_code = '''
# 在实际脚本中使用：
def load_2006_2020_data(project_root: Path):
    """加载2006-2020年数据（私有函数）"""
    data_dir = str(project_root / 'data' / '2006_2020')
    instruments_file = project_root / 'data' / '2006_2020' / 'instruments' / 'all.txt'
    
    instruments = get_instruments_from_file(instruments_file)
    if not instruments:
        return None
    
    # 使用所有3875只股票
    instruments = instruments[:3875]
    
    return load_stock_data_qlib(
        data_dir=data_dir,
        instruments=instruments,
        calc_start='2005-10-01',  # 提前一些以计算指标
        calc_end='2020-09-25',
        target_start='2006-01-01',
        target_end='2020-09-25'
    )

def zsqsx_2006_2020_backtest():
    """2006-2020年ZSQSX回测"""
    project_root = Path(__file__).parent.parent
    
    # 1. 加载数据
    df = load_2006_2020_data(project_root)
    if df is None or df.empty:
        print("数据加载失败")
        return
    
    # 2. 计算ZSQSX指标
    df_with_zsdkx = calc_zsdkx(df, output_stats=True)
    
    # 3. 生成信号
    df_with_signals = get_zsdkx_signal_conditions(df_with_zsdkx)
    
    # 4. 计算成功率（需要添加成功标志）
    # df_with_signals['next_close'] = df_with_signals.groupby('instrument')['close'].shift(-1)
    # df_with_signals['success'] = df_with_signals['next_close'] > df_with_signals['close']
    
    # 5. 分析性能
    # performance = analyze_zsdkx_performance(df_with_signals)
    '''
    
    print(template_code)
    print("\n✅ 模板代码已生成，可根据实际需求修改")


def main():
    """主函数：运行所有示例"""
    print("🚀 ZSQSX公式使用示例")
    print("=" * 60)
    
    try:
        # 运行示例1
        example_zsdkx_calculation()
        
        # 运行示例2
        example_signal_generation()
        
        # 运行示例3
        example_performance_analysis()
        
        # 运行示例4
        example_with_real_data_template()
        
        print("\n" + "=" * 60)
        print("✅ 所有示例运行完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()