#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FZTAlpha算子使用示例

展示如何在QLib中使用FZTAlpha算子，以及QLib不可用时如何使用简化版本。

作者: FZT项目组
创建日期: 2026-03-02
"""

import sys
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

import pandas as pd
import numpy as np

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# 忽略警告
warnings.filterwarnings('ignore')


def example_basic_usage():
    """基本使用示例"""
    print("=" * 80)
    print("FZTAlpha算子基本使用示例")
    print("=" * 80)
    
    # 导入FZTAlpha
    try:
        from qlib_operators.fzt_alpha import FZTAlpha
        print("✓ 成功导入FZTAlpha")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return
    
    # 创建测试数据
    print("\n1. 创建测试数据...")
    np.random.seed(42)
    n_days = 50
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    base_price = 100
    trend = np.linspace(0, 0.1, n_days)
    noise = np.random.normal(0, 0.02, n_days)
    returns = 0.001 + trend/100 + noise
    prices = base_price * np.exp(np.cumsum(returns))
    
    highs = prices * (1 + np.abs(np.random.normal(0.01, 0.005, n_days)))
    lows = prices * (1 - np.abs(np.random.normal(0.01, 0.005, n_days)))
    
    test_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': np.random.lognormal(mean=10, sigma=1, size=n_days)
    }, index=dates)
    
    print(f"   数据形状: {test_data.shape}")
    print(f"   时间范围: {test_data.index.min()} 到 {test_data.index.max()}")
    
    # 创建FZTAlpha实例
    print("\n2. 创建FZTAlpha实例...")
    fzt_alpha = FZTAlpha()
    print(f"   实例类型: {type(fzt_alpha)}")
    print(f"   实例表示: {fzt_alpha}")
    
    # 计算特征
    print("\n3. 计算特征...")
    features = fzt_alpha(test_data)
    print(f"   特征形状: {features.shape}")
    print(f"   特征列: {list(features.columns)}")
    
    # 显示特征示例
    print("\n4. 特征示例（最后5天）:")
    sample_features = features.tail()
    print(sample_features[['fzt_var6a', 'fzt_brick_chart', 'fzt_selection_condition']])
    
    # 分析选股信号
    print("\n5. 选股信号分析:")
    if 'fzt_selection_condition' in features.columns:
        signals = features['fzt_selection_condition']
        signal_count = signals.sum()
        signal_dates = signals[signals].index
        
        print(f"   总信号数: {signal_count}")
        print(f"   信号率: {signal_count/len(signals):.2%}")
        
        if signal_count > 0:
            print(f"   信号日期:")
            for date in signal_dates:
                # 获取当天的特征值
                date_features = features.loc[date]
                print(f"     {date.date()}: var6a={date_features['fzt_var6a']:.2f}, "
                      f"brick_chart={date_features['fzt_brick_chart']:.2f}")
    
    return features


def example_with_config():
    """使用配置参数的示例"""
    print("\n" + "=" * 80)
    print("FZTAlpha算子配置使用示例")
    print("=" * 80)
    
    from qlib_operators.fzt_alpha import FZTAlpha
    
    # 自定义配置
    config = {
        'var1_window': 5,      # 修改VAR1A计算窗口
        'var2_sma_window': 5,  # 修改VAR2A平滑窗口
        'brick_threshold': 3,  # 降低砖型图阈值
    }
    
    print(f"使用配置: {config}")
    
    # 创建带配置的实例
    fzt_alpha_custom = FZTAlpha(config=config)
    
    # 创建简单测试数据
    np.random.seed(123)
    n_days = 30
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    test_data = pd.DataFrame({
        'open': np.random.normal(100, 2, n_days),
        'high': np.random.normal(102, 2, n_days),
        'low': np.random.normal(98, 2, n_days),
        'close': np.random.normal(100, 2, n_days),
        'volume': np.random.lognormal(10, 1, n_days)
    }, index=dates)
    
    # 计算特征
    features_custom = fzt_alpha_custom(test_data)
    
    print(f"\n自定义配置特征形状: {features_custom.shape}")
    
    # 比较默认配置和自定义配置的信号差异
    fzt_alpha_default = FZTAlpha()
    features_default = fzt_alpha_default(test_data)
    
    if 'fzt_selection_condition' in features_custom.columns and 'fzt_selection_condition' in features_default.columns:
        signals_custom = features_custom['fzt_selection_condition'].sum()
        signals_default = features_default['fzt_selection_condition'].sum()
        
        print(f"\n信号比较:")
        print(f"   默认配置信号数: {signals_default}")
        print(f"   自定义配置信号数: {signals_custom}")
        print(f"   差异: {signals_custom - signals_default}")
    
    return features_custom


def example_qlib_integration():
    """QLib集成示例（如果QLib可用）"""
    print("\n" + "=" * 80)
    print("QLib集成示例")
    print("=" * 80)
    
    from qlib_operators.fzt_alpha import FZTAlpha, QLIB_AVAILABLE
    
    print(f"QLib可用: {QLIB_AVAILABLE}")
    
    if QLIB_AVAILABLE:
        try:
            from qlib.data.ops import Operator
            
            # 检查是否是Operator子类
            fzt_alpha = FZTAlpha()
            is_operator = isinstance(fzt_alpha, Operator)
            is_operator_subclass = issubclass(FZTAlpha, Operator)
            
            print(f"   是Operator实例: {is_operator}")
            print(f"   是Operator子类: {is_operator_subclass}")
            
            # 演示在QLib管道中使用
            print("\n   可以在QLib管道中使用FZTAlpha作为特征提取算子")
            print("   示例代码:")
            print("   ```python")
            print("   from qlib_operators.fzt_alpha import FZTAlpha")
            print("   from qlib.data.dataset import DatasetH")
            print("   ")
            print("   # 创建FZTAlpha算子")
            print("   fzt_op = FZTAlpha()")
            print("   ")
            print("   # 在DatasetH中使用")
            print("   dataset = DatasetH(")
            print("       handler={")
            print("           'start_time': '2020-01-01',")
            print("           'end_time': '2023-12-31',")
            print("           'fit_start_time': '2020-01-01',")
            print("           'fit_end_time': '2022-12-31',")
            print("           'instruments': 'csi300',")
            print("           'labels': ['Ref($close, -2)/Ref($close, -1)-1'],")
            print("           'feature': [fzt_op],  # 使用FZTAlpha算子")
            print("       },")
            print("       segments={")
            print("           'train': ('2020-01-01', '2021-12-31'),")
            print("           'valid': ('2022-01-01', '2022-12-31'),")
            print("           'test': ('2023-01-01', '2023-12-31'),")
            print("       }")
            print("   )")
            print("   ```")
            
        except ImportError as e:
            print(f"   QLib导入错误: {e}")
    else:
        print("   QLib不可用，使用简化版本")
        print("   简化版本提供相同的接口，可以在QLib可用时无缝切换")
        
        # 演示简化版本的使用
        fzt_alpha = FZTAlpha()
        print(f"\n   简化版本实例: {fzt_alpha}")
        print(f"   特征名称: {fzt_alpha.get_feature_names()}")


def example_performance_benchmark():
    """性能基准测试"""
    print("\n" + "=" * 80)
    print("性能基准测试")
    print("=" * 80)
    
    from qlib_operators.fzt_alpha import FZTAlpha
    import time
    
    # 创建不同规模的数据
    sizes = [100, 1000, 5000, 10000]
    
    results = []
    
    for n_days in sizes:
        print(f"\n测试数据规模: {n_days} 天")
        
        # 创建数据
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        
        test_data = pd.DataFrame({
            'open': np.random.normal(100, 5, n_days),
            'high': np.random.normal(105, 5, n_days),
            'low': np.random.normal(95, 5, n_days),
            'close': np.random.normal(100, 5, n_days),
            'volume': np.random.lognormal(10, 1, n_days)
        }, index=dates)
        
        # 创建算子
        fzt_alpha = FZTAlpha()
        
        # 预热
        _ = fzt_alpha(test_data.iloc[:100])
        
        # 性能测试
        start_time = time.time()
        features = fzt_alpha(test_data)
        end_time = time.time()
        
        compute_time = end_time - start_time
        rows_per_second = n_days / compute_time if compute_time > 0 else 0
        
        results.append({
            'n_days': n_days,
            'compute_time': compute_time,
            'rows_per_second': rows_per_second,
            'features_shape': features.shape
        })
        
        print(f"   计算时间: {compute_time:.4f} 秒")
        print(f"   每秒处理行数: {rows_per_second:.0f}")
        print(f"   特征形状: {features.shape}")
    
    # 总结
    print("\n" + "-" * 80)
    print("性能总结:")
    for result in results:
        print(f"  {result['n_days']:6d} 天: {result['compute_time']:7.4f} 秒 "
              f"({result['rows_per_second']:8.0f} 行/秒)")
    
    return results


def main():
    """主函数"""
    try:
        # 基本使用示例
        features = example_basic_usage()
        
        # 配置使用示例
        features_custom = example_with_config()
        
        # QLib集成示例
        example_qlib_integration()
        
        # 性能基准测试
        performance_results = example_performance_benchmark()
        
        print("\n" + "=" * 80)
        print("示例完成！")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n示例执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())