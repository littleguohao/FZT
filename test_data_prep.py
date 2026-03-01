#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据准备模块测试脚本

用于测试data_prep.py模块的基本功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_prep import DataPreprocessor


def test_config_loading():
    """测试配置文件加载"""
    print("测试配置文件加载...")
    try:
        preprocessor = DataPreprocessor()
        print("✓ 配置文件加载成功")
        print(f"  数据源: {preprocessor.config['data']['source']}")
        print(f"  数据路径: {preprocessor.data_path}")
        return True
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return False


def test_data_loading():
    """测试数据加载"""
    print("\n测试数据加载...")
    try:
        preprocessor = DataPreprocessor()
        raw_data = preprocessor.load_raw_data()
        
        print(f"✓ 数据加载成功")
        print(f"  总行数: {raw_data.shape[0]}")
        print(f"  股票数量: {raw_data['code'].nunique()}")
        print(f"  时间范围: {raw_data['date'].min()} 到 {raw_data['date'].max()}")
        print(f"  列名: {list(raw_data.columns)}")
        
        # 显示前几行数据
        print("\n  数据示例:")
        print(raw_data.head(3).to_string())
        
        return True
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False


def test_data_preprocessing():
    """测试数据预处理"""
    print("\n测试数据预处理...")
    try:
        preprocessor = DataPreprocessor()
        raw_data = preprocessor.load_raw_data()
        processed_data = preprocessor.preprocess_data(raw_data)
        
        print(f"✓ 数据预处理成功")
        print(f"  处理后行数: {processed_data.shape[0]}")
        print(f"  处理后列数: {processed_data.shape[1]}")
        
        # 检查数据类型
        print(f"  日期类型: {processed_data['date'].dtype}")
        print(f"  价格列类型: {processed_data['close'].dtype}")
        
        # 检查缺失值
        missing_values = processed_data.isnull().sum().sum()
        print(f"  缺失值数量: {missing_values}")
        
        return True
    except Exception as e:
        print(f"✗ 数据预处理失败: {e}")
        return False


def test_stock_filtering():
    """测试股票筛选"""
    print("\n测试股票筛选...")
    try:
        preprocessor = DataPreprocessor()
        raw_data = preprocessor.load_raw_data()
        processed_data = preprocessor.preprocess_data(raw_data)
        filtered_data = preprocessor.filter_stocks(processed_data)
        
        print(f"✓ 股票筛选成功")
        print(f"  原始股票数量: {processed_data['code'].nunique()}")
        print(f"  筛选后股票数量: {filtered_data['code'].nunique()}")
        print(f"  筛选后行数: {filtered_data.shape[0]}")
        
        return True
    except Exception as e:
        print(f"✗ 股票筛选失败: {e}")
        return False


def test_time_splitting():
    """测试时间划分"""
    print("\n测试时间划分...")
    try:
        preprocessor = DataPreprocessor()
        raw_data = preprocessor.load_raw_data()
        processed_data = preprocessor.preprocess_data(raw_data)
        filtered_data = preprocessor.filter_stocks(processed_data)
        split_data = preprocessor.split_by_time(filtered_data)
        
        print(f"✓ 时间划分成功")
        for split_name, data in split_data.items():
            if not data.empty:
                print(f"  {split_name}: {data.shape[0]} 行, {data['code'].nunique()} 只股票")
                print(f"    时间范围: {data['date'].min().date()} 到 {data['date'].max().date()}")
        
        return True
    except Exception as e:
        print(f"✗ 时间划分失败: {e}")
        return False


def test_pipeline():
    """测试完整管道"""
    print("\n测试完整数据处理管道...")
    try:
        preprocessor = DataPreprocessor()
        split_data = preprocessor.run_pipeline()
        
        print(f"✓ 完整管道运行成功")
        
        # 生成报告
        report = preprocessor.generate_data_report(split_data)
        print("\n数据质量报告摘要:")
        for line in report.split('\n')[:20]:  # 只显示前20行
            print(line)
        
        return True
    except Exception as e:
        print(f"✗ 完整管道运行失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 80)
    print("数据准备模块测试")
    print("=" * 80)
    
    tests = [
        ("配置文件加载", test_config_loading),
        ("数据加载", test_data_loading),
        ("数据预处理", test_data_preprocessing),
        ("股票筛选", test_stock_filtering),
        ("时间划分", test_time_splitting),
        ("完整管道", test_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except KeyboardInterrupt:
            print("\n测试被用户中断")
            return 1
        except Exception as e:
            print(f"测试 {test_name} 发生异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("测试结果汇总:")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name:20} {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n所有测试通过！数据准备模块功能正常。")
        return 0
    else:
        print("\n部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())