#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据准备模块使用示例

展示如何使用data_prep.py模块进行数据处理
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_prep import DataPreprocessor


def example_basic_usage():
    """基本使用示例"""
    print("=" * 80)
    print("数据准备模块 - 基本使用示例")
    print("=" * 80)
    
    # 1. 初始化数据预处理器
    print("\n1. 初始化数据预处理器...")
    preprocessor = DataPreprocessor()
    print(f"   数据源: {preprocessor.config['data']['source']}")
    print(f"   数据路径: {preprocessor.data_path}")
    print(f"   输出目录: {preprocessor.output_dir}")
    
    # 2. 加载原始数据
    print("\n2. 加载原始数据...")
    raw_data = preprocessor.load_raw_data()
    print(f"   原始数据行数: {raw_data.shape[0]:,}")
    print(f"   股票数量: {raw_data['code'].nunique():,}")
    print(f"   时间范围: {raw_data['date'].min()} 到 {raw_data['date'].max()}")
    
    # 3. 数据预处理
    print("\n3. 数据预处理...")
    processed_data = preprocessor.preprocess_data(raw_data)
    print(f"   处理后数据行数: {processed_data.shape[0]:,}")
    print(f"   处理后列数: {processed_data.shape[1]}")
    
    # 4. 股票筛选
    print("\n4. 股票筛选...")
    filtered_data = preprocessor.filter_stocks(processed_data)
    print(f"   筛选后股票数量: {filtered_data['code'].nunique():,}")
    print(f"   筛选后数据行数: {filtered_data.shape[0]:,}")
    
    # 5. 时间划分
    print("\n5. 时间划分...")
    split_data = preprocessor.split_by_time(filtered_data)
    
    for split_name, data in split_data.items():
        if not data.empty:
            print(f"   {split_name}: {data.shape[0]:,} 行, {data['code'].nunique():,} 只股票")
            print(f"     时间范围: {data['date'].min().date()} 到 {data['date'].max().date()}")
    
    # 6. 生成数据报告
    print("\n6. 生成数据质量报告...")
    report = preprocessor.generate_data_report(split_data)
    
    # 只显示报告的一部分
    print("\n数据质量报告（摘要）:")
    lines = report.split('\n')
    for line in lines[:30]:  # 显示前30行
        print(line)
    
    return split_data


def example_pipeline_usage():
    """管道使用示例"""
    print("\n" + "=" * 80)
    print("数据准备模块 - 管道使用示例")
    print("=" * 80)
    
    # 使用管道模式（推荐）
    print("\n使用管道模式处理数据...")
    
    try:
        preprocessor = DataPreprocessor()
        split_data = preprocessor.run_pipeline()
        
        print("✓ 数据处理管道运行成功")
        print(f"  输出文件保存在: {preprocessor.output_dir}")
        
        # 列出生成的文件
        print("\n生成的文件:")
        for file_path in preprocessor.output_dir.glob("*"):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  - {file_path.name} ({size_mb:.2f} MB)")
        
        return split_data
        
    except Exception as e:
        print(f"✗ 管道运行失败: {e}")
        return None


def example_custom_config():
    """自定义配置示例"""
    print("\n" + "=" * 80)
    print("数据准备模块 - 自定义配置示例")
    print("=" * 80)
    
    # 创建自定义配置文件
    custom_config = """
data:
  source: "local_csv"
  local_csv:
    enabled: true
    data_path: "/Users/lucky/Downloads/O_DATA/"
    file_pattern: "*-all-latest.csv"
    field_mapping:
      date: "Date"
      code: "Code"
      open: "Open"
      high: "High"
      low: "Low"
      close: "Close"
      volume: "Volume"
      amount: "Amount"

time_ranges:
  local_csv:
    data_start: "2021-08-01"
    data_end: "2026-02-06"
    train_start: "2021-08-01"
    train_end: "2024-12-31"   # 自定义训练集结束时间
    valid_start: "2025-01-01"
    valid_end: "2025-06-30"   # 自定义验证集时间
    test_start: "2025-07-01"
    test_end: "2026-02-06"    # 自定义测试集时间

processing:
  normalize: true
  fill_na: true
  remove_outliers: true

stock_filter:
  min_days: 30  # 减少最小交易日数要求
  exclude_st: true
  min_price: 0.5  # 降低最低价格要求
  max_price: 2000.0  # 提高最高价格限制
"""
    
    # 保存自定义配置
    custom_config_path = Path("custom_data_config.yaml")
    with open(custom_config_path, 'w', encoding='utf-8') as f:
        f.write(custom_config)
    
    print(f"自定义配置文件已创建: {custom_config_path}")
    
    # 使用自定义配置
    try:
        preprocessor = DataPreprocessor(str(custom_config_path))
        print("✓ 自定义配置加载成功")
        
        # 运行管道
        split_data = preprocessor.run_pipeline()
        
        print("✓ 使用自定义配置的数据处理完成")
        
        # 清理临时文件
        custom_config_path.unlink()
        
        return split_data
        
    except Exception as e:
        print(f"✗ 自定义配置使用失败: {e}")
        if custom_config_path.exists():
            custom_config_path.unlink()
        return None


def main():
    """主函数"""
    print("FZT项目 - 数据准备模块使用示例")
    print("本示例展示如何使用data_prep.py模块进行数据处理")
    print()
    
    # 示例1：基本使用
    example_basic_usage()
    
    # 示例2：管道使用
    example_pipeline_usage()
    
    # 示例3：自定义配置（可选）
    # example_custom_config()
    
    print("\n" + "=" * 80)
    print("示例执行完成")
    print("=" * 80)
    print("\n更多功能请参考:")
    print("1. src/data_prep.py - 数据准备模块源代码")
    print("2. test_data_prep.py - 测试脚本")
    print("3. config/data_config.yaml - 配置文件")
    
    return 0


if __name__ == "__main__":
    # 创建examples目录
    examples_dir = Path(__file__).parent
    examples_dir.mkdir(exist_ok=True)
    
    sys.exit(main())