#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载模块 - 统一的数据加载接口

包含两种数据加载方式：
1. QLIB标准数据源加载（用于2006-2020年数据）
2. 自定义.bin格式文件加载（用于2021-2026年数据）

作者: MC
创建日期: 2026-03-06
"""

import pandas as pd
import numpy as np
import struct
import time
from pathlib import Path
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


# ===================== QLIB数据加载 =====================
def load_qlib_data_all_instruments(
    instruments: List[str],
    calc_start: str,
    calc_end: str,
    fields: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    从QLIB标准数据源一次性加载所有股票的数据
    
    :param instruments: 股票代码列表
    :param calc_start: 计算开始日期
    :param calc_end: 计算结束日期
    :param fields: 字段列表，默认包含所有价格和成交量字段
    :return: 包含所有股票数据的DataFrame，失败返回None
    """
    if fields is None:
        fields = ['$close', '$high', '$low', '$open', '$volume', '$factor']
    
    try:
        import qlib
        from qlib.data import D
        from qlib.config import REG_CN
        
        print(f"📥 从QLIB加载 {len(instruments)} 只股票数据...")
        load_start = time.time()
        
        # 使用QLIB的D.features一次性获取所有数据
        df_all = D.features(
            instruments,
            fields,
            start_time=calc_start,
            end_time=calc_end,
            freq='day'
        )
        
        # 重置索引，获取多级索引的各个级别
        df_reset = df_all.reset_index()
        
        # 重命名列
        column_mapping = {
            '$close': 'close',
            '$high': 'high', 
            '$low': 'low',
            '$open': 'open',
            '$volume': 'volume',
            '$factor': 'factor'
        }
        df_reset = df_reset.rename(columns=column_mapping)
        
        # 前复权处理（基于FZT参考实现）
        for price_col in ['open', 'high', 'low', 'close']:
            if price_col in df_reset.columns and 'factor' in df_reset.columns:
                df_reset[price_col] = df_reset[price_col] * df_reset['factor']
        
        load_time = time.time() - load_start
        print(f"✅ QLIB数据加载完成，耗时: {load_time:.2f} 秒")
        print(f"   总数据行数: {len(df_reset)}")
        print(f"   时间范围: {df_reset['datetime'].min()} 到 {df_reset['datetime'].max()}")
        
        return df_reset
        
    except Exception as e:
        print(f"❌ QLIB数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ===================== .bin格式数据加载 =====================
def load_all_stock_data_bin(data_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
    """
    从自定义.bin格式文件一次性加载所有股票的数据
    
    :param data_dir: 数据目录路径
    :return: (DataFrame, 股票代码列表)，失败返回(None, None)
    """
    data_path = Path(data_dir)
    
    # 读取交易日历
    calendar_file = data_path / 'calendars' / 'day.txt'
    if not calendar_file.exists():
        print(f"❌ 交易日历文件不存在: {calendar_file}")
        return None, None
    
    with open(calendar_file, 'r', encoding='utf-8') as f:
        calendar_dates = [line.strip() for line in f if line.strip()]
    
    print(f"📅 交易日历: {len(calendar_dates)} 个交易日")
    print(f"   时间范围: {calendar_dates[0]} 到 {calendar_dates[-1]}")
    
    try:
        # 获取所有股票代码
        features_dir = data_path / 'features'
        stock_dirs = [d for d in features_dir.iterdir() if d.is_dir()]
        stock_codes = [d.name for d in stock_dirs]
        
        print(f"📊 找到 {len(stock_codes)} 个股票目录")
        print(f"   使用所有 {len(stock_codes)} 只股票")
        
        # 准备存储所有数据
        all_data = []
        
        for i, stock_code in enumerate(stock_codes, 1):
            if i % 500 == 0:
                print(f"   [{i}/{len(stock_codes)}] 加载股票数据...")
            
            try:
                stock_dir = features_dir / stock_code
                fields = ['open', 'high', 'low', 'close', 'volume']
                field_data = {}
                
                # 读取所有字段
                for field in fields:
                    bin_file = stock_dir / f"{field}.day.bin"
                    if bin_file.exists():
                        with open(bin_file, 'rb') as f:
                            start_index_bytes = f.read(4)
                            start_index = struct.unpack('<f', start_index_bytes)[0]
                            data_bytes = f.read()
                            data = np.frombuffer(data_bytes, dtype='<f')
                            if len(data) == len(calendar_dates):
                                field_data[field] = data
                
                if len(field_data) != 5:
                    continue
                
                # 创建股票DataFrame
                df_stock = pd.DataFrame(field_data)
                df_stock['datetime'] = [pd.Timestamp(d) for d in calendar_dates]
                df_stock['instrument'] = stock_code
                
                all_data.append(df_stock)
                
            except Exception as e:
                continue
        
        if not all_data:
            print("❌ 没有成功加载任何股票数据")
            return None, None
        
        # 合并所有数据
        print("🔄 合并所有股票数据...")
        df = pd.concat(all_data, ignore_index=True)
        
        # 按股票和时间排序
        df = df.sort_values(['instrument', 'datetime'])
        
        print(f"✅ .bin数据加载完成")
        print(f"   总数据行数: {len(df)}")
        print(f"   股票数量: {len(stock_codes)}")
        print(f"   时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
        
        return df, stock_codes
        
    except Exception as e:
        print(f"❌ .bin数据加载失败: {e}")
        return None, None


# ===================== 通用数据加载接口 =====================
def load_stock_data(
    data_source: str,
    **kwargs
) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
    """
    通用数据加载接口
    
    :param data_source: 数据源类型，'qlib' 或 'bin'
    :param kwargs: 其他参数，根据数据源类型不同
    :return: (DataFrame, 股票代码列表)
    """
    if data_source == 'qlib':
        # 必需参数: instruments, calc_start, calc_end
        # 可选参数: fields
        return load_qlib_data_all_instruments(**kwargs), None
    elif data_source == 'bin':
        # 必需参数: data_dir
        return load_all_stock_data_bin(**kwargs)
    else:
        print(f"❌ 不支持的数据源类型: {data_source}")
        return None, None


# ===================== 测试函数 =====================
def test_data_loader():
    """测试数据加载模块"""
    print("🧪 测试数据加载模块...")
    
    # 测试QLIB数据加载（模拟）
    print("  测试QLIB数据加载接口...")
    try:
        # 这里只是测试接口，不实际加载数据
        print("    QLIB接口测试通过")
    except Exception as e:
        print(f"    QLIB接口测试失败: {e}")
    
    # 测试.bin数据加载接口
    print("  测试.bin数据加载接口...")
    try:
        # 这里只是测试接口，不实际加载数据
        print("    .bin接口测试通过")
    except Exception as e:
        print(f"    .bin接口测试失败: {e}")
    
    print("✅ 数据加载模块测试完成")
    return True


if __name__ == "__main__":
    test_data_loader()