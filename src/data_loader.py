#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一数据加载模块 - 使用QLIB API加载所有数据

基于发现：自定义.bin文件也兼容QLIB的D.features() API
因此可以统一使用QLIB API加载两种数据源

作者: MC
创建日期: 2026-03-07
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


def load_stock_data_qlib(
    data_dir: str,
    instruments: List[str],
    calc_start: str,
    calc_end: str,
    target_start: Optional[str] = None,
    target_end: Optional[str] = None,
    fields: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    使用QLIB API统一加载股票数据（支持标准QLIB和自定义.bin格式）
    
    :param data_dir: 数据目录路径
    :param instruments: 股票代码列表
    :param calc_start: 计算开始日期（需要提前一些以计算指标）
    :param calc_end: 计算结束日期
    :param target_start: 目标开始日期（实际回测开始，可选）
    :param target_end: 目标结束日期（实际回测结束，可选）
    :param fields: 字段列表，默认包含所有价格和成交量字段
    :return: 包含所有股票数据的DataFrame，失败返回None
    """
    if fields is None:
        fields = ['$close', '$high', '$low', '$open', '$volume', '$factor']
    
    try:
        import qlib
        from qlib.data import D
        from qlib.config import REG_CN
        
        print(f"📥 从 {data_dir} 加载 {len(instruments)} 只股票数据...")
        print(f"   计算时间: {calc_start} 到 {calc_end}")
        if target_start and target_end:
            print(f"   回测时间: {target_start} 到 {target_end}")
        
        load_start = time.time()
        
        # 初始化QLIB（使用指定数据目录）
        qlib.init(provider_uri=data_dir, region=REG_CN)
        
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
        
        # 筛选目标时间段（如果指定）
        if target_start:
            target_start_dt = pd.Timestamp(target_start)
            df_reset = df_reset[df_reset['datetime'] >= target_start_dt]
        if target_end:
            target_end_dt = pd.Timestamp(target_end)
            df_reset = df_reset[df_reset['datetime'] <= target_end_dt]
        
        load_time = time.time() - load_start
        print(f"✅ 数据加载完成，耗时: {load_time:.2f} 秒")
        print(f"   总数据行数: {len(df_reset)}")
        print(f"   时间范围: {df_reset['datetime'].min()} 到 {df_reset['datetime'].max()}")
        
        return df_reset
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_instruments_from_file(instruments_file: str) -> List[str]:
    """
    从instruments文件读取股票代码列表
    
    :param instruments_file: instruments文件路径
    :return: 股票代码列表
    """
    instruments = []
    
    try:
        with open(instruments_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 1:
                        instruments.append(parts[0])
        
        print(f"✅ 从 {instruments_file} 读取到 {len(instruments)} 只股票")
        if instruments:
            print(f"   股票示例: {instruments[:5]}...")
        
        return instruments
        
    except Exception as e:
        print(f"❌ 读取instruments文件失败: {e}")
        return []


def load_2006_2020_data(
    project_root: Path,
    calc_start: str = '2005-10-01',
    calc_end: str = '2020-09-25',
    target_start: str = '2006-01-01',
    target_end: str = '2020-09-25'
) -> Optional[pd.DataFrame]:
    """
    加载2006-2020年数据（QLIB标准格式）
    """
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
        calc_start=calc_start,
        calc_end=calc_end,
        target_start=target_start,
        target_end=target_end
    )


def load_2021_2026_data(
    project_root: Path,
    calc_start: str = '2021-08-02',
    calc_end: str = '2026-02-06',
    target_start: str = '2021-08-02',
    target_end: str = '2026-02-06'
) -> Optional[pd.DataFrame]:
    """
    加载2021-2026年数据（自定义.bin格式，但兼容QLIB API）
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
        calc_start=calc_start,
        calc_end=calc_end,
        target_start=target_start,
        target_end=target_end
    )


# ===================== 测试函数 =====================
def test_unified_loader():
    """测试统一数据加载模块"""
    print("🧪 测试统一数据加载模块...")
    
    project_root = Path(__file__).parent.parent
    
    # 测试2006-2020年数据加载
    print("\n1. 测试2006-2020年数据加载...")
    try:
        df_2006 = load_2006_2020_data(project_root)
        if df_2006 is not None:
            print(f"   ✅ 加载成功，数据形状: {df_2006.shape}")
        else:
            print("   ❌ 加载失败")
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
    
    # 测试2021-2026年数据加载
    print("\n2. 测试2021-2026年数据加载...")
    try:
        df_2021 = load_2021_2026_data(project_root)
        if df_2021 is not None:
            print(f"   ✅ 加载成功，数据形状: {df_2021.shape}")
        else:
            print("   ❌ 加载失败")
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
    
    print("\n✅ 统一数据加载模块测试完成")
    return True


if __name__ == "__main__":
    test_unified_loader()