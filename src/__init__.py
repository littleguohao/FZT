"""
FZT量化选股项目 - 源代码包

包含以下模块：
- fzt_core: FZT核心计算模块（通用函数）
- data_loader: 数据加载模块（统一接口）

作者: MC
创建日期: 2026-03-06
"""

__version__ = "1.0.0"
__author__ = "MC"
__email__ = ""

# 导出FZT核心模块
from .fzt_core import (
    tdx_sma_series,
    tdx_sma,
    calc_brick_pattern_final,
    calculate_fzt_features_vectorized,
    test_fzt_core
)

# 导出数据加载模块
from .data_loader import (
    load_qlib_data_all_instruments,
    load_all_stock_data_bin,
    load_stock_data,
    test_data_loader
)

__all__ = [
    # FZT核心函数
    "tdx_sma_series",
    "tdx_sma",
    "calc_brick_pattern_final",
    "calculate_fzt_features_vectorized",
    "test_fzt_core",
    
    # 数据加载函数
    "load_qlib_data_all_instruments",
    "load_all_stock_data_bin",
    "load_stock_data",
    "test_data_loader"
]