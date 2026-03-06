"""
FZT量化选股项目 - 源代码包

包含以下模块：
- fzt_core: FZT核心计算模块（通用函数）
- data_loader: 统一数据加载模块（使用QLIB API）

作者: MC
创建日期: 2026-03-07
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

# 导出统一数据加载模块
from .data_loader import (
    load_stock_data_qlib,
    get_instruments_from_file,
    load_2006_2020_data,
    load_2021_2026_data,
    test_unified_loader
)

__all__ = [
    # FZT核心函数
    "tdx_sma_series",
    "tdx_sma",
    "calc_brick_pattern_final",
    "calculate_fzt_features_vectorized",
    "test_fzt_core",
    
    # 统一数据加载函数
    "load_stock_data_qlib",
    "get_instruments_from_file",
    "load_2006_2020_data",
    "load_2021_2026_data",
    "test_unified_loader"
]