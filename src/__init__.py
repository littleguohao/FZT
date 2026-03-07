"""
FZT量化选股项目 - 源代码包

包含以下模块：
- fzt_core: FZT核心计算模块（通用函数）
- zsqsx_core: ZSQSX核心计算模块（通用函数）
- data_loader: 公共数据加载模块（不包含私有数据路径）

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

# 导出ZSQSX核心模块
from .zsqsx_core import (
    calc_zsdkx,
    calculate_zsdkx_features_vectorized,
    get_zsdkx_signal_conditions,
    analyze_zsdkx_performance,
    test_zsdkx_core
)

# 导出公共数据加载模块
from .data_loader import (
    load_stock_data_qlib,
    get_instruments_from_file,
    test_public_loader
)

__all__ = [
    # FZT核心函数
    "tdx_sma_series",
    "tdx_sma",
    "calc_brick_pattern_final",
    "calculate_fzt_features_vectorized",
    "test_fzt_core",
    
    # ZSQSX核心函数
    "calc_zsdkx",
    "calculate_zsdkx_features_vectorized",
    "get_zsdkx_signal_conditions",
    "analyze_zsdkx_performance",
    "test_zsdkx_core",
    
    # 公共数据加载函数
    "load_stock_data_qlib",
    "get_instruments_from_file",
    "test_public_loader"
]