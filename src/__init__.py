"""
FZT量化选股项目 - 源代码包

包含以下模块：
- fzt_core: FZT核心计算模块（通用函数）

作者: MC
创建日期: 2026-03-06
"""

__version__ = "1.0.0"
__author__ = "MC"
__email__ = ""

# 导出主要模块
from .fzt_core import (
    tdx_sma_series,
    tdx_sma,
    calc_brick_pattern_final,
    calculate_fzt_features_vectorized,
    test_fzt_core
)

__all__ = [
    "tdx_sma_series",
    "tdx_sma",
    "calc_brick_pattern_final",
    "calculate_fzt_features_vectorized",
    "test_fzt_core"
]