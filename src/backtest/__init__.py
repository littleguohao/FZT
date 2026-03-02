#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测模块 - 完整的策略回测系统

包含：
1. backtest_engine.py - 回测引擎
2. performance_evaluator.py - 绩效评估
3. report_generator.py - 报告生成
4. cost_model.py - 交易成本模型
5. enhanced_strategy.py - 增强回测策略

作者: FZT项目组
创建日期: 2026-03-01
更新日期: 2026-03-02
"""

from .backtest_engine import (
    BacktestEngine, BacktestConfig, SignalType, PositionStatus,
    Trade, Position
)
from .performance_evaluator import PerformanceEvaluator
from .report_generator import ReportGenerator
from .cost_model import CostModel
from .enhanced_strategy import (
    FZTEnhancedStrategy,
    load_config_from_yaml,
    create_sample_data
)

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "SignalType",
    "PositionStatus",
    "Trade",
    "Position",
    "PerformanceEvaluator",
    "ReportGenerator",
    "CostModel",
    "FZTEnhancedStrategy",
    "load_config_from_yaml",
    "create_sample_data"
]

__version__ = "1.2.0"