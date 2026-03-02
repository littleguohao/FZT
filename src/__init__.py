"""
FZT量化选股项目 - 源代码包

包含以下模块：
- data_prep: 数据准备模块
- fzt_formula: FZT选股公式计算模块
- model_train: 模型训练模块（待实现）
- backtest: 回测模块（待实现）
- feature_eng: 特征工程模块（待实现）
"""

__version__ = "0.1.0"
__author__ = "FZT项目组"
__email__ = "fzt-project@example.com"

# 导出主要类
from .data_prep import DataPreprocessor
from .fzt_formula import FZTFormula, calculate_target, create_sample_data
from .fzt_brick_formula import FZTBrickFormula
from .feature_eng import FeatureEngineer
from .model_train import ModelTrainer
from .evaluate import ModelEvaluator
from .hybrid_data_processor import HybridDataProcessor

# 导出回测模块
from .backtest import (
    BacktestEngine, BacktestConfig, SignalType, PositionStatus,
    Trade, Position, PerformanceEvaluator, ReportGenerator
)

__all__ = [
    "DataPreprocessor",
    "FZTFormula",
    "FZTBrickFormula",
    "FeatureEngineer",
    "ModelTrainer",
    "ModelEvaluator",
    "HybridDataProcessor",
    "BacktestEngine",
    "BacktestConfig",
    "SignalType",
    "PositionStatus",
    "Trade",
    "Position",
    "PerformanceEvaluator",
    "ReportGenerator",
    "calculate_target",
    "create_sample_data"
]