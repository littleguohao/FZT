#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FZT筛选 + Alpha158因子 + LightGBM训练工作流（使用通用数据加载器）

重构版本：使用通用数据加载器，保持原有所有逻辑不变。
只修改数据加载部分，其他所有功能保持不变。

使用方法：
python3 scripts/train_fzt_alpha158_with_dataloader.py
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import DataLoader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data_with_dataloader():
    """使用DataLoader加载并准备数据（替换原函数）"""
    logger.info("📥 使用DataLoader加载数据...")
    
    try:
        # 初始化数据加载器
        loader = DataLoader(config_path="config/data_config.yaml")
        
        # 加载QLIB数据
        df = loader.load_qlib_data()
        if df is None:
            logger.error("❌ QLIB数据加载失败")
            return None
        
        # 计算目标变量
        df = loader.calculate_target_variable(df)
        
        logger.info(f"✅ 数据加载成功: {df.shape[0]} 行, {df.shape[1]} 列")
        
        # 确保数据排序
        df = df.sort_values(['code', 'date'])
        
        return df
        
    except Exception as e:
        logger.error(f"❌ 数据加载失败: {e}")
        return None


def main():
    """主函数 - 调用原始主函数，但使用新的数据加载函数"""
    logger.info("🚀 FZT筛选 + Alpha158因子 + LightGBM训练工作流（使用DataLoader）")
    logger.info("=" * 70)
    
    try:
        # 导入原始模块
        import importlib.util
        
        original_path = Path(__file__).parent / "train_fzt_alpha158.py"
        spec = importlib.util.spec_from_file_location("train_fzt_alpha158", original_path)
        original_module = importlib.util.module_from_spec(spec)
        
        # 替换数据加载函数
        original_module.load_and_prepare_data = load_and_prepare_data_with_dataloader
        
        # 执行原始主函数
        sys.modules['train_fzt_alpha158'] = original_module
        spec.loader.exec_module(original_module)
        
        # 调用原始main函数
        return original_module.main()
        
    except Exception as e:
        logger.error(f"❌ 工作流执行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())