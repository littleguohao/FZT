#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行FZT因子发现流程

功能：
1. 使用扩展数据训练FZT模型
2. 实现因子发现功能
3. 使用指定参数优化训练

作者: FZT项目组
创建日期: 2026-03-02
"""

import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, str(Path.cwd()))


def run_fzt_with_expanded_data():
    """使用扩展数据运行FZT训练"""
    print("=" * 100)
    print("🚀 步骤1: 使用扩展数据训练FZT模型")
    print("=" * 100)
    
    try:
        # 导入训练器
        from train_fzt_optimized import FZTOptimizedTrainer
        
        trainer = FZTOptimizedTrainer()
        result = trainer.run_optimized_training()
        
        if result == 0:
            print("✅ 扩展数据FZT训练成功")
            return True
        else:
            print("❌ 扩展数据FZT训练失败")
            return False
            
    except Exception as e:
        logger.error(f"扩展数据FZT训练失败: {e}")
        return False


def run_factor_discovery():
    """运行因子发现流程"""
    print("\n" + "=" * 100)
    print("🔍 步骤2: 运行因子发现功能")
    print("=" * 100)
    
    try:
        # 检查是否有FZT特征数据
        fzt_features_path = Path("results/fzt_optimized/fzt_features_expanded.csv")
        
        if not fzt_features_path.exists():
            print("❌ FZT特征数据不存在，请先运行FZT训练")
            return False
        
        # 加载FZT特征数据
        print("加载FZT特征数据...")
        fzt_features = pd.read_csv(fzt_features_path)
        
        # 加载目标数据
        target_data_path = Path("results/expanded_stock_data/expanded_data_with_target.csv")
        target_data = pd.read_csv(target_data_path)
        
        # 合并数据
        merged_data = pd.merge(
            fzt_features,
            target_data[['date', 'code', 'target']],
            on=['date', 'code'],
            how='inner'
        )
        
        # 分离特征和目标
        exclude_cols = ['date', 'code', 'target']
        feature_cols = [col for col in merged_data.columns if col not in exclude_cols]
        
        X = merged_data[feature_cols].copy()
        y = merged_data['target'].copy()
        
        # 移除缺失值
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"✅ 数据加载完成")
        print(f"   特征数量: {X.shape[1]}")
        print(f"   样本数量: {X.shape[0]:,}")
        print()
        
        # 运行因子发现
        from src.factor_discovery import FactorDiscoverer
        
        discoverer = FactorDiscoverer()
        results = discoverer.run_discovery_pipeline(X, y)
        
        if results:
            print("✅ 因子发现流程成功")
            return True
        else:
            print("❌ 因子发现流程失败")
            return False
            
    except Exception as e:
        logger.error(f"因子发现流程失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🎯 FZT项目完整优化流程")
    print("=" * 100)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 需要导入pandas
    import pandas as pd
    
    # 步骤1: 使用扩展数据训练FZT模型
    step1_success = run_fzt_with_expanded_data()
    
    if not step1_success:
        print("❌ 步骤1失败，无法继续")
        return 1
    
    # 步骤2: 运行因子发现
    step2_success = run_factor_discovery()
    
    if not step2_success:
        print("⚠️  步骤2失败，但步骤1已完成")
    
    print("\n" + "=" * 100)
    print("🎉 FZT项目优化流程完成！")
    print("=" * 100)
    
    if step1_success and step2_success:
        print("✅ 所有步骤成功完成")
        print("\n📁 生成的文件:")
        print("   results/fzt_optimized/ - FZT优化模型和结果")
        print("   results/factor_discovery/ - 因子发现结果")
    elif step1_success:
        print("✅ 步骤1完成，步骤2部分完成")
        print("\n📁 生成的文件:")
        print("   results/fzt_optimized/ - FZT优化模型和结果")
    else:
        print("❌ 流程失败")
    
    print("\n🚀 下一步建议:")
    print("   1. 检查优化模型的选股成功率")
    print("   2. 查看发现的有效因子")
    print("   3. 进一步调优参数")
    print("   4. 测试在最新数据上的表现")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())