#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FZT排序增强策略 - 扩展数据训练脚本

使用190只股票的扩展数据进行模型训练

作者：FZT项目组
创建日期：2026年3月2日
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import yaml
import joblib
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_expanded_data():
    """加载扩展数据"""
    data_path = Path("results/expanded_stock_data/expanded_data_with_target.csv")
    
    if not data_path.exists():
        logger.error(f"❌ 扩展数据文件不存在: {data_path}")
        return None
    
    logger.info(f"📊 加载扩展数据: {data_path}")
    
    try:
        # 加载数据
        df = pd.read_csv(data_path)
        logger.info(f"✅ 数据加载成功: {df.shape[0]} 行, {df.shape[1]} 列")
        
        # 显示基本信息
        logger.info(f"📈 数据基本信息:")
        logger.info(f"  股票数量: {df['code'].nunique()}")
        logger.info(f"  日期范围: {df['date'].min()} 到 {df['date'].max()}")
        logger.info(f"  总样本数: {len(df)}")
        
        # 检查目标变量
        target_col = None
        for col in ['target', 'target_t1', 'future_return']:
            if col in df.columns:
                target_col = col
                break
        
        if target_col:
            target_series = df[target_col]
            # 如果是连续值，先转换为二元
            if target_series.dtype in ['float64', 'float32']:
                target_binary = (target_series > 0).astype(int)
                pos_ratio = target_binary.mean()
            else:
                pos_ratio = (target_series == 1).mean()
            
            logger.info(f"🎯 目标变量分布 ({target_col}):")
            logger.info(f"  正类比例 (上涨): {pos_ratio:.2%}")
            logger.info(f"  负类比例 (下跌): {1-pos_ratio:.2%}")
        else:
            logger.warning("⚠️  数据中无目标列")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ 数据加载失败: {e}")
        return None


def prepare_training_data(df):
    """准备训练数据"""
    logger.info("🔧 准备训练数据...")
    
    try:
        # 检查必要的列
        required_cols = ['code', 'date', 'close', 'volume', 'open', 'high', 'low']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"❌ 缺少必要列: {missing_cols}")
            return None, None, None
        
        # 检查目标列
        target_col = None
        for col in ['target', 'target_t1', 'future_return']:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            logger.error("❌ 未找到目标列")
            return None, None, None
        
        logger.info(f"🎯 使用目标列: {target_col}")
        
        # 提取特征列（排除非特征列）
        exclude_cols = ['code', 'date', target_col, 'next_close', 'future_return']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"📊 特征数量: {len(feature_cols)}")
        logger.info(f"📊 样本数量: {len(df)}")
        
        # 分割特征和目标
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # 如果目标是连续值，转换为二元分类
        if y.dtype in ['float64', 'float32']:
            logger.info("🎯 连续目标转换为二元分类")
            y = (y > 0).astype(int)
        
        # 检查数据质量
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"⚠️  特征中存在 {nan_count} 个NaN值，将进行填充")
            X = X.fillna(X.mean())
        
        # 检查目标变量
        unique_targets = y.unique()
        logger.info(f"🎯 目标变量唯一值: {unique_targets}")
        
        # 确保目标变量是整数
        y = y.astype(int)
        
        logger.info("✅ 训练数据准备完成")
        return X, y, feature_cols
        
    except Exception as e:
        logger.error(f"❌ 数据准备失败: {e}")
        return None, None, None


def train_fzt_model(X, y, feature_cols):
    """训练FZT模型"""
    logger.info("🤖 训练FZT模型...")
    
    try:
        # 尝试导入LightGBM
        try:
            import lightgbm as lgb
            LIGHTGBM_AVAILABLE = True
            logger.info("✅ LightGBM可用")
        except ImportError:
            LIGHTGBM_AVAILABLE = False
            logger.warning("⚠️  LightGBM不可用，将使用随机森林")
        
        if LIGHTGBM_AVAILABLE:
            # 使用LightGBM
            from sklearn.model_selection import train_test_split
            
            # 分割训练集和验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"📊 数据分割:")
            logger.info(f"  训练集: {X_train.shape[0]} 样本")
            logger.info(f"  验证集: {X_val.shape[0]} 样本")
            
            # 创建LightGBM数据集
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # 参数配置
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': 5,
                'seed': 42,
                'n_estimators': 100,
                'verbose': -1
            }
            
            # 训练模型
            logger.info("开始训练LightGBM模型...")
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(20)]
            )
            
            # 在验证集上评估
            y_pred = model.predict(X_val)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            from sklearn.metrics import accuracy_score, classification_report
            accuracy = accuracy_score(y_val, y_pred_binary)
            
            # 计算选股成功率
            selected_mask = y_pred_binary == 1
            if selected_mask.sum() > 0:
                selection_success_rate = (y_val[selected_mask] == 1).mean()
            else:
                selection_success_rate = 0
            
            logger.info(f"📈 模型性能:")
            logger.info(f"  准确率: {accuracy:.4f}")
            logger.info(f"  选股成功率: {selection_success_rate:.4f}")
            logger.info(f"  选股比例: {selected_mask.mean():.2%}")
            
            # 特征重要性
            importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importance()
            }).sort_values('importance', ascending=False)
            
            logger.info("🔝 Top 10 特征重要性:")
            for i, row in importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            return model, {
                'accuracy': accuracy,
                'selection_success_rate': selection_success_rate,
                'selection_ratio': selected_mask.mean(),
                'feature_importance': importance
            }
            
        else:
            # 使用随机森林作为备选
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            # 分割数据
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 训练随机森林
            logger.info("开始训练随机森林模型...")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # 评估
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            # 计算选股成功率
            selected_mask = y_pred == 1
            if selected_mask.sum() > 0:
                selection_success_rate = (y_val[selected_mask] == 1).mean()
            else:
                selection_success_rate = 0
            
            logger.info(f"📈 模型性能:")
            logger.info(f"  准确率: {accuracy:.4f}")
            logger.info(f"  选股成功率: {selection_success_rate:.4f}")
            
            # 特征重要性
            importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("🔝 Top 10 特征重要性:")
            for i, row in importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            return model, {
                'accuracy': accuracy,
                'selection_success_rate': selection_success_rate,
                'selection_ratio': selected_mask.mean(),
                'feature_importance': importance
            }
            
    except Exception as e:
        logger.error(f"❌ 模型训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None


def save_results(model, evaluation, feature_cols):
    """保存结果"""
    try:
        # 创建结果目录
        results_dir = Path("results") / "expanded_training"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存模型
        model_path = results_dir / f"fzt_expanded_model_{timestamp}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"💾 模型已保存: {model_path}")
        
        # 保存评估结果
        eval_path = results_dir / f"evaluation_results_{timestamp}.yaml"
        eval_data = {
            'accuracy': float(evaluation['accuracy']),
            'selection_success_rate': float(evaluation['selection_success_rate']),
            'selection_ratio': float(evaluation['selection_ratio']),
            'num_features': len(feature_cols),
            'model_type': model.__class__.__name__
        }
        
        with open(eval_path, 'w', encoding='utf-8') as f:
            yaml.dump(eval_data, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"📊 评估结果已保存: {eval_path}")
        
        # 保存特征重要性
        imp_path = results_dir / f"feature_importance_{timestamp}.csv"
        evaluation['feature_importance'].to_csv(imp_path, index=False, encoding='utf-8')
        logger.info(f"🔍 特征重要性已保存: {imp_path}")
        
        # 生成报告
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'data_source': 'expanded_stock_data',
                'num_stocks': 190,
                'num_samples': '~580,000',
                'model_type': model.__class__.__name__
            },
            'evaluation': eval_data,
            'files': {
                'model': str(model_path),
                'evaluation': str(eval_path),
                'feature_importance': str(imp_path)
            }
        }
        
        report_path = results_dir / f"training_report_{timestamp}.yaml"
        with open(report_path, 'w', encoding='utf-8') as f:
            yaml.dump(report, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"📋 训练报告已保存: {report_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 结果保存失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("🎯 FZT排序增强策略 - 扩展数据训练")
    logger.info("=" * 60)
    
    # 1. 加载扩展数据
    df = load_expanded_data()
    if df is None:
        return 1
    
    # 2. 准备训练数据
    X, y, feature_cols = prepare_training_data(df)
    if X is None:
        return 1
    
    # 3. 训练FZT模型
    result = train_fzt_model(X, y, feature_cols)
    if result[0] is None:
        return 1
    
    model, evaluation = result
    
    # 4. 保存结果
    if not save_results(model, evaluation, feature_cols):
        return 1
    
    # 5. 打印摘要
    print("\n" + "="*60)
    print("🎉 FZT扩展数据训练完成！")
    print("="*60)
    print(f"📊 数据统计:")
    print(f"  • 股票数量: {df['code'].nunique()}")
    print(f"  • 特征数量: {len(feature_cols)}")
    print(f"  • 样本数量: {len(df)}")
    print(f"  • 数据期间: {df['date'].min()} 到 {df['date'].max()}")
    print(f"\n📈 模型性能:")
    print(f"  • 准确率: {evaluation['accuracy']:.4f}")
    print(f"  • 选股成功率: {evaluation['selection_success_rate']:.4f}")
    print(f"  • 选股比例: {evaluation['selection_ratio']:.2%}")
    print(f"  • 模型类型: {model.__class__.__name__}")
    print(f"\n🔝 Top 5 特征重要性:")
    for i, row in evaluation['feature_importance'].head(5).iterrows():
        print(f"  • {row['feature']}: {row['importance']:.4f}")
    print(f"\n💾 结果保存:")
    print(f"  • 模型文件: results/expanded_training/")
    print(f"  • 评估报告: results/expanded_training/")
    print("="*60)
    
    logger.info("🎉 扩展数据训练完成")
    return 0


if __name__ == '__main__':
    sys.exit(main())