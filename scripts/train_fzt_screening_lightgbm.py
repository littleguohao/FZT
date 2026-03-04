#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FZT筛选 + LightGBM训练工作流

正确流程：
1. 使用FZT公式筛选符合条件的股票
2. 只用筛选后的股票进行LightGBM训练
3. 关注gain得分发掘有效因子

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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
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


def load_and_prepare_data():
    """加载并准备数据"""
    logger.info("📥 加载数据...")
    
    try:
        # 加载扩展数据
        data_file = Path("results/expanded_stock_data/expanded_data_with_target.csv")
        if not data_file.exists():
            logger.error(f"❌ 数据文件不存在: {data_file}")
            return None
        
        df = pd.read_csv(data_file)
        logger.info(f"✅ 数据加载成功: {df.shape[0]} 行, {df.shape[1]} 列")
        
        # 确保数据排序
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['code', 'date'])
        
        # 计算基础特征（用于LightGBM训练）
        df['returns_1d'] = df.groupby('code')['close'].pct_change(fill_method=None)
        df['returns_5d'] = df.groupby('code')['close'].pct_change(5, fill_method=None)
        df['returns_10d'] = df.groupby('code')['close'].pct_change(10, fill_method=None)
        
        # 价格特征
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # 成交量特征
        df['volume_ratio'] = df['volume'] / df.groupby('code')['volume'].shift(1)
        df['volume_ma5'] = df.groupby('code')['volume'].rolling(5).mean().reset_index(level=0, drop=True)
        
        # 波动率特征
        df['volatility_5d'] = df.groupby('code')['close'].pct_change().rolling(5).std()
        df['volatility_20d'] = df.groupby('code')['close'].pct_change().rolling(20).std()
        
        # 处理缺失值
        df = df.fillna(method='ffill').fillna(0)
        
        logger.info(f"✅ 特征准备完成: {df.shape[1]} 列")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ 数据准备失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def apply_fzt_screening(data: pd.DataFrame):
    """应用FZT公式筛选"""
    logger.info("🔍 应用FZT公式筛选...")
    
    try:
        from src.fzt_brick_formula import FZTBrickFormula
        
        fzt_calculator = FZTBrickFormula()
        
        # 按股票分组处理
        screened_data_list = []
        total_stocks = data['code'].nunique()
        
        for i, (stock_code, stock_data) in enumerate(data.groupby('code')):
            if i % 20 == 0:
                logger.info(f"  处理股票 {i+1}/{total_stocks}: {stock_code}")
            
            if len(stock_data) < 20:  # 需要足够的数据计算FZT
                continue
            
            # 计算FZT指标
            try:
                indicators = fzt_calculator.calculate_all_indicators(stock_data)
                
                # 获取选股条件
                if 'selection_condition' in indicators:
                    selection_signal = indicators['selection_condition']
                    
                    # 筛选符合条件的日期
                    screened_dates = stock_data.index[selection_signal.fillna(False)]
                    
                    if len(screened_dates) > 0:
                        screened_data = stock_data.loc[screened_dates].copy()
                        screened_data_list.append(screened_data)
                        
            except Exception as e:
                logger.warning(f"   股票 {stock_code} FZT计算失败: {str(e)[:50]}")
                continue
        
        if not screened_data_list:
            logger.warning("⚠️  没有股票通过FZT筛选")
            return pd.DataFrame()
        
        # 合并所有筛选后的数据
        screened_df = pd.concat(screened_data_list, ignore_index=True)
        
        logger.info(f"✅ FZT筛选完成")
        logger.info(f"   原始数据: {len(data)} 行")
        logger.info(f"   筛选后数据: {len(screened_df)} 行")
        logger.info(f"   筛选比例: {len(screened_df)/len(data):.2%}")
        
        return screened_df
        
    except Exception as e:
        logger.error(f"❌ FZT筛选失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def prepare_training_features(data: pd.DataFrame):
    """准备训练特征（不使用FZT特征）"""
    logger.info("🔧 准备训练特征...")
    
    try:
        # 特征列表（不使用FZT特征）
        feature_cols = [
            # 收益率特征
            'returns_1d', 'returns_5d', 'returns_10d',
            
            # 价格特征
            'high_low_ratio', 'close_open_ratio',
            
            # 成交量特征
            'volume_ratio', 'volume_ma5',
            
            # 波动率特征
            'volatility_5d', 'volatility_20d',
            
            # 基础价格（可选）
            'open', 'high', 'low', 'close', 'volume'
        ]
        
        # 只保留存在的特征
        available_features = [col for col in feature_cols if col in data.columns]
        
        X = data[available_features].fillna(0).values
        y = data['target'].values
        
        logger.info(f"✅ 特征准备完成")
        logger.info(f"   特征数量: {len(available_features)}")
        logger.info(f"   样本数量: {len(X)}")
        logger.info(f"   正类比例: {y.mean():.2%}")
        
        return X, y, available_features
        
    except Exception as e:
        logger.error(f"❌ 特征准备失败: {e}")
        return None, None, []


def train_lightgbm_with_gain_analysis(X_train, y_train, X_val, y_val, feature_names):
    """训练LightGBM并分析gain得分"""
    logger.info("🤖 训练LightGBM（关注gain得分）...")
    
    try:
        import lightgbm as lgb
        
        # LightGBM参数
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 5,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_estimators': 100,
            'verbose': -1
        }
        
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_names)
        
        # 训练模型
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(10),
                lgb.log_evaluation(10)
            ]
        )
        
        # 获取特征重要性（gain得分）
        importance = model.feature_importance(importance_type='gain')
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'gain': importance,
            'split': model.feature_importance(importance_type='split'),
            'cover': model.feature_importance(importance_type='cover')
        })
        
        # 按gain得分排序
        importance_df = importance_df.sort_values('gain', ascending=False)
        
        logger.info("✅ 模型训练完成")
        logger.info(f"   最佳迭代次数: {model.best_iteration}")
        
        print("\n📊 特征重要性分析（按gain得分排序）:")
        print("="*80)
        print(f"{'特征':<20} {'gain得分':<15} {'split次数':<15} {'cover占比':<15} {'解读':<30}")
        print("-"*80)
        
        for _, row in importance_df.head(10).iterrows():
            # 特征解读
            interpretation = ""
            if 'return' in row['feature']:
                interpretation = "收益率特征"
            elif 'volume' in row['feature']:
                interpretation = "成交量特征"
            elif 'volatility' in row['feature']:
                interpretation = "波动率特征"
            elif row['feature'] in ['open', 'high', 'low', 'close']:
                interpretation = "价格特征"
            elif 'ratio' in row['feature']:
                interpretation = "价格比率特征"
            
            print(f"{row['feature']:<20} {row['gain']:<15.2f} {row['split']:<15} {row['cover']:<15.2f} {interpretation:<30}")
        
        print("="*80)
        
        # 分析gain得分的含义
        print("\n🎯 gain得分分析（选股场景解读）:")
        print("   gain（增益）: 该因子在所有决策树中带来的误差减少总量（核心！）")
        print("                → 最高（直接反映因子对胜率的贡献）")
        print("   split（分裂）: 该因子被用作决策树分裂节点的次数")
        print("                → 辅助（次数多≠贡献大）")
        print("   cover（覆盖）: 该因子涉及的样本数占比")
        print("                → 参考（仅看因子覆盖度）")
        
        return model, importance_df
        
    except Exception as e:
        logger.error(f"❌ 模型训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None


def evaluate_model(model, X_test, y_test, feature_names):
    """评估模型性能"""
    logger.info("📈 评估模型性能...")
    
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # 预测
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, y_pred_prob)
        except:
            roc_auc = 0
        
        # 计算选股成功率（预测上涨且实际上涨的比例）
        success_mask = (y_pred == 1) & (y_test == 1)
        if (y_pred == 1).sum() > 0:
            success_rate = success_mask.sum() / (y_pred == 1).sum()
        else:
            success_rate = 0
        
        logger.info(f"✅ 模型评估完成")
        
        print("\n📊 模型性能指标:")
        print("="*60)
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"选股成功率: {success_rate:.4f}")
        print("="*60)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'success_rate': success_rate
        }
        
    except Exception as e:
        logger.error(f"❌ 模型评估失败: {e}")
        return {}


def discover_effective_factors(importance_df, top_n=10):
    """发掘有效因子"""
    logger.info("🔍 发掘有效因子...")
    
    try:
        # 选择gain得分高的因子
        top_factors = importance_df.head(top_n)
        
        print(f"\n🎯 Top-{top_n} 有效因子（按gain得分）:")
        print("="*80)
        
        for i, (_, row) in enumerate(top_factors.iterrows(), 1):
            print(f"{i:2d}. {row['feature']}")
            print(f"    gain得分: {row['gain']:.2f} (误差减少总量)")
            print(f"    split次数: {row['split']} (分裂节点次数)")
            print(f"    cover占比: {row['cover']:.2f} (样本覆盖度)")
            
            # 因子类型判断
            if 'return' in row['feature']:
                print(f"    类型: 收益率特征")
                print(f"    实战价值: 反映股票短期动量")
            elif 'volume' in row['feature']:
                print(f"    类型: 成交量特征")
                print(f"    实战价值: 反映市场关注度和流动性")
            elif 'volatility' in row['feature']:
                print(f"    类型: 波动率特征")
                print(f"    实战价值: 反映股票风险水平")
            elif 'ratio' in row['feature']:
                print(f"    类型: 价格比率特征")
                print(f"    实战价值: 反映价格结构和强度")
            
            print()
        
        print("💡 因子发掘建议:")
        print("   1. 关注gain得分高的因子（直接贡献预测能力）")
        print("   2. 结合split次数判断因子稳定性")
        print("   3. 避免过度依赖单一因子")
        print("   4. 定期更新因子有效性分析")
        print("="*80)
        
        return top_factors
        
    except Exception as e:
        logger.error(f"❌ 因子发掘失败: {e}")
        return pd.DataFrame()


def main():
    """主函数"""
    logger.info("🚀 FZT筛选 + LightGBM训练工作流开始")
    logger.info("=" * 60)
    
    try:
        # 1. 加载并准备数据
        data = load_and_prepare_data()
        if data is None or data.empty:
            logger.error("❌ 数据加载失败")
            return 1
        
        print(f"\n📊 原始数据统计:")
        print(f"  总行数: {len(data):,}")
        print(f"  股票数量: {data['code'].nunique()}")
        print(f"  日期范围: {data['date'].min()} 到 {data['date'].max()}")
        print(f"  正类比例: {data['target'].mean():.2%}")
        
        # 2. 应用FZT筛选
        screened_data = apply_fzt_screening(data)
        if screened_data.empty:
            logger.error("❌ FZT筛选后无数据")
            return 1
        
        print(f"\n🔍 FZT筛选结果:")
        print(f"  筛选后行数: {len(screened_data):,}")
        print(f"  筛选比例: {len(screened_data)/len(data):.2%}")
        print(f"  筛选后正类比例: {screened_data['target'].mean():.2%}")
        
        # 3. 准备训练特征（不使用FZT特征）
        X, y, feature_names = prepare_training_features(screened_data)
        if X is None or len(X) == 0:
            logger.error("❌ 特征准备失败")
            return 1
        
        # 4. 划分训练集和测试集（时间序列划分）
        train_ratio = 0.7
        split_idx = int(len(X) * train_ratio)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 进一步划分验证集
        val_ratio = 0.2
        val_split_idx = int(len(X_train) * (1 - val_ratio))
        X_train, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
        y_train, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
        
        print(f"\n📈 数据划分:")
        print(f"  训练集: {len(X_train):,} 样本")
        print(f"  验证集: {len(X_val):,} 样本")
        print(f"  测试集: {len(X_test):,} 样本")
        
        # 5. 训练LightGBM并分析gain得分
        model, importance_df = train_lightgbm_with_gain_analysis(
            X_train, y_train, X_val, y_val, feature_names
        )
        
        if model is None:
            logger.error("❌ 模型训练失败")
            return 1
        
        # 6. 评估模型
        metrics = evaluate_model(model, X_test, y_test, feature_names)
        
        # 7. 发掘有效因子
        top_factors = discover_effective_factors(importance_df, top_n=15)
        
        # 8. 保存结果
        logger.info("💾 保存训练结果...")
        
        # 创建输出目录
        output_dir = Path("results") / "fzt_screening_training"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存模型
        model_path = output_dir / f"fzt_screening_model_{timestamp}.txt"
        model.save_model(str(model_path))
        
        # 保存特征重要性
        importance_path = output_dir / f"feature_importance_{timestamp}.csv"
        importance_df.to_csv(importance_path, index=False)
        
        # 保存训练结果
        result = {
            'training_date': timestamp,
            'data_info': {
                'original_samples': len(data),
                'screened_samples': len(screened_data),
                'screening_ratio': len(screened_data) / len(data),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'positive_ratio': float(y.mean())
            },
            'model_info': {
                'best_iteration': model.best_iteration,
                'feature_count': len(feature_names)
            },
            'performance': metrics,
            'top_factors': top_factors[['feature', 'gain', 'split', 'cover']].to_dict('records')
        }
        
        result_path = output_dir / f"training_result_{timestamp}.yaml"
        with open(result_path, 'w', encoding='utf-8') as f:
            yaml.dump(result, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n🎉 训练完成！")
        print(f"📁 结果保存到: {output_dir}")
        print(f"  模型文件: {model_path.name}")
        print(f"  特征重要性: {importance_path.name}")
        print(f"  训练结果: {result_path.name}")
        
        print(f"\n🚀 工作流总结:")
        print(f"  1. ✅ FZT公式筛选: 从 {len(data):,} 行中筛选出 {len(screened_data):,} 行")
        print(f"  2. ✅ LightGBM训练: 使用 {len(feature_names)} 个非FZT特征")
        print(f"  3. ✅ gain得分分析: 识别出 {len(top_factors)} 个有效因子")
        print(f"  4. ✅ 模型性能: 选股成功率 {metrics.get('success_rate', 0):.4f}")
        
        logger.info("🎯 FZT筛选 + LightGBM训练工作流完成")
        return 0
        
    except Exception as e:
        logger.error(f"❌ 工作流执行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())