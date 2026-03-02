"""
LambdaRank训练器示例

演示如何使用LambdaRankTrainer训练排序模型。
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ranking_model.lambdarank_trainer import LambdaRankTrainer
from ranking_model.label_engineering import create_ranking_labels


def create_sample_data(n_dates=20, n_instruments=50):
    """创建示例数据"""
    # 创建日期范围
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='D')
    instruments = [f'STOCK_{i:03d}' for i in range(n_instruments)]
    
    # 创建多索引
    index = pd.MultiIndex.from_product(
        [dates, instruments],
        names=['date', 'instrument']
    )
    
    # 创建特征
    np.random.seed(42)
    n_samples = len(index)
    
    # 基础特征
    features = pd.DataFrame({
        'open': np.random.randn(n_samples) * 0.1 + 100,
        'high': np.random.randn(n_samples) * 0.1 + 105,
        'low': np.random.randn(n_samples) * 0.1 + 95,
        'close': np.random.randn(n_samples) * 0.1 + 102,
        'volume': np.random.randn(n_samples) * 1000 + 10000,
        'amount': np.random.randn(n_samples) * 10000 + 1000000,
    }, index=index)
    
    # 添加技术指标特征
    features['returns_1'] = np.random.randn(n_samples) * 0.01
    features['returns_5'] = np.random.randn(n_samples) * 0.02
    features['returns_20'] = np.random.randn(n_samples) * 0.03
    features['volume_ma_5'] = np.random.randn(n_samples) * 1000 + 10000
    features['volume_ma_20'] = np.random.randn(n_samples) * 1000 + 10000
    features['rsi_14'] = np.random.uniform(0, 100, n_samples)
    
    # 创建未来收益率（标签基础）
    future_returns = pd.Series(
        np.random.randn(n_samples) * 0.02,
        index=index
    )
    
    # 创建FZT选股条件（模拟）
    fzt_condition = pd.Series(
        np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),
        index=index
    )
    
    return features, future_returns, fzt_condition


def main():
    """主函数"""
    print("=" * 60)
    print("LambdaRank训练器示例")
    print("=" * 60)
    
    # 1. 创建示例数据
    print("\n1. 创建示例数据...")
    features, future_returns, fzt_condition = create_sample_data(
        n_dates=30, n_instruments=100
    )
    
    print(f"特征形状: {features.shape}")
    print(f"特征列: {list(features.columns)}")
    print(f"未来收益率形状: {future_returns.shape}")
    print(f"FZT条件分布: {fzt_condition.value_counts().to_dict()}")
    
    # 2. 创建排序标签
    print("\n2. 创建排序标签...")
    labels = create_ranking_labels(
        future_returns=future_returns,
        fzt_condition=fzt_condition,
        top_k=10,  # 每天选择前10只股票
        label_type='binary'
    )
    
    print(f"标签形状: {labels.shape}")
    print(f"标签分布: {labels.value_counts().to_dict()}")
    
    # 3. 创建训练器
    print("\n3. 创建LambdaRank训练器...")
    
    # 使用自定义配置
    custom_config = {
        'model': {
            'type': 'lightgbm_lambdarank',
            'objective': 'ranking',
            'description': '示例LambdaRank模型'
        },
        'lightgbm': {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 3, 5, 10],
            'learning_rate': 0.1,
            'num_leaves': 15,
            'max_depth': 3,
            'min_data_in_leaf': 10,
            'num_iterations': 50,
            'early_stopping_rounds': 10,
            'verbose': -1,
            'random_state': 42
        },
        'data_split': {
            'time_series_cv': True,
            'rolling_window': {
                'enabled': False  # 禁用滚动窗口，使用简单分割
            },
            'query_group_column': 'date'
        }
    }
    
    trainer = LambdaRankTrainer(config_dict=custom_config)
    
    print("训练器创建成功")
    print(f"模型类型: {trainer.config['model']['type']}")
    print(f"LightGBM目标: {trainer.config['lightgbm']['objective']}")
    
    # 4. 训练模型
    print("\n4. 训练模型...")
    
    # 分割训练集和验证集
    dates = features.index.get_level_values('date').unique()
    split_idx = int(len(dates) * 0.7)
    
    train_dates = dates[:split_idx]
    val_dates = dates[split_idx:]
    
    train_mask = features.index.get_level_values('date').isin(train_dates)
    val_mask = features.index.get_level_values('date').isin(val_dates)
    
    train_features = features[train_mask]
    train_labels = labels[train_mask]
    val_features = features[val_mask]
    val_labels = labels[val_mask]
    
    print(f"训练集大小: {len(train_features)}")
    print(f"验证集大小: {len(val_features)}")
    
    # 训练
    result = trainer.train(
        features=train_features,
        labels=train_labels,
        validation_data=(val_features, val_labels)
    )
    
    print("训练完成!")
    print(f"最佳迭代: {result['best_iteration']}")
    
    # 5. 评估模型
    print("\n5. 评估模型...")
    
    # 训练集评估
    train_metrics = result['metrics']['train']
    print("训练集指标:")
    for metric_name, metric_value in train_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # 验证集评估
    if result['metrics']['val']:
        val_metrics = result['metrics']['val']
        print("\n验证集指标:")
        for metric_name, metric_value in val_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
    
    # 6. 特征重要性
    print("\n6. 特征重要性分析...")
    importance = trainer.get_feature_importance()
    
    # 按重要性排序
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 10重要特征:")
    for i, (feature, importance_score) in enumerate(sorted_importance[:10], 1):
        print(f"  {i:2d}. {feature:15s}: {importance_score:.4f}")
    
    # 7. 预测
    print("\n7. 进行预测...")
    predictions = trainer.predict(features)
    
    print(f"预测形状: {predictions.shape}")
    print(f"预测统计:")
    print(f"  最小值: {predictions.min():.4f}")
    print(f"  最大值: {predictions.max():.4f}")
    print(f"  均值: {predictions.mean():.4f}")
    print(f"  标准差: {predictions.std():.4f}")
    
    # 8. 交叉验证（跳过，因为数据量较小）
    print("\n8. 跳过交叉验证（数据量较小）...")
    print("  在实际应用中，可以使用更多数据进行时间序列交叉验证")
    
    # 9. 保存和加载模型
    print("\n9. 保存和加载模型...")
    
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'lambdarank_model.pkl')
        
        # 保存模型
        trainer.save_model(model_path)
        print(f"模型已保存到: {model_path}")
        
        # 加载模型
        loaded_trainer = LambdaRankTrainer()
        loaded_trainer.load_model(model_path)
        print("模型加载成功")
        
        # 验证加载的模型
        loaded_predictions = loaded_trainer.predict(features)
        print(f"加载模型的预测形状: {loaded_predictions.shape}")
        
        # 验证预测一致性
        prediction_diff = (predictions - loaded_predictions).abs().max()
        print(f"预测最大差异: {prediction_diff:.6f}")
        if prediction_diff < 1e-6:
            print("✓ 模型保存和加载验证通过")
        else:
            print("⚠ 模型保存和加载存在差异")
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()