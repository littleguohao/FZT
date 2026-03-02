"""
LambdaRank训练器测试模块

测试LambdaRankTrainer类的功能。
"""

import pytest
import pandas as pd
import numpy as np
import yaml
import os
import tempfile
from pathlib import Path
import sys

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ranking_model.lambdarank_trainer import LambdaRankTrainer


class TestLambdaRankTrainer:
    """LambdaRankTrainer测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        # 创建日期范围
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        instruments = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        # 创建多索引
        index = pd.MultiIndex.from_product(
            [dates, instruments],
            names=['date', 'instrument']
        )
        
        # 创建特征
        np.random.seed(42)
        n_samples = len(index)
        features = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'feature4': np.random.randn(n_samples),
            'feature5': np.random.randn(n_samples)
        }, index=index)
        
        # 创建标签（排序标签，LambdaRank需要整数标签）
        # 为每个日期创建排名标签
        labels_list = []
        for date in dates:
            date_mask = index.get_level_values('date') == date
            n_instruments = date_mask.sum()
            # 为每个日期创建1-5的排名标签
            date_labels = np.random.randint(1, 6, size=n_instruments)
            labels_list.extend(date_labels)
        
        labels = pd.Series(
            labels_list,
            index=index
        )
        
        return features, labels
    
    @pytest.fixture
    def sample_config(self):
        """创建示例配置"""
        config = {
            'model': {
                'type': 'lightgbm_lambdarank',
                'objective': 'ranking',
                'description': '测试模型'
            },
            'lightgbm': {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_eval_at': [1, 3, 5],
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': 5,
                'num_iterations': 10,  # 测试时减少迭代次数
                'early_stopping_rounds': 5,
                'verbose': -1  # 静默模式
            },
            'data_split': {
                'time_series_cv': True,
                'rolling_window': {
                    'enabled': True,
                    'window_size': 5,
                    'step_size': 2,
                    'min_train_size': 3
                },
                'query_group_column': 'date'
            },
            'training': {
                'early_stopping': {
                    'enabled': True,
                    'patience': 5,
                    'min_delta': 0.001
                }
            }
        }
        
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        yield config_path
        
        # 清理
        if os.path.exists(config_path):
            os.unlink(config_path)
    
    def test_initialization(self, sample_config):
        """测试初始化"""
        trainer = LambdaRankTrainer(config_path=sample_config)
        
        assert trainer is not None
        assert hasattr(trainer, 'config')
        assert hasattr(trainer, 'model')
        assert hasattr(trainer, 'feature_names')
        
        # 验证配置加载
        assert trainer.config['model']['type'] == 'lightgbm_lambdarank'
        assert trainer.config['lightgbm']['objective'] == 'lambdarank'
    
    def test_prepare_dataset(self, sample_data, sample_config):
        """测试数据集准备"""
        features, labels = sample_data
        trainer = LambdaRankTrainer(config_path=sample_config)
        
        # 准备数据集
        dataset = trainer._prepare_dataset(features, labels)
        
        assert dataset is not None
        assert 'features' in dataset
        assert 'labels' in dataset
        assert 'groups' in dataset
        
        # 验证数据形状
        assert len(dataset['features']) == len(features)
        assert len(dataset['labels']) == len(labels)
        
        # 验证分组信息
        assert isinstance(dataset['groups'], np.ndarray)
        assert dataset['groups'].sum() == len(features)
        
        # 验证分组按日期
        unique_dates = features.index.get_level_values('date').unique()
        assert len(dataset['groups']) == len(unique_dates)
    
    def test_train_single_fold(self, sample_data, sample_config):
        """测试单折训练"""
        features, labels = sample_data
        trainer = LambdaRankTrainer(config_path=sample_config)
        
        # 训练模型
        result = trainer.train(features, labels)
        
        assert result is not None
        assert 'model' in result
        assert 'metrics' in result
        assert 'feature_importance' in result
        
        # 验证模型存在
        assert result['model'] is not None
        
        # 验证指标
        metrics = result['metrics']
        assert isinstance(metrics, dict)
        assert 'train' in metrics
        assert 'val' in metrics
        assert isinstance(metrics['train'], dict)
        assert 'ndcg' in metrics['train']
        
        # 验证特征重要性
        feature_importance = result['feature_importance']
        assert isinstance(feature_importance, dict)
        assert len(feature_importance) == len(features.columns)
    
    def test_cross_validation(self, sample_data, sample_config):
        """测试交叉验证"""
        features, labels = sample_data
        trainer = LambdaRankTrainer(config_path=sample_config)
        
        # 进行交叉验证
        cv_results = trainer.cross_validate(features, labels, n_folds=3)
        
        assert cv_results is not None
        assert 'fold_metrics' in cv_results
        assert 'mean_metrics' in cv_results
        assert 'std_metrics' in cv_results
        
        # 验证折叠数量
        fold_metrics = cv_results['fold_metrics']
        assert len(fold_metrics) == 3
        
        # 验证指标计算
        mean_metrics = cv_results['mean_metrics']
        assert isinstance(mean_metrics, dict)
        assert 'ndcg' in mean_metrics
        assert 'ndcg@1' in mean_metrics
        assert 'ndcg@3' in mean_metrics
        assert 'ndcg@5' in mean_metrics
    
    def test_predict(self, sample_data, sample_config):
        """测试预测"""
        features, labels = sample_data
        trainer = LambdaRankTrainer(config_path=sample_config)
        
        # 训练模型
        result = trainer.train(features, labels)
        
        # 预测
        predictions = trainer.predict(features)
        
        assert predictions is not None
        assert len(predictions) == len(features)
        assert isinstance(predictions, pd.Series)
        assert predictions.index.equals(features.index)
        
        # 验证预测值范围（排序分数可以是任意实数）
        assert not predictions.isna().any()
    
    def test_save_load_model(self, sample_data, sample_config):
        """测试模型保存和加载"""
        features, labels = sample_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.pkl')
            
            # 训练并保存模型
            trainer1 = LambdaRankTrainer(config_path=sample_config)
            result1 = trainer1.train(features, labels)
            trainer1.save_model(model_path)
            
            # 验证模型文件存在
            assert os.path.exists(model_path)
            
            # 加载模型
            trainer2 = LambdaRankTrainer(config_path=sample_config)
            trainer2.load_model(model_path)
            
            # 使用两个模型进行预测
            predictions1 = trainer1.predict(features)
            predictions2 = trainer2.predict(features)
            
            # 验证预测结果一致
            pd.testing.assert_series_equal(predictions1, predictions2)
    
    def test_feature_importance(self, sample_data, sample_config):
        """测试特征重要性计算"""
        features, labels = sample_data
        trainer = LambdaRankTrainer(config_path=sample_config)
        
        # 训练模型
        result = trainer.train(features, labels)
        
        # 获取特征重要性
        importance = trainer.get_feature_importance()
        
        assert importance is not None
        assert isinstance(importance, dict)
        assert len(importance) == len(features.columns)
        
        # 验证重要性值非负
        importance_values = list(importance.values())
        assert all(v >= 0 for v in importance_values)
        
        # 验证特征名称
        feature_names = list(importance.keys())
        assert set(feature_names) == set(features.columns)
    
    def test_early_stopping(self, sample_data, sample_config):
        """测试早停机制"""
        features, labels = sample_data
        trainer = LambdaRankTrainer(config_path=sample_config)
        
        # 训练模型
        result = trainer.train(features, labels)
        
        # 验证早停相关信息
        assert 'best_iteration' in result
        assert 'train_history' in result
        
        train_history = result['train_history']
        assert isinstance(train_history, dict)
        assert 'train_ndcg' in train_history
        assert 'val_ndcg' in train_history
        
        # 验证迭代次数不超过配置
        assert result['best_iteration'] <= trainer.config['lightgbm']['num_iterations']
    
    def test_invalid_input(self, sample_config):
        """测试无效输入处理"""
        trainer = LambdaRankTrainer(config_path=sample_config)
        
        # 测试空特征
        empty_features = pd.DataFrame()
        empty_labels = pd.Series(dtype=float)
        
        with pytest.raises(ValueError):
            trainer.train(empty_features, empty_labels)
        
        # 测试特征和标签形状不匹配
        features = pd.DataFrame({'feature1': [1, 2, 3]})
        labels = pd.Series([1, 2])  # 长度不匹配
        
        with pytest.raises(ValueError):
            trainer.train(features, labels)
        
        # 测试无效配置路径
        with pytest.raises(FileNotFoundError):
            LambdaRankTrainer(config_path='nonexistent_config.yaml')
    
    def test_time_series_split(self, sample_data, sample_config):
        """测试时间序列分割"""
        features, labels = sample_data
        trainer = LambdaRankTrainer(config_path=sample_config)
        
        # 创建时间序列分割
        splits = trainer._create_time_series_splits(features, n_folds=3)
        
        assert splits is not None
        assert len(splits) == 3
        
        for split in splits:
            assert 'train_indices' in split
            assert 'val_indices' in split
            
            # 验证训练集和验证集不重叠
            train_set = set(split['train_indices'])
            val_set = set(split['val_indices'])
            assert len(train_set.intersection(val_set)) == 0
            
            # 验证时间顺序：训练集日期早于验证集
            if len(split['train_indices']) > 0 and len(split['val_indices']) > 0:
                train_dates = features.index.get_level_values('date')[split['train_indices']]
                val_dates = features.index.get_level_values('date')[split['val_indices']]
                assert train_dates.max() <= val_dates.min()


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v'])