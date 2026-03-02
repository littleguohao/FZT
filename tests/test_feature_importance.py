"""
测试特征重要性分析模块
"""
import pytest
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import ndcg_score
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

from src.evaluation.feature_importance import FeatureImportanceAnalyzer


class TestFeatureImportanceAnalyzer:
    """测试FeatureImportanceAnalyzer类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        # 生成特征数据
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # 生成目标变量（排序标签）
        # 让前3个特征对目标有较强影响
        y = (
            0.5 * X['feature_0'] + 
            0.3 * X['feature_1'] + 
            0.2 * X['feature_2'] + 
            0.1 * np.random.randn(n_samples)
        )
        
        # 转换为排序标签（1-5）
        y_ranked = pd.qcut(y, q=5, labels=False) + 1
        
        return X, y_ranked
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """训练一个LightGBM模型"""
        X, y = sample_data
        
        # 创建LightGBM数据集
        train_data = lgb.Dataset(X, label=y)
        
        # 训练参数（使用回归任务，因为排序任务需要query信息）
        params = {
            'objective': 'regression',
            'metric': 'l2',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'verbose': -1,
            'seed': 42
        }
        
        # 训练模型
        model = lgb.train(
            params,
            train_data,
            num_boost_round=50,
            valid_sets=[train_data],
            valid_names=['train'],
            callbacks=[lgb.log_evaluation(0)]
        )
        
        return model
    
    def test_initialization(self):
        """测试初始化"""
        analyzer = FeatureImportanceAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'calculate_importance')
        assert hasattr(analyzer, 'get_feature_ranking')
        assert hasattr(analyzer, 'generate_report')
    
    def test_calculate_gain_importance(self, trained_model, sample_data):
        """测试计算gain重要性"""
        X, y = sample_data
        analyzer = FeatureImportanceAnalyzer()
        
        # 计算gain重要性
        importance_df = analyzer.calculate_importance(
            model=trained_model,
            X=X,
            y=y,
            method='gain'
        )
        
        # 验证结果
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert 'importance_type' in importance_df.columns
        assert len(importance_df) == X.shape[1]
        
        # 验证重要性值非负
        assert (importance_df['importance'] >= 0).all()
        
        # 验证特征名称
        expected_features = [f'feature_{i}' for i in range(X.shape[1])]
        assert set(importance_df['feature']) == set(expected_features)
    
    def test_calculate_split_importance(self, trained_model, sample_data):
        """测试计算split重要性"""
        X, y = sample_data
        analyzer = FeatureImportanceAnalyzer()
        
        # 计算split重要性
        importance_df = analyzer.calculate_importance(
            model=trained_model,
            X=X,
            y=y,
            method='split'
        )
        
        # 验证结果
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == X.shape[1]
        assert (importance_df['importance'] >= 0).all()
    
    def test_calculate_cover_importance(self, trained_model, sample_data):
        """测试计算cover重要性（LightGBM 4.6.0不支持，跳过）"""
        # LightGBM 4.6.0不支持cover重要性，跳过此测试
        pytest.skip("LightGBM 4.6.0不支持cover重要性")
    
    def test_calculate_permutation_importance(self, trained_model, sample_data):
        """测试计算permutation重要性"""
        X, y = sample_data
        analyzer = FeatureImportanceAnalyzer()
        
        # 计算permutation重要性
        importance_df = analyzer.calculate_importance(
            model=trained_model,
            X=X,
            y=y,
            method='permutation',
            n_repeats=3,
            scoring='ndcg'
        )
        
        # 验证结果
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == X.shape[1]
        assert 'importance_mean' in importance_df.columns
        assert 'importance_std' in importance_df.columns
        
        # 前3个特征应该更重要（因为我们生成数据时让它们对目标有影响）
        top_features = importance_df.nlargest(3, 'importance_mean')['feature'].tolist()
        expected_important = ['feature_0', 'feature_1', 'feature_2']
        # 至少有一个重要特征在前3名中
        assert any(feat in top_features for feat in expected_important)
    
    def test_calculate_correlation_importance(self, sample_data):
        """测试计算相关性重要性"""
        X, y = sample_data
        analyzer = FeatureImportanceAnalyzer()
        
        # 计算相关性重要性
        importance_df = analyzer.calculate_importance(
            model=None,  # 不需要模型
            X=X,
            y=y,
            method='correlation'
        )
        
        # 验证结果
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == X.shape[1]
        assert 'correlation' in importance_df.columns
        assert 'abs_correlation' in importance_df.columns
        
        # 相关性值应该在[-1, 1]范围内
        assert (importance_df['correlation'].between(-1, 1)).all()
        assert (importance_df['abs_correlation'].between(0, 1)).all()
    
    def test_get_feature_ranking(self, trained_model, sample_data):
        """测试获取特征排名"""
        X, y = sample_data
        analyzer = FeatureImportanceAnalyzer()
        
        # 计算重要性
        importance_df = analyzer.calculate_importance(
            model=trained_model,
            X=X,
            y=y,
            method='gain'
        )
        
        # 获取排名
        ranking_df = analyzer.get_feature_ranking(importance_df)
        
        # 验证结果
        assert isinstance(ranking_df, pd.DataFrame)
        assert 'rank' in ranking_df.columns
        assert 'cumulative_importance' in ranking_df.columns
        assert 'normalized_importance' in ranking_df.columns
        
        # 验证排名是1到n_features
        assert ranking_df['rank'].min() == 1
        assert ranking_df['rank'].max() == len(ranking_df)
        
        # 验证排名没有重复
        assert ranking_df['rank'].nunique() == len(ranking_df)
        
        # 验证累积重要性
        assert (ranking_df['cumulative_importance'] >= 0).all()
        assert (ranking_df['cumulative_importance'] <= 1).all()
        
        # 验证归一化重要性
        assert (ranking_df['normalized_importance'] >= 0).all()
        assert np.isclose(ranking_df['normalized_importance'].sum(), 1.0)
    
    def test_generate_report(self, trained_model, sample_data):
        """测试生成报告"""
        X, y = sample_data
        analyzer = FeatureImportanceAnalyzer()
        
        # 计算重要性
        importance_df = analyzer.calculate_importance(
            model=trained_model,
            X=X,
            y=y,
            method='gain'
        )
        
        # 生成报告
        report = analyzer.generate_report(importance_df)
        
        # 验证报告内容
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'top_features' in report
        assert 'statistics' in report
        
        # 验证摘要信息
        summary = report['summary']
        assert 'total_features' in summary
        assert 'analysis_method' in summary
        assert 'top_feature' in summary
        
        # 验证统计信息
        stats = report['statistics']
        assert 'mean_importance' in stats
        assert 'std_importance' in stats
        assert 'max_importance' in stats
        assert 'min_importance' in stats
    
    def test_visualize_importance(self, trained_model, sample_data):
        """测试可视化功能"""
        X, y = sample_data
        analyzer = FeatureImportanceAnalyzer()
        
        # 计算重要性
        importance_df = analyzer.calculate_importance(
            model=trained_model,
            X=X,
            y=y,
            method='gain'
        )
        
        # 测试条形图
        fig_bar = analyzer.visualize_importance(
            importance_df,
            plot_type='bar',
            top_n=5
        )
        assert fig_bar is not None
        
        # 测试累积重要性图
        ranking_df = analyzer.get_feature_ranking(importance_df)
        fig_cumulative = analyzer.visualize_importance(
            ranking_df,
            plot_type='cumulative'
        )
        assert fig_cumulative is not None
    
    def test_feature_selection_suggestions(self, trained_model, sample_data):
        """测试特征选择建议"""
        X, y = sample_data
        analyzer = FeatureImportanceAnalyzer()
        
        # 计算重要性
        importance_df = analyzer.calculate_importance(
            model=trained_model,
            X=X,
            y=y,
            method='gain'
        )
        
        # 获取特征选择建议
        suggestions = analyzer.get_feature_selection_suggestions(
            importance_df,
            threshold=0.8  # 保留累积重要性80%的特征
        )
        
        # 验证建议
        assert isinstance(suggestions, dict)
        assert 'selected_features' in suggestions
        assert 'num_selected' in suggestions
        assert 'cumulative_importance' in suggestions
        assert 'threshold' in suggestions
        
        # 验证选择的特征数量合理
        assert suggestions['num_selected'] <= len(importance_df)
        assert suggestions['num_selected'] >= 1
        
        # 验证累积重要性达到阈值
        assert suggestions['cumulative_importance'] >= suggestions['threshold']
    
    def test_multiple_importance_methods(self, trained_model, sample_data):
        """测试多种重要性方法"""
        X, y = sample_data
        analyzer = FeatureImportanceAnalyzer()
        
        # 测试所有方法（排除cover，因为LightGBM 4.6.0不支持）
        methods = ['gain', 'split', 'permutation', 'correlation']
        
        for method in methods:
            if method == 'correlation':
                # 相关性分析不需要模型
                importance_df = analyzer.calculate_importance(
                    model=None,
                    X=X,
                    y=y,
                    method=method
                )
            else:
                importance_df = analyzer.calculate_importance(
                    model=trained_model,
                    X=X,
                    y=y,
                    method=method
                )
            
            # 验证每种方法都能成功计算
            assert isinstance(importance_df, pd.DataFrame)
            assert len(importance_df) == X.shape[1]
            
            # 验证特征名称一致
            expected_features = [f'feature_{i}' for i in range(X.shape[1])]
            assert set(importance_df['feature']) == set(expected_features)
    
    def test_invalid_method(self, trained_model, sample_data):
        """测试无效方法"""
        X, y = sample_data
        analyzer = FeatureImportanceAnalyzer()
        
        # 应该抛出ValueError
        with pytest.raises(ValueError):
            analyzer.calculate_importance(
                model=trained_model,
                X=X,
                y=y,
                method='invalid_method'
            )
    
    def test_missing_model_for_correlation(self, sample_data):
        """测试相关性分析不需要模型"""
        X, y = sample_data
        analyzer = FeatureImportanceAnalyzer()
        
        # 相关性分析应该可以没有模型
        importance_df = analyzer.calculate_importance(
            model=None,
            X=X,
            y=y,
            method='correlation'
        )
        
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == X.shape[1]
    
    def test_empty_data(self):
        """测试空数据"""
        analyzer = FeatureImportanceAnalyzer()
        
        # 空DataFrame
        X = pd.DataFrame()
        y = pd.Series([], dtype=float)
        
        # 应该抛出ValueError
        with pytest.raises(ValueError):
            analyzer.calculate_importance(
                model=None,
                X=X,
                y=y,
                method='correlation'
            )
    
    def test_large_feature_set(self):
        """测试大规模特征集"""
        np.random.seed(42)
        n_samples = 1000
        n_features = 100  # 较大特征集
        
        # 生成特征数据
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # 生成目标变量
        y = pd.Series(np.random.randn(n_samples))
        
        # 训练一个简单模型
        train_data = lgb.Dataset(X, label=y)
        params = {
            'objective': 'regression',
            'metric': 'l2',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'verbose': -1,
            'seed': 42
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=20,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        analyzer = FeatureImportanceAnalyzer()
        
        # 计算重要性（应该能处理大规模特征）
        importance_df = analyzer.calculate_importance(
            model=model,
            X=X,
            y=y,
            method='gain'
        )
        
        # 验证结果
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == n_features
        
        # 验证性能：应该在合理时间内完成
        # 这里我们只验证功能，不验证具体时间


if __name__ == '__main__':
    pytest.main([__file__, '-v'])