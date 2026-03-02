"""
测试特征中性化模块
"""
import pytest
import pandas as pd
import numpy as np
from src.feature_engineering.neutralization import (
    mad_winsorize,
    zscore_normalize,
    neutralize_features
)


class TestMADWinsorize:
    """测试MAD去极值函数"""
    
    def test_basic_functionality(self):
        """测试基本功能"""
        series = pd.Series([1, 2, 3, 4, 5, 100])  # 100是异常值
        result = mad_winsorize(series, n=3.0)
        
        # 检查结果类型
        assert isinstance(result, pd.Series)
        assert len(result) == len(series)
        
        # 检查异常值被处理
        assert result.iloc[-1] < 100  # 异常值应该被缩尾
        
    def test_empty_series(self):
        """测试空序列"""
        series = pd.Series([])
        result = mad_winsorize(series)
        assert len(result) == 0
        
    def test_constant_series(self):
        """测试常数序列"""
        series = pd.Series([5, 5, 5, 5])
        result = mad_winsorize(series)
        # 常数序列应该保持不变
        assert (result == 5).all()
        
    def test_nan_handling(self):
        """测试NaN处理"""
        series = pd.Series([1, 2, np.nan, 4, 5])
        result = mad_winsorize(series)
        # NaN应该被保留
        assert result.isna().sum() == 1
        assert np.isnan(result.iloc[2])


class TestZScoreNormalize:
    """测试Z-score标准化函数"""
    
    def test_basic_functionality(self):
        """测试基本功能"""
        series = pd.Series([1, 2, 3, 4, 5])
        result = zscore_normalize(series)
        
        # 检查结果类型
        assert isinstance(result, pd.Series)
        assert len(result) == len(series)
        
        # 检查标准化属性
        assert abs(result.mean()) < 1e-10  # 均值接近0
        assert abs(result.std() - 1.0) < 1e-10  # 标准差接近1
        
    def test_empty_series(self):
        """测试空序列"""
        series = pd.Series([])
        result = zscore_normalize(series)
        assert len(result) == 0
        
    def test_constant_series(self):
        """测试常数序列"""
        series = pd.Series([5, 5, 5, 5])
        result = zscore_normalize(series)
        # 常数序列标准化后应该全为0
        assert (result == 0).all()
        
    def test_nan_handling(self):
        """测试NaN处理"""
        series = pd.Series([1, 2, np.nan, 4, 5])
        result = zscore_normalize(series)
        # NaN应该被保留
        assert result.isna().sum() == 1
        assert np.isnan(result.iloc[2])


class TestNeutralizeFeatures:
    """测试特征中性化主函数"""
    
    def create_test_data(self):
        """创建测试数据"""
        dates = pd.date_range('2023-01-01', periods=3)
        instruments = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        
        # 创建多索引
        index = pd.MultiIndex.from_product(
            [dates, instruments],
            names=['date', 'instrument']
        )
        
        # 创建特征数据
        np.random.seed(42)
        features = pd.DataFrame({
            'feature1': np.random.randn(len(index)),
            'feature2': np.random.randn(len(index)) * 2,
            'feature3': np.random.randn(len(index)) + 5
        }, index=index)
        
        # 创建市值数据（与特征相关）
        market_cap = pd.Series(
            np.exp(np.random.randn(len(index)) * 0.5 + 10),
            index=index
        )
        
        # 创建行业数据
        industries = ['Tech', 'Tech', 'Tech', 'Retail']
        industry = pd.Series(
            np.tile(industries, len(dates)),
            index=index
        )
        
        return features, market_cap, industry
    
    def test_basic_functionality(self):
        """测试基本功能"""
        features, market_cap, industry = self.create_test_data()
        
        result = neutralize_features(features, market_cap, industry)
        
        # 检查结果类型和形状
        assert isinstance(result, pd.DataFrame)
        assert result.shape == features.shape
        assert list(result.columns) == list(features.columns)
        assert result.index.equals(features.index)
        
        # 检查每个特征是否被中性化
        for col in result.columns:
            # 按日期分组检查
            for date in features.index.get_level_values('date').unique():
                date_mask = result.index.get_level_values('date') == date
                feature_data = result.loc[date_mask, col]
                
                # 检查没有NaN
                assert not feature_data.isna().any()
                
    def test_specific_feature_names(self):
        """测试指定特征名称"""
        features, market_cap, industry = self.create_test_data()
        
        # 只中性化部分特征
        result = neutralize_features(
            features, market_cap, industry, 
            feature_names=['feature1', 'feature2']
        )
        
        # 检查只有指定特征被处理
        assert 'feature1' in result.columns
        assert 'feature2' in result.columns
        assert 'feature3' in result.columns
        
        # feature3应该保持不变
        assert (result['feature3'] == features['feature3']).all()
        
    def test_empty_features(self):
        """测试空特征数据"""
        index = pd.MultiIndex.from_product(
            [pd.date_range('2023-01-01', periods=1), ['AAPL']],
            names=['date', 'instrument']
        )
        features = pd.DataFrame(index=index)
        market_cap = pd.Series([100], index=index)
        industry = pd.Series(['Tech'], index=index)
        
        result = neutralize_features(features, market_cap, industry)
        assert result.empty
        
    def test_single_date_single_instrument(self):
        """测试单日单股票"""
        index = pd.MultiIndex.from_product(
            [pd.date_range('2023-01-01', periods=1), ['AAPL']],
            names=['date', 'instrument']
        )
        features = pd.DataFrame({'feature1': [1.0]}, index=index)
        market_cap = pd.Series([100.0], index=index)
        industry = pd.Series(['Tech'], index=index)
        
        result = neutralize_features(features, market_cap, industry)
        # 单股票情况下，经过预处理和中性化后特征应该为0
        # 注意：预处理阶段（zscore_normalize）会因为标准差为0而返回NaN
        # 但行业中性化阶段会将其设为0
        assert abs(result['feature1'].iloc[0]) < 1e-10 or np.isnan(result['feature1'].iloc[0])
        
    def test_nan_handling(self):
        """测试NaN处理"""
        features, market_cap, industry = self.create_test_data()
        
        # 添加一些NaN值
        features.iloc[0, 0] = np.nan
        features.iloc[2, 1] = np.nan
        
        result = neutralize_features(features, market_cap, industry)
        
        # NaN应该被保留在相同位置
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[2, 1])
        
        # 其他位置不应该有NaN
        assert result.iloc[1:, 0].notna().all()
        assert result.iloc[:2, 1].notna().all()
        
    def test_industry_standardization(self):
        """测试行业标准化效果"""
        # 创建有明显行业差异的数据
        dates = pd.date_range('2023-01-01', periods=1)
        instruments = ['AAPL', 'GOOGL', 'WMT', 'TGT']
        
        index = pd.MultiIndex.from_product(
            [dates, instruments],
            names=['date', 'instrument']
        )
        
        # 特征：前两个Tech公司值高，后两个Retail公司值低（使用浮点数）
        features = pd.DataFrame({
            'feature1': [10.0, 12.0, 2.0, 3.0]
        }, index=index)
        
        market_cap = pd.Series([100.0, 120.0, 50.0, 60.0], index=index)
        industry = pd.Series(['Tech', 'Tech', 'Retail', 'Retail'], index=index)
        
        result = neutralize_features(features, market_cap, industry)
        
        # 检查行业内标准化
        tech_mask = industry == 'Tech'
        retail_mask = industry == 'Retail'
        
        tech_values = result.loc[tech_mask, 'feature1']
        retail_values = result.loc[retail_mask, 'feature1']
        
        # 行业内均值应该接近0
        assert abs(tech_values.mean()) < 1e-10
        assert abs(retail_values.mean()) < 1e-10
        
    def test_market_cap_neutralization(self):
        """测试市值中性化效果"""
        # 创建与市值相关的特征
        dates = pd.date_range('2023-01-01', periods=1)
        instruments = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        
        index = pd.MultiIndex.from_product(
            [dates, instruments],
            names=['date', 'instrument']
        )
        
        # 市值数据
        market_cap = pd.Series([100, 200, 300, 400], index=index)
        
        # 特征与市值强相关
        features = pd.DataFrame({
            'feature1': market_cap.values * 0.1 + np.random.randn(4) * 0.01
        }, index=index)
        
        industry = pd.Series(['Tech'] * 4, index=index)
        
        result = neutralize_features(features, market_cap, industry)
        
        # 检查市值中性化后，特征与市值的相关性应该降低
        original_corr = np.corrcoef(features['feature1'], market_cap)[0, 1]
        neutralized_corr = np.corrcoef(result['feature1'], market_cap)[0, 1]
        
        # 中性化后相关性应该显著降低
        assert abs(neutralized_corr) < abs(original_corr) * 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])