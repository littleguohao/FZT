"""
测试排序标签工程模块
"""

import pandas as pd
import numpy as np
import pytest
import sys
import os
from datetime import datetime

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestLabelEngineering:
    """测试标签工程功能"""
    
    def setup_method(self):
        """创建测试数据"""
        # 创建测试数据 - 3天，每天5只股票
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        instruments = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        # 创建多索引
        index = pd.MultiIndex.from_product(
            [dates, instruments],
            names=['date', 'instrument']
        )
        
        # 创建特征数据
        np.random.seed(42)
        self.features = pd.DataFrame({
            'feature1': np.random.randn(len(index)),
            'feature2': np.random.randn(len(index)),
            'feature3': np.random.randn(len(index))
        }, index=index)
        
        # 创建未来收益率数据
        self.future_returns = pd.Series(
            np.random.randn(len(index)) * 0.1,  # 模拟收益率
            index=index
        )
        
        # 创建FZT选股条件（随机选择一些股票）
        self.fzt_condition = pd.Series(
            np.random.choice([0, 1], size=len(index), p=[0.3, 0.7]),
            index=index
        )
        
        # 打印测试数据信息
        print(f"测试数据形状: {self.features.shape}")
        print(f"FZT条件分布: {self.fzt_condition.value_counts().to_dict()}")
    
    def test_create_ranking_labels_binary(self):
        """测试创建二分类标签"""
        try:
            from src.ranking_model.label_engineering import create_ranking_labels
        except ImportError:
            pytest.skip("模块尚未实现")
        
        # 测试二分类标签
        labels = create_ranking_labels(
            future_returns=self.future_returns,
            fzt_condition=self.fzt_condition,
            top_k=2,
            label_type='binary'
        )
        
        # 验证标签形状
        assert labels.shape == self.future_returns.shape
        assert labels.index.equals(self.future_returns.index)
        
        # 验证标签值（应为0或1）
        assert labels.isin([0, 1]).all()
        
        # 验证每天选中的股票数量
        for date in self.future_returns.index.get_level_values('date').unique():
            date_mask = labels.index.get_level_values('date') == date
            fzt_mask = self.fzt_condition[date_mask] == 1
            
            # 只考虑FZT选中的股票
            valid_labels = labels[date_mask][fzt_mask]
            
            # 验证每天最多有top_k个标签为1
            assert valid_labels.sum() <= 2  # top_k=2
            
            # 验证标签为1的数量等于min(top_k, 当天FZT选中股票数量)
            fzt_count = fzt_mask.sum()
            expected_ones = min(2, fzt_count)
            assert valid_labels.sum() == expected_ones
        
        print("✓ 二分类标签测试通过")
    
    def test_create_ranking_labels_continuous(self):
        """测试创建连续标签"""
        try:
            from src.ranking_model.label_engineering import create_ranking_labels
        except ImportError:
            pytest.skip("模块尚未实现")
        
        # 测试连续标签
        labels = create_ranking_labels(
            future_returns=self.future_returns,
            fzt_condition=self.fzt_condition,
            top_k=2,
            label_type='continuous'
        )
        
        # 验证标签形状
        assert labels.shape == self.future_returns.shape
        assert labels.index.equals(self.future_returns.index)
        
        # 验证标签值范围（应为0到1之间）
        assert (labels >= 0).all() and (labels <= 1).all()
        
        # 验证每天FZT选中的股票标签在0-1范围内
        for date in self.future_returns.index.get_level_values('date').unique():
            date_mask = labels.index.get_level_values('date') == date
            fzt_mask = self.fzt_condition[date_mask] == 1
            
            # 只考虑FZT选中的股票
            valid_labels = labels[date_mask][fzt_mask]
            
            if len(valid_labels) > 0:
                # 验证标签在0-1范围内
                assert (valid_labels >= 0).all() and (valid_labels <= 1).all()
                
                # 验证排名顺序：收益率高的标签值应该更大
                date_returns = self.future_returns[date_mask][fzt_mask]
                if len(valid_labels) > 1:
                    # 检查标签与收益率的单调性
                    sorted_pairs = sorted(zip(date_returns.values, valid_labels.values))
                    returns_sorted, labels_sorted = zip(*sorted_pairs)
                    
                    # 收益率增加时，标签应该非递减
                    for i in range(1, len(labels_sorted)):
                        assert labels_sorted[i] >= labels_sorted[i-1] - 1e-10
        
        print("✓ 连续标签测试通过")
    
    def test_create_ranking_labels_edge_cases(self):
        """测试边缘情况"""
        try:
            from src.ranking_model.label_engineering import create_ranking_labels
        except ImportError:
            pytest.skip("模块尚未实现")
        
        # 测试没有FZT选中的情况
        empty_fzt = pd.Series(0, index=self.future_returns.index)
        labels = create_ranking_labels(
            future_returns=self.future_returns,
            fzt_condition=empty_fzt,
            top_k=2,
            label_type='binary'
        )
        
        # 所有标签应为0
        assert (labels == 0).all()
        
        # 测试top_k大于FZT选中股票数量
        all_fzt = pd.Series(1, index=self.future_returns.index)
        labels = create_ranking_labels(
            future_returns=self.future_returns,
            fzt_condition=all_fzt,
            top_k=10,  # 大于每天股票数量
            label_type='binary'
        )
        
        # 验证每天所有股票标签都为1（因为top_k大于股票数量）
        for date in self.future_returns.index.get_level_values('date').unique():
            date_mask = labels.index.get_level_values('date') == date
            assert labels[date_mask].sum() == 5  # 每天5只股票
        
        print("✓ 边缘情况测试通过")
    
    def test_create_lambdarank_dataset(self):
        """测试创建LambdaRank数据集"""
        try:
            from src.ranking_model.label_engineering import create_lambdarank_dataset
        except ImportError:
            pytest.skip("模块尚未实现")
        
        # 首先创建标签
        from src.ranking_model.label_engineering import create_ranking_labels
        
        labels = create_ranking_labels(
            future_returns=self.future_returns,
            fzt_condition=self.fzt_condition,
            top_k=2,
            label_type='binary'
        )
        
        # 创建LambdaRank数据集
        dataset = create_lambdarank_dataset(
            features=self.features,
            labels=labels
        )
        
        # 验证返回的数据结构
        assert 'features' in dataset
        assert 'labels' in dataset
        assert 'groups' in dataset
        
        # 验证特征和标签形状匹配
        assert dataset['features'].shape[0] == dataset['labels'].shape[0]
        
        # 验证分组信息
        groups = dataset['groups']
        assert isinstance(groups, list) or isinstance(groups, np.ndarray)
        
        # 验证分组大小之和等于数据点数量
        if isinstance(groups, list):
            assert sum(groups) == len(labels)
        else:
            assert groups.sum() == len(labels)
        
        # 验证每天为一组
        unique_dates = self.features.index.get_level_values('date').unique()
        assert len(groups) == len(unique_dates)
        
        print("✓ LambdaRank数据集测试通过")
    
    def test_create_lambdarank_dataset_with_group_sizes(self):
        """测试使用自定义分组大小创建LambdaRank数据集"""
        try:
            from src.ranking_model.label_engineering import create_lambdarank_dataset
        except ImportError:
            pytest.skip("模块尚未实现")
        
        # 首先创建标签
        from src.ranking_model.label_engineering import create_ranking_labels
        
        labels = create_ranking_labels(
            future_returns=self.future_returns,
            fzt_condition=self.fzt_condition,
            top_k=2,
            label_type='binary'
        )
        
        # 自定义分组大小（例如：按周分组）
        group_sizes = [10, 5]  # 前10个样本一组，后5个样本一组
        
        # 创建LambdaRank数据集
        dataset = create_lambdarank_dataset(
            features=self.features,
            labels=labels,
            group_sizes=group_sizes
        )
        
        # 验证分组信息
        groups = dataset['groups']
        assert sum(groups) == len(labels)
        assert len(groups) == len(group_sizes)
        
        print("✓ 自定义分组LambdaRank数据集测试通过")
    
    def test_no_future_data_leakage(self):
        """测试无未来数据泄露"""
        try:
            from src.ranking_model.label_engineering import create_ranking_labels
        except ImportError:
            pytest.skip("模块尚未实现")
        
        # 创建有明显时间趋势的数据
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        instruments = ['AAPL', 'GOOGL']
        
        index = pd.MultiIndex.from_product(
            [dates, instruments],
            names=['date', 'instrument']
        )
        
        # 未来收益率：随时间增加
        future_returns = pd.Series(
            np.arange(len(index)) * 0.01,  # 线性增加
            index=index
        )
        
        # 所有股票都被FZT选中
        fzt_condition = pd.Series(1, index=index)
        
        labels = create_ranking_labels(
            future_returns=future_returns,
            fzt_condition=fzt_condition,
            top_k=1,
            label_type='binary'
        )
        
        # 验证每天只使用当天的信息
        for i, date in enumerate(dates):
            date_mask = labels.index.get_level_values('date') == date
            date_labels = labels[date_mask]
            
            # 每天应该只有1只股票标签为1（top_k=1）
            assert date_labels.sum() == 1
            
            # 标签为1的股票应该是当天未来收益率最高的
            date_returns = future_returns[date_mask]
            top_stock = date_returns.idxmax()[1]  # 获取instrument
            label_1_stock = date_labels[date_labels == 1].index[0][1]
            
            assert top_stock == label_1_stock
        
        print("✓ 无未来数据泄露测试通过")


if __name__ == '__main__':
    # 运行测试
    test = TestLabelEngineering()
    test.setup_method()
    
    print("开始运行标签工程测试...")
    
    # 测试二分类标签
    try:
        test.test_create_ranking_labels_binary()
    except Exception as e:
        print(f"二分类标签测试失败: {e}")
    
    # 测试连续标签
    try:
        test.test_create_ranking_labels_continuous()
    except Exception as e:
        print(f"连续标签测试失败: {e}")
    
    # 测试边缘情况
    try:
        test.test_create_ranking_labels_edge_cases()
    except Exception as e:
        print(f"边缘情况测试失败: {e}")
    
    # 测试LambdaRank数据集
    try:
        test.test_create_lambdarank_dataset()
    except Exception as e:
        print(f"LambdaRank数据集测试失败: {e}")
    
    # 测试自定义分组
    try:
        test.test_create_lambdarank_dataset_with_group_sizes()
    except Exception as e:
        print(f"自定义分组测试失败: {e}")
    
    # 测试无未来数据泄露
    try:
        test.test_no_future_data_leakage()
    except Exception as e:
        print(f"无未来数据泄露测试失败: {e}")
    
    print("测试完成！")