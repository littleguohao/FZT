"""
排序标签工程模块

为LambdaRank排序学习构造标签，用于训练排序模型。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import warnings


def create_ranking_labels(
    future_returns: pd.Series,
    fzt_condition: pd.Series,
    top_k: int = 5,
    label_type: str = 'binary'
) -> pd.Series:
    """
    创建排序标签
    
    根据未来收益率和FZT选股条件，为排序学习创建标签。
    
    参数:
    ----------
    future_returns : pd.Series
        未来收益率序列（T+1收益率），索引为(date, instrument)
    fzt_condition : pd.Series
        FZT选股条件序列 (0/1)，索引为(date, instrument)
    top_k : int, 默认=5
        每天选择的股票数量
    label_type : str, 默认='binary'
        标签类型：'binary'（二分类）或'continuous'（连续）
        
    返回:
    -------
    pd.Series
        排序标签序列，索引与输入相同
        
    注意:
    -----
    1. 只考虑FZT选出的股票（fzt_condition == 1）
    2. 按日期分组处理，每天独立
    3. 根据未来收益率排序生成标签
    4. 确保无未来函数，只使用当日信息
    """
    
    # 参数验证
    if not isinstance(future_returns, pd.Series):
        raise TypeError("future_returns必须是pandas Series")
    if not isinstance(fzt_condition, pd.Series):
        raise TypeError("fzt_condition必须是pandas Series")
    
    # 验证索引一致性
    if not future_returns.index.equals(fzt_condition.index):
        raise ValueError("future_returns和fzt_condition必须具有相同的索引")
    
    # 验证多索引格式
    if not (isinstance(future_returns.index, pd.MultiIndex) and 
            future_returns.index.names == ['date', 'instrument']):
        raise ValueError("索引必须是(date, instrument)格式的多索引")
    
    if label_type not in ['binary', 'continuous']:
        raise ValueError("label_type必须是'binary'或'continuous'")
    
    if top_k <= 0:
        raise ValueError("top_k必须大于0")
    
    # 初始化标签序列（全为0或NaN）
    if label_type == 'binary':
        labels = pd.Series(0, index=future_returns.index)
    else:
        labels = pd.Series(np.nan, index=future_returns.index)
    
    # 按日期分组处理
    dates = future_returns.index.get_level_values('date').unique()
    
    for date in dates:
        # 获取当天的数据
        date_mask = future_returns.index.get_level_values('date') == date
        
        date_returns = future_returns[date_mask]
        date_fzt = fzt_condition[date_mask]
        
        # 只考虑FZT选中的股票
        fzt_mask = date_fzt == 1
        fzt_returns = date_returns[fzt_mask]
        
        if len(fzt_returns) == 0:
            # 当天没有FZT选中的股票
            if label_type == 'binary':
                labels.loc[date_mask] = 0
            else:
                labels.loc[date_mask] = 0  # 连续标签也为0
            continue
        
        # 根据未来收益率排序（降序，收益率高的排前面）
        sorted_indices = fzt_returns.sort_values(ascending=False).index
        
        # 确定实际选择的股票数量
        actual_k = min(top_k, len(fzt_returns))
        
        if label_type == 'binary':
            # 二分类标签：Top K为1，其余为0
            top_indices = sorted_indices[:actual_k]
            
            # 设置标签
            labels.loc[top_indices] = 1
            
            # 确保非FZT选中的股票标签为0
            non_fzt_indices = date_fzt[date_fzt == 0].index
            if len(non_fzt_indices) > 0:
                labels.loc[non_fzt_indices] = 0
        
        else:  # continuous
            # 连续标签：标准化排名（0到1之间）
            # 使用逆排名：收益率最高的为1，最低的为0
            ranks = pd.Series(range(len(fzt_returns)), index=sorted_indices)
            
            if len(fzt_returns) > 1:
                # 标准化到0-1范围
                normalized_ranks = 1 - ranks / (len(fzt_returns) - 1)
            else:
                # 只有一只股票时，标签为1
                normalized_ranks = pd.Series(1.0, index=sorted_indices)
            
            # 设置标签
            labels.loc[sorted_indices] = normalized_ranks
            
            # 非FZT选中的股票标签为0
            non_fzt_indices = date_fzt[date_fzt == 0].index
            if len(non_fzt_indices) > 0:
                labels.loc[non_fzt_indices] = 0
    
    # 验证标签
    if label_type == 'binary':
        assert labels.isin([0, 1]).all(), "二分类标签必须为0或1"
    else:
        # 检查NaN值
        if labels.isna().any():
            warnings.warn(f"连续标签中存在{labels.isna().sum()}个NaN值，已填充为0")
            labels = labels.fillna(0)
        
        # 检查范围
        if not ((labels >= 0).all() and (labels <= 1).all()):
            warnings.warn("连续标签超出0-1范围，已进行裁剪")
            labels = labels.clip(0, 1)
    
    return labels


def create_lambdarank_dataset(
    features: pd.DataFrame,
    labels: pd.Series,
    group_sizes: Optional[List[int]] = None
) -> Dict[str, Union[pd.DataFrame, pd.Series, np.ndarray]]:
    """
    创建LambdaRank数据集
    
    将特征、标签和分组信息组织成LambdaRank所需的格式。
    
    参数:
    ----------
    features : pd.DataFrame
        特征DataFrame，索引为(date, instrument)
    labels : pd.Series
        标签序列，索引为(date, instrument)
    group_sizes : List[int], 可选
        自定义分组大小。如果为None，则按日期分组
        
    返回:
    -------
    Dict
        包含以下键的字典：
        - 'features': 特征DataFrame（可能重新排序）
        - 'labels': 标签序列（与特征对齐）
        - 'groups': 分组信息数组
        
    注意:
    -----
    1. 所有数据必须使用(date, instrument)多索引
    2. 默认按日期分组，每天为一组
    3. 支持缺失值处理
    4. 确保特征、标签、分组信息对齐
    """
    
    # 参数验证
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features必须是pandas DataFrame")
    if not isinstance(labels, pd.Series):
        raise TypeError("labels必须是pandas Series")
    
    # 验证索引一致性
    if not features.index.equals(labels.index):
        raise ValueError("features和labels必须具有相同的索引")
    
    # 验证多索引格式
    if not (isinstance(features.index, pd.MultiIndex) and 
            features.index.names == ['date', 'instrument']):
        raise ValueError("索引必须是(date, instrument)格式的多索引")
    
    # 处理缺失值
    # 检查特征中的缺失值
    feature_missing = features.isna().any(axis=1)
    label_missing = labels.isna()
    
    # 合并缺失掩码
    missing_mask = feature_missing | label_missing
    
    if missing_mask.any():
        warnings.warn(f"数据中存在{missing_mask.sum()}个缺失值，已删除")
        
        # 删除缺失值
        features_clean = features[~missing_mask].copy()
        labels_clean = labels[~missing_mask].copy()
    else:
        features_clean = features.copy()
        labels_clean = labels.copy()
    
    # 如果没有数据剩余，返回空数据集
    if len(features_clean) == 0:
        warnings.warn("所有数据都包含缺失值，返回空数据集")
        return {
            'features': pd.DataFrame(),
            'labels': pd.Series(dtype=float),
            'groups': np.array([], dtype=int)
        }
    
    # 确定分组
    if group_sizes is not None:
        # 使用自定义分组大小
        if not isinstance(group_sizes, list):
            raise TypeError("group_sizes必须是列表")
        
        if sum(group_sizes) != len(features_clean):
            raise ValueError(f"分组大小之和({sum(group_sizes)})必须等于数据点数量({len(features_clean)})")
        
        groups = np.array(group_sizes, dtype=int)
        
        # 按分组大小重新排序数据（如果需要）
        # 这里假设数据已经按分组顺序排列
        
    else:
        # 按日期分组
        dates = features_clean.index.get_level_values('date').unique()
        groups = []
        
        for date in dates:
            date_mask = features_clean.index.get_level_values('date') == date
            group_size = date_mask.sum()
            groups.append(group_size)
        
        groups = np.array(groups, dtype=int)
        
        # 按日期排序数据（确保分组顺序）
        if not features_clean.index.is_monotonic_increasing:
            features_clean = features_clean.sort_index()
            labels_clean = labels_clean.sort_index()
    
    # 验证分组信息
    assert sum(groups) == len(features_clean), "分组大小之和必须等于数据点数量"
    
    # 准备返回的数据集
    dataset = {
        'features': features_clean,
        'labels': labels_clean,
        'groups': groups
    }
    
    return dataset


def validate_label_quality(
    labels: pd.Series,
    future_returns: pd.Series,
    fzt_condition: pd.Series
) -> Dict[str, float]:
    """
    验证标签质量
    
    计算标签与未来收益率的相关性等质量指标。
    
    参数:
    ----------
    labels : pd.Series
        排序标签序列
    future_returns : pd.Series
        未来收益率序列
    fzt_condition : pd.Series
        FZT选股条件序列
        
    返回:
    -------
    Dict
        质量指标字典
    """
    
    # 只考虑FZT选中的股票
    fzt_mask = fzt_condition == 1
    valid_labels = labels[fzt_mask]
    valid_returns = future_returns[fzt_mask]
    
    if len(valid_labels) == 0:
        return {
            'correlation': 0.0,
            'label_mean': 0.0,
            'label_std': 0.0,
            'coverage_ratio': 0.0
        }
    
    # 计算相关性
    correlation = valid_labels.corr(valid_returns)
    
    # 标签统计
    label_mean = valid_labels.mean()
    label_std = valid_labels.std()
    
    # 覆盖率（FZT选中股票中有标签的比例）
    coverage_ratio = (valid_labels.notna() & valid_returns.notna()).sum() / len(valid_labels)
    
    # 按日期计算平均相关性
    dates = valid_labels.index.get_level_values('date').unique()
    daily_correlations = []
    
    for date in dates:
        date_mask = valid_labels.index.get_level_values('date') == date
        date_labels = valid_labels[date_mask]
        date_returns = valid_returns[date_mask]
        
        if len(date_labels) > 1:
            corr = date_labels.corr(date_returns)
            if not pd.isna(corr):
                daily_correlations.append(corr)
    
    avg_daily_correlation = np.mean(daily_correlations) if daily_correlations else 0.0
    
    return {
        'correlation': correlation,
        'avg_daily_correlation': avg_daily_correlation,
        'label_mean': label_mean,
        'label_std': label_std,
        'coverage_ratio': coverage_ratio,
        'num_samples': len(valid_labels),
        'num_dates': len(dates)
    }


def create_cross_validation_folds(
    features: pd.DataFrame,
    labels: pd.Series,
    n_folds: int = 5,
    strategy: str = 'time_series'
) -> List[Dict[str, Union[pd.DataFrame, pd.Series]]]:
    """
    创建交叉验证折叠
    
    为排序学习创建时间序列交叉验证折叠。
    
    参数:
    ----------
    features : pd.DataFrame
        特征DataFrame
    labels : pd.Series
        标签序列
    n_folds : int, 默认=5
        折叠数量
    strategy : str, 默认='time_series'
        分割策略：'time_series'（时间序列）或'random'（随机）
        
    返回:
    -------
    List[Dict]
        每个折叠的字典列表，包含'train'和'val'数据
    """
    
    if not features.index.equals(labels.index):
        raise ValueError("features和labels必须具有相同的索引")
    
    # 获取所有日期
    dates = features.index.get_level_values('date').unique()
    dates_sorted = sorted(dates)
    
    if strategy == 'time_series':
        # 时间序列交叉验证
        fold_size = len(dates_sorted) // n_folds
        
        folds = []
        for i in range(n_folds):
            # 确定验证集日期范围
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < n_folds - 1 else len(dates_sorted)
            
            val_dates = dates_sorted[val_start:val_end]
            train_dates = [d for d in dates_sorted if d not in val_dates]
            
            # 创建掩码
            train_mask = features.index.get_level_values('date').isin(train_dates)
            val_mask = features.index.get_level_values('date').isin(val_dates)
            
            folds.append({
                'train': {
                    'features': features[train_mask],
                    'labels': labels[train_mask]
                },
                'val': {
                    'features': features[val_mask],
                    'labels': labels[val_mask]
                }
            })
    
    else:  # random
        # 随机交叉验证（不推荐用于时间序列数据）
        np.random.seed(42)
        indices = np.arange(len(features))
        np.random.shuffle(indices)
        
        fold_size = len(indices) // n_folds
        
        folds = []
        for i in range(n_folds):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < n_folds - 1 else len(indices)
            
            val_indices = indices[val_start:val_end]
            train_indices = [idx for idx in indices if idx not in val_indices]
            
            folds.append({
                'train': {
                    'features': features.iloc[train_indices],
                    'labels': labels.iloc[train_indices]
                },
                'val': {
                    'features': features.iloc[val_indices],
                    'labels': labels.iloc[val_indices]
                }
            })
    
    return folds


# 示例使用函数
def example_usage():
    """示例使用函数"""
    
    # 创建示例数据
    dates = pd.date_range('2024-01-01', periods=3, freq='D')
    instruments = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    index = pd.MultiIndex.from_product(
        [dates, instruments],
        names=['date', 'instrument']
    )
    
    # 特征
    np.random.seed(42)
    features = pd.DataFrame({
        'feature1': np.random.randn(len(index)),
        'feature2': np.random.randn(len(index)),
        'feature3': np.random.randn(len(index))
    }, index=index)
    
    # 未来收益率
    future_returns = pd.Series(
        np.random.randn(len(index)) * 0.1,
        index=index
    )
    
    # FZT选股条件
    fzt_condition = pd.Series(
        np.random.choice([0, 1], size=len(index), p=[0.3, 0.7]),
        index=index
    )
    
    print("示例数据:")
    print(f"特征形状: {features.shape}")
    print(f"FZT条件分布: {fzt_condition.value_counts().to_dict()}")
    
    # 创建二分类标签
    binary_labels = create_ranking_labels(
        future_returns=future_returns,
        fzt_condition=fzt_condition,
        top_k=2,
        label_type='binary'
    )
    
    print(f"\n二分类标签统计:")
    print(f"标签分布: {binary_labels.value_counts().to_dict()}")
    
    # 创建连续标签
    continuous_labels = create_ranking_labels(
        future_returns=future_returns,
        fzt_condition=fzt_condition,
        top_k=2,
        label_type='continuous'
    )
    
    print(f"\n连续标签统计:")
    print(f"最小值: {continuous_labels.min():.4f}")
    print(f"最大值: {continuous_labels.max():.4f}")
    print(f"均值: {continuous_labels.mean():.4f}")
    
    # 创建LambdaRank数据集
    dataset = create_lambdarank_dataset(
        features=features,
        labels=binary_labels
    )
    
    print(f"\nLambdaRank数据集:")
    print(f"特征形状: {dataset['features'].shape}")
    print(f"标签形状: {dataset['labels'].shape}")
