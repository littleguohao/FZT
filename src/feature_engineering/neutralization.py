"""
特征中性化模块

实现"市值回归 + 行业标准化"混合中性化方案
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import warnings


def mad_winsorize(series: pd.Series, n: float = 3.0) -> pd.Series:
    """
    MAD去极值处理
    
    使用中位数绝对偏差(MAD)方法识别和处理异常值
    
    Parameters
    ----------
    series : pd.Series
        输入序列
    n : float, default=3.0
        MAD倍数阈值，默认3倍MAD
    
    Returns
    -------
    pd.Series
        去极值处理后的序列
    """
    if len(series) == 0:
        return series.copy()
    
    # 创建副本以避免修改原始数据
    result = series.copy()
    
    # 计算中位数
    median = np.nanmedian(result)
    
    # 计算MAD（中位数绝对偏差）
    mad = np.nanmedian(np.abs(result - median))
    
    # 如果MAD为0（常数序列），直接返回
    if mad == 0:
        return result
    
    # 计算上下界
    upper_bound = median + n * mad
    lower_bound = median - n * mad
    
    # 缩尾处理：将超出界限的值替换为边界值
    result = result.clip(lower=lower_bound, upper=upper_bound)
    
    return result


def zscore_normalize(series: pd.Series) -> pd.Series:
    """
    横截面Z-score标准化
    
    将序列标准化为均值为0，标准差为1
    
    Parameters
    ----------
    series : pd.Series
        输入序列
    
    Returns
    -------
    pd.Series
        标准化后的序列
    """
    if len(series) == 0:
        return series.copy()
    
    # 创建副本
    result = series.copy()
    
    # 计算均值和标准差（忽略NaN）
    mean_val = np.nanmean(result)
    std_val = np.nanstd(result, ddof=1)  # 使用样本标准差
    
    # 如果标准差为0（常数序列），将所有非NaN值设为0
    if std_val == 0:
        result = result - mean_val  # 减去均值，使均值为0
        return result
    
    # Z-score标准化
    result = (result - mean_val) / std_val
    
    return result


def neutralize_features(
    features: pd.DataFrame,
    market_cap: pd.Series,
    industry: pd.Series,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    特征中性化主函数
    
    实现"市值回归 + 行业标准化"混合中性化方案：
    1. 预处理：去极值(MAD) + 标准化(Z-score)
    2. 市值中性化：回归法 (特征 = α + β * log(市值) + ε)
    3. 行业中性化：分组标准化 (行业内均值为0，标准差为1)
    
    Parameters
    ----------
    features : pd.DataFrame
        特征数据，索引为(date, instrument)多索引
    market_cap : pd.Series
        市值数据，索引与features相同
    industry : pd.Series
        行业数据，索引与features相同
    feature_names : Optional[List[str]], default=None
        需要中性化的特征名称列表，如果为None则处理所有特征
    
    Returns
    -------
    pd.DataFrame
        中性化后的特征数据
    """
    # 验证输入数据
    _validate_inputs(features, market_cap, industry)
    
    # 如果没有指定特征名称，使用所有特征
    if feature_names is None:
        feature_names = list(features.columns)
    else:
        # 确保指定的特征存在
        missing_features = set(feature_names) - set(features.columns)
        if missing_features:
            raise ValueError(f"指定的特征不存在: {missing_features}")
    
    # 创建结果DataFrame
    result = features.copy()
    
    # 获取日期列表
    dates = features.index.get_level_values('date').unique()
    
    # 对每个日期独立处理
    for date in dates:
        # 获取当天的数据
        date_mask = features.index.get_level_values('date') == date
        date_features = features.loc[date_mask, feature_names].copy()
        date_market_cap = market_cap.loc[date_mask].copy()
        date_industry = industry.loc[date_mask].copy()
        
        # 如果当天没有数据，跳过
        if len(date_features) == 0:
            continue
        
        # 对每个特征进行中性化
        for feature_name in feature_names:
            feature_series = date_features[feature_name].copy()
            
            # 步骤1: 预处理 - 去极值 + 标准化
            processed = _preprocess_feature(feature_series)
            
            # 步骤2: 市值中性化
            market_neutralized = _market_cap_neutralization(
                processed, date_market_cap
            )
            
            # 步骤3: 行业中性化
            industry_neutralized = _industry_neutralization(
                market_neutralized, date_industry
            )
            
            # 将结果保存回原DataFrame
            result.loc[date_mask, feature_name] = industry_neutralized
    
    return result


def _validate_inputs(
    features: pd.DataFrame,
    market_cap: pd.Series,
    industry: pd.Series
) -> None:
    """验证输入数据格式"""
    # 检查索引是否匹配
    if not features.index.equals(market_cap.index):
        raise ValueError("features和market_cap的索引不匹配")
    
    if not features.index.equals(industry.index):
        raise ValueError("features和industry的索引不匹配")
    
    # 检查索引是否为多索引
    if not isinstance(features.index, pd.MultiIndex):
        raise ValueError("features的索引必须是多索引(date, instrument)")
    
    # 检查索引名称
    if features.index.names != ['date', 'instrument']:
        warnings.warn(
            "features索引名称不是['date', 'instrument']，但将继续处理",
            UserWarning
        )
    
    # 检查是否有NaN值（只警告）
    if features.isna().any().any():
        warnings.warn("features中包含NaN值，中性化过程中会保留NaN", UserWarning)
    
    if market_cap.isna().any():
        warnings.warn("market_cap中包含NaN值，可能影响中性化效果", UserWarning)
    
    if industry.isna().any():
        warnings.warn("industry中包含NaN值，可能影响中性化效果", UserWarning)


def _preprocess_feature(series: pd.Series) -> pd.Series:
    """
    特征预处理：去极值 + 标准化
    
    Parameters
    ----------
    series : pd.Series
        原始特征序列
    
    Returns
    -------
    pd.Series
        预处理后的特征序列
    """
    # 创建副本
    result = series.copy()
    
    # 步骤1: MAD去极值
    result = mad_winsorize(result)
    
    # 步骤2: Z-score标准化
    result = zscore_normalize(result)
    
    return result


def _market_cap_neutralization(
    feature_series: pd.Series,
    market_cap: pd.Series
) -> pd.Series:
    """
    市值中性化：回归法
    
    通过回归去除特征中的市值暴露：
    特征 = α + β * log(市值) + ε
    返回残差ε作为市值中性化后的特征
    
    Parameters
    ----------
    feature_series : pd.Series
        预处理后的特征序列
    market_cap : pd.Series
        市值序列
    
    Returns
    -------
    pd.Series
        市值中性化后的特征序列
    """
    # 创建副本
    result = feature_series.copy()
    
    # 获取非NaN的索引
    valid_mask = feature_series.notna() & market_cap.notna()
    
    if valid_mask.sum() < 2:
        # 如果有效数据不足，无法进行回归，返回原始值
        warnings.warn(
            f"有效数据不足({valid_mask.sum()})，跳过市值中性化",
            UserWarning
        )
        return result
    
    # 提取有效数据
    valid_features = feature_series[valid_mask]
    valid_market_cap = market_cap[valid_mask]
    
    # 对市值取对数（避免负值或零值）
    log_market_cap = np.log(valid_market_cap.clip(lower=1e-10))
    
    try:
        # 使用最小二乘法进行回归
        # 构建设计矩阵：常数项 + log(市值)
        X = np.column_stack([np.ones_like(log_market_cap), log_market_cap])
        y = valid_features.values
        
        # 求解回归系数
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # 计算预测值
        y_pred = X @ beta
        
        # 计算残差（市值中性化后的特征）
        residuals = y - y_pred
        
        # 将残差放回原序列
        result.loc[valid_mask] = residuals
        
    except np.linalg.LinAlgError:
        # 如果回归失败（如共线性），返回原始值
        warnings.warn("市值回归失败，跳过市值中性化", UserWarning)
    
    return result


def _industry_neutralization(
    feature_series: pd.Series,
    industry: pd.Series
) -> pd.Series:
    """
    行业中性化：分组标准化
    
    在每个行业内进行标准化，使行业内均值为0，标准差为1
    
    Parameters
    ----------
    feature_series : pd.Series
        市值中性化后的特征序列
    industry : pd.Series
        行业序列
    
    Returns
    -------
    pd.Series
        行业中性化后的特征序列
    """
    # 创建副本
    result = feature_series.copy()
    
    # 获取所有行业
    industries = industry.unique()
    
    for ind in industries:
        # 获取该行业的股票
        industry_mask = (industry == ind) & feature_series.notna()
        
        if industry_mask.sum() == 0:
            # 如果没有有效数据，跳过
            continue
        
        # 提取该行业的特征值
        industry_features = feature_series[industry_mask]
        
        if industry_mask.sum() == 1:
            # 如果只有一个股票，设为0（减去均值）
            normalized = industry_features - industry_features.iloc[0]
        else:
            # 行业内标准化
            mean_val = industry_features.mean()
            std_val = industry_features.std()
            
            if std_val == 0:
                # 如果标准差为0，将所有值设为0（减去均值）
                normalized = industry_features - mean_val
            else:
                # Z-score标准化
                normalized = (industry_features - mean_val) / std_val
        
        # 将标准化后的值放回原序列
        result.loc[industry_mask] = normalized
    
    return result


# 为了方便使用，提供批量中性化函数
def neutralize_batch(
    features: pd.DataFrame,
    market_cap: pd.Series,
    industry: pd.Series,
    batch_size: int = 100
) -> pd.DataFrame:
    """
    批量中性化特征（适用于大量特征）
    
    Parameters
    ----------
    features : pd.DataFrame
        特征数据
    market_cap : pd.Series
        市值数据
    industry : pd.Series
        行业数据
    batch_size : int, default=100
        每批处理的特征数量
    
    Returns
    -------
    pd.DataFrame
        中性化后的特征数据
    """
    result = pd.DataFrame(index=features.index)
    
    # 分批处理特征
    feature_columns = list(features.columns)
    num_batches = (len(feature_columns) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(feature_columns))
        batch_features = feature_columns[start_idx:end_idx]
        
        print(f"处理批次 {i+1}/{num_batches}: 特征 {start_idx+1}-{end_idx}")
        
        # 中性化当前批次
        batch_result = neutralize_features(
            features[batch_features],
            market_cap,
            industry,
            feature_names=batch_features
        )
        
        # 合并结果
        result = pd.concat([result, batch_result], axis=1)
    
    return result