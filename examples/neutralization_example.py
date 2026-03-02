"""
特征中性化模块使用示例
"""
import pandas as pd
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.feature_engineering.neutralization import (
    mad_winsorize,
    zscore_normalize,
    neutralize_features,
    neutralize_batch
)


def create_sample_data():
    """创建示例数据"""
    # 创建日期和股票
    dates = pd.date_range('2023-01-01', periods=5)
    instruments = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']
    
    # 创建多索引
    index = pd.MultiIndex.from_product(
        [dates, instruments],
        names=['date', 'instrument']
    )
    
    # 创建特征数据（包含一些异常值）
    np.random.seed(42)
    n_samples = len(index)
    
    features = pd.DataFrame({
        'momentum': np.random.randn(n_samples) * 2 + 10,
        'value': np.random.randn(n_samples) * 3 + 20,
        'growth': np.random.randn(n_samples) * 1.5 + 15,
        'quality': np.random.randn(n_samples) * 2.5 + 18,
        'volatility': np.random.randn(n_samples) * 4 + 5
    }, index=index)
    
    # 添加一些异常值
    features.iloc[10, 0] = 100  # 异常大的动量值
    features.iloc[25, 1] = -50  # 异常小的价值值
    
    # 创建市值数据（与特征相关）
    market_cap = pd.Series(
        np.exp(np.random.randn(n_samples) * 0.8 + 12),
        index=index
    )
    
    # 创建行业数据
    industries = ['Tech', 'Tech', 'Tech', 'Retail', 'Auto', 'Tech', 'Tech', 'Finance']
    industry = pd.Series(
        np.tile(industries, len(dates)),
        index=index
    )
    
    return features, market_cap, industry


def demonstrate_mad_winsorize():
    """演示MAD去极值"""
    print("=" * 60)
    print("1. MAD去极值演示")
    print("=" * 60)
    
    # 创建包含异常值的序列
    data = pd.Series([1, 2, 3, 4, 5, 100])
    print(f"原始数据: {data.values}")
    print(f"原始数据统计: 均值={data.mean():.2f}, 标准差={data.std():.2f}")
    
    # 应用MAD去极值
    winsorized = mad_winsorize(data, n=3.0)
    print(f"去极值后: {winsorized.values}")
    print(f"去极值后统计: 均值={winsorized.mean():.2f}, 标准差={winsorized.std():.2f}")
    print()


def demonstrate_zscore_normalize():
    """演示Z-score标准化"""
    print("=" * 60)
    print("2. Z-score标准化演示")
    print("=" * 60)
    
    # 创建序列
    data = pd.Series([10, 20, 30, 40, 50])
    print(f"原始数据: {data.values}")
    print(f"原始数据统计: 均值={data.mean():.2f}, 标准差={data.std():.2f}")
    
    # 应用Z-score标准化
    normalized = zscore_normalize(data)
    print(f"标准化后: {normalized.values}")
    print(f"标准化后统计: 均值={normalized.mean():.2f}, 标准差={normalized.std():.2f}")
    print()


def demonstrate_neutralization():
    """演示完整的中性化流程"""
    print("=" * 60)
    print("3. 完整特征中性化演示")
    print("=" * 60)
    
    # 创建示例数据
    features, market_cap, industry = create_sample_data()
    
    print(f"特征数据形状: {features.shape}")
    print(f"特征列: {list(features.columns)}")
    print(f"市值数据形状: {market_cap.shape}")
    print(f"行业数据形状: {industry.shape}")
    print()
    
    # 显示原始数据统计
    print("原始特征统计:")
    for col in features.columns:
        print(f"  {col}: 均值={features[col].mean():.2f}, 标准差={features[col].std():.2f}")
    print()
    
    # 应用中性化
    print("应用特征中性化...")
    neutralized_features = neutralize_features(features, market_cap, industry)
    
    # 显示中性化后数据统计
    print("中性化后特征统计:")
    for col in neutralized_features.columns:
        print(f"  {col}: 均值={neutralized_features[col].mean():.2f}, 标准差={neutralized_features[col].std():.2f}")
    print()
    
    # 检查中性化效果
    print("中性化效果检查:")
    
    # 1. 检查市值中性化效果
    print("1. 市值中性化效果:")
    for col in features.columns:
        original_corr = features[col].corr(market_cap)
        neutralized_corr = neutralized_features[col].corr(market_cap)
        print(f"  {col}: 原始相关性={original_corr:.4f}, 中性化后={neutralized_corr:.4f}")
    print()
    
    # 2. 检查行业中性化效果
    print("2. 行业中性化效果:")
    for col in features.columns:
        # 按行业分组计算均值
        industry_means = neutralized_features[col].groupby(industry).mean()
        max_industry_mean = industry_means.abs().max()
        print(f"  {col}: 最大行业均值绝对值={max_industry_mean:.4f}")
    print()
    
    return features, neutralized_features, market_cap, industry


def demonstrate_batch_neutralization():
    """演示批量中性化"""
    print("=" * 60)
    print("4. 批量中性化演示")
    print("=" * 60)
    
    # 创建更多特征的数据
    features, market_cap, industry = create_sample_data()
    
    # 添加更多特征列
    for i in range(10):
        features[f'extra_feature_{i}'] = np.random.randn(len(features)) * (i+1)
    
    print(f"批量处理前特征数量: {len(features.columns)}")
    print(f"特征列: {list(features.columns)[:5]}...")  # 只显示前5个
    
    # 使用批量中性化
    print("应用批量中性化（每批3个特征）...")
    batch_neutralized = neutralize_batch(features, market_cap, industry, batch_size=3)
    
    print(f"批量处理后特征数量: {len(batch_neutralized.columns)}")
    print("批量中性化完成！")
    print()


def main():
    """主函数"""
    print("特征中性化模块演示")
    print("=" * 60)
    
    # 演示各个功能
    demonstrate_mad_winsorize()
    demonstrate_zscore_normalize()
    features, neutralized_features, market_cap, industry = demonstrate_neutralization()
    demonstrate_batch_neutralization()
    
    print("=" * 60)
    print("演示完成！")
    print("=" * 60)
    
    # 保存示例结果
    output_dir = '../results/neutralization_example'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存原始和中性化后的特征
    features.to_csv(f'{output_dir}/original_features.csv')
    neutralized_features.to_csv(f'{output_dir}/neutralized_features.csv')
    market_cap.to_csv(f'{output_dir}/market_cap.csv')
    industry.to_csv(f'{output_dir}/industry.csv')
    
    print(f"结果已保存到: {output_dir}/")
    
    # 创建汇总报告
    report = pd.DataFrame({
        'feature': features.columns,
        'original_mean': [features[col].mean() for col in features.columns],
        'original_std': [features[col].std() for col in features.columns],
        'neutralized_mean': [neutralized_features[col].mean() for col in neutralized_features.columns],
        'neutralized_std': [neutralized_features[col].std() for col in neutralized_features.columns],
        'market_cap_corr_original': [features[col].corr(market_cap) for col in features.columns],
        'market_cap_corr_neutralized': [neutralized_features[col].corr(market_cap) for col in neutralized_features.columns]
    })
    
    report.to_csv(f'{output_dir}/neutralization_report.csv', index=False)
    print(f"汇总报告已保存到: {output_dir}/neutralization_report.csv")


if __name__ == '__main__':
    main()