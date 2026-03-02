"""
特征重要性分析示例

展示如何使用FeatureImportanceAnalyzer进行特征重要性分析。
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.feature_importance import FeatureImportanceAnalyzer, analyze_feature_importance


def create_sample_data(n_samples=1000, n_features=20):
    """创建样本数据"""
    np.random.seed(42)
    
    # 生成特征数据
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i:02d}' for i in range(n_features)]
    )
    
    # 生成目标变量 - 让前5个特征对目标有影响
    y = (
        0.3 * X['feature_00'] + 
        0.25 * X['feature_01'] + 
        0.2 * X['feature_02'] + 
        0.15 * X['feature_03'] + 
        0.1 * X['feature_04'] + 
        0.05 * np.random.randn(n_samples)
    )
    
    return X, y


def train_model(X, y):
    """训练LightGBM模型"""
    # 创建数据集
    train_data = lgb.Dataset(X, label=y)
    
    # 训练参数
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
        num_boost_round=100,
        valid_sets=[train_data],
        valid_names=['train'],
        callbacks=[lgb.log_evaluation(0)]
    )
    
    return model


def main():
    """主函数"""
    print("=" * 60)
    print("特征重要性分析示例")
    print("=" * 60)
    
    # 1. 创建样本数据
    print("\n1. 创建样本数据...")
    X, y = create_sample_data(n_samples=1000, n_features=20)
    print(f"   特征数据形状: {X.shape}")
    print(f"   目标变量形状: {y.shape}")
    
    # 2. 训练模型
    print("\n2. 训练LightGBM模型...")
    model = train_model(X, y)
    print("   模型训练完成")
    
    # 3. 创建特征重要性分析器
    print("\n3. 创建特征重要性分析器...")
    analyzer = FeatureImportanceAnalyzer(random_state=42)
    
    # 4. 计算gain重要性
    print("\n4. 计算gain重要性...")
    gain_importance = analyzer.calculate_importance(
        model=model,
        X=X,
        y=y,
        method='gain'
    )
    print(f"   前5个重要特征 (gain):")
    print(gain_importance.head(5).to_string(index=False))
    
    # 5. 计算split重要性
    print("\n5. 计算split重要性...")
    split_importance = analyzer.calculate_importance(
        model=model,
        X=X,
        y=y,
        method='split'
    )
    print(f"   前5个重要特征 (split):")
    print(split_importance.head(5).to_string(index=False))
    
    # 6. 计算permutation重要性
    print("\n6. 计算permutation重要性...")
    permutation_importance = analyzer.calculate_importance(
        model=model,
        X=X,
        y=y,
        method='permutation',
        n_repeats=3,
        scoring='mse'
    )
    print(f"   前5个重要特征 (permutation):")
    print(permutation_importance.head(5).to_string(index=False))
    
    # 7. 计算相关性重要性
    print("\n7. 计算相关性重要性...")
    correlation_importance = analyzer.calculate_importance(
        model=None,
        X=X,
        y=y,
        method='correlation'
    )
    print(f"   前5个重要特征 (correlation):")
    print(correlation_importance.head(5).to_string(index=False))
    
    # 8. 生成报告
    print("\n8. 生成特征重要性报告...")
    report = analyzer.generate_report(gain_importance, top_n=10)
    
    print(f"   摘要:")
    print(f"     总特征数: {report['summary']['total_features']}")
    print(f"     最重要特征: {report['summary']['top_feature']}")
    print(f"     分析方法: {report['summary']['analysis_method']}")
    
    print(f"\n   统计信息:")
    stats = report['statistics']
    print(f"     平均重要性: {stats['mean_importance']:.4f}")
    print(f"     重要性标准差: {stats['std_importance']:.4f}")
    print(f"     最大重要性: {stats['max_importance']:.4f}")
    print(f"     最小重要性: {stats['min_importance']:.4f}")
    
    print(f"\n   累积重要性分析:")
    cum_stats = report['cumulative_stats']
    print(f"     达到50%重要性所需特征数: {cum_stats['features_for_50pct']}")
    print(f"     达到80%重要性所需特征数: {cum_stats['features_for_80pct']}")
    print(f"     达到90%重要性所需特征数: {cum_stats['features_for_90pct']}")
    
    # 9. 特征选择建议
    print("\n9. 特征选择建议...")
    suggestions = analyzer.get_feature_selection_suggestions(
        gain_importance,
        threshold=0.8,
        min_features=5,
        max_features=15
    )
    
    print(f"   建议选择 {suggestions['num_selected']} 个特征")
    print(f"   累积重要性: {suggestions['cumulative_importance']:.2%}")
    print(f"   效率比: {suggestions['efficiency_ratio']:.4f}")
    print(f"   选择的特征: {', '.join(suggestions['selected_features'][:5])}...")
    
    # 10. 可视化
    print("\n10. 生成可视化图表...")
    
    # 重要性条形图
    fig_bar = analyzer.visualize_importance(
        gain_importance,
        plot_type='bar',
        top_n=15,
        title='特征重要性排名 (Gain方法)'
    )
    plt.savefig('feature_importance_bar.png', dpi=150, bbox_inches='tight')
    print("   已保存: feature_importance_bar.png")
    
    # 累积重要性图
    ranking_df = analyzer.get_feature_ranking(gain_importance)
    fig_cumulative = analyzer.visualize_importance(
        ranking_df,
        plot_type='cumulative',
        title='累积特征重要性'
    )
    plt.savefig('feature_importance_cumulative.png', dpi=150, bbox_inches='tight')
    print("   已保存: feature_importance_cumulative.png")
    
    # 11. 比较不同方法
    print("\n11. 比较不同重要性方法...")
    results = analyzer.compare_importance_methods(
        model=model,
        X=X,
        y=y,
        methods=['gain', 'split', 'permutation', 'correlation']
    )
    
    print(f"   比较了 {len(results)} 种方法: {list(results.keys())}")
    
    # 生成比较图
    fig_comparison = analyzer.plot_comparison(results, top_n=10)
    plt.savefig('feature_importance_comparison.png', dpi=150, bbox_inches='tight')
    print("   已保存: feature_importance_comparison.png")
    
    # 12. 使用工具函数
    print("\n12. 使用analyze_feature_importance工具函数...")
    analysis_results = analyze_feature_importance(
        model=model,
        X=X,
        y=y,
        methods=['gain', 'split', 'permutation'],
        save_path='feature_importance_report.json'
    )
    
    if analysis_results:
        print("   分析完成，报告已保存到: feature_importance_report.json")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)
    
    # 显示图表
    plt.show()


if __name__ == '__main__':
    main()