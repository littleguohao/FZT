"""
FZT排序增强策略完整管道示例

演示如何将LambdaRank训练器集成到FZT排序增强策略中。
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ranking_model.lambdarank_trainer import LambdaRankTrainer
from ranking_model.label_engineering import create_ranking_labels


class FZTRankingPipeline:
    """FZT排序增强策略管道"""
    
    def __init__(self, config_path=None):
        """
        初始化管道
        
        参数:
        ----------
        config_path : str, 可选
            配置文件路径
        """
        self.config_path = config_path
        self.trainer = None
        self.feature_columns = None
        
    def prepare_features(self, stock_data):
        """
        准备特征数据
        
        参数:
        ----------
        stock_data : pd.DataFrame
            股票数据，包含以下列：
            - date: 日期
            - instrument: 股票代码
            - open, high, low, close, volume: 基础数据
            - fzt_score: FZT分数
            - 其他技术指标
            
        返回:
        -------
        pd.DataFrame
            特征DataFrame
        """
        # 确保数据有多索引
        if not isinstance(stock_data.index, pd.MultiIndex):
            stock_data = stock_data.set_index(['date', 'instrument'])
        
        # 选择特征列
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume', 'amount',
            'returns_1', 'returns_5', 'returns_20',
            'volume_ma_5', 'volume_ma_20', 'rsi_14',
            'fzt_score', 'fzt_momentum', 'fzt_strength'
        ]
        
        # 只保留存在的列
        available_columns = [col for col in feature_columns if col in stock_data.columns]
        self.feature_columns = available_columns
        
        features = stock_data[available_columns].copy()
        
        # 处理缺失值
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def prepare_labels(self, stock_data, future_horizon=1):
        """
        准备标签数据
        
        参数:
        ----------
        stock_data : pd.DataFrame
            股票数据
        future_horizon : int
            未来预测周期
            
        返回:
        -------
        pd.Series
            标签序列
        """
        # 确保数据有多索引
        if not isinstance(stock_data.index, pd.MultiIndex):
            stock_data_indexed = stock_data.set_index(['date', 'instrument'])
        else:
            stock_data_indexed = stock_data
        
        # 计算未来收益率
        if 'close' in stock_data_indexed.columns:
            # 按股票分组计算未来收益率
            future_returns = stock_data_indexed.groupby('instrument')['close'].pct_change(future_horizon).shift(-future_horizon)
        else:
            # 如果没有收盘价，使用随机数据
            future_returns = pd.Series(
                np.random.randn(len(stock_data_indexed)) * 0.02,
                index=stock_data_indexed.index
            )
        
        # 创建FZT选股条件（假设fzt_score > 0.5为选中）
        if 'fzt_score' in stock_data_indexed.columns:
            fzt_condition = (stock_data_indexed['fzt_score'] > 0.5).astype(int)
        else:
            # 如果没有FZT分数，随机选择
            fzt_condition = pd.Series(
                np.random.choice([0, 1], size=len(stock_data_indexed), p=[0.3, 0.7]),
                index=stock_data_indexed.index
            )
        
        # 创建排序标签
        labels = create_ranking_labels(
            future_returns=future_returns,
            fzt_condition=fzt_condition,
            top_k=10,  # 每天选择前10只股票
            label_type='binary'
        )
        
        return labels
    
    def train(self, train_data, val_data=None):
        """
        训练排序模型
        
        参数:
        ----------
        train_data : pd.DataFrame
            训练数据
        val_data : pd.DataFrame, 可选
            验证数据
            
        返回:
        -------
        Dict
            训练结果
        """
        print("准备训练数据...")
        train_features = self.prepare_features(train_data)
        train_labels = self.prepare_labels(train_data)
        
        # 创建训练器
        self.trainer = LambdaRankTrainer(config_path=self.config_path)
        
        if val_data is not None:
            print("准备验证数据...")
            val_features = self.prepare_features(val_data)
            val_labels = self.prepare_labels(val_data)
            
            validation_data = (val_features, val_labels)
        else:
            validation_data = None
        
        print("开始训练LambdaRank模型...")
        result = self.trainer.train(
            features=train_features,
            labels=train_labels,
            validation_data=validation_data
        )
        
        return result
    
    def predict(self, stock_data):
        """
        预测排序分数
        
        参数:
        ----------
        stock_data : pd.DataFrame
            股票数据
            
        返回:
        -------
        pd.DataFrame
            包含预测分数的数据
        """
        if self.trainer is None:
            raise ValueError("模型未训练，请先调用train()方法")
        
        features = self.prepare_features(stock_data)
        predictions = self.trainer.predict(features)
        
        # 将预测结果添加到数据中
        result_data = stock_data.copy()
        result_data['ranking_score'] = predictions
        
        # 按日期排序
        result_data['rank'] = result_data.groupby('date')['ranking_score'].rank(ascending=False)
        
        return result_data
    
    def evaluate_portfolio(self, predictions, top_k=10):
        """
        评估投资组合表现
        
        参数:
        ----------
        predictions : pd.DataFrame
            包含预测分数的数据
        top_k : int
            每天选择的股票数量
            
        返回:
        -------
        Dict
            投资组合表现指标
        """
        # 选择每天前K只股票
        selected_stocks = predictions[predictions['rank'] <= top_k].copy()
        
        # 计算投资组合收益率（简化）
        if 'future_return' in selected_stocks.columns:
            portfolio_return = selected_stocks.groupby('date')['future_return'].mean()
        else:
            # 如果没有未来收益率，使用随机数据
            portfolio_return = pd.Series(
                np.random.randn(len(selected_stocks['date'].unique())) * 0.001,
                index=selected_stocks['date'].unique()
            )
        
        # 计算指标
        total_return = (1 + portfolio_return).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_return)) - 1
        volatility = portfolio_return.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'num_dates': len(portfolio_return),
            'avg_stocks_per_day': selected_stocks.groupby('date').size().mean()
        }


def create_sample_stock_data(n_dates=100, n_stocks=200):
    """创建示例股票数据"""
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='D')
    stocks = [f'STOCK_{i:04d}' for i in range(n_stocks)]
    
    data = []
    for date in dates:
        for stock in stocks:
            # 基础价格数据
            base_price = 100 + np.random.randn() * 10
            returns = np.random.randn() * 0.02
            
            row = {
                'date': date,
                'instrument': stock,
                'open': base_price * (1 + np.random.randn() * 0.01),
                'high': base_price * (1 + abs(np.random.randn()) * 0.02),
                'low': base_price * (1 - abs(np.random.randn()) * 0.02),
                'close': base_price * (1 + returns),
                'volume': np.random.randint(10000, 1000000),
                'amount': np.random.randint(100000, 10000000),
                'returns_1': np.random.randn() * 0.01,
                'returns_5': np.random.randn() * 0.02,
                'returns_20': np.random.randn() * 0.03,
                'volume_ma_5': np.random.randint(10000, 1000000),
                'volume_ma_20': np.random.randint(10000, 1000000),
                'rsi_14': np.random.uniform(0, 100),
                'fzt_score': np.random.uniform(0, 1),
                'fzt_momentum': np.random.randn(),
                'fzt_strength': np.random.uniform(0, 1),
                'future_return': np.random.randn() * 0.02  # 未来收益率
            }
            data.append(row)
    
    return pd.DataFrame(data)


def main():
    """主函数"""
    print("=" * 70)
    print("FZT排序增强策略完整管道示例")
    print("=" * 70)
    
    # 1. 创建示例数据
    print("\n1. 创建示例股票数据...")
    stock_data = create_sample_stock_data(n_dates=60, n_stocks=100)
    print(f"数据形状: {stock_data.shape}")
    print(f"日期范围: {stock_data['date'].min()} 到 {stock_data['date'].max()}")
    print(f"股票数量: {stock_data['instrument'].nunique()}")
    
    # 2. 分割数据
    print("\n2. 分割训练集和测试集...")
    dates = sorted(stock_data['date'].unique())
    split_idx = int(len(dates) * 0.7)
    
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    
    train_data = stock_data[stock_data['date'].isin(train_dates)].copy()
    test_data = stock_data[stock_data['date'].isin(test_dates)].copy()
    
    print(f"训练集: {len(train_dates)}天, {len(train_data)}条记录")
    print(f"测试集: {len(test_dates)}天, {len(test_data)}条记录")
    
    # 3. 创建和训练管道
    print("\n3. 创建FZT排序增强管道...")
    pipeline = FZTRankingPipeline()
    
    print("训练模型...")
    train_result = pipeline.train(train_data, val_data=None)
    
    print(f"训练完成! 最佳迭代: {train_result['best_iteration']}")
    
    # 4. 在测试集上预测
    print("\n4. 在测试集上进行预测...")
    test_predictions = pipeline.predict(test_data)
    
    print(f"预测完成! 数据形状: {test_predictions.shape}")
    print(f"预测分数统计:")
    print(f"  最小值: {test_predictions['ranking_score'].min():.4f}")
    print(f"  最大值: {test_predictions['ranking_score'].max():.4f}")
    print(f"  均值: {test_predictions['ranking_score'].mean():.4f}")
    
    # 5. 评估投资组合表现
    print("\n5. 评估投资组合表现...")
    portfolio_stats = pipeline.evaluate_portfolio(test_predictions, top_k=10)
    
    print("投资组合表现:")
    print(f"  总收益率: {portfolio_stats['total_return']:.2%}")
    print(f"  年化收益率: {portfolio_stats['annual_return']:.2%}")
    print(f"  年化波动率: {portfolio_stats['volatility']:.2%}")
    print(f"  夏普比率: {portfolio_stats['sharpe_ratio']:.2f}")
    print(f"  交易天数: {portfolio_stats['num_dates']}")
    print(f"  日均股票数: {portfolio_stats['avg_stocks_per_day']:.1f}")
    
    # 6. 特征重要性分析
    print("\n6. 特征重要性分析...")
    if pipeline.trainer:
        importance = pipeline.trainer.get_feature_importance()
        
        if importance:
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            print("Top 10重要特征:")
            for i, (feature, importance_score) in enumerate(sorted_importance[:10], 1):
                print(f"  {i:2d}. {feature:20s}: {importance_score:.4f}")
        else:
            print("  特征重要性为空，可能需要更多数据或调整参数")
    
    # 7. 保存模型
    print("\n7. 保存模型...")
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'fzt_ranking_model.pkl')
        
        if pipeline.trainer:
            pipeline.trainer.save_model(model_path)
            print(f"模型已保存到: {model_path}")
            
            # 验证模型文件大小
            file_size = os.path.getsize(model_path) / 1024  # KB
            print(f"模型文件大小: {file_size:.1f} KB")
    
    print("\n" + "=" * 70)
    print("管道示例完成!")
    print("=" * 70)
    
    # 8. 使用建议
    print("\n使用建议:")
    print("1. 在实际应用中，使用更多历史数据（至少2-3年）")
    print("2. 使用真实股票数据和FZT因子")
    print("3. 调整LightGBM参数以获得更好的性能")
    print("4. 使用时间序列交叉验证评估模型稳定性")
    print("5. 定期重新训练模型以适应市场变化")


if __name__ == '__main__':
    main()