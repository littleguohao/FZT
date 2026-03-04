# 排序标签工程模块使用指南

## 概述

`label_engineering.py` 模块为LambdaRank排序学习提供标签构造功能。它根据未来收益率和FZT选股条件创建排序标签，并组织成LambdaRank所需的数据格式。

## 核心功能

### 1. 创建排序标签
```python
from src.ranking_model.label_engineering import create_ranking_labels

# 二分类标签
binary_labels = create_ranking_labels(
    future_returns=future_returns_series,
    fzt_condition=fzt_condition_series,
    top_k=5,
    label_type='binary'
)

# 连续标签
continuous_labels = create_ranking_labels(
    future_returns=future_returns_series,
    fzt_condition=fzt_condition_series,
    top_k=5,
    label_type='continuous'
)
```

### 2. 创建LambdaRank数据集
```python
from src.ranking_model.label_engineering import create_lambdarank_dataset

dataset = create_lambdarank_dataset(
    features=features_df,
    labels=labels_series
)

# 使用自定义分组
dataset = create_lambdarank_dataset(
    features=features_df,
    labels=labels_series,
    group_sizes=[10, 15, 20]  # 自定义分组大小
)
```

### 3. 验证标签质量
```python
from src.ranking_model.label_engineering import validate_label_quality

quality_metrics = validate_label_quality(
    labels=labels_series,
    future_returns=future_returns_series,
    fzt_condition=fzt_condition_series
)
```

### 4. 创建交叉验证折叠
```python
from src.ranking_model.label_engineering import create_cross_validation_folds

folds = create_cross_validation_folds(
    features=features_df,
    labels=labels_series,
    n_folds=5,
    strategy='time_series'  # 或 'random'
)
```

## 数据格式要求

### 输入数据格式
所有数据必须使用 `(date, instrument)` 多索引：

```python
import pandas as pd

# 创建多索引
index = pd.MultiIndex.from_product(
    [dates_list, instruments_list],
    names=['date', 'instrument']
)

# 特征DataFrame
features = pd.DataFrame(data, index=index)

# 未来收益率Series
future_returns = pd.Series(data, index=index)

# FZT选股条件Series
fzt_condition = pd.Series(data, index=index)
```

### 输出数据格式

#### 标签格式
- **二分类标签**: 值为0或1
- **连续标签**: 值为0到1之间的标准化排名

#### LambdaRank数据集格式
```python
{
    'features': pd.DataFrame,  # 特征数据
    'labels': pd.Series,       # 标签数据
    'groups': np.ndarray       # 分组信息
}
```

## 使用示例

### 完整工作流程
```python
import pandas as pd
import numpy as np
from src.ranking_model.label_engineering import *

# 1. 准备数据
dates = pd.date_range('2024-01-01', periods=20, freq='D')
instruments = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

index = pd.MultiIndex.from_product([dates, instruments], names=['date', 'instrument'])

# 特征
features = pd.DataFrame(np.random.randn(len(index), 10), index=index)

# 未来收益率
future_returns = pd.Series(np.random.randn(len(index)) * 0.1, index=index)

# FZT选股条件（模拟）
fzt_condition = pd.Series(np.random.choice([0, 1], size=len(index), p=[0.4, 0.6]), index=index)

# 2. 创建标签
labels = create_ranking_labels(
    future_returns=future_returns,
    fzt_condition=fzt_condition,
    top_k=3,
    label_type='binary'
)

# 3. 验证标签质量
metrics = validate_label_quality(labels, future_returns, fzt_condition)
print(f"标签与收益率相关性: {metrics['correlation']:.4f}")

# 4. 创建LambdaRank数据集
dataset = create_lambdarank_dataset(features, labels)

# 5. 创建交叉验证
folds = create_cross_validation_folds(features, labels, n_folds=5, strategy='time_series')
```

### 与LightGBM LambdaRank集成
```python
import lightgbm as lgb

# 准备LambdaRank数据
dataset = create_lambdarank_dataset(features, labels)

# 转换为LightGBM Dataset
lgb_train = lgb.Dataset(
    dataset['features'].values,
    label=dataset['labels'].values,
    group=dataset['groups']
)

# 训练参数
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [5, 10],
    'learning_rate': 0.1,
    'num_leaves': 31,
    'verbosity': -1
}

# 训练模型
gbm = lgb.train(params, lgb_train, num_boost_round=100)
```

## 注意事项

### 1. 无未来数据泄露
- 标签构造只使用当日信息
- 按日期分组独立处理
- 确保训练时不会看到未来的信息

### 2. 数据质量
- 自动处理缺失值
- 验证索引一致性
- 检查标签范围

### 3. 边缘情况处理
- 无FZT选中股票时，所有标签为0
- top_k大于可用股票数量时，所有选中股票标签为1
- 单只股票时，连续标签为1

### 4. 性能考虑
- 使用向量化操作
- 按日期分组处理，避免全量排序
- 支持大规模数据

## 测试

运行测试确保功能正常：
```bash
# 运行单元测试
python -m pytest tests/test_label_engineering.py -v

# 运行集成测试
python test_label_integration.py
```

## 常见问题

### Q: 标签与未来收益率相关性低怎么办？
A: 检查FZT选股条件的有效性，调整top_k参数，或尝试连续标签。

### Q: 如何选择top_k值？
A: 根据实际投资策略确定，通常选择每天实际买入的股票数量。

### Q: 二分类和连续标签哪个更好？
A: 连续标签包含更多排序信息，但二分类标签更稳定。建议都尝试。

### Q: 数据量很大时性能如何？
A: 模块使用向量化操作和分组处理，适合大规模数据。如遇性能问题，可考虑分批处理。