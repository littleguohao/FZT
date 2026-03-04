# LambdaRank训练器文档

## 概述

LambdaRank训练器是一个基于LightGBM的排序学习模型训练器，专门为FZT排序增强策略设计。它实现了LambdaRank算法，用于训练股票排序模型。

## 功能特性

### 核心功能
1. **LambdaRank模型训练** - 使用LightGBM的LambdaRank目标函数
2. **时间序列交叉验证** - 支持滚动窗口和时间序列分割
3. **早停机制** - 防止过拟合，提高泛化能力
4. **模型评估** - 计算NDCG@k等排序指标
5. **特征重要性分析** - 识别关键特征
6. **模型保存和加载** - 支持持久化存储
7. **预测功能** - 对新数据进行排序预测

### 技术特点
- 支持多索引数据格式 (date, instrument)
- 自动处理缺失值和数据验证
- 可配置的LightGBM参数
- 支持自定义标签增益(label_gain)
- 时间序列无未来函数保证

## 安装要求

```bash
pip install lightgbm>=4.0.0
pip install pandas>=1.5.0
pip install numpy>=1.21.0
pip install pyyaml>=6.0
```

## 快速开始

### 基本使用

```python
import pandas as pd
import numpy as np
from ranking_model.lambdarank_trainer import LambdaRankTrainer

# 1. 准备数据
# 特征DataFrame，索引为(date, instrument)
features = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000)
}, index=pd.MultiIndex.from_product([
    pd.date_range('2024-01-01', periods=100),
    ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
], names=['date', 'instrument']))

# 标签Series，索引与特征相同（LambdaRank需要整数标签）
labels = pd.Series(np.random.randint(1, 6, size=1000), index=features.index)

# 2. 创建训练器
trainer = LambdaRankTrainer()

# 3. 训练模型
result = trainer.train(features, labels)

# 4. 进行预测
predictions = trainer.predict(features)

# 5. 评估模型
metrics = trainer.evaluate(features, labels)
```

### 使用配置文件

```python
from ranking_model.lambdarank_trainer import LambdaRankTrainer

# 从配置文件创建训练器
trainer = LambdaRankTrainer(config_path='config/ranking_config.yaml')

# 或者使用配置字典
config = {
    'model': {
        'type': 'lightgbm_lambdarank',
        'objective': 'ranking'
    },
    'lightgbm': {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'num_iterations': 100
    }
}
trainer = LambdaRankTrainer(config_dict=config)
```

### 交叉验证

```python
# 时间序列交叉验证
cv_results = trainer.cross_validate(features, labels, n_folds=5)

print(f"平均NDCG: {cv_results['mean_metrics']['ndcg']:.4f}")
print(f"平均NDCG@5: {cv_results['mean_metrics']['ndcg@5']:.4f}")
```

## 配置说明

### 默认配置

```yaml
model:
  type: lightgbm_lambdarank
  objective: ranking
  description: LightGBM LambdaRank排序模型

lightgbm:
  objective: lambdarank
  metric: ndcg
  ndcg_eval_at: [1, 3, 5, 10]
  learning_rate: 0.05
  num_leaves: 31
  max_depth: 5
  min_data_in_leaf: 20
  num_iterations: 100
  early_stopping_rounds: 10
  random_state: 42

data_split:
  time_series_cv: true
  rolling_window:
    enabled: true
    window_size: 252
    step_size: 63
    min_train_size: 126
  query_group_column: date

training:
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.001
```

### LambdaRank特定参数

```yaml
lightgbm:
  lambdarank:
    norm: true           # 是否标准化Lambda梯度
    truncation_level: 10 # 截断级别
    sigmoid: 1.0         # Sigmoid参数
    
  # LambdaRank需要label_gain参数
  # 默认使用指数增益: 0, 1, 3, 7, 15, 31, ...
  label_gain: [0, 1, 3, 7, 15, 31]
```

## API参考

### LambdaRankTrainer类

#### 初始化
```python
trainer = LambdaRankTrainer(config_path=None, config_dict=None)
```

#### 主要方法

**train(features, labels, validation_data=None)**
- 训练LambdaRank模型
- 返回包含模型、指标和特征重要性的字典

**cross_validate(features, labels, n_folds=5)**
- 执行时间序列交叉验证
- 返回每个折叠的指标和平均指标

**predict(features)**
- 对特征数据进行预测
- 返回排序分数Series

**evaluate(features, labels)**
- 评估模型性能
- 返回指标字典

**get_feature_importance()**
- 获取特征重要性
- 返回特征名到重要性分数的字典

**save_model(filepath)**
- 保存模型到文件

**load_model(filepath)**
- 从文件加载模型

**get_config()**
- 获取当前配置

**update_config(config_updates)**
- 更新配置

### 工具函数

**create_lambdarank_trainer_from_config(config_path)**
- 从配置文件创建训练器

**train_lambdarank_model(features, labels, config_path=None)**
- 训练模型的便捷函数

## 数据格式要求

### 特征数据
- 类型: pandas DataFrame
- 索引: MultiIndex (date, instrument)
- 列: 特征值，数值类型
- 示例:
  ```python
  features = pd.DataFrame({
      'feature1': [1.0, 2.0, ...],
      'feature2': [0.5, 1.5, ...]
  }, index=pd.MultiIndex.from_tuples([
      ('2024-01-01', 'AAPL'),
      ('2024-01-01', 'GOOGL'),
      ...
  ], names=['date', 'instrument']))
  ```

### 标签数据
- 类型: pandas Series
- 索引: 与特征数据相同
- 值: 整数标签（LambdaRank要求）
- 示例:
  ```python
  labels = pd.Series([1, 2, 3, 4, 5, ...], index=features.index)
  ```

## 性能指标

### NDCG (Normalized Discounted Cumulative Gain)
- **NDCG@1**: 排名第一的准确度
- **NDCG@3**: 前三名的准确度
- **NDCG@5**: 前五名的准确度
- **NDCG@10**: 前十名的准确度
- **平均NDCG**: 所有k值的平均

### 计算公式
```
DCG@k = Σ(i=1 to k) (2^rel_i - 1) / log2(i + 1)
IDCG@k = 理想排序下的DCG
NDCG@k = DCG@k / IDCG@k
```

## 最佳实践

### 1. 数据准备
```python
# 确保数据按日期排序
features = features.sort_index()

# 处理缺失值
features = features.fillna(method='ffill').fillna(0)

# 验证数据格式
assert isinstance(features.index, pd.MultiIndex)
assert features.index.names == ['date', 'instrument']
```

### 2. 参数调优
```python
# 调整学习率
config_updates = {
    'lightgbm': {
        'learning_rate': 0.1,  # 尝试0.01, 0.05, 0.1
        'num_leaves': 63,      # 尝试31, 63, 127
        'max_depth': 7         # 尝试3, 5, 7, -1(无限制)
    }
}
trainer.update_config(config_updates)
```

### 3. 早停策略
```python
# 使用验证集进行早停
result = trainer.train(
    train_features, 
    train_labels,
    validation_data=(val_features, val_labels)
)

print(f"最佳迭代: {result['best_iteration']}")
print(f"训练历史: {result['train_history']}")
```

### 4. 特征重要性分析
```python
importance = trainer.get_feature_importance()

# 按重要性排序
sorted_importance = sorted(
    importance.items(), 
    key=lambda x: x[1], 
    reverse=True
)

for feature, score in sorted_importance[:10]:
    print(f"{feature}: {score:.4f}")
```

## 故障排除

### 常见问题

**1. 标签类型错误**
```
LightGBMError: label should be int type for ranking task
```
解决方案: 确保标签是整数类型
```python
labels = labels.astype(int)
```

**2. 分组信息错误**
```
ValueError: 分组大小之和必须等于数据点数量
```
解决方案: 确保数据按日期正确分组
```python
features = features.sort_index()
```

**3. 特征列不匹配**
```
ValueError: 特征缺少以下列: [...]
```
解决方案: 确保预测时使用与训练相同的特征列
```python
features_aligned = features[trainer.feature_names].copy()
```

**4. 模型过早停止**
```
最佳迭代: 1
```
解决方案: 
- 增加数据量
- 调整学习率
- 减少正则化参数
- 检查特征质量

### 调试建议

```python
# 1. 检查数据
print(f"特征形状: {features.shape}")
print(f"标签分布: {labels.value_counts()}")

# 2. 验证数据格式
print(f"索引类型: {type(features.index)}")
print(f"索引名称: {features.index.names}")

# 3. 检查配置
config = trainer.get_config()
print(f"LightGBM参数: {config['lightgbm']}")

# 4. 监控训练过程
result = trainer.train(features, labels)
print(f"训练历史: {result['train_history']}")
```

## 示例脚本

项目包含以下示例:
1. **examples/lambdarank_example.py** - 基础使用示例
2. **examples/fzt_ranking_pipeline.py** - 完整管道示例

运行示例:
```bash
python examples/lambdarank_example.py
```

## 测试

运行测试:
```bash
python -m pytest tests/test_lambdarank_trainer.py -v
```

测试覆盖率:
- 初始化测试
- 数据准备测试
- 训练测试
- 预测测试
- 交叉验证测试
- 模型保存/加载测试
- 特征重要性测试
- 错误处理测试

## 与FZT策略集成

### 集成步骤
1. 使用FZT因子作为特征
2. 创建基于未来收益率的排序标签
3. 使用LambdaRank训练器训练模型
4. 预测股票排序分数
5. 构建投资组合

### 示例代码
```python
from ranking_model.lambdarank_trainer import LambdaRankTrainer
from ranking_model.label_engineering import create_ranking_labels

# 准备FZT特征和标签
fzt_features = prepare_fzt_features(stock_data)
future_returns = calculate_future_returns(stock_data)
fzt_condition = (stock_data['fzt_score'] > 0.5).astype(int)

labels = create_ranking_labels(
    future_returns=future_returns,
    fzt_condition=fzt_condition,
    top_k=10,
    label_type='binary'
)

# 训练模型
trainer = LambdaRankTrainer()
result = trainer.train(fzt_features, labels)

# 预测和选股
predictions = trainer.predict(fzt_features)
selected_stocks = predictions.groupby('date').nlargest(10)
```

## 性能优化

### 内存优化
- 使用适当的数据类型
- 分批处理大数据
- 及时释放内存

### 速度优化
- 使用LightGBM的并行训练
- 调整树参数
- 使用特征子采样

### 精度优化
- 特征工程
- 参数调优
- 集成学习

## 版本历史

### v1.0.0 (2024-03-02)
- 初始版本
- 实现LambdaRank训练器核心功能
- 支持时间序列交叉验证
- 添加完整测试套件
- 创建示例和文档

## 贡献指南

1. Fork仓库
2. 创建功能分支
3. 提交更改
4. 添加测试
5. 更新文档
6. 创建Pull Request

## 许可证

MIT License

## 支持

如有问题，请提交Issue或联系维护者。