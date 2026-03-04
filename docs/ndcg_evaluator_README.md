# NDCG评估器文档

## 概述
NDCG评估器是FZT排序增强策略的核心评估组件，用于评估排序模型的质量。它实现了多种排序评估指标，支持单个查询和批量分组评估。

## 功能特性

### 支持的评估指标
1. **NDCG@K** - 归一化折扣累积增益（主要指标）
2. **MAP@K** - 平均精度
3. **MRR** - 平均倒数排名
4. **Precision@K** - 精度
5. **Recall@K** - 召回率
6. **F1@K** - F1分数

### 核心功能
- 单个查询评估
- 批量分组评估（按日期/查询）
- 详细评估报告生成
- 阈值二值化评估
- 便捷函数接口

## 使用方法

### 基本使用
```python
from src.evaluation.ndcg_evaluator import NDCGEvaluator

# 创建评估器
evaluator = NDCGEvaluator(k_values=[1, 3, 5, 10])

# 单个查询评估
results = evaluator.evaluate_single_query(y_true, y_pred)

# 批量评估
batch_results = evaluator.evaluate_batch(y_true, y_pred, groups)

# 生成报告
report = evaluator.get_report(batch_results)
print(report)
```

### 便捷函数
```python
from src.evaluation.ndcg_evaluator import (
    compute_ndcg, compute_map, compute_mrr, 
    compute_precision_recall
)

# 计算单个指标
ndcg = compute_ndcg(y_true, y_pred, k=10)
map_score = compute_map(y_true, y_pred, k=10)
mrr = compute_mrr(y_true, y_pred, k=10)
precision, recall = compute_precision_recall(y_true, y_pred, k=10)
```

### 与LambdaRank训练器集成
```python
from src.ranking_model.lambdarank_trainer import LambdaRankTrainer

# 创建训练器
trainer = LambdaRankTrainer(config_path='config/ranking_config.yaml')

# 全面评估
comprehensive_results = trainer.evaluate_comprehensive(features, labels)

# 获取评估报告
report = trainer.get_evaluation_report(features, labels)
```

## 数据格式

### 输入数据
- `y_true`: 真实相关性分数（未来收益率）
- `y_pred`: 预测排序分数
- `groups`: 分组信息数组（按日期分组）

### 输出格式
```python
{
    'summary': {
        'ndcg': {
            1: {'mean': 0.85, 'std': 0.05, 'min': 0.75, 'max': 0.95, 'median': 0.86},
            3: {'mean': 0.82, 'std': 0.06, 'min': 0.70, 'max': 0.92, 'median': 0.83},
            # ...
        },
        'map': { ... },
        'mrr': { ... },
        # ...
    },
    'per_query': [...],  # 每个查询的详细结果
    'n_queries': 10,
    'total_samples': 1000
}
```

## 技术细节

### NDCG计算公式
```
DCG@K = Σ(i=1 to K) (rel_i / log2(i + 1))
IDCG@K = 理想排序下的DCG@K
NDCG@K = DCG@K / IDCG@K
```

### MAP计算公式
```
AP@K = Σ(i=1 to K) (precision@i × rel_i) / min(K, total_relevant)
MAP@K = 平均所有查询的AP@K
```

### MRR计算公式
```
MRR = 1 / rank_of_first_relevant
```

## 测试验证
所有功能都有完整的单元测试：
```bash
python -m pytest tests/test_ndcg_evaluator.py -v
```

## 文件结构
```
src/evaluation/
├── __init__.py          # 模块导出
├── ndcg_evaluator.py    # 主实现文件
└── feature_importance.py # 特征重要性评估

tests/
└── test_ndcg_evaluator.py  # 测试文件
```

## 性能考虑
- 使用NumPy向量化计算提高性能
- 支持大规模批量评估
- 内存友好的分组处理
- 避免重复计算

## 扩展性
- 可轻松添加新的评估指标
- 支持自定义K值列表
- 可配置的评估参数
- 模块化设计便于集成