# 特征重要性分析模块

## 概述

特征重要性分析模块提供了多种方法来计算和分析特征在机器学习模型中的重要性。该模块专门为FZT排序增强策略设计，支持大规模特征集的高效分析。

## 功能特性

### 1. 多种重要性计算方法
- **LightGBM原生重要性**: `gain`（增益）和 `split`（分裂次数）
- **Permutation重要性**: 通过打乱特征值计算性能下降
- **SHAP值**: SHapley Additive exPlanations（需要安装shap包）
- **相关性分析**: 特征与目标变量的相关性

### 2. 特征重要性可视化
- **重要性条形图**: 显示特征重要性排名
- **累积重要性图**: 显示累积重要性曲线和帕累托分析
- **相关性热力图**: 显示特征与目标的相关性
- **SHAP摘要图**: 显示SHAP值分布

### 3. 分析报告与建议
- **详细报告**: 包含摘要、统计信息和累积重要性分析
- **特征选择建议**: 基于累积重要性阈值提供特征选择建议
- **方法比较**: 比较不同重要性方法的结果

## 快速开始

### 安装依赖
```bash
pip install lightgbm pandas numpy matplotlib seaborn scikit-learn
# 可选：SHAP支持
pip install shap
```

### 基本用法
```python
from src.evaluation.feature_importance import FeatureImportanceAnalyzer

# 创建分析器
analyzer = FeatureImportanceAnalyzer(random_state=42)

# 计算gain重要性
importance_df = analyzer.calculate_importance(
    model=model,
    X=X_features,
    y=y_target,
    method='gain'
)

# 生成报告
report = analyzer.generate_report(importance_df, top_n=10)

# 可视化
fig = analyzer.visualize_importance(importance_df, plot_type='bar')
```

### 完整示例
```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.evaluation.feature_importance import FeatureImportanceAnalyzer

# 1. 准备数据
np.random.seed(42)
X = pd.DataFrame(np.random.randn(1000, 20))
y = 0.3 * X[0] + 0.2 * X[1] + 0.1 * np.random.randn(1000)

# 2. 训练模型
train_data = lgb.Dataset(X, label=y)
params = {'objective': 'regression', 'metric': 'l2', 'verbose': -1}
model = lgb.train(params, train_data, num_boost_round=50)

# 3. 分析特征重要性
analyzer = FeatureImportanceAnalyzer()

# 计算不同方法的重要性
gain_importance = analyzer.calculate_importance(model, X, y, 'gain')
permutation_importance = analyzer.calculate_importance(model, X, y, 'permutation')
correlation_importance = analyzer.calculate_importance(None, X, y, 'correlation')

# 生成报告
report = analyzer.generate_report(gain_importance)

# 特征选择建议
suggestions = analyzer.get_feature_selection_suggestions(
    gain_importance,
    threshold=0.8,
    min_features=5
)
```

## API参考

### FeatureImportanceAnalyzer类

#### 初始化
```python
analyzer = FeatureImportanceAnalyzer(random_state=42)
```

#### 主要方法

##### `calculate_importance(model, X, y, method='gain', **kwargs)`
计算特征重要性。

**参数**:
- `model`: 训练好的LightGBM模型（相关性分析可为None）
- `X`: 特征DataFrame
- `y`: 目标变量Series
- `method`: 计算方法，可选 'gain', 'split', 'permutation', 'shap', 'correlation'
- `**kwargs`: 方法特定参数

**返回**: 重要性DataFrame

##### `get_feature_ranking(importance_df, importance_col='importance')`
获取特征排名。

**返回**: 包含排名、归一化重要性、累积重要性的DataFrame

##### `generate_report(importance_df, top_n=10)`
生成特征重要性报告。

**返回**: 包含摘要、统计信息、累积分析的字典

##### `visualize_importance(importance_df, plot_type='bar', **kwargs)`
可视化特征重要性。

**参数**:
- `plot_type`: 图表类型，可选 'bar', 'cumulative', 'heatmap', 'shap_summary'
- `**kwargs`: 图表特定参数

**返回**: matplotlib Figure对象

##### `get_feature_selection_suggestions(importance_df, threshold=0.8, ...)`
获取特征选择建议。

**返回**: 包含选择特征、数量、累积重要性的字典

##### `compare_importance_methods(model, X, y, methods=None, **kwargs)`
比较不同重要性方法。

**返回**: 字典，键为方法名，值为重要性DataFrame

### 工具函数

#### `analyze_feature_importance(model, X, y, methods=None, save_path=None, **kwargs)`
一站式特征重要性分析工具。

**返回**: 包含分析器、结果、报告、比较图的字典

## 在FZT项目中的使用

### 与现有模块集成
```python
# 在训练流程后添加特征重要性分析
from src.evaluation.feature_importance import analyze_feature_importance

# 假设已有训练好的模型和特征数据
analysis_results = analyze_feature_importance(
    model=trained_model,
    X=X_train_features,
    y=y_train_labels,
    methods=['gain', 'permutation', 'correlation'],
    save_path='results/feature_importance_report.json'
)

# 获取特征选择建议用于后续训练
selected_features = analysis_results['analyzer'].get_feature_selection_suggestions(
    analysis_results['results']['gain'],
    threshold=0.85
)['selected_features']
```

### 支持排序任务
模块特别优化了排序任务的支持：
- Permutation重要性使用NDCG作为评分指标
- 支持大规模特征集的高效计算
- 提供特征选择建议以优化模型性能

## 配置选项

### 重要性计算参数
- `n_repeats`: Permutation重要性的重复次数（默认5）
- `sample_fraction`: 采样比例用于加速计算（默认0.3）
- `scoring`: 评分指标，支持 'mse', 'ndcg' 或自定义函数

### 可视化参数
- `top_n`: 显示的特征数量（默认20）
- `figsize`: 图表大小（默认(12, 8)）
- `plot_type`: 图表类型

### 特征选择参数
- `threshold`: 累积重要性阈值（默认0.8）
- `min_features`: 最小特征数量（默认5）
- `max_features`: 最大特征数量（可选）

## 性能优化

### 大规模特征集处理
1. **采样计算**: 支持采样以减少计算时间
2. **并行处理**: Permutation重要性可并行化（需用户实现）
3. **内存优化**: 分批处理大规模数据

### 建议配置
```python
# 对于大规模特征集（>1000个特征）
analyzer = FeatureImportanceAnalyzer()
importance_df = analyzer.calculate_importance(
    model=model,
    X=X_large,
    y=y,
    method='permutation',
    n_repeats=3,           # 减少重复次数
    sample_fraction=0.2,   # 减少采样比例
    scoring='ndcg'         # 使用排序指标
)
```

## 测试与验证

模块包含完整的测试套件：
```bash
# 运行测试
pytest tests/test_feature_importance.py -v

# 运行特定测试
pytest tests/test_feature_importance.py::TestFeatureImportanceAnalyzer::test_calculate_gain_importance
```

## 注意事项

1. **SHAP依赖**: SHAP分析需要额外安装 `shap` 包
2. **LightGBM版本**: 不同版本支持的importance_type可能不同
3. **内存使用**: 大规模特征集可能需要大量内存
4. **计算时间**: Permutation和SHAP分析可能较慢

## 扩展开发

### 添加新的重要性方法
继承 `FeatureImportanceAnalyzer` 类并实现新的计算方法：

```python
class CustomFeatureImportanceAnalyzer(FeatureImportanceAnalyzer):
    def _calculate_custom_importance(self, model, X, y, **kwargs):
        # 实现自定义方法
        pass
```

### 自定义可视化
重写 `visualize_importance` 方法或添加新的图表类型。

## 故障排除

### 常见问题
1. **ImportError: No module named 'shap'**
   - 解决方案: `pip install shap` 或使用其他方法

2. **KeyError: 'cover'**
   - 原因: LightGBM版本不支持cover重要性
   - 解决方案: 使用 'gain' 或 'split'

3. **内存不足**
   - 解决方案: 减小 `sample_fraction` 或使用分批处理

### 调试模式
```python
import warnings
warnings.filterwarnings('default')  # 显示所有警告
```

## 版本历史

- v1.0.0: 初始版本，支持多种重要性计算方法和可视化
- 集成到FZT排序增强策略项目

## 贡献指南

欢迎提交Issue和Pull Request来改进模块功能。