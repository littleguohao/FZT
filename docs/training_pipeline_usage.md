# FZT排序增强策略训练流水线使用说明

## 概述

本训练流水线提供了端到端的FZT排序增强策略训练功能，包括数据加载、特征工程、标签构造、模型训练、评估和结果保存。

## 文件结构

```
FZT_Quant/
├── config/
│   ├── training_pipeline.yaml      # 训练流水线主配置文件
│   ├── ranking_config.yaml         # 排序模型配置
│   └── feature_config.yaml         # 特征工程配置
├── scripts/
│   └── train_ranking_model.py      # 训练脚本主文件
├── src/                            # 源代码模块
├── results/                        # 训练结果输出目录
└── docs/                           # 文档目录
```

## 快速开始

### 1. 基本训练

```bash
# 使用默认配置进行训练
python scripts/train_ranking_model.py

# 指定配置文件
python scripts/train_ranking_model.py --config config/training_pipeline.yaml
```

### 2. 指定数据范围

```bash
# 指定训练数据时间范围
python scripts/train_ranking_model.py \
  --start-date 2005-01-01 \
  --end-date 2020-12-31
```

### 3. 指定模型参数

```bash
# 自定义模型参数
python scripts/train_ranking_model.py \
  --learning-rate 0.05 \
  --num-leaves 64 \
  --num-iterations 200
```

### 4. 交叉验证训练

```bash
# 使用交叉验证
python scripts/train_ranking_model.py \
  --mode cv \
  --cv-folds 5
```

### 5. 仅评估模式

```bash
# 加载已有模型进行评估
python scripts/train_ranking_model.py \
  --eval-only \
  --model-path results/models/fzt_model.pkl
```

### 6. 详细输出模式

```bash
# 显示详细日志
python scripts/train_ranking_model.py --verbose
```

### 7. 干运行模式

```bash
# 测试配置但不保存结果
python scripts/train_ranking_model.py --dry-run
```

## 配置文件说明

### training_pipeline.yaml

主配置文件包含以下主要部分：

1. **data**: 数据配置
   - `source`: 数据源 (qlib/csv/hybrid)
   - `time_range`: 时间范围配置
   - `stock_pool`: 股票池配置

2. **features**: 特征工程配置
   - `feature_types`: 启用的特征类型
   - `neutralization`: 特征中性化配置
   - `selection`: 特征选择配置

3. **label**: 标签工程配置
   - `type`: 标签类型 (future_return/rank/binary)
   - `fzt_condition`: FZT选股条件

4. **model**: 模型配置
   - `trainer_class`: 训练器类
   - `params`: 模型参数

5. **training**: 训练配置
   - `mode`: 训练模式 (single/cv/rolling)
   - `cross_validation`: 交叉验证配置

6. **evaluation**: 评估配置
   - `metrics`: 评估指标
   - `feature_importance`: 特征重要性评估

7. **output**: 输出配置
   - `model`: 模型保存配置
   - `evaluation_report`: 评估报告配置

### 环境变量支持

支持以下环境变量覆盖配置：

```bash
# 设置数据路径
export DATA_PATH=/path/to/data

# 设置设备
export DEVICE=cpu  # 或 gpu

# 设置并行工作数
export NUM_WORKERS=4

# 设置日志级别
export LOG_LEVEL=DEBUG
```

## 训练流程

### 1. 数据加载阶段
- 根据配置加载QLib/CSV/混合数据
- 应用股票池过滤
- 数据预处理和清洗

### 2. 特征工程阶段
- 生成基础价格特征
- 计算FZT衍生特征
- 生成技术指标特征
- 特征中性化处理
- 特征选择和标准化

### 3. 标签构造阶段
- 计算未来收益率
- 应用FZT选股条件
- 创建排序标签
- 标签后处理

### 4. 模型训练阶段
- 初始化LambdaRank训练器
- 配置训练参数
- 执行训练（支持早停）
- 获取特征重要性

### 5. 模型评估阶段
- 在测试集上进行预测
- 计算排序评估指标（NDCG, MAP等）
- 计算投资评估指标（IC, Sharpe等）
- 生成评估报告

### 6. 结果保存阶段
- 保存训练好的模型
- 保存特征重要性
- 保存训练历史
- 生成HTML评估报告

## 输出文件

训练完成后，会在以下目录生成文件：

### results/models/
- `fzt_ranking_model_YYYYMMDD_HHMMSS.pkl` - 训练好的模型

### results/feature_importance/
- `feature_importance_YYYYMMDD_HHMMSS.csv` - 特征重要性排名

### results/training_history/
- `training_history_YYYYMMDD_HHMMSS.json` - 训练历史记录

### results/reports/
- `evaluation_report_YYYYMMDD_HHMMSS.html` - HTML格式评估报告

### results/visualizations/
- 各种可视化图表（如果启用）

### logs/
- `training_pipeline.log` - 训练日志文件

## 错误处理

### 常见错误及解决方法

1. **数据加载失败**
   - 检查QLib数据路径是否正确
   - 确认数据文件存在
   - 检查网络连接（如果需要下载数据）

2. **特征工程失败**
   - 检查特征配置是否正确
   - 确认数据包含必要的字段
   - 检查内存是否充足

3. **模型训练失败**
   - 检查训练参数是否合理
   - 确认数据格式正确
   - 检查依赖库版本

4. **内存不足**
   - 减少数据量或特征数量
   - 增加系统内存
   - 使用更小的批次大小

### 错误处理策略

配置文件支持以下错误处理策略：

```yaml
error_handling:
  strategy: "continue"  # continue | stop | skip
  retry:
    enabled: true
    max_attempts: 3
    delay: 5
```

## 性能优化建议

### 1. 数据层面
- 使用适当的数据采样
- 启用数据缓存
- 使用合适的数据格式（parquet > csv）

### 2. 特征层面
- 减少特征维度
- 启用特征选择
- 使用批量特征计算

### 3. 训练层面
- 调整批次大小
- 使用早停机制
- 启用并行处理

### 4. 硬件层面
- 使用GPU加速（如果支持）
- 增加内存
- 使用SSD存储

## 扩展和定制

### 添加新特征类型

1. 在`src/feature_eng.py`中添加特征计算函数
2. 在`config/feature_config.yaml`中配置新特征
3. 在训练脚本中启用新特征

### 添加新模型类型

1. 在`src/ranking_model/`中添加新的训练器类
2. 实现`train()`和`predict()`方法
3. 在配置文件中添加新模型配置

### 添加新评估指标

1. 在`src/backtest/performance_evaluator.py`中添加指标计算函数
2. 在配置文件中添加新指标配置
3. 在评估报告中显示新指标

## 监控和调试

### 训练监控

启用资源监控：

```yaml
monitoring:
  resources:
    enabled: true
    monitor_memory: true
    monitor_cpu: true
```

### 调试模式

使用详细日志和检查点：

```bash
python scripts/train_ranking_model.py --verbose --dry-run
```

### 性能分析

使用Python分析工具：

```bash
# 使用cProfile进行性能分析
python -m cProfile -o profile.stats scripts/train_ranking_model.py

# 分析结果
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('time').print_stats(20)"
```

## 常见问题解答

### Q1: 训练时间太长怎么办？
A: 可以尝试以下方法：
- 减少数据量（缩短时间范围）
- 减少特征数量
- 使用更简单的模型
- 启用并行处理

### Q2: 模型过拟合怎么办？
A: 可以尝试以下方法：
- 增加正则化参数
- 使用更早的早停
- 增加训练数据
- 使用交叉验证

### Q3: 如何评估模型效果？
A: 查看以下指标：
- 排序指标：NDCG@5, MAP@5
- 投资指标：IC, Sharpe Ratio
- 特征重要性：前20个重要特征

### Q4: 如何部署训练好的模型？
A: 使用以下步骤：
1. 加载保存的模型文件
2. 准备新的数据
3. 使用相同的特征工程流程
4. 调用模型的predict()方法

## 联系方式

如有问题或建议，请联系：
- 项目仓库: [FZT_Quant]
- 邮箱: [your-email@example.com]
- 文档更新: 2026-03-02