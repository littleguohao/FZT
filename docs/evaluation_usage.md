# FZT排序增强策略评估脚本使用说明

## 概述

`evaluate_strategy.py` 是FZT排序增强策略的完整评估流水线脚本，提供从数据收集到报告生成的全流程评估功能。

## 功能特点

1. **多维度评估**：覆盖模型性能、回测结果、风险评估、特征分析四个维度
2. **多种评估模式**：支持全面评估、快速评估、专项评估等模式
3. **智能报告生成**：自动生成HTML、YAML、JSON、Markdown格式报告
4. **改进建议**：基于评估结果提供具体的改进建议
5. **对比评估**：支持多策略对比分析

## 安装依赖

```bash
# 确保在项目根目录
cd /Users/lucky/.openclaw/workspace/.worktrees/fzt-quant/FZT_Quant

# 安装核心依赖
pip install -r requirements.txt

# 安装评估相关依赖
pip install scipy matplotlib seaborn plotly
```

## 快速开始

### 1. 全面评估（推荐）

```bash
python scripts/evaluate_strategy.py --mode comprehensive
```

### 2. 快速评估

```bash
python scripts/evaluate_strategy.py --mode quick
```

### 3. 专项评估

```bash
# 仅模型评估
python scripts/evaluate_strategy.py --mode model-only

# 仅回测评估
python scripts/evaluate_strategy.py --mode backtest-only

# 仅风险评估
python scripts/evaluate_strategy.py --mode risk-only

# 仅特征分析
python scripts/evaluate_strategy.py --mode feature-only
```

## 命令行参数详解

### 基本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 评估模式 | `comprehensive` |
| `--config` | 配置文件路径 | `config/evaluation_pipeline.yaml` |
| `--output-dir` | 输出目录 | `results/evaluation_reports` |
| `--verbose` | 详细输出模式 | `False` |
| `--dry-run` | 干运行模式 | `False` |

### 数据源参数

| 参数 | 说明 |
|------|------|
| `--train-data` | 训练数据路径 |
| `--backtest-data` | 回测数据路径 |
| `--model-data` | 模型数据路径 |

### 输出控制

| 参数 | 说明 |
|------|------|
| `--no-report` | 不生成报告 |
| `--compare` | 对比评估的策略列表 |

## 评估模式说明

### 1. comprehensive（全面评估）
- **描述**：执行所有评估模块
- **包含内容**：
  - 模型性能评估
  - 回测结果评估
  - 风险评估
  - 特征分析
  - 综合报告生成
- **适用场景**：完整的策略评估、定期全面检查

### 2. quick（快速评估）
- **描述**：只评估关键指标
- **包含内容**：
  - 核心模型指标（NDCG@5, MAP@5）
  - 关键回测指标（年化收益、夏普比率、最大回撤）
  - 简要报告
- **适用场景**：快速检查、日常监控

### 3. model-only（仅模型评估）
- **描述**：专注于模型性能评估
- **包含内容**：
  - 排序指标评估
  - 模型稳定性测试
  - 过拟合检测
  - 模型对比分析
- **适用场景**：模型调优、算法改进

### 4. backtest-only（仅回测评估）
- **描述**：专注于回测结果评估
- **包含内容**：
  - 收益指标评估
  - 风险指标评估
  - 成本分析
  - 基准对比
  - 绩效归因
- **适用场景**：策略优化、绩效分析

### 5. risk-only（仅风险评估）
- **描述**：专注于风险评估
- **包含内容**：
  - 市场风险评估
  - 流动性风险评估
  - 操作风险评估
  - 压力测试
- **适用场景**：风险控制、合规检查

### 6. feature-only（仅特征分析）
- **描述**：专注于特征分析
- **包含内容**：
  - 特征重要性分析
  - 特征相关性分析
  - 特征稳定性分析
  - 特征组合分析
- **适用场景**：特征工程优化、因子挖掘

## 配置文件说明

配置文件位于 `config/evaluation_pipeline.yaml`，包含完整的评估配置。

### 主要配置项

#### 1. 评估模式配置
```yaml
evaluation:
  mode: "comprehensive"  # 评估模式
```

#### 2. 数据源配置
```yaml
data_sources:
  training:
    path: "results/training_reports"
    format: "yaml"
    required: true
```

#### 3. 模型评估配置
```yaml
model_evaluation:
  enabled: true
  metrics: ["ndcg@5", "map@5", "mrr", "precision@5", "recall@5"]
  stability_test: true
  overfitting_test: true
```

#### 4. 回测评估配置
```yaml
backtest_evaluation:
  enabled: true
  metrics:
    return: ["total_return", "annual_return", "sharpe_ratio"]
    risk: ["max_drawdown", "volatility", "var_95"]
    cost: ["total_cost", "cost_ratio"]
```

#### 5. 报告生成配置
```yaml
reporting:
  output_dir: "results/evaluation_reports"
  formats: ["html", "yaml", "json", "markdown"]
  include_charts: true
  include_recommendations: true
```

## 输出文件说明

评估完成后，会在输出目录生成以下文件：

### 1. 报告文件
- `evaluation_report_YYYYMMDD_HHMMSS.html` - HTML格式报告（可视化）
- `evaluation_report_YYYYMMDD_HHMMSS.yaml` - YAML格式报告（结构化）
- `evaluation_report_YYYYMMDD_HHMMSS.json` - JSON格式报告（程序可读）
- `evaluation_report_YYYYMMDD_HHMMSS.md` - Markdown格式报告（文档）

### 2. 日志文件
- `logs/evaluation.log` - 评估过程日志

### 3. 缓存文件
- `.cache/evaluation/` - 评估缓存目录

## 报告内容详解

### HTML报告包含以下部分：

1. **总体评估**
   - 综合评分（0-1分）
   - 优势列表
   - 劣势列表
   - 关键发现

2. **模型评估摘要**
   - 排序指标（NDCG@5, MAP@5等）
   - 模型稳定性
   - 过拟合检测

3. **回测评估摘要**
   - 收益指标（年化收益、夏普比率等）
   - 风险指标（最大回撤、波动率等）
   - 成本分析

4. **风险评估摘要**
   - 市场风险评分
   - 流动性风险评分
   - 操作风险评分

5. **改进建议**
   - 按优先级排序的建议
   - 具体行动方案

## 使用示例

### 示例1：完整评估流程
```bash
# 1. 运行全面评估
python scripts/evaluate_strategy.py --mode comprehensive

# 2. 查看生成的报告
open results/evaluation_reports/evaluation_report_*.html
```

### 示例2：自定义数据源
```bash
# 使用特定数据源进行评估
python scripts/evaluate_strategy.py \
  --mode comprehensive \
  --train-data results/fzt_core_training/ \
  --backtest-data results/backtest/ \
  --output-dir results/custom_evaluation/
```

### 示例3：对比评估
```bash
# 对比两个策略版本
python scripts/evaluate_strategy.py \
  --compare strategy_v1 strategy_v2 \
  --output-dir results/comparison/
```

### 示例4：定期监控
```bash
# 使用快速评估进行日常监控
python scripts/evaluate_strategy.py --mode quick --verbose
```

## 高级用法

### 1. 集成到工作流
```python
# 在Python代码中调用评估器
from scripts.evaluate_strategy import StrategyEvaluator

# 创建评估器
evaluator = StrategyEvaluator(config_path="config/evaluation_pipeline.yaml")

# 运行评估
results = evaluator.run_evaluation(mode="comprehensive")

# 使用评估结果
print(f"总体评分: {results['summary']['overall_score']:.2f}")
```

### 2. 自定义评估模块
```python
# 扩展评估器
class CustomEvaluator(StrategyEvaluator):
    def _custom_evaluation(self):
        """自定义评估逻辑"""
        # 实现自定义评估
        pass
    
    def run_evaluation(self, mode=None, **kwargs):
        # 调用父类方法
        results = super().run_evaluation(mode, **kwargs)
        
        # 添加自定义评估
        custom_results = self._custom_evaluation()
        results['custom_evaluation'] = custom_results
        
        return results
```

### 3. 批量评估
```bash
#!/bin/bash
# 批量评估脚本

# 定义评估列表
EVALUATIONS=(
    "comprehensive"
    "model-only"
    "backtest-only"
    "risk-only"
)

# 遍历执行
for mode in "${EVALUATIONS[@]}"; do
    echo "执行评估: $mode"
    python scripts/evaluate_strategy.py --mode "$mode" --output-dir "results/batch_$mode"
    echo "完成评估: $mode"
    echo "---"
done
```

## 故障排除

### 常见问题

1. **导入错误**
   ```
   错误: 导入项目模块时出错
   ```
   **解决方案**：确保在项目根目录运行，或正确设置PYTHONPATH

2. **数据源不存在**
   ```
   警告: 训练数据路径不存在
   ```
   **解决方案**：检查数据路径，或使用`--train-data`参数指定正确路径

3. **内存不足**
   ```
   错误: 内存不足
   ```
   **解决方案**：调整配置文件中的内存设置，或使用快速评估模式

4. **报告生成失败**
   ```
   错误: 生成报告失败
   ```
   **解决方案**：检查输出目录权限，或使用`--no-report`跳过报告生成

### 调试模式

```bash
# 启用详细日志
python scripts/evaluate_strategy.py --mode comprehensive --verbose

# 干运行模式（检查配置）
python scripts/evaluate_strategy.py --mode comprehensive --dry-run
```

## 性能优化建议

1. **使用快速评估模式**：对于日常监控，使用`--mode quick`
2. **启用缓存**：配置文件中的`caching.enabled: true`
3. **并行处理**：配置文件中的`parallel_processing.enabled: true`
4. **限制数据量**：调整配置文件中的数据加载选项
5. **使用专用评估服务器**：对于大规模评估，考虑使用专用服务器

## 更新日志

### v1.0.0 (2026-03-02)
- 初始版本发布
- 支持6种评估模式
- 生成4种格式报告
- 提供改进建议功能

## 联系支持

如有问题或建议，请联系：
- 项目负责人：FZT项目组
- 文档维护：评估脚本开发团队
- 问题反馈：通过项目Issue系统