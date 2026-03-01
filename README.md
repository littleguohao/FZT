# FZT量化选股模型

基于FZT选股公式和LightGBM机器学习的量化选股系统。

## 项目结构

```
FZT_Quant/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖包
├── .gitignore                   # Git忽略文件
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据（不提交到Git）
│   ├── processed/               # 处理后的数据（不提交到Git）
│   └── features/                # 特征数据（不提交到Git）
├── src/                         # 源代码
│   ├── __init__.py              # Python包初始化
│   ├── data_prep.py             # 数据准备模块
│   ├── model_train.py           # 模型训练模块
│   └── backtest.py              # 回测模块
├── config/                      # 配置文件
│   └── config.yaml              # 主配置文件
├── notebooks/                   # Jupyter笔记本
│   └── analysis.ipynb           # 数据分析笔记本
├── results/                     # 结果输出（不提交到Git）
│   ├── models/                  # 训练好的模型
│   ├── predictions/             # 预测结果
│   └── reports/                 # 分析报告
└── docs/                        # 文档
    └── methodology.md           # 方法论文档
```

## 安装说明

### 1. 环境要求
- Python 3.8+
- pip 20.0+

### 2. 安装依赖
```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖包
pip install -r requirements.txt
```

### 3. 配置环境
```bash
# 复制示例配置文件
cp config/config.example.yaml config/config.yaml

# 编辑配置文件，设置数据源和参数
# vim config/config.yaml
```

## 依赖说明

项目主要依赖以下Python包：

### 核心依赖
- **qlib (>=0.9.0)** - 微软量化投资平台
- **lightgbm (>=4.0.0)** - 微软梯度提升框架
- **pandas (>=2.0.0)** - 数据分析库
- **numpy (>=1.24.0)** - 数值计算库
- **scikit-learn (>=1.3.0)** - 机器学习库

### 可视化
- **matplotlib (>=3.7.0)** - 绘图库
- **seaborn (>=0.12.0)** - 统计可视化

### 开发工具
- **jupyter (>=1.0.0)** - 交互式笔记本
- **ipython (>=8.0.0)** - 增强的Python交互环境

### 配置和工具
- **pyyaml (>=6.0)** - YAML配置文件处理
- **tqdm (>=4.65.0)** - 进度条显示

### 测试
- **pytest (>=7.0.0)** - 测试框架
- **pytest-cov (>=4.0.0)** - 测试覆盖率

## 配置说明

### 配置文件结构
配置文件位于 `config/config.yaml`，包含以下主要部分：

```yaml
# 数据配置
data:
  source: "qlib"  # 数据源：qlib/yahoo/本地
  start_date: "2010-01-01"
  end_date: "2023-12-31"
  universe: "csi300"  # 股票池：csi300/csi500/all
  
# 特征工程
features:
  fzt_formula: true  # 是否使用FZT选股公式
  technical_indicators: true  # 是否计算技术指标
  fundamental_factors: true  # 是否使用基本面因子
  
# 模型训练
model:
  algorithm: "lightgbm"  # 算法：lightgbm/xgboost/random_forest
  params:  # 模型参数
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 7
    
# 回测设置
backtest:
  initial_capital: 1000000  # 初始资金
  commission: 0.0003  # 交易佣金
  slippage: 0.0001  # 滑点
```

## 数据说明

### 数据来源
- **Qlib数据**：需要先安装并配置Qlib数据源
- **雅虎财经**：备用数据源，需要网络连接
- **本地数据**：支持CSV格式的历史数据

### 📊 数据源选项
项目支持多种数据源配置：

#### 1. 本地CSV数据（默认）
- **位置**：`/Users/lucky/Downloads/O_DATA/`
- **时间范围**：2021年8月 - 2026年2月（约4.5年）
- **数据量**：充足的交易日数据（约1000+交易日/股票）
- **格式**：CSV文件，包含OHLCV日频数据

#### 2. Qlib数据（可选）
- **测试数据集**：2005年 - 2020年（16年历史数据）
- **完整数据集**：2007年至今（持续更新）
- **数据质量**：经过清洗和标准化的专业量化数据
- **安装**：需要单独下载Qlib数据包
- **下载脚本**：`scripts/download_qlib_data.py`

### 3. 混合数据模式（推荐）
- **训练验证**：Qlib数据 (2005-2020年，16年历史数据)
- **测试**：本地CSV数据 (2021-2026年，约5年最新数据)
- **优势**：充足历史数据训练 + 最新数据测试，避免数据泄露

### 4. 数据合并（可选）
- **时间范围**：2005年 - 2026年（约21年完整数据）
- **优势**：结合历史数据和最新数据
- **用途**：长期策略验证和模型训练

### Qlib数据下载
如果使用Qlib数据，需要先下载数据包：

```bash
# 方法1: 使用下载脚本
python scripts/download_qlib_data.py

# 方法2: 手动下载
python -c "from qlib.tests.data import GetData; GetData().qlib_data()"

# 方法3: 自动下载
python -c "import qlib; qlib.init(auto_mount=True)"
```

**注意**：
- 数据大小约0.45GB
- 需要稳定的网络连接
- 下载时间取决于网络速度

#### 3. 数据合并（可选）
- **时间范围**：2005年 - 2026年（约21年完整数据）
- **优势**：结合历史数据和最新数据
- **用途**：长期策略验证和模型训练

#### 4. 数据优势
1. **充足的数据量**：支持复杂的机器学习模型训练
2. **完整的时间跨度**：包含不同市场周期的数据
3. **灵活的配置**：可根据需要选择数据源
4. **可靠的验证**：可以使用传统的训练/验证/测试划分

### 数据预处理
1. **数据清洗**：处理缺失值、异常值
2. **特征计算**：计算FZT选股公式、技术指标、基本面因子
3. **标签生成**：根据未来收益率生成训练标签
4. **数据集划分**：按时间划分训练集、验证集、测试集

## 使用示例

### 1. 数据准备
```bash
# 下载数据
python src/data_prep.py --download --start_date 2010-01-01 --end_date 2023-12-31

# 特征工程
python src/data_prep.py --features --universe csi300
```

### 2. 模型训练
```bash
# 训练模型
python src/model_train.py --train --model lightgbm

# 模型评估
python src/model_train.py --evaluate --model_path results/models/latest.pkl
```

### 3. 回测分析
```bash
# 运行回测
python src/backtest.py --backtest --initial_capital 1000000

# 生成报告
python src/backtest.py --report --output results/reports/backtest_report.html
```

### 4. 完整流程
```bash
# 一键运行完整流程
python src/data_prep.py --all
python src/model_train.py --all
python src/backtest.py --all
```

## 详细目录结构说明

### data/ 数据目录
- **raw/** - 原始数据，从数据源下载的原始文件
- **processed/** - 处理后的数据，经过清洗和预处理
- **features/** - 特征数据，计算好的特征矩阵

### src/ 源代码
- **data_prep.py** - 数据准备模块，负责数据下载、清洗、特征工程
- **model_train.py** - 模型训练模块，负责模型训练、评估、保存
- **backtest.py** - 回测模块，负责策略回测、绩效分析

### config/ 配置目录
- **config.yaml** - 主配置文件，所有参数集中管理
- **config.example.yaml** - 配置文件示例

### notebooks/ 笔记本目录
- **analysis.ipynb** - 数据分析笔记本，用于探索性分析

### results/ 结果目录
- **models/** - 训练好的模型文件
- **predictions/** - 模型预测结果
- **reports/** - 分析报告和可视化图表

### docs/ 文档目录
- **methodology.md** - 方法论文档，详细说明FZT选股公式和模型原理

## 开发指南

### 代码规范
- 遵循PEP 8编码规范
- 使用类型注解
- 编写单元测试

### 版本控制
- 使用Git进行版本控制
- 提交信息遵循约定式提交规范
- 使用分支进行功能开发

### 测试
```bash
# 运行所有测试
pytest

# 运行测试并生成覆盖率报告
pytest --cov=src tests/
```

## 常见问题

### Q: 如何获取Qlib数据？
A: 参考Qlib官方文档安装和配置数据源。

### Q: 如何调整模型参数？
A: 修改 `config/config.yaml` 中的模型参数部分。

### Q: 如何添加新的特征？
A: 在 `src/data_prep.py` 的 `calculate_features` 函数中添加特征计算逻辑。

### Q: 如何扩展新的机器学习算法？
A: 在 `src/model_train.py` 的 `train_model` 函数中添加对新算法的支持。

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 贡献指南

欢迎提交Issue和Pull Request来改进本项目。