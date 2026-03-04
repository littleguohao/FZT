# FZT公式计算模块文档

## 概述

FZT公式计算模块是FZT量化选股系统的核心组件，负责计算FZT选股信号和相关特征。该模块基于FZT公式规范，实现了完整的选股信号计算、特征工程和批量处理功能。

## 功能特性

### 1. 核心功能
- **FZT公式值计算**: 基于价格动量、成交量确认、价格强度和波动率调整的综合评分
- **特征工程**: 生成完整的特征DataFrame，包含FZT值、信号、强度及各子因子
- **批量计算**: 支持多股票批量处理，性能优化
- **配置管理**: 支持配置文件加载和动态参数调整

### 2. 公式组件
- **价格动量**: 多时间窗口收益率计算（5, 10, 20, 60天），加权动量组合
- **成交量确认**: 成交量变化率计算，成交量与价格一致性验证
- **价格强度**: 相对价格位置计算，突破信号检测
- **波动率调整**: 历史波动率计算，风险调整评分
- **综合评分**: 各因子加权组合，时间序列标准化

### 3. 技术特性
- **向量化计算**: 使用Pandas和NumPy进行高效向量化操作
- **类型注解**: 完整的类型提示，提高代码可读性和IDE支持
- **错误处理**: 完善的异常处理和日志记录
- **单元测试**: 完整的测试套件，覆盖所有核心功能

## 模块结构

### 核心类: `FZTFormula`

```python
class FZTFormula:
    """FZT选股公式计算器"""
    
    def __init__(self, config: Optional[Dict] = None, config_path: Optional[str] = None):
        """初始化公式计算器"""
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算FZT公式值"""
        
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算FZT相关特征"""
        
    def batch_calculate(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """批量计算多只股票的FZT特征"""
        
    def generate_signal(self, fzt_value: float) -> Tuple[str, float]:
        """根据FZT值生成交易信号"""
        
    def analyze_fzt_distribution(self, fzt_values: pd.Series) -> Dict[str, Any]:
        """分析FZT值分布"""
```

### 辅助函数

```python
def create_sample_data(n_days: int = 100, start_date: str = '2023-01-01') -> pd.DataFrame:
    """创建样本数据用于测试"""

def calculate_target(data: pd.DataFrame, horizon: int = 2) -> pd.Series:
    """计算目标变量：未来N天收益率"""
```

## 输入输出规范

### 输入数据格式
```python
# 输入DataFrame应包含以下列：
# - date: 日期 (datetime)
# - code: 股票代码 (str)
# - open: 开盘价 (float)
# - high: 最高价 (float)
# - low: 最低价 (float)
# - close: 收盘价 (float)
# - volume: 成交量 (float)
# - amount: 成交额 (float, 可选)
```

### 输出格式
```python
# 1. FZT值序列
fzt_values = fzt_calculator.calculate(data)  # 返回pd.Series

# 2. 完整特征DataFrame
features = fzt_calculator.calculate_features(data)  # 返回pd.DataFrame
# 包含列：fzt_value, fzt_signal, fzt_strength, momentum_5d, volume_change, 等
```

## 配置系统

### 配置文件格式 (`config/fzt_config.yaml`)
```yaml
# 动量配置
momentum:
  windows: [5, 10, 20, 60]
  weights: [0.4, 0.3, 0.2, 0.1]
  weight: 0.5

# 成交量确认配置
volume:
  window: 5
  weight: 0.2

# 价格强度配置
strength:
  window: 20
  weight: 0.2

# 波动率调整配置
volatility:
  window: 20
  weight: 0.1
  adjustment_factor: 0.01

# 信号阈值配置
signals:
  strong_buy: 1.0
  weak_buy: 0.2
  weak_sell: -0.2
  strong_sell: -1.0
```

### 配置加载方式
```python
# 方式1: 使用默认配置
fzt_calculator = FZTFormula()

# 方式2: 使用自定义配置字典
custom_config = {
    'momentum': {
        'windows': [3, 10, 30],
        'weights': [0.5, 0.3, 0.2]
    }
}
fzt_calculator = FZTFormula(config=custom_config)

# 方式3: 从配置文件加载
fzt_calculator = FZTFormula(config_path="config/fzt_config.yaml")
```

## 使用示例

### 基本使用
```python
from src.fzt_formula import FZTFormula, create_sample_data

# 创建样本数据
data = create_sample_data(n_days=200)

# 初始化计算器
fzt_calculator = FZTFormula()

# 计算FZT值
fzt_values = fzt_calculator.calculate(data)

# 计算完整特征
features = fzt_calculator.calculate_features(data)

# 生成信号
signal, strength = fzt_calculator.generate_signal(1.5)
print(f"信号: {signal}, 强度: {strength}")
```

### 批量计算
```python
# 准备多股票数据
stock_data = {
    '000001.SZ': data1,
    '000002.SZ': data2,
    '000858.SZ': data3
}

# 批量计算
batch_results = fzt_calculator.batch_calculate(stock_data)

# 处理结果
for stock_code, features in batch_results.items():
    print(f"{stock_code}: FZT均值={features['fzt_value'].mean():.3f}")
```

### 与目标变量集成
```python
from src.fzt_formula import calculate_target

# 计算目标变量（未来5天收益率）
target = calculate_target(data, horizon=5)

# 创建训练数据集
dataset = features.copy()
dataset['target_5d'] = target
dataset_clean = dataset.dropna()
```

## 测试验证

### 单元测试
```bash
# 运行所有测试
python3 tests/test_fzt_formula.py

# 运行特定测试
python3 -m pytest tests/test_fzt_formula.py::test_fzt_calculation -v
```

### 测试覆盖率
- ✅ 配置文件加载测试
- ✅ 样本数据创建测试
- ✅ 各子因子计算测试
- ✅ FZT公式计算测试
- ✅ 信号生成测试
- ✅ 特征计算测试
- ✅ 批量计算测试
- ✅ 目标变量计算测试
- ✅ 分布分析测试
- ✅ 配置保存测试
- ✅ 数据准备集成测试

## 性能优化

### 向量化计算
- 所有核心计算使用Pandas向量化操作
- 避免循环，提高大数据集处理效率

### 批量处理
- 支持多股票并行计算（配置可选）
- 中间结果缓存机制
- 进度显示和日志记录

### 内存管理
- 使用适当的数据类型减少内存占用
- 及时释放中间变量
- 支持分块处理大数据集

## 集成指南

### 与数据准备模块集成
```python
from src.data_prep import DataPreprocessor
from src.fzt_formula import FZTFormula

# 加载和处理数据
preprocessor = DataPreprocessor()
raw_data = preprocessor.load_raw_data()
processed_data = preprocessor.preprocess_data(raw_data)

# 计算FZT特征
fzt_calculator = FZTFormula()
features = fzt_calculator.calculate_features(processed_data)
```

### 与模型训练模块集成
```python
# 准备特征和目标变量
features = fzt_calculator.calculate_features(data)
target = calculate_target(data, horizon=2)

# 创建训练数据集
X = features.drop(columns=['fzt_signal'])  # 特征
y = target  # 目标变量

# 划分训练测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 部署说明

### 环境要求
```bash
# 核心依赖
pandas>=1.5.0
numpy>=1.21.0
pyyaml>=6.0

# 可选依赖（用于高级功能）
scikit-learn>=1.0.0  # 机器学习集成
joblib>=1.2.0        # 并行计算
```

### 安装
```bash
# 从项目根目录安装
pip install -e .

# 或直接使用
python3 -m pip install -r requirements.txt
```

## 维护和扩展

### 添加新因子
1. 在`FZTFormula`类中添加新的计算方法
2. 更新配置验证逻辑
3. 在`calculate_features`方法中集成新因子
4. 添加相应的单元测试

### 修改公式权重
1. 更新配置文件中的权重参数
2. 重新验证配置有效性
3. 运行测试确保功能正常

### 性能调优
1. 监控计算时间和内存使用
2. 优化向量化操作
3. 考虑使用缓存机制
4. 支持并行计算

## 故障排除

### 常见问题

#### 1. 数据格式错误
**症状**: `ValueError: 输入数据缺少必要列`
**解决方案**: 确保输入DataFrame包含`close`, `volume`, `high`, `low`列

#### 2. 配置加载失败
**症状**: `FileNotFoundError` 或 `yaml.YAMLError`
**解决方案**: 检查配置文件路径和格式，或使用默认配置

#### 3. 计算结果异常
**症状**: FZT值超出正常范围（如±10以上）
**解决方案**: 检查输入数据质量，验证各子因子计算逻辑

#### 4. 性能问题
**症状**: 大数据集计算缓慢
**解决方案**: 启用批量计算，考虑使用并行处理，优化数据预处理

### 调试建议
1. 启用DEBUG级别日志
2. 使用样本数据验证基本功能
3. 逐步检查各子因子计算结果
4. 验证配置参数合理性

## 版本历史

### v1.0.0 (2026-03-01)
- 初始版本发布
- 实现FZT公式核心计算逻辑
- 完成特征工程和批量计算功能
- 提供完整的测试套件和文档
- 支持配置文件管理和动态参数调整

## 贡献指南

欢迎提交Issue和Pull Request来改进FZT公式计算模块。在提交代码前，请确保：

1. 所有测试通过
2. 代码符合PEP 8规范
3. 添加必要的类型注解和文档字符串
4. 更新相关文档

## 许可证

本项目采用MIT许可证。详见LICENSE文件。