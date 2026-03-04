# 数据准备模块 (data_prep.py)

## 概述

数据准备模块是FZT项目的核心数据处理组件，负责从原始CSV数据加载、预处理、筛选到最终数据集划分的全流程处理。

## 功能特性

### 1. 数据加载
- 支持从本地CSV文件批量加载
- 自动合并多股票数据
- 字段映射和重命名
- 进度显示和错误处理

### 2. 数据预处理
- 日期解析和排序
- 缺失值处理（前向填充）
- 异常值检测和移除
- 数据标准化（对数收益率）
- 技术指标计算（可选）

### 3. 股票筛选
- 最小交易日数过滤
- ST股票排除
- 价格范围过滤
- 数据完整性检查

### 4. 时间划分
- 按配置划分训练/验证/测试集
- 支持多种数据源时间范围
- 滚动窗口功能（可选）

### 5. 数据保存
- 支持多种格式：CSV、Parquet、Feather
- 数据版本控制
- 数据质量报告生成

## 快速开始

### 安装依赖
```bash
pip install pandas numpy pyyaml tqdm pyarrow
```

### 基本使用
```python
from src.data_prep import DataPreprocessor

# 初始化预处理器
preprocessor = DataPreprocessor()

# 运行完整管道
split_data = preprocessor.run_pipeline()

# 获取划分后的数据
train_data = split_data['train']
valid_data = split_data['valid']
test_data = split_data['test']
```

### 分步使用
```python
from src.data_prep import DataPreprocessor

# 1. 初始化
preprocessor = DataPreprocessor()

# 2. 加载原始数据
raw_data = preprocessor.load_raw_data()

# 3. 数据预处理
processed_data = preprocessor.preprocess_data(raw_data)

# 4. 股票筛选
filtered_data = preprocessor.filter_stocks(processed_data)

# 5. 时间划分
split_data = preprocessor.split_by_time(filtered_data)

# 6. 保存数据
preprocessor.save_processed_data(split_data)

# 7. 生成报告
report = preprocessor.generate_data_report(split_data)
print(report)
```

## 配置文件说明

配置文件位于 `config/data_config.yaml`，主要包含以下部分：

### 数据源配置
```yaml
data:
  source: "local_csv"  # 数据源类型
  local_csv:
    enabled: true
    data_path: "/Users/lucky/Downloads/O_DATA/"
    file_pattern: "*-all-latest.csv"
    field_mapping:  # CSV字段映射
      date: "Date"
      code: "Code"
      open: "Open"
      high: "High"
      low: "Low"
      close: "Close"
      volume: "Volume"
      amount: "Amount"
```

### 时间范围配置
```yaml
time_ranges:
  local_csv:
    data_start: "2021-08-01"
    data_end: "2026-02-06"
    train_start: "2021-08-01"
    train_end: "2023-12-31"
    valid_start: "2024-01-01"
    valid_end: "2024-12-31"
    test_start: "2025-01-01"
    test_end: "2026-02-06"
```

### 数据处理配置
```yaml
processing:
  normalize: true      # 数据标准化
  fill_na: true       # 缺失值填充
  remove_outliers: true  # 移除异常值
```

### 股票筛选配置
```yaml
stock_filter:
  min_days: 60        # 最小交易日数
  exclude_st: true    # 排除ST股票
  min_price: 1.0      # 最低价格
  max_price: 1000.0   # 最高价格
```

## 输出文件结构

处理后的数据保存在 `data/processed/` 目录下：

```
data/processed/
├── train_data_v20260301_143022.csv
├── train_data_v20260301_143022.parquet
├── train_data_v20260301_143022.feather
├── valid_data_v20260301_143022.csv
├── test_data_v20260301_143022.csv
└── data_statistics_v20260301_143022.yaml
```

## 数据质量报告

模块会自动生成数据质量报告，包含：

1. **数据集统计**：行数、股票数量、交易日数
2. **时间范围**：每个数据集的时间跨度
3. **数据完整性**：缺失值比例
4. **价格统计**：价格范围、平均值、标准差
5. **总体统计**：总数据量、股票覆盖情况

## 测试和验证

### 运行测试
```bash
python test_data_prep.py
```

### 测试内容
1. 配置文件加载测试
2. 数据加载测试
3. 数据预处理测试
4. 股票筛选测试
5. 时间划分测试
6. 完整管道测试

### 查看示例
```bash
python examples/use_data_prep.py
```

## 高级功能

### 自定义配置
```python
# 使用自定义配置文件
preprocessor = DataPreprocessor("custom_config.yaml")
```

### 添加技术指标
在配置文件中启用技术指标：
```yaml
features:
  technical_factors:
    - "momentum"
    - "volatility"
    - "volume"
    - "price_action"
```

### 滚动窗口
```yaml
time_ranges:
  use_rolling_window: true
  rolling_window:
    window_size: 252  # 1年交易日
    step_size: 63     # 1季度
    min_train_size: 126  # 半年
```

## 错误处理

模块内置完善的错误处理机制：

1. **配置文件验证**：检查必要配置项
2. **数据完整性检查**：验证字段映射和数据类型
3. **异常值处理**：自动检测和处理异常数据
4. **日志记录**：详细的处理日志，便于调试

## 性能优化

1. **批量处理**：使用pandas向量化操作
2. **内存管理**：及时释放不需要的数据
3. **进度显示**：使用tqdm显示处理进度
4. **格式选择**：Parquet格式提供更好的IO性能

## 注意事项

1. **数据路径**：确保配置的数据路径存在且可访问
2. **内存需求**：处理大量数据时可能需要足够的内存
3. **依赖安装**：Parquet格式需要pyarrow或fastparquet
4. **时间范围**：确保配置的时间范围在数据实际时间范围内

## 更新日志

### v1.0.0 (2026-03-01)
- 初始版本发布
- 支持本地CSV数据加载和处理
- 实现完整的数据处理管道
- 添加数据质量报告功能
- 提供测试和示例代码

## 贡献指南

1. 遵循PEP 8编码规范
2. 添加类型注解
3. 编写详细的文档字符串
4. 添加单元测试
5. 更新相关文档

## 许可证

本项目遵循MIT许可证。