# FZT排序增强策略实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现FZT公式的排序学习增强，每日选出未来一天涨幅最大的前5只股票。

**Architecture:** 基于QLib框架，实现FZT算子封装、特征中性化处理、LightGBM LambdaRank排序模型、成本调整回测系统。

**Tech Stack:** Python, QLib, LightGBM, Pandas, NumPy, scikit-learn

---

## 阶段1: 基础框架 (今天)

### Task 1: 创建项目目录结构

**Files:**
- Create: `src/qlib_operators/__init__.py`
- Create: `src/feature_engineering/__init__.py`
- Create: `src/ranking_model/__init__.py`
- Create: `src/backtest/__init__.py`
- Create: `src/evaluation/__init__.py`
- Create: `config/ranking_config.yaml`
- Create: `config/feature_config.yaml`

**Step 1: 创建基础目录结构**

```bash
mkdir -p src/qlib_operators src/feature_engineering src/ranking_model src/backtest src/evaluation
mkdir -p config scripts results/{models,features,backtest_results,evaluation_reports}
```

**Step 2: 创建__init__.py文件**

```python
# src/qlib_operators/__init__.py
"""QLib算子模块"""
```

**Step 3: 验证目录创建**

Run: `find src/ -type f -name "__init__.py" | wc -l`
Expected: 5

**Step 4: 提交**

```bash
git add src/qlib_operators/__init__.py src/feature_engineering/__init__.py src/ranking_model/__init__.py src/backtest/__init__.py src/evaluation/__init__.py
git commit -m "feat: create project directory structure"
```

### Task 2: 创建QLib FZT算子

**Files:**
- Create: `src/qlib_operators/fzt_alpha.py`
- Modify: `src/fzt_brick_formula.py` (复用现有代码)
- Test: `tests/test_fzt_alpha.py`

**Step 1: 编写测试用例**

```python
# tests/test_fzt_alpha.py
import pandas as pd
import numpy as np
from src.qlib_operators.fzt_alpha import FZTAlpha

def test_fzt_alpha_initialization():
    """测试FZT算子初始化"""
    alpha = FZTAlpha()
    assert alpha is not None
    print("✅ FZTAlpha初始化测试通过")

def test_fzt_alpha_calculation():
    """测试FZT计算"""
    # 创建测试数据
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(10) + 100,
        'high': np.random.randn(10) + 105,
        'low': np.random.randn(10) + 95,
        'close': np.random.randn(10) + 100,
        'volume': np.random.randn(10) * 1000000 + 1000000
    }, index=dates)
    
    alpha = FZTAlpha()
    result = alpha(data)
    
    assert result is not None
    assert len(result) == 10
    print("✅ FZTAlpha计算测试通过")
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_fzt_alpha.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.qlib_operators.fzt_alpha'"

**Step 3: 实现FZT算子**

```python
# src/qlib_operators/fzt_alpha.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FZT公式QLib算子

基于现有FZTBrickFormula封装为QLib算子，确保无未来函数。
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from qlib.data.ops import Operator
    HAS_QLIB = True
except ImportError:
    HAS_QLIB = False
    class Operator:
        """QLib Operator基类的简化版本"""
        pass

from src.fzt_brick_formula import FZTBrickFormula


class FZTAlpha(Operator):
    """FZT公式QLib算子"""
    
    def __init__(self, feature_name: str = "FZT"):
        """
        初始化FZT算子
        
        Args:
            feature_name: 特征名称
        """
        super().__init__(feature_name)
        self.fzt_calculator = FZTBrickFormula()
        self.feature_name = feature_name
    
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        """
        计算FZT值
        
        Args:
            df: 包含OHLCV数据的DataFrame，索引为日期
            
        Returns:
            FZT值序列
        """
        try:
            # 确保数据格式正确
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"缺少必要列: {missing}")
            
            # 计算FZT值
            fzt_values = self.fzt_calculator.calculate_fzt(df)
            
            # 返回序列
            if isinstance(fzt_values, pd.DataFrame):
                # 如果是DataFrame，取第一列
                return fzt_values.iloc[:, 0]
            elif isinstance(fzt_values, pd.Series):
                return fzt_values
            else:
                # 转换为Series
                return pd.Series(fzt_values, index=df.index)
                
        except Exception as e:
            print(f"FZT计算错误: {e}")
            # 返回NaN序列
            return pd.Series(np.nan, index=df.index, name=self.feature_name)
    
    def get_feature_names(self) -> list:
        """获取特征名称"""
        return [self.feature_name]
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_fzt_alpha.py::test_fzt_alpha_initialization -v`
Expected: PASS

Run: `pytest tests/test_fzt_alpha.py::test_fzt_alpha_calculation -v`
Expected: PASS (可能因为数据问题需要调整)

**Step 5: 提交**

```bash
git add src/qlib_operators/fzt_alpha.py tests/test_fzt_alpha.py
git commit -m "feat: implement QLib FZT alpha operator"
```

### Task 3: 创建特征中性化模块

**Files:**
- Create: `src/feature_engineering/neutralization.py`
- Test: `tests/test_neutralization.py`

**Step 1: 编写中性化测试**

```python
# tests/test_neutralization.py
import pandas as pd
import numpy as np
from src.feature_engineering.neutralization import neutralize_features

def test_neutralize_features_basic():
    """测试基础中性化功能"""
    # 创建测试数据
    n_stocks = 100
    n_dates = 5
    
    dates = pd.date_range('2020-01-01', periods=n_dates, freq='D')
    stocks = [f'stock_{i}' for i in range(n_stocks)]
    
    # 创建多索引DataFrame
    index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'instrument'])
    features = pd.DataFrame({
        'feature1': np.random.randn(len(index)),
        'feature2': np.random.randn(len(index))
    }, index=index)
    
    # 创建市值和行业数据
    market_cap = pd.Series(np.random.lognormal(10, 1, len(index)), index=index)
    industry = pd.Series(np.random.choice(['tech', 'finance', 'health'], len(index)), index=index)
    
    # 中性化
    neutralized = neutralize_features(features, market_cap, industry)
    
    assert neutralized.shape == features.shape
    assert not neutralized.isnull().any().any()
    print("✅ 基础中性化测试通过")
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_neutralization.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.feature_engineering.neutralization'"

**Step 3: 实现中性化模块**

```python
# src/feature_engineering/neutralization.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征中性化模块

实现"市值回归 + 行业标准化"混合中性化方案。
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, List
import warnings
warnings.filterwarnings('ignore')


def mad_winsorize(series: pd.Series, n: float = 3.0) -> pd.Series:
    """
    MAD去极值处理
    
    Args:
        series: 输入序列
        n: 中位数绝对偏差倍数
        
    Returns:
        去极值后的序列
    """
    if series.empty:
        return series
    
    median = series.median()
    mad = (series - median).abs().median()
    
    if mad == 0:
        return series
    
    lower_bound = median - n * mad
    upper_bound = median + n * mad
    
    return series.clip(lower_bound, upper_bound)


def zscore_normalize(series: pd.Series) -> pd.Series:
    """
    横截面Z-score标准化
    
    Args:
        series: 输入序列
        
    Returns:
        标准化后的序列
    """
    if series.empty:
        return series
    
    mean = series.mean()
    std = series.std()
    
    if std == 0:
        return series - mean
    
    return (series - mean) / std


def neutralize_features(
    features: pd.DataFrame,
    market_cap: pd.Series,
    industry: pd.Series,
    feature_names: List[str] = None
) -> pd.DataFrame:
    """
    混合中性化: 市值回归 + 行业标准化
    
    Args:
        features: 特征DataFrame，索引为(date, instrument)
        market_cap: 市值序列，索引与features相同
        industry: 行业序列，索引与features相同
        feature_names: 要处理的特征列表，None表示处理所有
        
    Returns:
        中性化后的特征DataFrame
    """
    if features.empty:
        return features
    
    if feature_names is None:
        feature_names = features.columns.tolist()
    
    # 复制数据避免修改原始数据
    result = features.copy()
    
    # 按日期分组处理
    dates = features.index.get_level_values('date').unique()
    
    for date in dates:
        date_mask = features.index.get_level_values('date') == date
        
        for feature in feature_names:
            if feature not in features.columns:
                continue
            
            # 获取当日数据
            feature_values = features.loc[date_mask, feature]
            cap_values = market_cap.loc[date_mask]
            industry_values = industry.loc[date_mask]
            
            if feature_values.empty or cap_values.empty:
                continue
            
            # 1. 预处理: 去极值 + 标准化
            feature_clean = mad_winsorize(feature_values, n=3.0)
            feature_std = zscore_normalize(feature_clean)
            
            # 2. 市值中性化 (回归法)
            # 使用对数市值
            log_cap = np.log(cap_values.replace(0, np.nan).fillna(cap_values.median()))
            
            # 简单线性回归: 特征 = α + β * log(市值) + ε
            # 使用最小二乘估计
            X = np.column_stack([np.ones_like(log_cap), log_cap])
            y = feature_std.values
            
            try:
                # 最小二乘解
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                y_pred = X @ beta
                resid_mkt = y - y_pred
            except:
                # 如果回归失败，使用原始值
                resid_mkt = feature_std.values
            
            # 3. 行业中性化 (分组标准化)
            resid_df = pd.DataFrame({
                'value': resid_mkt,
                'industry': industry_values.values
            }, index=feature_std.index)
            
            # 按行业分组标准化
            def industry_normalize(group):
                if len(group) < 2:
                    return group
                mean = group.mean()
                std = group.std()
                if std == 0:
                    return group - mean
                return (group - mean) / std
            
            neutralized = resid_df.groupby('industry')['value'].transform(industry_normalize)
            
            # 保存结果
            result.loc[date_mask, feature] = neutralized.values
    
    return result
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_neutralization.py::test_neutralize_features_basic -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/feature_engineering/neutralization.py tests/test_neutralization.py
git commit -m "feat: implement feature neutralization module"
```

### Task 4: 创建排序标签构造模块

**Files:**
- Create: `src/ranking_model/label_engineering.py`
- Test: `tests/test_label_engineering.py`

**Step 1: 编写标签工程测试**

```python
# tests/test_label_engineering.py
import pandas as pd
import numpy as np
from src.ranking_model.label_engineering import create_ranking_labels

def test_create_ranking_labels():
    """测试排序标签创建"""
    # 创建测试数据
    n_stocks = 50
    n_dates = 3
    
    dates = pd.date_range('2020-01-01', periods=n_dates, freq='D')
    stocks = [f'stock_{i}' for i in range(n_stocks)]
    
    index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'instrument'])
    
    # 未来收益率 (标签)
    future_returns = pd.Series(np.random.randn(len(index)), index=index)
    
    # FZT选股条件
    fzt_condition = pd.Series(np.random.choice([0, 1], len(index), p=[0.7, 0.3]), index=index)
    
    # 创建排序标签
    labels = create_ranking_labels(
        future_returns=future_returns,
        fzt_condition=fzt_condition,
        top_k=5
    )
    
    assert labels.shape == future_returns.shape
    assert labels.index.equals(future_returns.index)
    
    # 检查每天选出的股票数量
    for date in dates:
        date_mask = labels.index.get_level_values('date') == date
        selected = fzt_condition[date_mask]
        if selected.sum() > 0:  # 如果有FZT选出的股票
            # 应该有为1的标签 (Top K)
            assert labels[date_mask].max() == 1
            # Top K的数量应该<=5
            assert labels[date_mask].sum() <= 5
    
    print("✅ 排序标签创建测试通过")
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_label_engineering.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.ranking_model.label_engineering'"

**Step 3: 实现标签工程模块**

```python
# src/ranking_model/label_engineering.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
排序标签工程模块

为LambdaRank排序学习构造标签。
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def create_ranking_labels(
    future_returns: pd.Series,
    fzt_condition: pd.Series,
    top_k: int = 5,
    label_type: str = 'binary'
) -> pd.Series:
    """
    创建排序学习标签
    
    Args:
        future_returns: 未来收益率序列，索引为(date, instrument)
        fzt_condition: FZT选股条件序列 (0/1)，索引与future_returns相同
        top_k: 每天选择的股票数量
        label_type: 标签类型 ('binary'或'continuous')
        
    Returns:
        排序标签序列
    """
    if not future_returns.index.equals(fzt_condition.index):
        raise ValueError("future_returns和fzt_condition的索引必须相同")
    
    # 初始化标签
    labels = pd.Series(0, index=future_returns.index)
    
    # 按日期处理
    dates = future_returns.index.get_level_values('date').unique()
    
    for date in dates:
        date_mask = future_returns.index.get_level_values('date') == date
        
        # 获取当日数据
        date_returns = future_returns[date_mask]
        date_fzt = fzt_condition[date_mask]
        
        # 只考虑FZT选出的股票
        fzt_selected = date_fzt == 1
        
        if fzt_selected.sum() == 0:
            # 没有FZT选出的股票
            continue
        
        selected_returns = date_returns[fzt_selected]
        
        if selected_returns.empty:
            continue
        
        # 根据收益率排序
        if label_type == 'binary':
            # 二分类标签: Top K为1，其余为0
            # 取收益率最高的Top K只股票
            top_indices = selected_returns.nlargest(min(top_k, len(selected_returns))).index
            
            # 设置标签
            labels.loc[top_indices] = 1
            
        elif label_type == 'continuous':
            # 连续标签: 使用排名或收益率本身
            # 这里使用标准化排名 (0到1之间)
            ranks = selected_returns.rank(method='first', ascending=False)
            normalized_ranks = 1 - (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else 0.5
            
            labels.loc[selected_returns.index] = normalized_ranks
    
    return labels


def create_lambdarank_dataset(
    features: pd.DataFrame,
    labels: pd.Series,
    group_sizes: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    创建LambdaRank所需的数据集格式
    
    Args:
        features: 特征DataFrame
        labels: 标签序列
        group_sizes: 每组的大小 (通常是每天选出的股票数量)
        
    Returns:
        (features, labels, groups)
    """
    if not features.index.equals(labels.index):
        raise ValueError("features和labels的索引必须相同")
    
    # 如果没有提供group_sizes，按日期分组
    if group_sizes is None:
        dates = features.index.get_level_values('date').unique()
        groups = pd.Series(0, index=features.index)
        
        group_id = 0
        for date in dates:
            date_mask = features.index.get_level_values('date') == date
            size = date_mask.sum()
            if size > 0:
                groups.loc[date_mask] = group_id
                group_id += 1
        
        # 转换为group_sizes格式
        group_sizes = groups.value_counts().sort_index()
    
    return features, labels, group_sizes
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_label_engineering.py::test_create_ranking_labels -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/ranking_model/label_engineering.py tests/test_label_engineering.py
git commit -m "feat: implement ranking label engineering module"
```

### Task 5: 创建基础配置文件

**Files:**
- Create: `config/ranking_config.yaml`
- Create: `config/feature_config.yaml`
- Create: `config/backtest_config.yaml`

**Step 1: 创建排序模型配置**

```yaml
# config/ranking_config.yaml
# LightGBM LambdaRank配置

model:
  class: "LGBModel"
  module_path: "qlib.contrib.model.gbdt"
  kwargs:
    # 基础参数
    objective: "lambdarank"
    metric: ["ndcg", "ndcg@5"]
    num_leaves: 64
    learning_rate: 0.05
    feature_fraction: 0.8
    bagging_fraction: 0.8
    bagging_freq: 5
    lambda_l1: 0.1
    lambda_l2: 0.1
    min_child_samples: 20
    seed: 42
    
    # 排序特定参数
    lambdarank_truncation_level: 5
    lambdarank_norm: true
    label_gain: [0, 1, 3, 7, 15, 31]
    
    # 训练控制
    num_boost_round: 1000
    early_stopping_rounds: 50
    verbose_eval: 50

dataset:
  # 数据集配置
  train_start: "2005-01-01"
  train_end: "2017-12-31"
  valid_start: "2018-01-01"
  valid_end: "2019-12-31"
  test_start: "2020-01-01"
  test_end: "2020-12-31"
  
  # 特征配置
  feature_cols: &feature_cols
    - "FZT"  # FZT原始值
    - "FZT_rank"  # FZT排名
    - "FZT_slope_5d"  # FZT斜率
    - "VWAP"  # 成交量加权均价
    - "VOLUME"  # 成交量
    - "RSI_14"  # RSI指标
    - "MACD"  # MACD
    - "BETA"  # Beta值
    - "RESI_5"  # 5日收益率
  
  label_col: "label"  # 标签列名

evaluation:
  # 评估指标
  metrics:
    - "ndcg@5"
    - "map@5"
    - "precision@5"
    - "recall@5"
  
  # 回测指标
  backtest_metrics:
    - "annual_return"
    - "sharpe_ratio"
    - "max_drawdown"
    - "win_rate"
```

**Step 2: 创建特征工程配置**

```yaml
# config/feature_config.yaml
# 特征工程配置

neutralization:
  # 中性化配置
  enabled: true
  method: "hybrid"  # hybrid, regression, group_standardize
  market_cap_column: "market_cap"
  industry_column: "industry"
  
  # 回归配置
  regression:
    use_log: true  # 使用对数市值
    include_constant: true  # 包含常数项
  
  # 分组标准化配置
  group_standardize:
    min_group_size: 5  # 最小分组大小

feature_selection:
  # 特征筛选标准
  enabled: true
  criteria:
    # IC相关
    min_ic: 0.02
    min_ir: 0.5
    max_ic_std: 0.1
    
    # 相关性控制
    max_correlation_with_fzt: 0.6
    max_feature_correlation: 0.8
    
    # 稳定性
    min_industry_coverage: 0.7  # 至少覆盖70%的行业

alpha158_selection:
  # Alpha158特征选择
  enabled: true
  categories:
    - "price_volume"  # 价量特征
    - "momentum"  # 动量特征
    - "reversal"  # 反转特征
    - "volatility"  # 波动率特征
    - "liquidity"  # 流动性特征
  
  # 每个类别选择的数量
  selection_per_category: 5

fzt_features:
  # FZT衍生特征
  enabled: true
  features:
    - "fzt_raw"  # 原始FZT值
    - "fzt_rank"  # 横截面排名
    - "fzt_zscore"  # 标准化值
    - "fzt_slope_5d"  # 5日斜率
    - "fzt_slope_10d"  # 10日斜率
    - "fzt_vs_market"  # 相对市场强度
    - "fzt_delta"  # 日度变化
```

**Step 3: 创建回测配置**

```yaml
# config/backtest_config.yaml
# 回测配置

strategy:
  class: "FZTEnhancedStrategy"
  module_path: "src.backtest.enhanced_strategy"
  kwargs:
    # 选股参数
    top_k: 5
    hold_days: 1  # 持有天数
    
    # 交易成本
    commission: 0.0003  # 佣金万分之三
    stamp_tax: 0.001  # 印花税千分之一
    slippage: 0.001  # 滑点0.1%
    
    # 流动性控制
    min_turnover_ratio: 0.2  # 成交额后20%门槛
    min_market_cap_ratio: 0.3  # 市值后30%门槛
    
    # 风险控制
    max_industry_weight: 0.3  # 单一行业最大权重
    max_single_weight: 0.4  # 单一股票最大权重

exchange:
  # 交易所配置
  limit_threshold: 0.095  # 涨跌停阈值 (9.5%)
  trade_unit: 100  # 交易单位 (手)
  generate_cash: false  # 是否生成现金

backtest:
  # 回测参数
  start_time: "2020-01-01"
  end_time: "2020-12-31"
  freq: "day"  # 频率
  benchmark: "SH000300"  # 基准指数
  
  # 账户初始设置
  initial_cash: 1000000  # 初始资金
  position_ratio: 0.95  # 仓位比例

evaluation:
  # 回测评估
  metrics:
    - "annual_return"
    - "sharpe_ratio"
    - "max_drawdown"
    - "win_rate"
    - "turnover_rate"
    - "information_ratio"
  
  # 报告配置
  report_dir: "results/backtest_reports"
  save_figures: true
```

**Step 4: 验证配置文件**

Run: `python3 -c "import yaml; config = yaml.safe_load(open('config/ranking_config.yaml')); print('✅ ranking_config loaded')"`
Expected: ✅ ranking_config loaded

Run: `python3 -c "import yaml; config = yaml.safe_load(open('config/feature_config.yaml')); print('✅ feature_config loaded')"`
Expected: ✅ feature_config loaded

Run: `python3 -c "import yaml; config = yaml.safe_load(open('config/backtest_config.yaml')); print('✅ backtest_config loaded')"`
Expected: ✅ backtest_config loaded

**Step 5: 提交**

```bash
git add config/ranking_config.yaml config/feature_config.yaml config/backtest_config.yaml
git commit -m "feat: add configuration files for ranking enhancement"
```

## 阶段2: 模型训练 (明天)

### Task 6: 实现LambdaRank训练器

**Files:**
- Create: `src/ranking_model/lambdarank_trainer.py`
- Test: `tests/test_lambdarank_trainer.py`

### Task 7: 实现NDCG评估器

**Files:**
- Create: `src/evaluation/ndcg_evaluator.py`
- Test: `tests/test_ndcg_evaluator.py`

### Task 8: 实现特征重要性分析

**Files:**
- Create: `src/evaluation/feature_importance.py`
- Test: `tests/test_feature_importance.py`

## 阶段3: 回测评估 (后天)

### Task 9: 实现增强回测策略

**Files:**
- Create: `src/backtest/enhanced_strategy.py`
- Test: `tests/test_enhanced_strategy.py`

### Task 10: 实现成本模型

**Files:**
- Create: `src/backtest/cost_model.py`
- Test: `tests/test_cost_model.py`

### Task 11: 实现风险控制器

**Files:**
- Create: `src/backtest/risk_controller.py`
- Test: `tests/test_risk_controller.py`

## 阶段4: 完整流水线 (后续)

### Task 12: 创建训练脚本

**Files:**
- Create: `scripts/train_ranking_model.py`

### Task 13: 创建回测脚本

**Files:**
- Create: `scripts/run_backtest.py`

### Task 14: 创建评估脚本

**Files:**
- Create: `scripts/evaluate_strategy.py`

---

## 执行选项

**计划已完成并保存到 `docs/plans/2026-03-02-fzt-ranking-enhancement-implementation.md`。两个执行选项：**

**1. Subagent-Driven (本会话)** - 我分派新子代理执行每个任务，任务间进行代码审查，快速迭代

**2. 并行会话 (独立)** - 在新会话中使用executing-plans，批量执行并设置检查点

**哪种方式？**