# FZT选股公式回测系统

基于原始FZT公式（VAR1A-VAR6A复杂计算）的量化回测系统，覆盖2006-2026年共20.5年A股数据。支持多种技术因子组合和参数化控制。

## 🎯 项目特点

### **📊 数据完整性**
- **时间跨度**：2006-01-01 至 2026-02-06（20.5年）
- **股票覆盖**：9359只A股股票
- **数据格式**：统一的QLIB .bin格式
- **数据分割**：
  - **2006-2020年**：3875只股票（QLIB标准数据）
  - **2021-2026年**：5484只股票（自定义转换数据）

### **🚀 技术优化**
- **向量化计算**：1000倍性能提升（从数小时到几十秒）
- **通达信SMA算法**：精准复刻原始FZT公式计算
- **复权处理**：保证价格连续性
- **批量处理**：全市场一次性计算，避免循环

### **🔧 模块化设计**
- **清晰职责分离**：公共函数在`src/`中，私有配置在脚本中
- **极度精简**：只有6个Python文件（2脚本 + 4模块）
- **高度可重用**：`src/`中的模块可以被其他项目使用
- **完全自包含**：每个脚本包含完整的数据配置

### **🎛️ 参数化控制**
- **公式组合控制**：FZT、ZSQSX公式独立启用/禁用
- **技术因子控制**：成交量比、乖离率、RSI、OBV因子独立控制
- **筛选参数控制**：所有阈值和参数可配置
- **TOP排序控制**：按FZT面积排序取TOP N/K

## 📁 项目结构

### **GitHub仓库内容（极度精简）**
```
FZT_Quant/                              # 主目录
├── README.md                           # 项目说明（本文件）
├── requirements.txt                    # Python依赖包
├── .gitignore                          # Git忽略规则（排除大数据文件）
├── scripts/                            # ✅ 回测脚本目录
│   ├── fzt_factor_backtest.py          # ✅ FZT统一回测脚本（支持所有时期）
│   ├── b1_unified_backtest.py          # ✅ B1因子统一回测脚本（支持纯B1和B1+新因子）
│   ├── b2_factor_backtest.py           # ✅ B2因子独立回测脚本
│   └── csv_to_bin_converter.py         # CSV转QLIB二进制格式工具
└── src/                                # ✅ 核心模块目录
    ├── __init__.py                     # 模块导出文件
    ├── fzt_core.py                     # ✅ FZT核心计算模块（通用函数）
    ├── zsqsx_core.py                   # ✅ ZSQSX公式计算模块
    ├── factors.py                      # ✅ 技术因子计算模块
    ├── b2_core.py                      # ✅ B2因子核心模块
    └── data_loader.py                  # ✅ 公共数据加载模块（不包含私有数据路径）
```

### **本地额外文件（被.gitignore忽略）**
```
data/                                   # 数据目录（约120MB，不推送到GitHub）
├── 2006_2020/                          # 2006-2020年QLIB标准数据
│   ├── calendars/day.txt               # 交易日历
│   ├── features/                       # 3875只股票.bin数据
│   └── instruments/all.txt             # 股票列表
└── 2021_2026/                          # 2021-2026年自定义转换数据
    ├── calendars/day.txt               # 自定义交易日历
    ├── features/                       # 5484只股票.bin数据
    └── instruments/all.txt             # 股票列表

logs/                                   # 执行日志目录（自动生成）
results/                                # 回测结果目录（自动生成）
```

## 🚀 快速开始

### **1. 克隆仓库**
```bash
git clone git@github.com:littleguohao/FZT.git
cd FZT
```

### **2. 安装依赖**
```bash
pip install -r requirements.txt
```

### **3. 准备数据**
数据文件较大（约120MB），需要单独下载或从原始来源转换：
- **2006-2020年数据**：QLIB标准cn_data
- **2021-2026年数据**：自定义转换的.bin格式数据

### **4. 运行回测**
```bash
# 查看帮助
python scripts/fzt_factor_backtest.py --help

# 基础回测（2006-2020年，单独FZT公式）
python scripts/fzt_factor_backtest.py --period 2006_2020 --verify-fzt

# 基础回测（2021-2026年，单独FZT公式）
python scripts/fzt_factor_backtest.py --period 2021_2026 --verify-fzt

# 组合回测（FZT + ZSQSX）
python scripts/fzt_factor_backtest.py --period 2006_2020 --fzt --zsqsx --verify-fzt

# 添加技术因子
python scripts/fzt_factor_backtest.py --period 2006_2020 --verify-fzt --volume-ratio-enable --bias-enable

# TOP排序筛选
python scripts/fzt_factor_backtest.py --period 2006_2020 --verify-fzt --top-n 4 --top-k 4

# 复杂组合
python scripts/fzt_factor_backtest.py \
  --period 2006_2020 \
  --fzt \
  --zsqsx \
  --verify-fzt \
  --volume-ratio-enable \
  --bias-enable \
  --rsi-enable \
  --obv-enable \
  --top-n 4 \
  --top-k 4

# 两个时期同时回测
python scripts/fzt_factor_backtest.py --period both --top-n 0
```

## 📊 回测结果

### **原始FZT公式表现（基于完整回测）：**

| 时期 | 股票数量 | 总信号数 | 成功率 | 执行时间 |
|------|----------|----------|--------|----------|
| **2006-2020年** | 3875只 | 52,295个 | **52.90%** | 30秒 |
| **2021-2026年** | 5484只 | 573,237个 | **46.75%** | 35秒 |
| **总计** | 9359只 | 625,532个 | **47.25%** | 65秒 |

### **B1因子策略表现（基于完整回测）：**

#### **纯B1因子策略**
| 时期 | 总信号数 | 成功率 | 执行时间 |
|------|----------|--------|----------|
| **2006-2020年** | 165,158个 | **83.56%** | 8秒 |
| **2021-2026年** | 251,317个 | **79.05%** | 8秒 |
| **总计** | 416,475个 | **80.84%** | 16秒 |

#### **B1+新增因子策略（默认45%/30%配置）**
| 时期 | 总信号数 | 成功率 | 过滤比例 | 执行时间 |
|------|----------|--------|----------|----------|
| **2006-2020年** | 108,195个 | **83.95%** | 65.51% | 8秒 |
| **2021-2026年** | 208,316个 | **79.17%** | 82.89% | 8秒 |
| **总计** | 316,511个 | **80.80%** | 76.00% | 16秒 |

#### **不同阈值配置对比**
| 配置 | 成功率 | 信号保留 | 特点 |
|------|--------|----------|------|
| **纯B1因子** | 80.84% | 100% | 基础策略 |
| **B1+45%/30%** | 80.80% | 76.0% | ✅ 推荐配置 |
| **B1+35%/30%** | 80.66% | 68.8% | 平衡配置 |
| **B1+30%/30%** | 80.54% | 63.6% | 严格配置 |
| **B1+20%/20%** | 79.91% | 48.6% | 最严格配置 |

### **B2因子策略表现**
| 时期 | 总信号数 | 成功率 | 执行时间 |
|------|----------|--------|----------|
| **2006-2020年** | 约50,000个 | **~53%** | 10秒 |
| **2021-2026年** | 约70,000个 | **~53%** | 10秒 |

### **技术因子筛选效果（2006-2020年）：**

| 因子组合 | 总信号数 | 成功率 | 信号减少 |
|----------|----------|--------|----------|
| **单独FZT** | 52,295 | **52.90%** | - |
| **FZT+成交量比** | 13,845 | 50.12% | -73.5% |
| **FZT+乖离率** | 22,156 | 51.23% | -57.6% |
| **FZT+RSI** | 20,418 | 50.89% | -61.0% |
| **FZT+OBV** | 2,894 | 48.48% | -94.5% |
| **FZT+成交量比乖离率** | 7,934 | 48.89% | -84.8% |
| **FZT+RSI-OBV** | 5,897 | 49.62% | -88.7% |
| **FZT+ZSQSX** | 11,720 | 50.84% | -77.6% |
| **FZT+TOP4** | 10,419 | 53.41% | -80.1% |

### **年度成功率趋势：**
- **2006-2020年**：43.07% - 65.62%（平均52.90%）
- **2021-2026年**：44.68% - 48.74%（平均46.75%）
- **整体**：接近随机水平（50%），作为独立选股策略效果有限

## 🏗️ 模块设计

### **1. `src/fzt_core.py` - FZT核心计算模块**
```python
# 通达信SMA递归算法（精准复刻）
def tdx_sma_series(series_vals: np.ndarray, N: int, M: int = 1) -> np.ndarray
def tdx_sma(group_series: pd.Series, N: int, M: int = 1) -> pd.Series

# 完整FZT公式计算
def calc_brick_pattern_final(df_raw: pd.DataFrame, ...) -> pd.DataFrame

# 向量化计算
def calculate_fzt_features_vectorized(df: pd.DataFrame) -> pd.DataFrame
```

### **2. `src/zsqsx_core.py` - ZSQSX公式计算模块**
```python
# ZSQSX指标计算
def calc_zsdkx(df, M1=14, M2=28, M3=57, M4=114) -> pd.DataFrame

# ZSQSX选股信号条件
def get_zsdkx_signal_conditions(df_zsdkx) -> pd.DataFrame
```

### **3. `src/factors.py` - 技术因子计算模块**
```python
# 成交量比因子
def add_volume_ratio_factor(df) -> pd.DataFrame
def filter_by_volume_ratio_factor(df, volume_ratio_threshold=1.2) -> pd.DataFrame

# 乖离率因子
def add_bias_factor(df) -> pd.DataFrame
def filter_by_bias_factor(df, bias_lower=-0.05, bias_upper=0.2) -> pd.DataFrame

# RSI因子
def add_rsi_factor(df, rsi_period=14) -> pd.DataFrame
def filter_by_rsi_factor(df, rsi_low=40, rsi_high=70) -> pd.DataFrame

# OBV因子
def add_obv_factor(df) -> pd.DataFrame
def filter_by_obv_factor(df, obv_lookback=20) -> pd.DataFrame

# 短期暴涨过滤因子
def add_max_gain_condition(df, lookback=35, threshold=0.7) -> pd.DataFrame
def filter_by_max_gain_condition(df, lookback=35, threshold=0.7) -> pd.DataFrame

# 组合因子（向后兼容）
def add_volume_and_bias_factors(df) -> pd.DataFrame
def filter_by_volume_bias_factors(df, ...) -> pd.DataFrame
def filter_by_rsi_obv_factors(df, ...) -> pd.DataFrame
```

### **4. `src/b2_core.py` - B2因子核心模块**
```python
# KDJ指标计算
def compute_kdj(df, n=9, m1=3, m2=3) -> pd.DataFrame

# B2因子计算
def add_B2_factor(df, j_prev_thresh=13, j_today_thresh=55, ...) -> pd.DataFrame

# B2因子筛选
def filter_by_B2_factor(df, j_prev_thresh=13, j_today_thresh=55, ...) -> pd.DataFrame

# B2成功率计算
def calculate_b2_success_rate(df, j_prev_thresh=13, j_today_thresh=55, ...) -> Dict[str, Any]
```

### **5. `src/factors.py` - 新增可配置因子模块**
```python
# 新增可配置因子（最大涨幅限制 + 累计换手率限制）
def calc_custom_factors(df, N=20, M=0.45, Y=20, X=0.30, exclude_today=True) -> pd.DataFrame

# 添加自定义因子
def add_custom_factors(df, N=20, M=0.45, Y=20, X=0.30, exclude_today=True) -> pd.DataFrame

# 使用自定义因子筛选
def filter_by_custom_factors(df, N=20, M=0.45, Y=20, X=0.30, exclude_today=True) -> pd.DataFrame
```

### **5. `src/data_loader.py` - 公共数据加载模块**
```python
# 通用QLIB数据加载（不包含私有数据路径）
def load_stock_data_qlib(data_dir, instruments, calc_start, calc_end, ...)

# 读取股票列表
def get_instruments_from_file(instruments_file)
```

### **3. 脚本中的私有数据配置**
```python
# 在scripts/fzt_final_2006_2020.py中
def load_2006_2020_data(project_root):
    data_dir = str(project_root / 'data' / '2006_2020')  # 私有路径
    instruments = instruments[:3875]  # 私有配置
    return load_stock_data_qlib(data_dir, instruments, ...)  # 调用公共函数

# 在scripts/fzt_final_2021_2026.py中  
def load_2021_2026_data(project_root):
    data_dir = str(project_root / 'data' / '2021_2026')  # 私有路径
    instruments = instruments[:5484]  # 私有配置
    return load_stock_data_qlib(data_dir, instruments, ...)  # 调用公共函数
```

### **设计原则：**
- **公共函数在模块中**：可重用、不依赖具体数据路径
- **私有配置在脚本中**：自包含、完整的数据配置
- **清晰的职责分离**：通用功能 vs 具体实现
- **参数化控制**：所有功能通过命令行参数控制
- **模块化扩展**：易于添加新的技术因子或策略公式

## 📈 性能对比

| 优化阶段 | 2006-2020年 | 2021-2026年 | 性能提升 |
|----------|-------------|-------------|----------|
| **原始循环** | 数小时 | 数小时 | 1× |
| **批量处理** | 数十分钟 | 数十分钟 | 10× |
| **向量化优化** | **27.03秒** | **24.43秒** | **1000×** |

## 🔧 技术实现

### **📈 原始FZT公式**
```python
# 通达信公式（VAR1A-VAR6A复杂计算）
VAR1A:=(HHV(HIGH,4)-CLOSE)/(HHV(HIGH,4)-LLV(LOW,4))*100-90;
VAR2A:=SMA(VAR1A,4,1)+100;
VAR3A:=(CLOSE-LLV(LOW,4))/(HHV(HIGH,4)-LLV(LOW,4))*100;
VAR4A:=SMA(VAR3A,6,1);
VAR5A:=SMA(VAR4A,6,1)+100;
VAR6A:=VAR5A-VAR2A;
砖型图:=IF(VAR6A>4,VAR6A-4,0);
砖型图面积:=ABS(砖型图 - REF(砖型图,1));
AA:=(REF(砖型图,1)<砖型图);
首次多头增强:=(REF(AA,1)=0) AND (AA=1);
砖型图面积增幅:=砖型图面积 > REF(砖型图面积,1) * 2/3;
选股条件:=首次多头增强 AND 砖型图面积增幅;
```

### **📊 ZSQSX公式**
```python
# 趋势线公式
QSX = EMA(EMA(CLOSE,10),10)
MA1 = MA(CLOSE,60)
MA2 = EMA(CLOSE,13)
DKS = (MA(CLOSE,M1)+MA(CLOSE,M2)+MA(CLOSE,M3)+MA(CLOSE,M4))/4
选股条件: QSX > DKS
```

### **🔧 技术因子定义**
```python
# 成交量比 = volume / MA(volume, 5)
# 乖离率 = (close - MA(close, 60)) / MA(close, 60)
# RSI(14)在[40,70]之间
# OBV创20日新高
```

### **⚡ 向量化优化**
- **从"分批逐股票循环"升级为"全市场向量化一次性计算"**
- **使用`groupby('instrument') + transform`实现向量化滚动计算**
- **避免Python级循环，全部用pandas内置函数**
- **性能提升：1000倍以上（从数小时到几十秒）**

### **🎯 成功条件**
- **信号条件**：原始FZT公式的"选股条件"为True
- **成功判断**：次日收盘价 > 当日收盘价
- **实现方式**：`df['success'] = df['next_close'] > df['close']`

## 📋 依赖说明

### **核心依赖**
```txt
qlib>=0.9.0          # 量化投资平台
pandas>=2.0.0        # 数据分析
numpy>=1.24.0        # 数值计算
```

### **开发依赖**
```txt
matplotlib>=3.7.0    # 数据可视化
seaborn>=0.12.0      # 统计可视化
jupyter>=1.0.0       # 交互式笔记本
```

## 🔍 使用示例

### **1. 运行完整回测**
```bash
# 2006-2020年回测
python scripts/fzt_final_2006_2020.py

# 2021-2026年回测
python scripts/fzt_final_2021_2026.py
```

### **2. 参数化控制示例**
```bash
# 查看完整参数帮助
python scripts/fzt_factor_backtest.py --help

# 时期选择控制
python scripts/fzt_factor_backtest.py --period 2006_2020  # 2006-2020年
python scripts/fzt_factor_backtest.py --period 2021_2026  # 2021-2026年
python scripts/fzt_factor_backtest.py --period both        # 两个时期都回测

# 公式组合控制
python scripts/fzt_factor_backtest.py --period 2006_2020 --fzt --no-zsqsx --verify-fzt  # 只使用FZT
python scripts/fzt_factor_backtest.py --period 2006_2020 --no-fzt --zsqsx              # 只使用ZSQSX
python scripts/fzt_factor_backtest.py --period 2006_2020 --fzt --zsqsx --verify-fzt    # 同时使用FZT和ZSQSX

# 技术因子控制
python scripts/fzt_factor_backtest.py --period 2006_2020 --verify-fzt --volume-ratio-enable  # 成交量比因子
python scripts/fzt_factor_backtest.py --period 2006_2020 --verify-fzt --bias-enable          # 乖离率因子
python scripts/fzt_factor_backtest.py --period 2006_2020 --verify-fzt --rsi-enable           # RSI因子
python scripts/fzt_factor_backtest.py --period 2006_2020 --verify-fzt --obv-enable           # OBV因子
python scripts/fzt_factor_backtest.py --period 2006_2020 --verify-fzt --max-gain-enable      # 短期暴涨过滤因子

# 参数调整
python scripts/fzt_factor_backtest.py --period 2006_2020 --verify-fzt --volume-ratio-enable --volume-ratio-threshold 1.5
python scripts/fzt_factor_backtest.py --period 2006_2020 --verify-fzt --bias-enable --bias-lower -0.1 --bias-upper 0.3
python scripts/fzt_factor_backtest.py --period 2006_2020 --verify-fzt --rsi-enable --rsi-low 30 --rsi-high 80
python scripts/fzt_factor_backtest.py --period 2006_2020 --verify-fzt --obv-enable --obv-lookback 30
python scripts/fzt_factor_backtest.py --period 2006_2020 --verify-fzt --max-gain-enable --max-gain-lookback 20 --max-gain-threshold 0.5

# TOP排序控制
python scripts/fzt_factor_backtest.py --period 2006_2020 --verify-fzt --top-n 4 --top-k 4    # TOP4筛选
python scripts/fzt_factor_backtest.py --period 2006_2020 --verify-fzt --top-n 10 --top-k 10  # TOP10筛选

# 复杂组合
python scripts/fzt_factor_backtest.py \
  --period 2006_2020 \
  --fzt \
  --zsqsx \
  --verify-fzt \
  --volume-ratio-enable \
  --bias-enable \
  --rsi-enable \
  --obv-enable \
  --max-gain-enable \
  --top-n 4 \
  --top-k 4
```

### **3. 使用核心模块（在其他项目中）**
```python
# 导入公共模块
from src.fzt_core import calc_brick_pattern_final, tdx_sma
from src.zsqsx_core import calc_zsdkx, get_zsdkx_signal_conditions
from src.factors import (
    add_volume_ratio_factor, filter_by_volume_ratio_factor,
    add_bias_factor, filter_by_bias_factor,
    add_rsi_factor, filter_by_rsi_factor,
    add_obv_factor, filter_by_obv_factor
)
from src.data_loader import load_stock_data_qlib, get_instruments_from_file

# 加载数据（使用你自己的数据路径）
data_dir = 'your/data/path'
instruments = ['SH000300', 'SZ002790']
df = load_stock_data_qlib(
    data_dir=data_dir,
    instruments=instruments,
    calc_start='2020-01-01',
    calc_end='2020-12-31'
)

# 计算FZT信号
df_with_fzt = calc_brick_pattern_final(df)

# 计算ZSQSX信号
df_zsdkx = calc_zsdkx(df)
df_zsqsx = get_zsdkx_signal_conditions(df_zsdkx)

# 添加技术因子
df_with_factors = add_volume_ratio_factor(df)
df_with_factors = add_bias_factor(df_with_factors)
df_with_factors = add_rsi_factor(df_with_factors, rsi_period=14)
df_with_factors = add_obv_factor(df_with_factors)

# 筛选信号
df_filtered = filter_by_volume_ratio_factor(df_with_factors, volume_ratio_threshold=1.2)
df_filtered = filter_by_bias_factor(df_filtered, bias_lower=-0.05, bias_upper=0.2)
df_filtered = filter_by_rsi_factor(df_filtered, rsi_low=40, rsi_high=70)
df_filtered = filter_by_obv_factor(df_filtered, obv_lookback=20)

# 使用通达信SMA算法
sma_result = tdx_sma(data_series, N=4, M=1)
```

### **3. 创建新的回测脚本**
```python
# 基于现有模板创建新脚本
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fzt_core import calc_brick_pattern_final
from src.data_loader import load_stock_data_qlib

def load_my_data(project_root: Path):
    """你的私有数据加载函数"""
    data_dir = str(project_root / 'data' / 'my_data')
    instruments = get_instruments_from_file(data_dir + '/instruments/all.txt')
    return load_stock_data_qlib(
        data_dir=data_dir,
        instruments=instruments,
        calc_start='2010-01-01',
        calc_end='2015-12-31'
    )

def my_backtest():
    """你的回测函数"""
    project_root = Path(__file__).parent.parent
    df = load_my_data(project_root)
    df = calc_brick_pattern_final(df)
    # ... 你的回测逻辑
```

## 🎯 使用场景

### **1. 策略研究**
- 验证原始FZT公式的有效性
- 分析因子在不同市场周期的表现
- 作为基准策略对比其他选股方法
- 研究技术因子组合效果

### **2. 技术学习**
- 学习向量化优化技术
- 理解通达信公式实现
- 掌握大规模数据回测方法
- 学习参数化控制系统设计

### **3. 因子开发**
- 基于FZT公式开发衍生因子
- 结合其他技术指标构建复合策略
- 参数优化和敏感性分析
- 多因子组合策略研究

### **4. 参数化研究**
- 研究不同参数组合的效果
- 分析因子阈值对信号质量的影响
- 优化TOP排序参数
- 探索最优因子组合

## 🔮 未来扩展

### **1. 策略优化**
- 参数调优（周期参数、阈值参数）
- 结合其他技术指标（MACD、KDJ、布林带等）
- 多因子组合策略
- 机器学习因子挖掘

### **2. 功能增强**
- 风险调整收益分析
- 交易成本模型
- 组合优化和仓位管理
- 实时信号监控

### **3. 可视化**
- 策略表现图表
- 因子相关性分析
- 回撤和风险指标可视化
- 参数敏感性热力图

### **4. 扩展因子**
- 添加更多技术因子（MACD、KDJ、布林带等）
- 添加基本面因子（PE、PB、ROE等）
- 添加市场情绪因子
- 添加行业轮动因子

## 📝 注意事项

### **数据要求**
- 数据文件较大（约120MB），需要足够磁盘空间
- 需要QLIB标准数据格式（.bin文件）
- 交易日历必须与数据时间范围匹配

### **运行环境**
- Python 3.8+ 环境
- 建议8GB+内存（处理全市场数据）
- 需要安装QLIB和相关依赖

### **结果解释**
- 原始FZT公式成功率接近随机水平（50%）
- 技术因子筛选会大幅减少信号数量，成功率变化不大
- TOP排序筛选可能提高成功率，但信号数量减少
- 回测结果仅供参考，不构成投资建议

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进本项目：

1. **报告问题**：在GitHub Issues中描述问题
2. **功能建议**：提出改进建议或新功能想法
3. **代码贡献**：遵循现有代码风格，添加测试
4. **文档改进**：完善文档或添加使用示例

## 📄 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 🙏 致谢

- **QLIB团队**：提供优秀的量化投资平台
- **通达信公式社区**：原始FZT公式的实现参考
- **向量化优化参考**：用户提供的优化方案文档

---

**最后更新：2026-03-09**

**项目状态：✅ 完成 - 极度精简、模块化、参数化、可维护**

**版本：v4.0 - B1因子统一版本**

**主要更新：**
1. ✅ **B1因子统一脚本**：合并`b1_factor_backtest.py`和`b1_with_custom_factors.py`为`b1_unified_backtest.py`
2. ✅ **新增可配置因子**：最大涨幅限制 + 累计换手率限制因子
3. ✅ **优化默认配置**：默认45%/30%配置表现最佳
4. ✅ **清理冗余脚本**：删除临时测试脚本，保持最小化设计
5. ✅ **完整参数化控制**：所有功能通过命令行参数控制

**策略表现：**
- **B1因子成功率**：80.84%（纯B1）→ 80.80%（B1+新因子）
- **信号保留比例**：76.0%（过滤24%高风险信号）
- **执行效率**：16秒完成2006-2026年全市场回测
- **推荐配置**：最大涨幅45% + 换手率30%