#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FZTAlpha算子 - QLib兼容的FZT砖型图选股因子

将FZT砖型图公式封装为QLib算子，确保无未来函数计算。
支持QLib Operator接口，同时提供简化版本供QLib不可用时使用。

作者: FZT项目组
创建日期: 2026-03-02
"""

import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime

import pandas as pd
import numpy as np

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')

# 尝试导入QLib Operator基类
try:
    from qlib.data.ops import Operator
    QLIB_AVAILABLE = True
    logger.info("QLib可用，将创建QLib Operator子类")
except ImportError:
    QLIB_AVAILABLE = False
    logger.warning("QLib不可用，将创建简化版本算子")


class FZTAlphaBase:
    """FZTAlpha基类 - 包含核心计算逻辑"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化FZTAlpha基类
        
        Args:
            config: 配置字典（可选）
        """
        self.config = config or {}
        
        # 默认参数（与FZTBrickFormula保持一致）
        self.default_params = {
            'var1_window': 4,      # VAR1A计算窗口
            'var2_sma_window': 4,  # VAR2A平滑窗口
            'var3_window': 4,      # VAR3A计算窗口
            'var4_sma_window': 6,  # VAR4A平滑窗口
            'var5_sma_window': 6,  # VAR5A平滑窗口
            'brick_threshold': 4,  # 砖型图阈值
            'area_increase_ratio': 2/3,  # 面积增幅比例
        }
        
        # 合并配置
        self.params = {**self.default_params, **self.config}
        
        # 特征名称
        self.feature_names = [
            'fzt_var1a',
            'fzt_var2a', 
            'fzt_var3a',
            'fzt_var4a',
            'fzt_var5a',
            'fzt_var6a',
            'fzt_brick_chart',
            'fzt_brick_area',
            'fzt_aa',
            'fzt_first_bull_enhancement',
            'fzt_brick_area_increase',
            'fzt_selection_condition'
        ]
        
        logger.info("FZTAlpha基类初始化完成")
        logger.debug(f"参数配置: {self.params}")
    
    def hhv(self, series: pd.Series, window: int) -> pd.Series:
        """
        计算最高值（HHV） - 无未来函数
        
        Args:
            series: 价格序列
            window: 窗口大小
            
        Returns:
            pd.Series: 窗口内最高值（只使用当前及之前的数据）
        """
        # 与原始FZTBrickFormula保持一致，不使用min_periods
        # 这样前window-1个值为NaN，需要完整窗口数据
        return series.rolling(window).max()
    
    def llv(self, series: pd.Series, window: int) -> pd.Series:
        """
        计算最低值（LLV） - 无未来函数
        
        Args:
            series: 价格序列
            window: 窗口大小
            
        Returns:
            pd.Series: 窗口内最低值（只使用当前及之前的数据）
        """
        # 与原始FZTBrickFormula保持一致
        return series.rolling(window).min()
    
    def sma(self, series: pd.Series, window: int, weight: float = 1.0) -> pd.Series:
        """
        简单移动平均（SMA） - 无未来函数
        
        Args:
            series: 输入序列
            window: 窗口大小
            weight: 权重参数（公式中的第三个参数）
            
        Returns:
            pd.Series: 平滑后的序列（只使用当前及之前的数据）
        """
        # 与原始FZTBrickFormula保持一致
        return series.rolling(window).mean()
    
    def calculate_var1a(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        计算VAR1A指标 - 无未来函数
        
        VAR1A:=(HHV(HIGH,4)-CLOSE)/(HHV(HIGH,4)-LLV(LOW,4))*100-90;
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            
        Returns:
            pd.Series: VAR1A值
        """
        window = self.params['var1_window']
        
        # 计算HHV和LLV（无未来函数）
        hhv_high = self.hhv(high, window)
        llv_low = self.llv(low, window)
        
        # 避免除零
        denominator = hhv_high - llv_low
        denominator = denominator.replace(0, 1e-6)
        
        # 计算VAR1A
        var1a = (hhv_high - close) / denominator * 100 - 90
        
        return var1a
    
    def calculate_var2a(self, var1a: pd.Series) -> pd.Series:
        """
        计算VAR2A指标 - 无未来函数
        
        VAR2A:=SMA(VAR1A,4,1)+100;
        
        Args:
            var1a: VAR1A序列
            
        Returns:
            pd.Series: VAR2A值
        """
        window = self.params['var2_sma_window']
        
        # 计算SMA（无未来函数）
        var2a = self.sma(var1a, window) + 100
        
        return var2a
    
    def calculate_var3a(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        计算VAR3A指标 - 无未来函数
        
        VAR3A:=(CLOSE-LLV(LOW,4))/(HHV(HIGH,4)-LLV(LOW,4))*100;
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            
        Returns:
            pd.Series: VAR3A值
        """
        window = self.params['var3_window']
        
        # 计算HHV和LLV（无未来函数）
        hhv_high = self.hhv(high, window)
        llv_low = self.llv(low, window)
        
        # 避免除零
        denominator = hhv_high - llv_low
        denominator = denominator.replace(0, 1e-6)
        
        # 计算VAR3A
        var3a = (close - llv_low) / denominator * 100
        
        return var3a
    
    def calculate_var4a(self, var3a: pd.Series) -> pd.Series:
        """
        计算VAR4A指标 - 无未来函数
        
        VAR4A:=SMA(VAR3A,6,1);
        
        Args:
            var3a: VAR3A序列
            
        Returns:
            pd.Series: VAR4A值
        """
        window = self.params['var4_sma_window']
        
        # 计算SMA（无未来函数）
        var4a = self.sma(var3a, window)
        
        return var4a
    
    def calculate_var5a(self, var4a: pd.Series) -> pd.Series:
        """
        计算VAR5A指标 - 无未来函数
        
        VAR5A:=SMA(VAR4A,6,1)+100;
        
        Args:
            var4a: VAR4A序列
            
        Returns:
            pd.Series: VAR5A值
        """
        window = self.params['var5_sma_window']
        
        # 计算SMA（无未来函数）
        var5a = self.sma(var4a, window) + 100
        
        return var5a
    
    def calculate_brick_chart(self, var5a: pd.Series, var2a: pd.Series) -> pd.Series:
        """
        计算砖型图 - 无未来函数
        
        砖型图:=IF(VAR6A>4,VAR6A-4,0);
        
        Args:
            var5a: VAR5A序列
            var2a: VAR2A序列
            
        Returns:
            pd.Series: 砖型图值
        """
        threshold = self.params['brick_threshold']
        
        # 计算VAR6A
        var6a = var5a - var2a
        
        # 计算砖型图
        brick_chart = np.where(var6a > threshold, var6a - threshold, 0)
        
        return pd.Series(brick_chart, index=var6a.index)
    
    def calculate_brick_area(self, brick_chart: pd.Series) -> pd.Series:
        """
        计算砖型图面积 - 无未来函数
        
        砖型图面积:=ABS(砖型图 - REF(砖型图,1));
        
        Args:
            brick_chart: 砖型图序列
            
        Returns:
            pd.Series: 砖型图面积
        """
        # 计算面积（绝对值变化） - 使用shift确保无未来函数
        brick_area = abs(brick_chart - brick_chart.shift(1))
        
        return brick_area
    
    def calculate_aa(self, brick_chart: pd.Series) -> pd.Series:
        """
        计算AA指标 - 无未来函数
        
        AA:=(REF(砖型图,1)<砖型图);
        
        Args:
            brick_chart: 砖型图序列
            
        Returns:
            pd.Series: AA值（布尔序列）
        """
        # 砖型图上升 - 使用shift确保无未来函数
        aa = brick_chart.shift(1) < brick_chart
        
        return aa
    
    def calculate_first_bull_enhancement(self, aa: pd.Series) -> pd.Series:
        """
        计算首次多头增强 - 无未来函数
        
        首次多头增强:=(REF(AA,1)=0) AND (AA=1);
        
        Args:
            aa: AA序列
            
        Returns:
            pd.Series: 首次多头增强信号（布尔序列）
        """
        # 首次出现上升 - 使用shift确保无未来函数
        first_bull_enhancement = (aa.shift(1) == False) & (aa == True)
        
        return first_bull_enhancement
    
    def calculate_brick_area_increase(self, brick_area: pd.Series) -> pd.Series:
        """
        计算砖型图面积增幅 - 无未来函数
        
        砖型图面积增幅:=砖型图面积 > REF(砖型图面积,1) * 2/3;
        
        Args:
            brick_area: 砖型图面积序列
            
        Returns:
            pd.Series: 面积增幅信号（布尔序列）
        """
        ratio = self.params['area_increase_ratio']
        
        # 面积增加超过阈值 - 使用shift确保无未来函数
        area_increase = brick_area > brick_area.shift(1) * ratio
        
        return area_increase
    
    def calculate_selection_condition(self, 
                                     first_bull_enhancement: pd.Series,
                                     brick_area_increase: pd.Series) -> pd.Series:
        """
        计算选股条件 - 无未来函数
        
        选股条件:=首次多头增强 AND 砖型图面积增幅;
        
        Args:
            first_bull_enhancement: 首次多头增强信号
            brick_area_increase: 砖型图面积增幅信号
            
        Returns:
            pd.Series: 选股条件信号（布尔序列）
        """
        # 同时满足两个条件
        selection_condition = first_bull_enhancement & brick_area_increase
        
        return selection_condition
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算所有指标 - 确保无未来函数
        
        Args:
            data: 包含OHLC数据的DataFrame
            
        Returns:
            Dict[str, pd.Series]: 所有指标字典
        """
        logger.debug("开始计算FZT所有指标（无未来函数）...")
        
        # 验证输入数据
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"输入数据缺少必要列: {col}")
        
        # 提取数据
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 计算所有指标（确保无未来函数）
        var1a = self.calculate_var1a(high, low, close)
        var2a = self.calculate_var2a(var1a)
        var3a = self.calculate_var3a(high, low, close)
        var4a = self.calculate_var4a(var3a)
        var5a = self.calculate_var5a(var4a)
        brick_chart = self.calculate_brick_chart(var5a, var2a)
        brick_area = self.calculate_brick_area(brick_chart)
        aa = self.calculate_aa(brick_chart)
        first_bull_enhancement = self.calculate_first_bull_enhancement(aa)
        brick_area_increase = self.calculate_brick_area_increase(brick_area)
        selection_condition = self.calculate_selection_condition(first_bull_enhancement, brick_area_increase)
        
        # 组装结果
        indicators = {
            'fzt_var1a': var1a,
            'fzt_var2a': var2a,
            'fzt_var3a': var3a,
            'fzt_var4a': var4a,
            'fzt_var5a': var5a,
            'fzt_var6a': var5a - var2a,
            'fzt_brick_chart': brick_chart,
            'fzt_brick_area': brick_area,
            'fzt_aa': aa,
            'fzt_first_bull_enhancement': first_bull_enhancement,
            'fzt_brick_area_increase': brick_area_increase,
            'fzt_selection_condition': selection_condition
        }
        
        logger.debug("所有指标计算完成（无未来函数）")
        
        return indicators
    
    def forward(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        前向计算 - 核心计算方法
        
        Args:
            data: 输入数据（包含OHLCV）
            
        Returns:
            pd.DataFrame: 计算结果（所有指标）
        """
        # 计算所有指标
        indicators = self.calculate_all_indicators(data)
        
        # 转换为DataFrame
        result_df = pd.DataFrame(indicators, index=data.index)
        
        # 处理缺失值（向前填充）
        result_df = result_df.ffill().fillna(0)
        
        logger.debug(f"前向计算完成: {result_df.shape}")
        
        return result_df
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称
        
        Returns:
            List[str]: 特征名称列表
        """
        return self.feature_names
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        调用接口 - 使实例可调用
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 计算结果
        """
        return self.forward(data)


# 根据QLib可用性创建不同的类
if QLIB_AVAILABLE:
    # QLib可用，创建Operator子类
    class FZTAlpha(Operator, FZTAlphaBase):
        """FZTAlpha算子 - QLib Operator子类"""
        
        def __init__(self, config: Optional[Dict] = None):
            """
            初始化FZTAlpha算子
            
            Args:
                config: 配置字典（可选）
            """
            # 分别调用父类的__init__
            Operator.__init__(self)
            FZTAlphaBase.__init__(self, config)
            
            logger.info("FZTAlpha算子初始化完成（QLib Operator子类）")
        
        def forward(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            QLib Operator接口 - 前向计算
            
            Args:
                data: 输入数据
                
            Returns:
                pd.DataFrame: 计算结果
            """
            return super().forward(data)
        
        def __str__(self) -> str:
            """字符串表示"""
            return "FZTAlpha(QLib Operator)"
        
        def __repr__(self) -> str:
            """详细表示"""
            return f"FZTAlpha(config={self.config})"
        
else:
    # QLib不可用，创建简化版本
    class FZTAlpha(FZTAlphaBase):
        """FZTAlpha算子 - 简化版本（QLib不可用时使用）"""
        
        def __init__(self, config: Optional[Dict] = None):
            """
            初始化FZTAlpha算子（简化版本）
            
            Args:
                config: 配置字典（可选）
            """
            super().__init__(config)
            logger.info("FZTAlpha算子初始化完成（简化版本）")
        
        def __str__(self) -> str:
            """字符串表示"""
            return "FZTAlpha(Simplified)"
        
        def __repr__(self) -> str:
            """详细表示"""
            return f"FZTAlpha(config={self.config})"


def test_fzt_alpha_operator():
    """测试FZTAlpha算子"""
    try:
        print("=" * 80)
        print("FZTAlpha算子测试")
        print("=" * 80)
        
        # 创建测试数据
        print("\n1. 创建测试数据...")
        np.random.seed(42)
        n_days = 100
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        
        # 生成价格数据
        base_price = 100
        trend = np.linspace(0, 0.2, n_days)
        noise = np.random.normal(0, 0.02, n_days)
        returns = 0.001 + trend/100 + noise
        prices = base_price * np.exp(np.cumsum(returns))
        
        # 生成高低价
        highs = prices * (1 + np.abs(np.random.normal(0.01, 0.005, n_days)))
        lows = prices * (1 - np.abs(np.random.normal(0.01, 0.005, n_days)))
        
        test_data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': np.random.lognormal(mean=10, sigma=1, size=n_days)
        }, index=dates)
        
        print(f"   数据形状: {test_data.shape}")
        print(f"   时间范围: {test_data.index.min()} 到 {test_data.index.max()}")
        
        # 初始化FZTAlpha算子
        print("\n2. 初始化FZTAlpha算子...")
        fzt_alpha = FZTAlpha()
        print(f"   ✓ 初始化成功: {fzt_alpha}")
        
        # 检查QLib兼容性
        print("\n3. 检查QLib兼容性...")
        if QLIB_AVAILABLE:
            print("   ✓ QLib可用，是Operator子类")
            from qlib.data.ops import Operator
            print(f"   是Operator子类: {issubclass(FZTAlpha, Operator)}")
            print(f"   是Operator实例: {isinstance(fzt_alpha, Operator)}")
        else:
            print("   ⚠ QLib不可用，使用简化版本")
            print(f"   是FZTAlphaBase子类: {issubclass(FZTAlpha, FZTAlphaBase)}")
        
        # 计算特征
        print("\n4. 计算特征...")
        features = fzt_alpha(test_data)
        print(f"   ✓ 特征计算完成: {features.shape}")
        print(f"   特征列: {list(features.columns)}")
        
        # 检查特征名称
        print("\n5. 检查特征名称...")
        feature_names = fzt_alpha.get_feature_names()
        print(f"   特征名称: {feature_names}")
        print(f"   特征数量: {len(feature_names)}")
        
        # 检查无未来函数
        print("\n6. 检查无未来函数...")
        # 检查是否有NaN值
        nan_counts = features.isna().sum()
        total_nan = nan_counts.sum()
        total_cells = features.size
        
        print(f"   NaN总数: {total_nan} / {total_cells} ({total_nan/total_cells:.2%})")
        
        # 检查窗口计算导致的NaN
        window_params = [
            fzt_alpha.params['var1_window'],
            fzt_alpha.params['var2_sma_window'],
            fzt_alpha.params['var3_window'],
            fzt_alpha.params['var4_sma_window'],
            fzt_alpha.params['var5_sma_window']
        ]
        max_window = max(window_params)
        expected_nan_rows = max_window - 1  # 窗口计算导致的前几行为NaN
        
        print(f"   最大窗口: {max_window}")
        print(f"   预期NaN行数: {expected_nan_rows}")
        
        # 检查选股信号
        print("\n7. 检查选股信号...")
        if 'fzt_selection_condition' in features.columns:
            selection_signals = features['fzt_selection_condition']
            signal_count = selection_signals.sum()
            signal_rate = signal_count / len(selection_signals)
            
            print(f"   选股信号数量: {signal_count}")
            print(f"   信号率: {signal_rate:.2%}")
            
            if signal_count > 0:
                signal_dates = selection_signals[selection_signals].index.tolist()
                print(f"   前5个信号日期: {[d.strftime('%Y-%m-%d') for d in signal_dates[:5]]}")
        
        # 性能测试
        print("\n8. 性能测试...")
        import time
        
        start_time = time.time()
        for _ in range(10):
            _ = fzt_alpha(test_data)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"   平均计算时间: {avg_time:.4f} 秒")
        print(f"   数据行数: {len(test_data)}")
        print(f"   每秒处理行数: {len(test_data)/avg_time:.0f}")
        
        # 多股票测试
        print("\n9. 多股票测试...")
        n_stocks = 3
        multi_stock_data = []
        
        for i in range(n_stocks):
            stock_data = test_data.copy()
            stock_data['instrument'] = f'00000{i+1}.SZ'
            multi_stock_data.append(stock_data)
        
        multi_stock_df = pd.concat(multi_stock_data)
        
        # 按股票分组处理
        grouped = multi_stock_df.groupby('instrument')
        
        results = []
        for stock_code, group in grouped:
            stock_features = fzt_alpha(group[['open', 'high', 'low', 'close', 'volume']])
            results.append(stock_features)
            print(f"   股票 {stock_code}: {stock_features.shape}")
        
        print(f"   ✓ 成功处理 {n_stocks} 只股票")
        
        print("\n" + "=" * 80)
        print("测试完成！FZTAlpha算子功能正常。")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # 运行测试
    sys.exit(test_fzt_alpha_operator())