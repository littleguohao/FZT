#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FZT砖型图选股公式计算模块

基于用户提供的FZT选股公式实现：
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

作者: FZT项目组
创建日期: 2026-03-01
"""

import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

import pandas as pd
import numpy as np
import yaml

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')


class FZTBrickFormula:
    """FZT砖型图选股公式计算器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化FZT砖型图公式计算器
        
        Args:
            config: 配置字典（可选）
        """
        self.config = config or {}
        
        # 默认参数
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
        
        logger.info("FZT砖型图公式计算器初始化完成")
        logger.info(f"参数配置: {self.params}")
    
    def hhv(self, series: pd.Series, window: int) -> pd.Series:
        """
        计算最高值（HHV）
        
        Args:
            series: 价格序列
            window: 窗口大小
            
        Returns:
            pd.Series: 窗口内最高值
        """
        return series.rolling(window).max()
    
    def llv(self, series: pd.Series, window: int) -> pd.Series:
        """
        计算最低值（LLV）
        
        Args:
            series: 价格序列
            window: 窗口大小
            
        Returns:
            pd.Series: 窗口内最低值
        """
        return series.rolling(window).min()
    
    def sma(self, series: pd.Series, window: int, weight: float = 1.0) -> pd.Series:
        """
        简单移动平均（SMA）
        
        Args:
            series: 输入序列
            window: 窗口大小
            weight: 权重参数（公式中的第三个参数）
            
        Returns:
            pd.Series: 平滑后的序列
        """
        # 在通达信公式中，SMA(X,N,M) = (M*X + (N-M)*REF(SMA,1))/N
        # 这里简化为简单移动平均
        return series.rolling(window).mean()
    
    def calculate_var1a(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        计算VAR1A指标
        
        VAR1A:=(HHV(HIGH,4)-CLOSE)/(HHV(HIGH,4)-LLV(LOW,4))*100-90;
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            
        Returns:
            pd.Series: VAR1A值
        """
        window = self.params['var1_window']
        
        # 计算HHV和LLV
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
        计算VAR2A指标
        
        VAR2A:=SMA(VAR1A,4,1)+100;
        
        Args:
            var1a: VAR1A序列
            
        Returns:
            pd.Series: VAR2A值
        """
        window = self.params['var2_sma_window']
        
        # 计算SMA
        var2a = self.sma(var1a, window) + 100
        
        return var2a
    
    def calculate_var3a(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        计算VAR3A指标
        
        VAR3A:=(CLOSE-LLV(LOW,4))/(HHV(HIGH,4)-LLV(LOW,4))*100;
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            
        Returns:
            pd.Series: VAR3A值
        """
        window = self.params['var3_window']
        
        # 计算HHV和LLV
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
        计算VAR4A指标
        
        VAR4A:=SMA(VAR3A,6,1);
        
        Args:
            var3a: VAR3A序列
            
        Returns:
            pd.Series: VAR4A值
        """
        window = self.params['var4_sma_window']
        
        # 计算SMA
        var4a = self.sma(var3a, window)
        
        return var4a
    
    def calculate_var5a(self, var4a: pd.Series) -> pd.Series:
        """
        计算VAR5A指标
        
        VAR5A:=SMA(VAR4A,6,1)+100;
        
        Args:
            var4a: VAR4A序列
            
        Returns:
            pd.Series: VAR5A值
        """
        window = self.params['var5_sma_window']
        
        # 计算SMA
        var5a = self.sma(var4a, window) + 100
        
        return var5a
    
    def calculate_brick_chart(self, var5a: pd.Series, var2a: pd.Series) -> pd.Series:
        """
        计算砖型图
        
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
        计算砖型图面积
        
        砖型图面积:=ABS(砖型图 - REF(砖型图,1));
        
        Args:
            brick_chart: 砖型图序列
            
        Returns:
            pd.Series: 砖型图面积
        """
        # 计算面积（绝对值变化）
        brick_area = abs(brick_chart - brick_chart.shift(1))
        
        return brick_area
    
    def calculate_aa(self, brick_chart: pd.Series) -> pd.Series:
        """
        计算AA指标
        
        AA:=(REF(砖型图,1)<砖型图);
        
        Args:
            brick_chart: 砖型图序列
            
        Returns:
            pd.Series: AA值（布尔序列）
        """
        # 砖型图上升
        aa = brick_chart.shift(1) < brick_chart
        
        return aa
    
    def calculate_first_bull_enhancement(self, aa: pd.Series) -> pd.Series:
        """
        计算首次多头增强
        
        首次多头增强:=(REF(AA,1)=0) AND (AA=1);
        
        Args:
            aa: AA序列
            
        Returns:
            pd.Series: 首次多头增强信号（布尔序列）
        """
        # 首次出现上升
        first_bull_enhancement = (aa.shift(1) == False) & (aa == True)
        
        return first_bull_enhancement
    
    def calculate_brick_area_increase(self, brick_area: pd.Series) -> pd.Series:
        """
        计算砖型图面积增幅
        
        砖型图面积增幅:=砖型图面积 > REF(砖型图面积,1) * 2/3;
        
        Args:
            brick_area: 砖型图面积序列
            
        Returns:
            pd.Series: 面积增幅信号（布尔序列）
        """
        ratio = self.params['area_increase_ratio']
        
        # 面积增加超过阈值
        area_increase = brick_area > brick_area.shift(1) * ratio
        
        return area_increase
    
    def calculate_selection_condition(self, 
                                     first_bull_enhancement: pd.Series,
                                     brick_area_increase: pd.Series) -> pd.Series:
        """
        计算选股条件
        
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
        计算所有指标
        
        Args:
            data: 包含OHLC数据的DataFrame
            
        Returns:
            Dict[str, pd.Series]: 所有指标字典
        """
        logger.info("开始计算FZT砖型图所有指标...")
        
        # 验证输入数据
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"输入数据缺少必要列: {col}")
        
        # 提取数据
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 计算所有指标
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
            'var1a': var1a,
            'var2a': var2a,
            'var3a': var3a,
            'var4a': var4a,
            'var5a': var5a,
            'var6a': var5a - var2a,
            'brick_chart': brick_chart,
            'brick_area': brick_area,
            'aa': aa,
            'first_bull_enhancement': first_bull_enhancement,
            'brick_area_increase': brick_area_increase,
            'selection_condition': selection_condition
        }
        
        logger.info("所有指标计算完成")
        
        return indicators
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成特征DataFrame
        
        Args:
            data: 原始数据
            
        Returns:
            pd.DataFrame: 包含所有指标的特征DataFrame
        """
        # 计算所有指标
        indicators = self.calculate_all_indicators(data)
        
        # 创建特征DataFrame
        features = pd.DataFrame(index=data.index)
        
        # 添加原始数据
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                features[col] = data[col]
        
        # 添加计算指标
        for name, series in indicators.items():
            features[name] = series
        
        # 添加衍生特征
        if 'close' in features.columns:
            features['returns_1d'] = features['close'].pct_change()
            features['returns_5d'] = features['close'].pct_change(5)
        
        # 处理缺失值
        features = features.fillna(method='ffill').fillna(0)
        
        logger.info(f"特征生成完成: {len(features)} 行, {len(features.columns)} 列")
        
        return features
    
    def analyze_selection_signals(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        分析选股信号
        
        Args:
            features: 特征DataFrame
            
        Returns:
            Dict[str, Any]: 信号分析结果
        """
        if 'selection_condition' not in features.columns:
            raise ValueError("特征中缺少selection_condition列")
        
        selection_signals = features['selection_condition']
        
        # 统计信号
        total_signals = selection_signals.sum()
        signal_rate = total_signals / len(selection_signals) if len(selection_signals) > 0 else 0
        
        # 信号分布
        signal_dates = selection_signals[selection_signals].index.tolist()
        
        # 信号间隔分析
        signal_intervals = []
        if len(signal_dates) > 1:
            for i in range(1, len(signal_dates)):
                interval = (signal_dates[i] - signal_dates[i-1]).days
                signal_intervals.append(interval)
        
        # 信号强度分析
        signal_strength = {}
        if total_signals > 0:
            signal_strength = {
                'avg_brick_chart': features.loc[selection_signals, 'brick_chart'].mean(),
                'avg_brick_area': features.loc[selection_signals, 'brick_area'].mean(),
                'avg_var6a': features.loc[selection_signals, 'var6a'].mean()
            }
        
        analysis = {
            'total_signals': int(total_signals),
            'signal_rate': float(signal_rate),
            'signal_dates': [d.strftime('%Y-%m-%d') for d in signal_dates],
            'signal_intervals': signal_intervals,
            'avg_interval': np.mean(signal_intervals) if signal_intervals else 0,
            'signal_strength': signal_strength
        }
        
        return analysis
    
    def backtest_signals(self, features: pd.DataFrame, horizon: int = 2) -> pd.DataFrame:
        """
        回测选股信号
        
        Args:
            features: 特征DataFrame
            horizon: 预测 horizon（天数）
            
        Returns:
            pd.DataFrame: 回测结果
        """
        if 'selection_condition' not in features.columns or 'close' not in features.columns:
            raise ValueError("特征中缺少必要列")
        
        # 创建回测结果DataFrame
        backtest_results = pd.DataFrame(index=features.index)
        
        # 复制信号
        backtest_results['signal'] = features['selection_condition']
        
        # 计算未来收益率
        backtest_results['future_return'] = features['close'].shift(-horizon) / features['close'] - 1
        
        # 标记最后horizon天为NaN
        if len(backtest_results) > horizon:
            backtest_results['future_return'].iloc[-horizon:] = np.nan
        
        # 计算信号收益率
        signal_returns = backtest_results.loc[backtest_results['signal'], 'future_return']
        
        # 计算基准收益率（所有日期的平均）
        benchmark_returns = backtest_results['future_return'].dropna()
        
        # 统计结果
        if len(signal_returns) > 0:
            signal_stats = {
                'signal_count': len(signal_returns),
                'avg_return': signal_returns.mean(),
                'std_return': signal_returns.std(),
                'win_rate': (signal_returns > 0).mean(),
                'max_return': signal_returns.max(),
                'min_return': signal_returns.min()
            }
        else:
            signal_stats = {}
        
        if len(benchmark_returns) > 0:
            benchmark_stats = {
                'avg_return': benchmark_returns.mean(),
                'std_return': benchmark_returns.std(),
                'win_rate': (benchmark_returns > 0).mean()
            }
        else:
            benchmark_stats = {}
        
        # 计算超额收益
        if signal_stats and benchmark_stats:
            excess_return = signal_stats['avg_return'] - benchmark_stats['avg_return']
            signal_stats['excess_return'] = excess_return
        
        backtest_results['signal_stats'] = pd.Series([signal_stats] * len(backtest_results))
        backtest_results['benchmark_stats'] = pd.Series([benchmark_stats] * len(backtest_results))
        
        logger.info(f"回测完成: {len(signal_returns)} 个信号")
        
        return backtest_results
    
    def batch_process(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        批量处理多只股票
        
        Args:
            stock_data: 股票数据字典 {stock_code: data}
            
        Returns:
            Dict[str, Dict]: 各股票的处理结果
        """
        logger.info(f"开始批量处理 {len(stock_data)} 只股票...")
        
        results = {}
        
        for stock_code, data in stock_data.items():
            try:
                # 计算特征
                features = self.generate_features(data)
                
                # 分析信号
                signal_analysis = self.analyze_selection_signals(features)
                
                # 回测信号
                backtest_results = self.backtest_signals(features, horizon=2)
                
                results[stock_code] = {
                    'features': features,
                    'signal_analysis': signal_analysis,
                    'backtest_results': backtest_results,
                    'success': True
                }
                
                logger.debug(f"股票 {stock_code} 处理完成: {signal_analysis['total_signals']} 个信号")
                
            except Exception as e:
                logger.warning(f"股票 {stock_code} 处理失败: {e}")
                results[stock_code] = {
                    'success': False,
                    'error': str(e)
                }
                continue
        
        logger.info(f"批量处理完成: {sum(1 for r in results.values() if r['success'])}/{len(stock_data)} 只股票成功")
        
        return results


def test_fzt_brick_formula():
    """测试FZT砖型图公式"""
    try:
        print("=" * 80)
        print("FZT砖型图公式测试")
        print("=" * 80)
        
        # 创建测试数据
        print("\n1. 创建测试数据...")
        np.random.seed(42)
        n_days = 100
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        
        # 生成价格数据（有一定趋势）
        base_price = 100
        trend = np.linspace(0, 0.2, n_days)  # 20%的上升趋势
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
        
        # 初始化公式计算器
        print("\n2. 初始化FZT砖型图公式计算器...")
        fzt_brick = FZTBrickFormula()
        print("   ✓ 初始化成功")
        
        # 计算所有指标
        print("\n3. 计算所有指标...")
        indicators = fzt_brick.calculate_all_indicators(test_data)
        print(f"   ✓ 指标计算完成: {len(indicators)} 个指标")
        print(f"   指标名称: {list(indicators.keys())}")
        
        # 生成特征
        print("\n4. 生成特征...")
        features = fzt_brick.generate_features(test_data)
        print(f"   ✓ 特征生成完成: {features.shape}")
        print(f"   特征列: {list(features.columns)}")
        
        # 分析选股信号
        print("\n5. 分析选股信号...")
        signal_analysis = fzt_brick.analyze_selection_signals(features)
        print(f"   总信号数: {signal_analysis['total_signals']}")
        print(f"   信号率: {signal_analysis['signal_rate']:.2%}")
        if signal_analysis['total_signals'] > 0:
            print(f"   平均间隔: {signal_analysis['avg_interval']:.1f} 天")
            print(f"   信号强度 - 平均砖型图: {signal_analysis['signal_strength']['avg_brick_chart']:.2f}")
        
        # 回测信号
        print("\n6. 回测选股信号...")
        backtest_results = fzt_brick.backtest_signals(features, horizon=2)
        print(f"   ✓ 回测完成")
        
        # 显示信号示例
        print("\n7. 信号示例:")
        if signal_analysis['total_signals'] > 0:
            signal_dates = signal_analysis['signal_dates'][:5]  # 显示前5个信号
            for i, date in enumerate(signal_dates, 1):
                print(f"   信号 {i}: {date}")
        else:
            print("   无选股信号")
        
        # 显示指标示例
        print("\n8. 指标值示例:")
        sample_idx = -10  # 最后10天的数据
        sample_data = features.iloc[sample_idx:]
        print(f"   最后10天指标值:")
        for col in ['var1a', 'var2a', 'var3a', 'var4a', 'var5a', 'var6a', 'brick_chart', 'selection_condition']:
            if col in sample_data.columns:
                value = sample_data[col].iloc[-1]
                if isinstance(value, (bool, np.bool_)):
                    print(f"   {col}: {value}")
                else:
                    print(f"   {col}: {value:.2f}")
        
        print("\n" + "=" * 80)
        print("测试完成！FZT砖型图公式功能正常。")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_fzt_brick_formula())
