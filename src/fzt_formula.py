#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FZT选股公式计算模块

基于FZT公式规范，计算选股信号和相关特征。

功能：
1. 计算FZT公式值（综合选股评分）
2. 生成FZT相关特征
3. 提供公式参数配置
4. 支持批量股票计算

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


class FZTFormula:
    """FZT选股公式计算器"""
    
    # 默认配置
    DEFAULT_CONFIG = {
        'momentum': {
            'windows': [5, 10, 20, 60],
            'weights': [0.4, 0.3, 0.2, 0.1]
        },
        'volume': {
            'window': 5,
            'weight': 0.2
        },
        'strength': {
            'window': 20,
            'weight': 0.2
        },
        'volatility': {
            'window': 20,
            'weight': 0.1,
            'adjustment_factor': 0.01
        },
        'normalization': {
            'window': 60
        },
        'signals': {
            'strong_buy': 1.0,
            'weak_buy': 0.2,
            'weak_sell': -0.2,
            'strong_sell': -1.0
        }
    }
    
    def __init__(self, config: Optional[Dict] = None, config_path: Optional[str] = None):
        """
        初始化FZT公式计算器
        
        Args:
            config: 配置字典（可选）
            config_path: 配置文件路径（可选）
        """
        # 加载配置
        if config_path:
            self.config = self._load_config_from_file(config_path)
        elif config:
            self.config = self._merge_configs(self.DEFAULT_CONFIG, config)
        else:
            self.config = self.DEFAULT_CONFIG.copy()
        
        # 验证配置
        self._validate_config()
        
        # 提取配置参数
        self.momentum_config = self.config['momentum']
        self.volume_config = self.config['volume']
        self.strength_config = self.config['strength']
        self.volatility_config = self.config['volatility']
        self.normalization_config = self.config['normalization']
        self.signal_config = self.config['signals']
        
        logger.info("FZT公式计算器初始化完成")
        logger.info(f"动量窗口: {self.momentum_config['windows']}")
        logger.info(f"动量权重: {self.momentum_config['weights']}")
    
    def _load_config_from_file(self, config_path: str) -> Dict[str, Any]:
        """从文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 合并默认配置
            merged_config = self._merge_configs(self.DEFAULT_CONFIG, config)
            logger.info(f"配置文件加载成功: {config_path}")
            return merged_config
            
        except FileNotFoundError:
            logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            return self.DEFAULT_CONFIG.copy()
        except yaml.YAMLError as e:
            logger.warning(f"配置文件解析错误: {e}，使用默认配置")
            return self.DEFAULT_CONFIG.copy()
    
    def _merge_configs(self, default: Dict, custom: Dict) -> Dict:
        """合并默认配置和自定义配置"""
        merged = default.copy()
        
        def deep_merge(base: Dict, update: Dict) -> Dict:
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        return deep_merge(merged, custom)
    
    def _validate_config(self) -> None:
        """验证配置的完整性"""
        required_sections = ['momentum', 'volume', 'strength', 'volatility', 'normalization', 'signals']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置中缺少必要部分: {section}")
        
        # 验证动量配置
        momentum = self.config['momentum']
        if 'windows' not in momentum or 'weights' not in momentum:
            raise ValueError("动量配置中缺少windows或weights")
        
        if len(momentum['windows']) != len(momentum['weights']):
            raise ValueError("动量窗口数量和权重数量不匹配")
        
        if not np.isclose(sum(momentum['weights']), 1.0):
            logger.warning(f"动量权重总和不为1: {sum(momentum['weights'])}")
        
        # 验证因子权重总和
        factor_weights = [
            self.config['momentum'].get('weight', 0.5),
            self.config['volume'].get('weight', 0.2),
            self.config['strength'].get('weight', 0.2),
            self.config['volatility'].get('weight', 0.1)
        ]
        
        if not np.isclose(sum(factor_weights), 1.0):
            logger.warning(f"因子权重总和不为1: {sum(factor_weights)}")
        
        logger.info("配置验证通过")
    
    def calculate_momentum(self, close: pd.Series) -> pd.Series:
        """
        计算价格动量因子
        
        Args:
            close: 收盘价序列
            
        Returns:
            pd.Series: 加权动量值
        """
        windows = self.momentum_config['windows']
        weights = self.momentum_config['weights']
        
        momentum_factors = []
        
        for window, weight in zip(windows, weights):
            if len(close) >= window:
                # 计算收益率
                returns = close / close.shift(window) - 1
                momentum_factors.append(weight * returns)
        
        if momentum_factors:
            # 组合不同窗口的动量
            combined_momentum = pd.concat(momentum_factors, axis=1).sum(axis=1)
        else:
            # 数据不足，返回0
            combined_momentum = pd.Series(0, index=close.index)
        
        # 处理缺失值
        combined_momentum = combined_momentum.fillna(0)
        
        return combined_momentum
    
    def calculate_volume_confirmation(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        计算成交量确认因子
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
            
        Returns:
            pd.Series: 成交量确认值
        """
        window = self.volume_config['window']
        
        if len(volume) < window:
            return pd.Series(0, index=volume.index)
        
        # 计算价格动量（短期）
        price_momentum = close.pct_change(window)
        
        # 计算成交量变化率
        volume_change = volume / volume.shift(window) - 1
        
        # 成交量确认：价格动量方向与成交量变化一致
        volume_confirmation = np.sign(price_momentum) * volume_change
        
        # 处理缺失值
        volume_confirmation = volume_confirmation.fillna(0)
        
        return volume_confirmation
    
    def calculate_price_strength(self, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """
        计算价格强度因子
        
        Args:
            close: 收盘价序列
            high: 最高价序列
            low: 最低价序列
            
        Returns:
            pd.Series: 价格强度值
        """
        window = self.strength_config['window']
        
        if len(close) < window:
            return pd.Series(0, index=close.index)
        
        # 计算相对价格位置
        min_price = close.rolling(window).min()
        max_price = close.rolling(window).max()
        
        # 避免除零
        price_range = max_price - min_price
        price_range = price_range.replace(0, 1e-6)
        
        # 价格强度：当前价格在近期范围中的位置
        strength = (close - min_price) / price_range
        
        # 添加突破信号
        breakout = (close > close.rolling(window).max()).astype(int)
        strength = strength + 0.1 * breakout  # 突破给予额外加分
        
        # 处理缺失值
        strength = strength.fillna(0.5)  # 中间位置
        
        return strength
    
    def calculate_volatility_adjustment(self, close: pd.Series) -> pd.Series:
        """
        计算波动率调整因子
        
        Args:
            close: 收盘价序列
            
        Returns:
            pd.Series: 波动率调整值
        """
        window = self.volatility_config['window']
        adjustment_factor = self.volatility_config.get('adjustment_factor', 0.01)
        
        if len(close) < window:
            return pd.Series(1.0, index=close.index)  # 默认无调整
        
        # 计算历史波动率
        returns = close.pct_change()
        volatility = returns.rolling(window).std()
        
        # 波动率调整：低波动给予更高权重
        # 使用双曲正切函数平滑调整
        adjustment = 1 / (volatility + adjustment_factor)
        
        # 标准化到合理范围
        adjustment = adjustment / adjustment.rolling(window).mean()
        adjustment = adjustment.clip(0.5, 2.0)  # 限制调整范围
        
        # 处理缺失值
        adjustment = adjustment.fillna(1.0)
        
        return adjustment
    
    def normalize_series(self, series: pd.Series) -> pd.Series:
        """
        标准化序列（Z-score标准化）
        
        Args:
            series: 需要标准化的序列
            
        Returns:
            pd.Series: 标准化后的序列
        """
        window = self.normalization_config['window']
        
        if len(series) < window:
            return pd.Series(0, index=series.index)
        
        # 滚动标准化
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        
        # 避免除零
        rolling_std = rolling_std.replace(0, 1e-6)
        
        normalized = (series - rolling_mean) / rolling_std
        
        # 处理缺失值
        normalized = normalized.fillna(0)
        
        return normalized
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算FZT公式值
        
        Args:
            data: 包含OHLCV数据的DataFrame，必须有以下列：
                  'close', 'volume', 'high', 'low'
        
        Returns:
            pd.Series: FZT公式值序列
        """
        logger.info("开始计算FZT公式值...")
        
        # 验证输入数据
        required_columns = ['close', 'volume', 'high', 'low']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"输入数据缺少必要列: {col}")
        
        # 提取数据
        close = data['close']
        volume = data['volume']
        high = data['high']
        low = data['low']
        
        # 1. 计算各子因子
        momentum = self.calculate_momentum(close)
        volume_confirmation = self.calculate_volume_confirmation(close, volume)
        price_strength = self.calculate_price_strength(close, high, low)
        volatility_adjustment = self.calculate_volatility_adjustment(close)
        
        # 2. 组合因子（使用配置权重）
        momentum_weight = self.momentum_config.get('weight', 0.5)
        volume_weight = self.volume_config.get('weight', 0.2)
        strength_weight = self.strength_config.get('weight', 0.2)
        volatility_weight = self.volatility_config.get('weight', 0.1)
        
        raw_fzt = (
            momentum_weight * momentum +
            volume_weight * volume_confirmation +
            strength_weight * price_strength +
            volatility_weight * volatility_adjustment
        )
        
        # 3. 标准化
        fzt_normalized = self.normalize_series(raw_fzt)
        
        logger.info(f"FZT公式计算完成，数据长度: {len(fzt_normalized)}")
        logger.info(f"FZT值范围: [{fzt_normalized.min():.2f}, {fzt_normalized.max():.2f}]")
        logger.info(f"FZT均值: {fzt_normalized.mean():.2f}, 标准差: {fzt_normalized.std():.2f}")
        
        return fzt_normalized
    
    def generate_signal(self, fzt_value: float) -> Tuple[str, float]:
        """
        根据FZT值生成交易信号
        
        Args:
            fzt_value: FZT公式值
            
        Returns:
            Tuple[str, float]: (信号类型, 信号强度)
        """
        thresholds = self.signal_config
        
        if fzt_value >= thresholds['strong_buy']:
            signal = 'STRONG_BUY'
            strength = (fzt_value - thresholds['strong_buy']) / 2.0 + 1.0
        elif fzt_value >= thresholds['weak_buy']:
            signal = 'WEAK_BUY'
            strength = (fzt_value - thresholds['weak_buy']) / (thresholds['strong_buy'] - thresholds['weak_buy'])
        elif fzt_value <= thresholds['strong_sell']:
            signal = 'STRONG_SELL'
            strength = (thresholds['strong_sell'] - fzt_value) / 2.0 + 1.0
        elif fzt_value <= thresholds['weak_sell']:
            signal = 'WEAK_SELL'
            strength = (thresholds['weak_sell'] - fzt_value) / (thresholds['weak_sell'] - thresholds['strong_sell'])
        else:
            signal = 'NEUTRAL'
            strength = 0.0
        
        # 限制强度在合理范围
        strength = max(0.0, min(strength, 2.0))
        
        return signal, strength
    
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算FZT相关特征
        
        Args:
            data: 原始数据
            
        Returns:
            pd.DataFrame: 包含FZT特征和基础特征的DataFrame
        """
        logger.info("开始计算FZT相关特征...")
        
        # 创建特征DataFrame
        features = pd.DataFrame(index=data.index)
        
        # 1. 基础价格特征
        if 'close' in data.columns:
            features['close'] = data['close']
            features['returns_1d'] = data['close'].pct_change()
            features['returns_5d'] = data['close'].pct_change(5)
            features['volatility_20d'] = data['close'].pct_change().rolling(20).std()
        
        # 2. 成交量特征
        if 'volume' in data.columns:
            features['volume'] = data['volume']
            features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            features['volume_change'] = data['volume'].pct_change()
        
        # 3. 计算FZT公式值
        fzt_values = self.calculate(data)
        features['fzt_value'] = fzt_values
        
        # 4. FZT衍生特征
        # 信号类型和强度
        signals = []
        strengths = []
        
        for value in fzt_values:
            signal, strength = self.generate_signal(value)
            signals.append(signal)
            strengths.append(strength)
        
        features['fzt_signal'] = signals
        features['fzt_strength'] = strengths
        
        # 5. 各子因子值（用于分析）
        close = data['close'] if 'close' in data.columns else None
        volume = data['volume'] if 'volume' in data.columns else None
        high = data['high'] if 'high' in data.columns else None
        low = data['low'] if 'low' in data.columns else None
        
        if all([close is not None, volume is not None, high is not None, low is not None]):
            features['momentum_factor'] = self.calculate_momentum(close)
            features['volume_factor'] = self.calculate_volume_confirmation(close, volume)
            features['strength_factor'] = self.calculate_price_strength(close, high, low)
            features['volatility_factor'] = self.calculate_volatility_adjustment(close)
        
        # 6. 技术指标特征
        if 'close' in data.columns:
            # 移动平均
            features['sma_5'] = data['close'].rolling(5).mean()
            features['sma_20'] = data['close'].rolling(20).mean()
            
            # 价格通道
            features['upper_band'] = data['close'].rolling(20).max()
            features['lower_band'] = data['close'].rolling(20).min()
            
            # RSI简化版
            returns = data['close'].pct_change()
            gain = returns.where(returns > 0, 0).rolling(14).mean()
            loss = -returns.where(returns < 0, 0).rolling(14).mean()
            rs = gain / (loss + 1e-6)
            features['rsi'] = 100 - (100 / (1 + rs))
        
        # 7. 时间特征
        if isinstance(data.index, pd.DatetimeIndex):
            features['day_of_week'] = data.index.dayofweek
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
        
        # 8. 处理缺失值
        features = features.fillna(method='ffill').fillna(0)
        
        logger.info(f"特征计算完成: {len(features)} 行, {len(features.columns)} 列")
        logger.info(f"特征列: {list(features.columns)}")
        
        return features
    
    def batch_calculate(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        批量计算多只股票的FZT特征
        
        Args:
            stock_data: 股票数据字典 {stock_code: data}
            
        Returns:
            Dict[str, pd.DataFrame]: 各股票的FZT特征
        """
        logger.info(f"开始批量计算 {len(stock_data)} 只股票的FZT特征...")
        
        results = {}
        
        for stock_code, data in stock_data.items():
            try:
                features = self.calculate_features(data)
                features['stock_code'] = stock_code
                results[stock_code] = features
                
                logger.debug(f"股票 {stock_code} 特征计算完成: {features.shape}")
                
            except Exception as e:
                logger.warning(f"股票 {stock_code} 特征计算失败: {e}")
                continue
        
        logger.info(f"批量计算完成: {len(results)}/{len(stock_data)} 只股票成功")
        
        return results
    
    def analyze_fzt_distribution(self, fzt_values: pd.Series) -> Dict[str, Any]:
        """
        分析FZT值分布
        
        Args:
            fzt_values: FZT值序列
            
        Returns:
            Dict[str, Any]: 分布统计信息
        """
        if len(fzt_values) == 0:
            return {}
        
        # 基本统计
        stats = {
            'count': len(fzt_values),
            'mean': float(fzt_values.mean()),
            'std': float(fzt_values.std()),
            'min': float(fzt_values.min()),
            'max': float(fzt_values.max()),
            'median': float(fzt_values.median()),
            'skewness': float(fzt_values.skew()),
            'kurtosis': float(fzt_values.kurtosis())
        }
        
        # 分位数
        quantiles = fzt_values.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        stats['quantiles'] = {f'q{int(q*100)}': float(v) for q, v in quantiles.items()}
        
        # 信号分布
        signals = []
        for value in fzt_values:
            signal, _ = self.generate_signal(value)
            signals.append(signal)
        
        signal_counts = pd.Series(signals).value_counts()
        stats['signal_distribution'] = signal_counts.to_dict()
        
        # 正态性检验（Jarque-Bera简化版）
        n = len(fzt_values)
        if n > 0:
            s = stats['skewness']
            k = stats['kurtosis']
            jb = n/6 * (s**2 + 0.25*(k-3)**2)
            stats['jarque_bera'] = float(jb)
        
        return stats
    
    def save_config(self, filepath: str) -> None:
        """
        保存当前配置到文件
        
        Args:
            filepath: 配置文件路径
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"配置已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise


def calculate_target(data: pd.DataFrame, horizon: int = 2) -> pd.Series:
    """
    计算目标变量：未来N天收益率
    
    Args:
        data: 包含价格数据的DataFrame，必须有'close'列
        horizon: 预测 horizon（天数）
        
    Returns:
        未来收益率序列
    """
    if 'close' not in data.columns:
        raise ValueError("输入数据必须包含'close'列")
    
    # 计算未来收益率
    future_return = data['close'].shift(-horizon) / data['close'] - 1
    
    # 将最后horizon天的值设为NaN（因为没有未来数据）
    if len(future_return) > horizon:
        future_return.iloc[-horizon:] = np.nan
    
    return future_return


def create_sample_data(n_days: int = 100, start_date: str = '2023-01-01') -> pd.DataFrame:
    """
    创建样本数据用于测试
    
    Args:
        n_days: 数据天数
        start_date: 开始日期
        
    Returns:
        pd.DataFrame: 样本数据
    """
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # 生成随机但有一定趋势的价格数据
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # 生成成交量数据（与价格有一定相关性）
    volumes = np.random.lognormal(mean=10, sigma=1, size=n_days) * (1 + 0.3 * np.sign(returns))
    
    # 生成高低价
    highs = prices * (1 + np.abs(np.random.normal(0.01, 0.01, n_days)))
    lows = prices * (1 - np.abs(np.random.normal(0.01, 0.01, n_days)))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes,
        'amount': volumes * prices  # 成交额 = 成交量 * 价格
    }, index=dates)
    
    return data


def main():
    """主函数，用于测试和演示"""
    try:
        print("=" * 80)
        print("FZT公式计算模块测试")
        print("=" * 80)
        
        # 1. 创建样本数据
        print("\n1. 创建样本数据...")
        sample_data = create_sample_data(200)
        print(f"   数据形状: {sample_data.shape}")
        print(f"   时间范围: {sample_data.index.min()} 到 {sample_data.index.max()}")
        print(f"   列名: {list(sample_data.columns)}")
        
        # 2. 初始化FZT公式计算器
        print("\n2. 初始化FZT公式计算器...")
        fzt_calculator = FZTFormula()
        print("   ✓ 初始化成功")
        
        # 3. 计算FZT值
        print("\n3. 计算FZT公式值...")
        fzt_values = fzt_calculator.calculate(sample_data)
        print(f"   ✓ 计算完成，长度: {len(fzt_values)}")
        print(f"   值范围: [{fzt_values.min():.2f}, {fzt_values.max():.2f}]")
        
        # 4. 计算完整特征
        print("\n4. 计算完整特征...")
        features = fzt_calculator.calculate_features(sample_data)
        print(f"   ✓ 特征计算完成，形状: {features.shape}")
        print(f"   特征列数: {len(features.columns)}")
        
        # 5. 分析FZT分布
        print("\n5. 分析FZT分布...")
        stats = fzt_calculator.analyze_fzt_distribution(fzt_values)
        print(f"   均值: {stats.get('mean', 0):.2f}")
        print(f"   标准差: {stats.get('std', 0):.2f}")
        print(f"   信号分布: {stats.get('signal_distribution', {})}")
        
        # 6. 生成信号示例
        print("\n6. 信号生成示例:")
        test_values = [-1.5, -0.5, 0, 0.5, 1.5]
        for value in test_values:
            signal, strength = fzt_calculator.generate_signal(value)
            print(f"   FZT={value:.1f} -> {signal} (强度: {strength:.2f})")
        
        # 7. 计算目标变量
        print("\n7. 计算目标变量...")
        target = calculate_target(sample_data, horizon=2)
        print(f"   ✓ 目标变量计算完成")
        print(f"   非空值数量: {target.notna().sum()}")
        
        # 8. 保存配置
        print("\n8. 保存配置...")
        config_path = "fzt_config.yaml"
        fzt_calculator.save_config(config_path)
        print(f"   ✓ 配置已保存到: {config_path}")
        
        print("\n" + "=" * 80)
        print("测试完成！所有功能正常。")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())