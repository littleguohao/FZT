#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程模块 - 集成FZT砖型图公式和技术指标

功能：
1. 集成FZT砖型图公式特征
2. 计算技术指标特征
3. 创建完整的数据集
4. 特征选择和重要性分析

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
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """特征工程师"""
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """
        初始化特征工程师
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # 导入FZT公式模块
        from .fzt_brick_formula import FZTBrickFormula
        from .fzt_formula import FZTFormula
        
        # 创建FZT公式计算器
        self.fzt_brick_calculator = FZTBrickFormula()
        self.fzt_general_calculator = FZTFormula()
        
        # 输出目录
        self.features_dir = Path("data/features")
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # 特征缓存
        self.feature_cache = {}
        
        logger.info("特征工程师初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"配置文件不存在: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"配置文件解析错误: {e}")
            raise
    
    def calculate_fzt_brick_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算FZT砖型图公式特征
        
        Args:
            data: 原始OHLCV数据
            
        Returns:
            pd.DataFrame: FZT砖型图特征
        """
        logger.info("计算FZT砖型图公式特征...")
        
        try:
            # 使用FZT砖型图公式计算器
            fzt_features = self.fzt_brick_calculator.generate_features(data)
            
            # 重命名特征，添加前缀
            fzt_features = fzt_features.add_prefix('fzt_')
            
            logger.info(f"FZT砖型图特征计算完成: {fzt_features.shape}")
            return fzt_features
            
        except Exception as e:
            logger.error(f"计算FZT砖型图特征失败: {e}")
            raise
    
    def calculate_fzt_general_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算通用FZT公式特征
        
        Args:
            data: 原始OHLCV数据
            
        Returns:
            pd.DataFrame: 通用FZT特征
        """
        logger.info("计算通用FZT公式特征...")
        
        try:
            # 使用通用FZT公式计算器
            fzt_features = self.fzt_general_calculator.calculate_features(data)
            
            # 重命名特征，添加前缀
            fzt_features = fzt_features.add_prefix('fzt_gen_')
            
            logger.info(f"通用FZT特征计算完成: {fzt_features.shape}")
            return fzt_features
            
        except Exception as e:
            logger.error(f"计算通用FZT特征失败: {e}")
            raise
    
    def calculate_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算价格相关特征
        
        Args:
            data: 原始数据
            
        Returns:
            pd.DataFrame: 价格特征
        """
        logger.info("计算价格特征...")
        
        features = pd.DataFrame(index=data.index)
        
        if 'close' not in data.columns:
            logger.warning("数据中缺少close列，跳过价格特征计算")
            return features
        
        close = data['close']
        
        # 收益率特征
        features['returns_1d'] = close.pct_change()
        features['returns_2d'] = close.pct_change(2)
        features['returns_5d'] = close.pct_change(5)
        features['returns_10d'] = close.pct_change(10)
        features['returns_20d'] = close.pct_change(20)
        features['returns_60d'] = close.pct_change(60)
        
        # 动量特征
        features['momentum_5d'] = close / close.shift(5) - 1
        features['momentum_10d'] = close / close.shift(10) - 1
        features['momentum_20d'] = close / close.shift(20) - 1
        features['momentum_60d'] = close / close.shift(60) - 1
        
        # 波动率特征
        returns = close.pct_change()
        features['volatility_5d'] = returns.rolling(5).std()
        features['volatility_10d'] = returns.rolling(10).std()
        features['volatility_20d'] = returns.rolling(20).std()
        features['volatility_60d'] = returns.rolling(60).std()
        
        # 价格位置特征
        if 'high' in data.columns and 'low' in data.columns:
            high = data['high']
            low = data['low']
            
            features['high_low_range'] = (high - low) / close
            features['close_position'] = (close - low) / (high - low + 1e-6)
            features['body_size'] = abs(close - data.get('open', close)) / close
        
        # 价格趋势特征
        features['sma_5'] = close.rolling(5).mean()
        features['sma_10'] = close.rolling(10).mean()
        features['sma_20'] = close.rolling(20).mean()
        features['sma_60'] = close.rolling(60).mean()
        
        features['ema_5'] = close.ewm(span=5).mean()
        features['ema_10'] = close.ewm(span=10).mean()
        features['ema_20'] = close.ewm(span=20).mean()
        
        # 价格通道
        features['upper_band_20'] = close.rolling(20).max()
        features['lower_band_20'] = close.rolling(20).min()
        features['price_channel_position'] = (close - features['lower_band_20']) / (
            features['upper_band_20'] - features['lower_band_20'] + 1e-6
        )
        
        # 价格突破特征
        features['breakout_high_20'] = (close > features['upper_band_20']).astype(int)
        features['breakout_low_20'] = (close < features['lower_band_20']).astype(int)
        
        logger.info(f"价格特征计算完成: {features.shape}")
        return features
    
    def calculate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量特征
        
        Args:
            data: 原始数据
            
        Returns:
            pd.DataFrame: 成交量特征
        """
        logger.info("计算成交量特征...")
        
        features = pd.DataFrame(index=data.index)
        
        if 'volume' not in data.columns:
            logger.warning("数据中缺少volume列，跳过成交量特征计算")
            return features
        
        volume = data['volume']
        
        # 成交量变化特征
        features['volume_change_1d'] = volume.pct_change()
        features['volume_change_5d'] = volume.pct_change(5)
        features['volume_change_10d'] = volume.pct_change(10)
        
        # 成交量均线
        features['volume_sma_5'] = volume.rolling(5).mean()
        features['volume_sma_10'] = volume.rolling(10).mean()
        features['volume_sma_20'] = volume.rolling(20).mean()
        features['volume_sma_60'] = volume.rolling(60).mean()
        
        # 成交量比率
        features['volume_ratio_5'] = volume / features['volume_sma_5']
        features['volume_ratio_20'] = volume / features['volume_sma_20']
        features['volume_ratio_60'] = volume / features['volume_sma_60']
        
        # 成交量与价格关系
        if 'close' in data.columns:
            close = data['close']
            price_change = close.pct_change()
            
            # 价量关系
            features['price_volume_corr_5'] = price_change.rolling(5).corr(volume.pct_change())
            features['price_volume_corr_10'] = price_change.rolling(10).corr(volume.pct_change())
            
            # 成交量确认
            features['volume_confirmation'] = np.sign(price_change) * volume.pct_change()
        
        # 成交量波动
        features['volume_volatility_5'] = volume.pct_change().rolling(5).std()
        features['volume_volatility_20'] = volume.pct_change().rolling(20).std()
        
        logger.info(f"成交量特征计算完成: {features.shape}")
        return features
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            data: 原始数据
            
        Returns:
            pd.DataFrame: 技术指标特征
        """
        logger.info("计算技术指标...")
        
        features = pd.DataFrame(index=data.index)
        
        if 'close' not in data.columns or 'high' not in data.columns or 'low' not in data.columns:
            logger.warning("数据中缺少必要列，跳过技术指标计算")
            return features
        
        close = data['close']
        high = data['high']
        low = data['low']
        
        # RSI (相对强弱指数)
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-6)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        features['rsi_14'] = calculate_rsi(close, 14)
        features['rsi_7'] = calculate_rsi(close, 7)
        features['rsi_21'] = calculate_rsi(close, 21)
        
        # MACD (移动平均收敛发散)
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # 布林带
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        features['bollinger_upper'] = sma_20 + 2 * std_20
        features['bollinger_lower'] = sma_20 - 2 * std_20
        features['bollinger_position'] = (close - features['bollinger_lower']) / (
            features['bollinger_upper'] - features['bollinger_lower'] + 1e-6
        )
        
        # ATR (平均真实范围)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean()
        features['atr_ratio'] = features['atr_14'] / close
        
        # 威廉指标
        highest_14 = high.rolling(14).max()
        lowest_14 = low.rolling(14).min()
        features['williams_r'] = (highest_14 - close) / (highest_14 - lowest_14 + 1e-6) * -100
        
        # 随机指标
        features['stochastic_k'] = (close - lowest_14) / (highest_14 - lowest_14 + 1e-6) * 100
        features['stochastic_d'] = features['stochastic_k'].rolling(3).mean()
        
        # CCI (商品通道指数)
        typical_price = (high + low + close) / 3
        sma_typical = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        features['cci'] = (typical_price - sma_typical) / (0.015 * mad + 1e-6)
        
        logger.info(f"技术指标计算完成: {features.shape}")
        return features
    
    def calculate_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算时间特征
        
        Args:
            data: 原始数据
            
        Returns:
            pd.DataFrame: 时间特征
        """
        logger.info("计算时间特征...")
        
        features = pd.DataFrame(index=data.index)
        
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("数据索引不是DatetimeIndex，跳过时间特征计算")
            return features
        
        # 日期特征
        features['day_of_week'] = data.index.dayofweek
        features['day_of_month'] = data.index.day
        features['week_of_year'] = data.index.isocalendar().week
        features['month'] = data.index.month
        features['quarter'] = data.index.quarter
        features['year'] = data.index.year
        
        # 季节性特征
        features['is_month_start'] = data.index.is_month_start.astype(int)
        features['is_month_end'] = data.index.is_month_end.astype(int)
        features['is_quarter_start'] = data.index.is_quarter_start.astype(int)
        features['is_quarter_end'] = data.index.is_quarter_end.astype(int)
        features['is_year_start'] = data.index.is_year_start.astype(int)
        features['is_year_end'] = data.index.is_year_end.astype(int)
        
        # 交易日特征
        features['days_since_start'] = (data.index - data.index[0]).days
        
        logger.info(f"时间特征计算完成: {features.shape}")
        return features
    
    def calculate_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有特征
        
        Args:
            data: 原始数据
            
        Returns:
            pd.DataFrame: 所有特征
        """
        logger.info("开始计算所有特征...")
        
        # 计算各类特征
        feature_sets = []
        
        # 1. FZT砖型图特征
        try:
            fzt_brick_features = self.calculate_fzt_brick_features(data)
            feature_sets.append(fzt_brick_features)
        except Exception as e:
            logger.warning(f"FZT砖型图特征计算失败: {e}")
        
        # 2. 通用FZT特征
        try:
            fzt_general_features = self.calculate_fzt_general_features(data)
            feature_sets.append(fzt_general_features)
        except Exception as e:
            logger.warning(f"通用FZT特征计算失败: {e}")
        
        # 3. 价格特征
        price_features = self.calculate_price_features(data)
        feature_sets.append(price_features)
        
        # 4. 成交量特征
        volume_features = self.calculate_volume_features(data)
        feature_sets.append(volume_features)
        
        # 5. 技术指标
        technical_features = self.calculate_technical_indicators(data)
        feature_sets.append(technical_features)
        
        # 6. 时间特征
        time_features = self.calculate_time_features(data)
        feature_sets.append(time_features)
        
        # 合并所有特征
        if feature_sets:
            all_features = pd.concat(feature_sets, axis=1)
        else:
            all_features = pd.DataFrame(index=data.index)
        
        # 处理缺失值
        all_features = all_features.fillna(method='ffill').fillna(0)
        
        # 移除完全相同的列
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        
        logger.info(f"所有特征计算完成: {all_features.shape}")
        logger.info(f"特征数量: {len(all_features.columns)}")
        logger.info(f"数据行数: {len(all_features)}")
        
        return all_features
    
    def calculate_target_variable(self, data: pd.DataFrame, horizon: int = 1) -> pd.Series:
        """
        计算目标变量：未来N天收益率（默认T+1）
        
        Args:
            data: 包含价格数据的DataFrame
            horizon: 预测 horizon（天数），默认1表示第二天
            
        Returns:
            pd.Series: 未来收益率序列
        """
        if 'close' not in data.columns:
            raise ValueError("输入数据必须包含'close'列")
        
        # 计算未来收益率（T+horizon）
        future_return = data['close'].shift(-horizon) / data['close'] - 1
        
        # 将最后horizon天的值设为NaN（因为没有未来数据）
        if len(future_return) > horizon:
            future_return.iloc[-horizon:] = np.nan
        
        # 转换为二分类：上涨为1，下跌或平盘为0
        target_binary = (future_return > 0).astype(int)
        target_binary.name = 'target'
        
        logger.info(f"目标变量计算完成 (T+{horizon}):")
        logger.info(f"  正类比例: {target_binary.mean():.2%}")
        logger.info(f"  有效样本数: {target_binary.notna().sum()}")
        
        return target_binary
    
    def normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        标准化特征
        
        Args:
            features: 原始特征
            
        Returns:
            pd.DataFrame: 标准化后的特征
        """
        logger.info("标准化特征...")
        
        # 分离数值特征和非数值特征
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        non_numeric_cols = features.select_dtypes(exclude=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.warning("没有数值特征需要标准化")
            return features
        
        # 标准化数值特征
        scaler = StandardScaler()
        numeric_features = features[numeric_cols].copy()
        
        # 处理无穷值和极大值
        numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
        numeric_features = numeric_features.fillna(0)
        
        # 标准化
        scaled_values = scaler.fit_transform(numeric_features)
        scaled_features = pd.DataFrame(
            scaled_values, 
            index=numeric_features.index, 
            columns=numeric_features.columns
        )
        
        # 合并所有特征
        if len(non_numeric_cols) > 0:
            non_numeric_features = features[non_numeric_cols].copy()
            normalized_features = pd.concat([scaled_features, non_numeric_features], axis=1)
        else:
            normalized_features = scaled_features
        
        logger.info(f"特征标准化完成: {normalized_features.shape}")
        
        return normalized_features
    
    def select_features(self, features: pd.DataFrame, target: pd.Series, 
                       k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """
        选择最重要的特征
        
        Args:
            features: 特征DataFrame
            target: 目标变量
            k: 选择的特征数量
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: 选择的特征和特征名称
        """
        logger.info(f"选择最重要的 {k} 个特征...")
        
        # 只处理数值特征
        numeric_features = features.select_dtypes(include=[np.number])
        
        if len(numeric_features.columns) == 0:
            logger.warning("没有数值特征可供选择")
            return features, list(features.columns)
        
        # 对齐索引
        common_idx = numeric_features.index.intersection(target.index)
        X = numeric_features.loc[common_idx]
        y = target.loc[common_idx]
        
        # 移除目标变量中的NaN
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0 or len(y) == 0:
            logger.warning("没有有效数据用于特征选择")
            return features, list(features.columns)
        
        # 特征选择
        selector = SelectKBest(score_func=f_regression, k=min(k, len(X.columns)))
        selector.fit(X, y)
        
        # 获取选择的特征
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # 获取特征分数
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_,
            'p_value': selector.pvalues_
        }).sort_values('score', ascending=False)
        
        logger.info(f"特征选择完成: {len(selected_features)} 个特征被选中")
        logger.info(f"Top 10 特征:")
        for i, row in feature_scores.head(10).iterrows():
            logger.info(f"  {row['feature']}: 分数={row['score']:.2f}, p值={row['p_value']:.4f}")
        
        # 返回选择的特征
        selected_data = features[selected_features].copy()
        
        return selected_data, selected_features
    
    def create_dataset(self, data: pd.DataFrame, horizon: int = 2, 
                      normalize: bool = True, select_features: bool = True) -> pd.DataFrame:
        """
        创建完整的数据集
        
        Args:
            data: 原始数据
            horizon: 预测 horizon
            normalize: 是否标准化特征
            select_features: 是否进行特征选择
            
        Returns:
            pd.DataFrame: 完整的数据集
        """
        logger.info("创建完整数据集...")
        
        # 1. 计算所有特征
        features = self.calculate_all_features(data)
        
        # 2. 计算目标变量
        target = self.calculate_target_variable(data, horizon)
        features['target'] = target
        
        # 3. 标准化特征
        if normalize:
            features_to_normalize = features.drop(columns=['target'], errors='ignore')
            normalized_features = self.normalize_features(features_to_normalize)
            normalized_features['target'] = features['target']
            features = normalized_features
        
        # 4. 特征选择
        if select_features and 'target' in features.columns:
            target_series = features['target']
            feature_cols = [col for col in features.columns if col != 'target']
            
            if len(feature_cols) > 0:
                selected_features, selected_names = self.select_features(
                    features[feature_cols], target_series, k=50
                )
                selected_features['target'] = target_series
                features = selected_features
        
        # 5. 添加元数据
        features['date'] = features.index
        if 'stock_code' not in features.columns and hasattr(data, 'name'):
            features['stock_code'] = data.name
        
        # 6. 处理缺失值
        features = features.fillna(method='ffill').fillna(0)
        
        logger.info(f"数据集创建完成: {features.shape}")
        logger.info(f"特征列: {list(features.columns)}")
        
        return features
    
    def batch_create_datasets(self, stock_data: Dict[str, pd.DataFrame], 
                             horizon: int = 2) -> Dict[str, pd.DataFrame]:
        """
        批量创建多只股票的数据集
        
        Args:
            stock_data: 股票数据字典
            horizon: 预测 horizon
            
        Returns:
            Dict[str, pd.DataFrame]: 各股票的数据集
        """
        logger.info(f"批量创建 {len(stock_data)} 只股票的数据集...")
        
        datasets = {}
        
        for stock_code, data in stock_data.items():
            try:
                # 设置数据名称（用于特征计算中的引用）
                data.name = stock_code
                
                # 创建数据集
                dataset = self.create_dataset(data, horizon)
                dataset['stock_code'] = stock_code
                
                datasets[stock_code] = dataset
                
                logger.debug(f"股票 {stock_code} 数据集创建完成: {dataset.shape}")
                
            except Exception as e:
                logger.warning(f"股票 {stock_code} 数据集创建失败: {e}")
                continue
        
        logger.info(f"批量数据集创建完成: {len(datasets)}/{len(stock_data)} 只股票成功")
        
        return datasets
    
    def save_dataset(self, dataset: pd.DataFrame, name: str):
        """
        保存数据集
        
        Args:
            dataset: 数据集
            name: 数据集名称
        """
        save_path = self.features_dir / f"{name}_dataset.parquet"
        
        # 保存为Parquet格式
        dataset.to_parquet(save_path)
        
        # 保存特征列表
        feature_list_path = self.features_dir / f"{name}_features.txt"
        with open(feature_list_path, 'w', encoding='utf-8') as f:
            for feature in dataset.columns:
                f.write(f"{feature}\n")
        
        logger.info(f"数据集已保存到: {save_path}")
        logger.info(f"特征列表已保存到: {feature_list_path}")
    
    def load_dataset(self, name: str) -> pd.DataFrame:
        """
        加载数据集
        
        Args:
            name: 数据集名称
            
        Returns:
            pd.DataFrame: 数据集
        """
        load_path = self.features_dir / f"{name}_dataset.parquet"
        
        if not load_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {load_path}")
        
        dataset = pd.read_parquet(load_path)
        logger.info(f"数据集已加载: {dataset.shape}")
        
        return dataset
    
    def analyze_dataset(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        分析数据集
        
        Args:
            dataset: 数据集
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        logger.info("分析数据集...")
        
        analysis = {
            'shape': dataset.shape,
            'columns': list(dataset.columns),
            'dtypes': dataset.dtypes.to_dict(),
            'missing_values': dataset.isnull().sum().to_dict(),
            'basic_stats': {}
        }
        
        # 数值特征的基本统计
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis['basic_stats'] = dataset[numeric_cols].describe().to_dict()
        
        # 目标变量分析
        if 'target' in dataset.columns:
            target = dataset['target']
            analysis['target_stats'] = {
                'count': len(target),
                'mean': float(target.mean()),
                'std': float(target.std()),
                'min': float(target.min()),
                'max': float(target.max()),
                'positive_ratio': float((target > 0).mean())
            }
        
        # 特征相关性
        if len(numeric_cols) > 1:
            correlation = dataset[numeric_cols].corr()
            analysis['correlation_matrix_shape'] = correlation.shape
        
        logger.info(f"数据集分析完成: {dataset.shape}")
        
        return analysis


def test_feature_engineer():
    """测试特征工程师"""
    try:
        print("=" * 80)
        print("特征工程模块测试")
        print("=" * 80)
        
        # 创建测试数据
        print("\n1. 创建测试数据...")
        np.random.seed(42)
        n_days = 100
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        
        base_price = 100
        trend = np.linspace(0, 0.2, n_days)
        noise = np.random.normal(0, 0.02, n_days)
        returns = 0.001 + trend/100 + noise
        prices = base_price * np.exp(np.cumsum(returns))
        
        test_data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0.01, 0.005, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0.01, 0.005, n_days))),
            'close': prices,
            'volume': np.random.lognormal(mean=10, sigma=1, size=n_days)
        }, index=dates)
        
        print(f"   数据形状: {test_data.shape}")
        
        # 初始化特征工程师
        print("\n2. 初始化特征工程师...")
        # 使用绝对导入避免相对导入问题
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        engineer = FeatureEngineer()
        print("   ✓ 初始化成功")
        
        # 计算所有特征
        print("\n3. 计算所有特征...")
        features = engineer.calculate_all_features(test_data)
        print(f"   ✓ 特征计算完成: {features.shape}")
        print(f"   特征数量: {len(features.columns)}")
        
        # 创建完整数据集
        print("\n4. 创建完整数据集...")
        dataset = engineer.create_dataset(test_data, horizon=2)
        print(f"   ✓ 数据集创建完成: {dataset.shape}")
        print(f"   包含目标变量: {'target' in dataset.columns}")
        
        # 分析数据集
        print("\n5. 分析数据集...")
        analysis = engineer.analyze_dataset(dataset)
        print(f"   数据集形状: {analysis['shape']}")
        print(f"   特征数量: {len(analysis['columns'])}")
        
        if 'target_stats' in analysis:
            target_stats = analysis['target_stats']
            print(f"   目标变量统计:")
            print(f"     均值: {target_stats['mean']:.4f}")
            print(f"     标准差: {target_stats['std']:.4f}")
            print(f"     正收益率比例: {target_stats['positive_ratio']:.2%}")
        
        # 保存和加载测试
        print("\n6. 测试保存和加载...")
        engineer.save_dataset(dataset, 'test')
        loaded_dataset = engineer.load_dataset('test')
        print(f"   ✓ 保存和加载测试通过")
        print(f"   原始数据集形状: {dataset.shape}")
        print(f"   加载数据集形状: {loaded_dataset.shape}")
        
        # 显示特征示例
        print("\n7. 特征示例:")
        sample_features = features.iloc[-5:]
        print(f"   最后5天的特征形状: {sample_features.shape}")
        print(f"   特征列示例: {list(features.columns)[:10]}...")
        
        print("\n" + "=" * 80)
        print("特征工程模块测试完成！所有功能正常。")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_feature_engineer())
