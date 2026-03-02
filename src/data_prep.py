#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据准备模块 - FZT项目数据预处理

功能：
1. 读取本地CSV数据文件
2. 数据预处理和清洗
3. 时间范围过滤和数据集划分
4. 股票筛选
5. 数据保存和版本控制

作者: FZT项目组
创建日期: 2026-03-01
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """数据预处理主类"""
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """
        初始化数据预处理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # 设置数据源
        self.source = self.config['data']['source']
        
        self._validate_config()
        
        # 设置数据路径（如果使用本地CSV数据）
        if self.source in ['local_csv', 'combined', 'hybrid']:
            self.data_path = Path(self.config['data']['local_csv']['data_path'])
            self.file_pattern = self.config['data']['local_csv']['file_pattern']
            self.field_mapping = self.config['data']['local_csv']['field_mapping']
        
        # 处理配置
        self.processing_config = self.config.get('processing', {})
        self.filter_config = self.config.get('stock_filter', {})
        
        # 时间范围配置
        self.time_ranges = self.config['time_ranges']
        self.source = self.config['data']['source']
        
        # 输出目录
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"数据预处理器初始化完成，数据源: {self.source}")
        logger.info(f"数据路径: {self.data_path}")
        logger.info(f"输出目录: {self.output_dir}")
    
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
    
    def _validate_config(self) -> None:
        """验证配置文件的完整性"""
        required_sections = ['data', 'time_ranges']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置文件中缺少必要部分: {section}")
        
        # 验证数据源配置
        valid_sources = ['local_csv', 'qlib', 'combined', 'hybrid']
        if self.config['data']['source'] not in valid_sources:
            raise ValueError(f"不支持的数据源: {self.config['data']['source']}，有效值: {valid_sources}")
        
        # 根据数据源验证配置
        source = self.config['data']['source']
        if source == 'local_csv':
            local_csv_config = self.config['data']['local_csv']
            if not local_csv_config.get('enabled', False):
                raise ValueError("本地CSV数据源未启用")
        elif source == 'qlib':
            qlib_config = self.config['data']['qlib']
            if not qlib_config.get('enabled', False):
                raise ValueError("Qlib数据源未启用")
        elif source == 'hybrid':
            hybrid_config = self.config['data']['hybrid']
            if not hybrid_config.get('enabled', False):
                raise ValueError("混合数据模式未启用")
            # 验证混合数据源配置
            train_val_source = hybrid_config.get('train_val_source')
            test_source = hybrid_config.get('test_source')
            if train_val_source not in ['qlib']:
                raise ValueError(f"混合模式训练验证数据源必须是qlib，当前: {train_val_source}")
            if test_source not in ['local_csv']:
                raise ValueError(f"混合模式测试数据源必须是local_csv，当前: {test_source}")
        
        # 验证字段映射（仅当使用本地CSV数据时）
        if self.source in ['local_csv', 'combined', 'hybrid']:
            required_fields = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']
            # 获取本地CSV配置
            local_csv_config = self.config['data']['local_csv']
            field_mapping = local_csv_config.get('field_mapping', {})
            for field in required_fields:
                if field not in field_mapping:
                    raise ValueError(f"字段映射中缺少必要字段: {field}")
        
        logger.info("配置文件验证通过")
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        加载原始CSV数据
        
        Returns:
            pd.DataFrame: 合并后的原始数据
        """
        logger.info("开始加载原始CSV数据...")
        
        # 查找匹配的文件
        data_files = list(self.data_path.glob(self.file_pattern))
        
        if not data_files:
            # 尝试另一种匹配方式
            data_files = list(self.data_path.glob("*.csv"))
            if not data_files:
                raise FileNotFoundError(f"未找到匹配的文件: {self.data_path}/{self.file_pattern}")
            else:
                logger.info(f"使用通配符找到 {len(data_files)} 个CSV文件")
        
        logger.info(f"找到 {len(data_files)} 个数据文件")
        
        # 读取并合并所有文件
        dfs = []
        for file_path in tqdm(data_files, desc="读取数据文件"):
            try:
                df = pd.read_csv(file_path)
                
                # 重命名列
                reverse_mapping = {v: k for k, v in self.field_mapping.items()}
                df = df.rename(columns=reverse_mapping)
                
                # 添加文件来源信息
                df['source_file'] = file_path.name
                
                dfs.append(df)
                
            except Exception as e:
                logger.warning(f"读取文件失败 {file_path}: {e}")
                continue
        
        if not dfs:
            raise ValueError("所有数据文件读取失败")
        
        # 合并所有数据
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # 数据基本信息
        logger.info(f"数据加载完成，总行数: {combined_df.shape[0]}")
        logger.info(f"股票数量: {combined_df['code'].nunique()}")
        logger.info(f"时间范围: {combined_df['date'].min()} 到 {combined_df['date'].max()}")
        
        return combined_df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理主流程
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        logger.info("开始数据预处理...")
        
        # 创建副本避免修改原始数据
        processed_df = df.copy()
        
        # 1. 日期解析和排序
        processed_df['date'] = pd.to_datetime(processed_df['date'])
        processed_df = processed_df.sort_values(['code', 'date']).reset_index(drop=True)
        
        # 2. 数据类型转换
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        # 3. 缺失值处理
        if self.processing_config.get('fill_na', True):
            # 按股票前向填充
            processed_df[numeric_columns] = processed_df.groupby('code')[numeric_columns].ffill()
            # 如果还有缺失值，用0填充
            processed_df[numeric_columns] = processed_df[numeric_columns].fillna(0)
        
        # 4. 异常值检测和处理
        if self.processing_config.get('remove_outliers', True):
            processed_df = self._remove_outliers(processed_df)
        
        # 5. 数据标准化（可选）
        if self.processing_config.get('normalize', True):
            processed_df = self._normalize_data(processed_df)
        
        # 6. 添加技术指标（可选）
        if self.config.get('features', {}).get('technical_factors'):
            processed_df = self._add_technical_indicators(processed_df)
        
        logger.info(f"数据预处理完成，处理后行数: {processed_df.shape[0]}")
        logger.info(f"处理后股票数量: {processed_df['code'].nunique()}")
        
        return processed_df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """移除异常值"""
        logger.info("检测并处理异常值...")
        
        # 价格异常值：价格在合理范围内
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            # 移除价格为0或负数的记录
            df = df[df[col] > 0]
            
            # 移除价格超过配置范围的记录
            min_price = self.filter_config.get('min_price', 1.0)
            max_price = self.filter_config.get('max_price', 1000.0)
            df = df[(df[col] >= min_price) & (df[col] <= max_price)]
        
        # 成交量异常值：移除成交量为0的记录
        df = df[df['volume'] > 0]
        
        # 价格关系异常：high >= low, high >= close >= low
        df = df[df['high'] >= df['low']]
        df = df[df['high'] >= df['close']]
        df = df[df['close'] >= df['low']]
        
        return df
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据标准化"""
        logger.info("数据标准化...")
        
        # 价格标准化：使用对数收益率
        price_columns = ['open', 'high', 'low', 'close']
        
        for col in price_columns:
            # 计算对数收益率
            df[f'{col}_log_return'] = df.groupby('code')[col].pct_change().add(1).apply(np.log)
        
        # 成交量标准化：使用对数变换
        df['volume_log'] = np.log1p(df['volume'])
        df['amount_log'] = np.log1p(df['amount'])
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        logger.info("添加技术指标...")
        
        # 按股票分组计算技术指标
        for code, group in df.groupby('code'):
            idx = group.index
            
            # 简单移动平均
            df.loc[idx, 'sma_5'] = group['close'].rolling(window=5).mean()
            df.loc[idx, 'sma_20'] = group['close'].rolling(window=20).mean()
            
            # 动量指标
            df.loc[idx, 'momentum_5'] = group['close'].pct_change(5)
            df.loc[idx, 'momentum_20'] = group['close'].pct_change(20)
            
            # 波动率
            df.loc[idx, 'volatility_20'] = group['close'].pct_change().rolling(window=20).std()
        
        return df
    
    def filter_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        股票筛选
        
        Args:
            df: 预处理后的数据
            
        Returns:
            pd.DataFrame: 筛选后的数据
        """
        logger.info("开始股票筛选...")
        
        # 1. 最小交易日数过滤
        min_days = self.filter_config.get('min_days', 60)
        stock_days = df.groupby('code')['date'].nunique()
        valid_stocks = stock_days[stock_days >= min_days].index.tolist()
        df = df[df['code'].isin(valid_stocks)]
        
        logger.info(f"最小交易日数过滤: {len(valid_stocks)}/{stock_days.shape[0]} 只股票通过")
        
        # 2. ST股票排除（如果可识别）
        if self.filter_config.get('exclude_st', True):
            # 假设ST股票代码以"ST"开头或包含"ST"
            # 注意：需要转义正则表达式特殊字符
            st_patterns = [r'^ST', r'^.*ST.*$', r'^\*ST', r'^S\*T']
            pattern = '|'.join(st_patterns)
            st_mask = df['code'].str.contains(pattern, case=False, regex=True)
            removed_count = st_mask.sum()
            df = df[~st_mask]
            if removed_count > 0:
                logger.info(f"排除ST股票: 移除 {removed_count} 条记录")
        
        # 3. 价格范围过滤（已在异常值处理中完成）
        
        # 4. 停牌股票排除（可选）
        if self.filter_config.get('exclude_suspended', True):
            # 假设停牌股票有连续多天价格不变
            # 这里简化处理：移除连续3天价格完全相同的记录
            df = df.sort_values(['code', 'date'])
            df['price_change'] = df.groupby('code')['close'].diff().abs()
            # 实际应用中需要更复杂的停牌检测逻辑
        
        logger.info(f"股票筛选完成，剩余股票数量: {df['code'].nunique()}")
        logger.info(f"剩余数据行数: {df.shape[0]}")
        
        return df
    
    def split_by_time(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        按时间划分数据集
        
        Args:
            df: 预处理和筛选后的数据
            
        Returns:
            Dict[str, pd.DataFrame]: 划分后的数据集
        """
        logger.info(f"按时间划分数据集 (数据源: {self.source})...")
        
        # 获取对应数据源的时间范围配置
        if self.source == 'local_csv':
            time_config = self.time_ranges['local_csv']
            
            # 转换为datetime
            train_start = pd.to_datetime(time_config['train_start'])
            train_end = pd.to_datetime(time_config['train_end'])
            valid_start = pd.to_datetime(time_config['valid_start'])
            valid_end = pd.to_datetime(time_config['valid_end'])
            test_start = pd.to_datetime(time_config['test_start'])
            test_end = pd.to_datetime(time_config['test_end'])
            
            # 划分数据集
            train_data = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
            valid_data = df[(df['date'] >= valid_start) & (df['date'] <= valid_end)]
            test_data = df[(df['date'] >= test_start) & (df['date'] <= test_end)]
            
        elif self.source == 'qlib':
            time_config = self.time_ranges['qlib']
            
            # 转换为datetime
            train_start = pd.to_datetime(time_config['train_start'])
            train_end = pd.to_datetime(time_config['train_end'])
            valid_start = pd.to_datetime(time_config['valid_start'])
            valid_end = pd.to_datetime(time_config['valid_end'])
            test_start = pd.to_datetime(time_config['test_start'])
            test_end = pd.to_datetime(time_config['test_end'])
            
            # 划分数据集
            train_data = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
            valid_data = df[(df['date'] >= valid_start) & (df['date'] <= valid_end)]
            test_data = df[(df['date'] >= test_start) & (df['date'] <= test_end)]
            
        elif self.source == 'combined':
            time_config = self.time_ranges['combined']
            
            # 转换为datetime
            train_start = pd.to_datetime(time_config['train_start'])
            train_end = pd.to_datetime(time_config['train_end'])
            valid_start = pd.to_datetime(time_config['valid_start'])
            valid_end = pd.to_datetime(time_config['valid_end'])
            test_start = pd.to_datetime(time_config['test_start'])
            test_end = pd.to_datetime(time_config['test_end'])
            
            # 划分数据集
            train_data = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
            valid_data = df[(df['date'] >= valid_start) & (df['date'] <= valid_end)]
            test_data = df[(df['date'] >= test_start) & (df['date'] <= test_end)]
            
        elif self.source == 'hybrid':
            # 混合模式：训练验证使用Qlib数据，测试使用本地CSV数据
            # 这里只处理一种数据源，混合逻辑在更高层处理
            if 'hybrid' not in self.time_ranges:
                raise ValueError("混合模式配置不存在")
            
            hybrid_config = self.time_ranges['hybrid']
            
            # 判断当前处理的是哪种数据
            if df.empty:
                # 空数据直接返回空结果
                train_data = pd.DataFrame()
                valid_data = pd.DataFrame()
                test_data = pd.DataFrame()
            else:
                data_start = df['date'].min()
                data_end = df['date'].max()
                
                # 如果是Qlib数据时间范围（2005-2020）
                if data_start.year <= 2020:
                    if 'qlib' not in hybrid_config:
                        raise ValueError("混合模式中缺少qlib配置")
                    
                    time_config = hybrid_config['qlib']
                    train_start = pd.to_datetime(time_config['train_start'])
                    train_end = pd.to_datetime(time_config['train_end'])
                    valid_start = pd.to_datetime(time_config['valid_start'])
                    valid_end = pd.to_datetime(time_config['valid_end'])
                    
                    train_data = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
                    valid_data = df[(df['date'] >= valid_start) & (df['date'] <= valid_end)]
                    test_data = pd.DataFrame()  # Qlib数据不用于测试
                    
                # 如果是本地CSV数据时间范围（2021-2026）
                else:
                    if 'local_csv' not in hybrid_config:
                        raise ValueError("混合模式中缺少local_csv配置")
                    
                    time_config = hybrid_config['local_csv']
                    test_start = pd.to_datetime(time_config['test_start'])
                    test_end = pd.to_datetime(time_config['test_end'])
                    
                    train_data = pd.DataFrame()  # 本地数据不用于训练
                    valid_data = pd.DataFrame()  # 本地数据不用于验证
                    test_data = df[(df['date'] >= test_start) & (df['date'] <= test_end)]
                    
        else:
            raise ValueError(f"不支持的数据源: {self.source}")
        
        # 验证划分结果
        total_days = df['date'].nunique() if not df.empty else 0
        train_days = train_data['date'].nunique() if not train_data.empty else 0
        valid_days = valid_data['date'].nunique() if not valid_data.empty else 0
        test_days = test_data['date'].nunique() if not test_data.empty else 0
        
        logger.info(f"时间划分结果:")
        if not train_data.empty:
            logger.info(f"  训练集: {train_start.date()} 到 {train_end.date()}, {train_days} 天, {train_data.shape[0]} 行")
        if not valid_data.empty:
            logger.info(f"  验证集: {valid_start.date()} 到 {valid_end.date()}, {valid_days} 天, {valid_data.shape[0]} 行")
        if not test_data.empty:
            logger.info(f"  测试集: {test_start.date()} 到 {test_end.date()}, {test_days} 天, {test_data.shape[0]} 行")
        
        if total_days > 0:
            coverage = (train_days + valid_days + test_days) / total_days
            logger.info(f"  总覆盖率: {coverage:.2%}")
        else:
            logger.warning("  原始数据为空，无法计算覆盖率")
        
        return {
            'train': train_data,
            'valid': valid_data,
            'test': test_data
        }
    
    def save_processed_data(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """
        保存处理后的数据
        
        Args:
            data_dict: 划分后的数据集字典
        """
        logger.info("保存处理后的数据...")
        
        # 生成版本标识
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for split_name, data in data_dict.items():
            if data.empty:
                logger.warning(f"{split_name}数据集为空，跳过保存")
                continue
            
            # 保存为CSV
            csv_path = self.output_dir / f"{split_name}_data_v{version}.csv"
            data.to_csv(csv_path, index=False)
            logger.info(f"保存CSV文件: {csv_path}")
            
            # 保存为Parquet（如果安装了pyarrow）
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                
                parquet_path = self.output_dir / f"{split_name}_data_v{version}.parquet"
                data.to_parquet(parquet_path, index=False)
                logger.info(f"保存Parquet文件: {parquet_path}")
            except ImportError:
                logger.warning("未安装pyarrow，跳过Parquet格式保存")
            
            # 保存为Feather（如果安装了pyarrow）
            try:
                import pyarrow.feather as feather
                
                feather_path = self.output_dir / f"{split_name}_data_v{version}.feather"
                feather.write_feather(data, feather_path)
                logger.info(f"保存Feather文件: {feather_path}")
            except ImportError:
                logger.warning("未安装pyarrow，跳过Feather格式保存")
        
        # 保存数据统计信息
        self._save_data_statistics(data_dict, version)
        
        logger.info(f"数据保存完成，版本: {version}")
    
    def _save_data_statistics(self, data_dict: Dict[str, pd.DataFrame], version: str) -> None:
        """保存数据统计信息"""
        stats = {}
        
        for split_name, data in data_dict.items():
            if data.empty:
                stats[split_name] = {"rows": 0, "stocks": 0, "days": 0}
                continue
            
            stats[split_name] = {
                "rows": data.shape[0],
                "stocks": data['code'].nunique(),
                "days": data['date'].nunique(),
                "date_range": {
                    "start": data['date'].min().strftime("%Y-%m-%d"),
                    "end": data['date'].max().strftime("%Y-%m-%d")
                },
                "columns": list(data.columns)
            }
        
        # 保存统计信息为YAML
        stats_path = self.output_dir / f"data_statistics_v{version}.yaml"
        with open(stats_path, 'w', encoding='utf-8') as f:
            yaml.dump(stats, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"数据统计信息保存到: {stats_path}")
    
    def run_pipeline(self) -> Dict[str, pd.DataFrame]:
        """
        运行完整的数据处理管道
        
        Returns:
            Dict[str, pd.DataFrame]: 划分后的数据集
        """
        logger.info("开始运行数据处理管道...")
        
        try:
            # 1. 加载原始数据
            raw_data = self.load_raw_data()
            
            # 2. 数据预处理
            processed_data = self.preprocess_data(raw_data)
            
            # 3. 股票筛选
            filtered_data = self.filter_stocks(processed_data)
            
            # 4. 时间划分
            split_data = self.split_by_time(filtered_data)
            
            # 5. 保存数据
            self.save_processed_data(split_data)
            
            logger.info("数据处理管道运行完成")
            
            return split_data
            
        except Exception as e:
            logger.error(f"数据处理管道运行失败: {e}")
            raise
    
    def generate_data_report(self, data_dict: Dict[str, pd.DataFrame]) -> str:
        """
        生成数据质量报告
        
        Args:
            data_dict: 划分后的数据集
            
        Returns:
            str: 数据质量报告
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("数据质量报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"数据源: {self.source}")
        report_lines.append("")
        
        for split_name, data in data_dict.items():
            report_lines.append(f"{split_name.upper()} 数据集:")
            report_lines.append(f"  - 数据行数: {data.shape[0]:,}")
            report_lines.append(f"  - 股票数量: {data['code'].nunique():,}")
            report_lines.append(f"  - 交易日数: {data['date'].nunique():,}")
            
            if not data.empty:
                date_range = f"{data['date'].min().date()} 到 {data['date'].max().date()}"
                report_lines.append(f"  - 时间范围: {date_range}")
                
                # 数据完整性检查
                missing_values = data.isnull().sum().sum()
                total_cells = data.size
                completeness = 1 - missing_values / total_cells if total_cells > 0 else 0
                report_lines.append(f"  - 数据完整性: {completeness:.2%}")
                
                # 价格统计
                price_stats = data[['open', 'high', 'low', 'close']].describe()
                report_lines.append(f"  - 价格范围: {price_stats.loc['min', 'close']:.2f} - {price_stats.loc['max', 'close']:.2f}")
                report_lines.append(f"  - 平均价格: {price_stats.loc['mean', 'close']:.2f}")
            
            report_lines.append("")
        
        # 总体统计
        total_rows = sum(data.shape[0] for data in data_dict.values())
        total_stocks = set()
        for data in data_dict.values():
            total_stocks.update(data['code'].unique())
        
        report_lines.append("总体统计:")
        report_lines.append(f"  - 总数据行数: {total_rows:,}")
        report_lines.append(f"  - 总股票数量: {len(total_stocks):,}")
        report_lines.append(f"  - 数据集划分: {', '.join(data_dict.keys())}")
        
        report = "\n".join(report_lines)
        logger.info("数据质量报告生成完成")
        
        return report


def main():
    """主函数，用于测试和演示"""
    try:
        # 初始化数据预处理器
        preprocessor = DataPreprocessor()
        
        # 运行完整管道
        split_data = preprocessor.run_pipeline()
        
        # 生成数据报告
        report = preprocessor.generate_data_report(split_data)
        print(report)
        
        # 保存报告到文件
        report_path = preprocessor.output_dir / "data_quality_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n数据质量报告已保存到: {report_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"主函数执行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())