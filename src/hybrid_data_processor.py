#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合数据处理器 - 处理Qlib训练验证 + 本地CSV测试的混合模式

功能：
1. 加载Qlib数据用于训练和验证
2. 加载本地CSV数据用于测试
3. 统一数据格式和预处理
4. 合并数据集并保存

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


class HybridDataProcessor:
    """混合数据处理器"""
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """
        初始化混合数据处理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        
        # 导入数据准备模块
        from .data_prep import DataPreprocessor
        
        # 创建Qlib数据预处理器
        self.qlib_preprocessor = self._create_qlib_preprocessor()
        
        # 创建本地CSV数据预处理器
        self.local_csv_preprocessor = self._create_local_csv_preprocessor()
        
        # 输出目录
        self.output_dir = Path("data/processed/hybrid")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("混合数据处理器初始化完成")
        logger.info(f"训练验证数据源: Qlib (2005-2020)")
        logger.info(f"测试数据源: 本地CSV (2021-2026)")
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
        if 'data' not in self.config:
            raise ValueError("配置文件中缺少data部分")
        
        if 'hybrid' not in self.config['data']:
            raise ValueError("配置文件中缺少hybrid配置")
        
        hybrid_config = self.config['data']['hybrid']
        if not hybrid_config.get('enabled', False):
            raise ValueError("混合数据模式未启用")
        
        # 验证数据源配置
        train_val_source = hybrid_config.get('train_val_source')
        test_source = hybrid_config.get('test_source')
        
        if train_val_source != 'qlib':
            raise ValueError(f"混合模式训练验证数据源必须是qlib，当前: {train_val_source}")
        
        if test_source != 'local_csv':
            raise ValueError(f"混合模式测试数据源必须是local_csv，当前: {test_source}")
        
        # 验证时间范围配置
        if 'time_ranges' not in self.config:
            raise ValueError("配置文件中缺少time_ranges部分")
        
        if 'hybrid' not in self.config['time_ranges']:
            raise ValueError("配置文件中缺少hybrid时间范围配置")
        
        logger.info("混合数据配置验证通过")
    
    def _create_qlib_preprocessor(self) -> 'DataPreprocessor':
        """创建Qlib数据预处理器"""
        # 创建临时配置用于Qlib数据
        qlib_config = self.config.copy()
        
        # 修改配置为qlib模式
        qlib_config['data']['source'] = 'qlib'
        qlib_config['data']['qlib']['enabled'] = True
        
        # 保存临时配置
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(qlib_config, f, default_flow_style=False, allow_unicode=True)
            temp_config_path = f.name
        
        try:
            from .data_prep import DataPreprocessor
            preprocessor = DataPreprocessor(temp_config_path)
            logger.info("Qlib数据预处理器创建成功")
            return preprocessor
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_config_path)
            except:
                pass
    
    def _create_local_csv_preprocessor(self) -> 'DataPreprocessor':
        """创建本地CSV数据预处理器"""
        # 创建临时配置用于本地CSV数据
        local_config = self.config.copy()
        
        # 修改配置为local_csv模式
        local_config['data']['source'] = 'local_csv'
        local_config['data']['local_csv']['enabled'] = True
        
        # 保存临时配置
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(local_config, f, default_flow_style=False, allow_unicode=True)
            temp_config_path = f.name
        
        try:
            from .data_prep import DataPreprocessor
            preprocessor = DataPreprocessor(temp_config_path)
            logger.info("本地CSV数据预处理器创建成功")
            return preprocessor
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_config_path)
            except:
                pass
    
    def load_qlib_data(self) -> Dict[str, pd.DataFrame]:
        """
        加载Qlib数据
        
        Returns:
            Dict[str, pd.DataFrame]: Qlib数据集
        """
        logger.info("开始加载Qlib数据...")
        
        try:
            # 使用新的Qlib数据加载器
            from .qlib_data_loader import QlibDataLoader
            
            qlib_loader = QlibDataLoader(self.config_path)
            split_data = qlib_loader.load_and_split_data()
            
            logger.info(f"✅ Qlib数据加载完成:")
            for split_name, data in split_data.items():
                if not data.empty:
                    logger.info(f"  {split_name}: {data.shape[0]} 行, {data['code'].nunique()} 只股票")
                else:
                    logger.warning(f"  {split_name}: 数据为空")
            
            return split_data
            
        except ImportError as e:
            logger.error(f"Qlib数据加载器导入失败: {e}")
            logger.warning("使用空数据作为占位符")
            
            # 创建空的DataFrame作为占位符
            empty_df = pd.DataFrame()
            
            return {
                'train': empty_df,
                'valid': empty_df,
                'test': empty_df
            }
        except Exception as e:
            logger.error(f"加载Qlib数据失败: {e}")
            raise
    
    def load_local_csv_data(self) -> Dict[str, pd.DataFrame]:
        """
        加载本地CSV数据
        
        Returns:
            Dict[str, pd.DataFrame]: 本地CSV数据集
        """
        logger.info("开始加载本地CSV数据...")
        
        try:
            # 使用数据预处理器加载数据
            split_data = self.local_csv_preprocessor.run_pipeline()
            
            logger.info(f"本地CSV数据加载完成:")
            for split_name, data in split_data.items():
                if not data.empty:
                    logger.info(f"  {split_name}: {data.shape[0]} 行, {data['code'].nunique()} 只股票")
            
            return split_data
            
        except Exception as e:
            logger.error(f"加载本地CSV数据失败: {e}")
            raise
    
    def process_hybrid_data(self) -> Dict[str, pd.DataFrame]:
        """
        处理混合数据
        
        Returns:
            Dict[str, pd.DataFrame]: 混合数据集
        """
        logger.info("开始处理混合数据...")
        
        try:
            # 1. 加载Qlib数据（训练验证）
            logger.info("1. 加载Qlib数据用于训练验证...")
            qlib_data = self.load_qlib_data()
            
            # 2. 加载本地CSV数据（测试）
            logger.info("2. 加载本地CSV数据用于测试...")
            local_csv_data = self.load_local_csv_data()
            
            # 3. 合并数据集
            logger.info("3. 合并数据集...")
            hybrid_data = {
                'train': qlib_data.get('train', pd.DataFrame()),
                'valid': qlib_data.get('valid', pd.DataFrame()),
                'test': local_csv_data.get('test', pd.DataFrame())
            }
            
            # 4. 验证数据集
            self._validate_hybrid_data(hybrid_data)
            
            # 5. 保存混合数据
            self._save_hybrid_data(hybrid_data)
            
            logger.info("混合数据处理完成")
            
            return hybrid_data
            
        except Exception as e:
            logger.error(f"处理混合数据失败: {e}")
            raise
    
    def _validate_hybrid_data(self, hybrid_data: Dict[str, pd.DataFrame]) -> None:
        """验证混合数据集"""
        logger.info("验证混合数据集...")
        
        train_data = hybrid_data.get('train')
        valid_data = hybrid_data.get('valid')
        test_data = hybrid_data.get('test')
        
        # 检查训练数据
        if train_data is not None and not train_data.empty:
            logger.info(f"训练集: {train_data.shape[0]} 行, {train_data['code'].nunique()} 只股票")
            if 'date' in train_data.columns:
                logger.info(f"  时间范围: {train_data['date'].min().date()} 到 {train_data['date'].max().date()}")
        
        # 检查验证数据
        if valid_data is not None and not valid_data.empty:
            logger.info(f"验证集: {valid_data.shape[0]} 行, {valid_data['code'].nunique()} 只股票")
            if 'date' in valid_data.columns:
                logger.info(f"  时间范围: {valid_data['date'].min().date()} 到 {valid_data['date'].max().date()}")
        
        # 检查测试数据
        if test_data is not None and not test_data.empty:
            logger.info(f"测试集: {test_data.shape[0]} 行, {test_data['code'].nunique()} 只股票")
            if 'date' in test_data.columns:
                logger.info(f"  时间范围: {test_data['date'].min().date()} 到 {test_data['date'].max().date()}")
        
        # 检查时间范围不重叠
        if (train_data is not None and not train_data.empty and 
            test_data is not None and not test_data.empty):
            
            train_max_date = train_data['date'].max()
            test_min_date = test_data['date'].min()
            
            if train_max_date >= test_min_date:
                logger.warning(f"训练集和测试集时间范围有重叠: 训练集结束于 {train_max_date.date()}, 测试集开始于 {test_min_date.date()}")
            else:
                logger.info(f"训练集和测试集时间范围无重叠: 训练集结束于 {train_max_date.date()}, 测试集开始于 {test_min_date.date()}")
        
        logger.info("混合数据集验证通过")
    
    def _save_hybrid_data(self, hybrid_data: Dict[str, pd.DataFrame]) -> None:
        """保存混合数据集"""
        logger.info("保存混合数据集...")
        
        # 生成版本标识
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for split_name, data in hybrid_data.items():
            if data.empty:
                logger.warning(f"{split_name}数据集为空，跳过保存")
                continue
            
            # 保存为CSV
            csv_path = self.output_dir / f"hybrid_{split_name}_data_v{version}.csv"
            data.to_csv(csv_path, index=False)
            logger.info(f"保存CSV文件: {csv_path}")
            
            # 保存为Parquet（如果安装了pyarrow）
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                
                parquet_path = self.output_dir / f"hybrid_{split_name}_data_v{version}.parquet"
                data.to_parquet(parquet_path, index=False)
                logger.info(f"保存Parquet文件: {parquet_path}")
            except ImportError:
                logger.warning("未安装pyarrow，跳过Parquet格式保存")
        
        # 保存数据统计信息
        self._save_hybrid_statistics(hybrid_data, version)
        
        logger.info(f"混合数据保存完成，版本: {version}")
    
    def _save_hybrid_statistics(self, hybrid_data: Dict[str, pd.DataFrame], version: str) -> None:
        """保存混合数据统计信息"""
        stats = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'data_sources': {
                'train_val': 'Qlib (2005-2020)',
                'test': 'Local CSV (2021-2026)'
            },
            'datasets': {}
        }
        
        for split_name, data in hybrid_data.items():
            if data.empty:
                stats['datasets'][split_name] = {"rows": 0, "stocks": 0, "days": 0}
                continue
            
            stats['datasets'][split_name] = {
                "rows": data.shape[0],
                "stocks": data['code'].nunique(),
                "days": data['date'].nunique(),
                "date_range": {
                    "start": data['date'].min().strftime("%Y-%m-%d"),
                    "end": data['date'].max().strftime("%Y-%m-%d")
                }
            }
        
        # 保存统计信息为YAML
        stats_path = self.output_dir / f"hybrid_statistics_v{version}.yaml"
        with open(stats_path, 'w', encoding='utf-8') as f:
            yaml.dump(stats, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"混合数据统计信息保存到: {stats_path}")
    
    def generate_hybrid_report(self, hybrid_data: Dict[str, pd.DataFrame]) -> str:
        """
        生成混合数据报告
        
        Args:
            hybrid_data: 混合数据集
            
        Returns:
            str: 混合数据报告
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("混合数据报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"数据模式: Qlib训练验证 + 本地CSV测试")
        report_lines.append("")
        
        report_lines.append("数据源配置:")
        report_lines.append(f"  - 训练验证: Qlib数据 (2005-2020)")
        report_lines.append(f"  - 测试: 本地CSV数据 (2021-2026)")
        report_lines.append("")
        
        for split_name, data in hybrid_data.items():
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
            
            report_lines.append("")
        
        # 总体统计
        total_rows = sum(data.shape[0] for data in hybrid_data.values())
        total_stocks = set()
        for data in hybrid_data.values():
            total_stocks.update(data['code'].unique())
        
        report_lines.append("总体统计:")
        report_lines.append(f"  - 总数据行数: {total_rows:,}")
        report_lines.append(f"  - 总股票数量: {len(total_stocks):,}")
        report_lines.append(f"  - 数据集划分: {', '.join(hybrid_data.keys())}")
        
        report = "\n".join(report_lines)
        logger.info("混合数据报告生成完成")
        
        return report


def main():
    """主函数，用于测试和演示"""
    try:
        print("=" * 80)
        print("混合数据处理器测试")
        print("=" * 80)
        
        # 初始化混合数据处理器
        processor = HybridDataProcessor()
        
        # 处理混合数据
        hybrid_data = processor.process_hybrid_data()
        
        # 生成混合数据报告
        report = processor.generate_hybrid_report(hybrid_data)
        print(report)
        
        # 保存报告到文件
        report_path = processor.output_dir / "hybrid_data_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n混合数据报告已保存到: {report_path}")
        
        print("\n" + "=" * 80)
        print("混合数据处理完成！")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"主函数执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())