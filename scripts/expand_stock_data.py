#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩展股票数据：加载1000只股票数据

功能：
1. 获取A股股票列表
2. 分批加载1000只股票数据
3. 保存扩展后的数据集

作者: FZT项目组
创建日期: 2026-03-02
"""

import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import time

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

# 添加项目路径
sys.path.insert(0, str(Path.cwd()))


class StockDataExpander:
    """扩展股票数据加载器"""
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """初始化扩展器"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.qlib_config = self.config['data']['qlib']
        
        # 获取时间范围
        self.data_start = self.qlib_config.get('data_start', '2005-01-01')
        self.data_end = self.qlib_config.get('data_end', '2020-12-31')
        
        # 输出目录
        self.results_dir = Path("results/expanded_stock_data")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("股票数据扩展器初始化完成")
        logger.info(f"数据范围: {self.data_start} 到 {self.data_end}")
        logger.info(f"目标股票数量: 1000只")
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_stock_list(self, max_stocks: int = 1000) -> List[str]:
        """获取A股股票列表"""
        logger.info(f"获取A股股票列表，目标: {max_stocks}只")
        
        try:
            # 导入Qlib
            import qlib
            from qlib.config import REG_CN
            from qlib.data import D
            
            # 初始化Qlib
            provider_uri = Path(self.qlib_config['provider_uri']).expanduser()
            logger.info(f"初始化Qlib: {provider_uri}")
            
            qlib.init(provider_uri=str(provider_uri), region=REG_CN)
            logger.info("✅ Qlib初始化成功")
            
            # 方法1: 尝试从Qlib获取股票列表
            try:
                from qlib.data import instruments as inst
                
                # 获取所有股票
                all_stocks = inst.list_instruments(
                    instruments='all',
                    start_time=self.data_start,
                    end_time=self.data_end,
                    as_list=True
                )
                
                logger.info(f"从Qlib获取到 {len(all_stocks)} 只股票")
                
                # 限制数量
                if len(all_stocks) > max_stocks:
                    selected_stocks = all_stocks[:max_stocks]
                    logger.info(f"选择前 {max_stocks} 只股票")
                else:
                    selected_stocks = all_stocks
                    logger.info(f"使用全部 {len(all_stocks)} 只股票")
                
                return selected_stocks
                
            except Exception as e:
                logger.warning(f"无法从Qlib获取股票列表: {e}")
                
                # 方法2: 使用预定义的股票代码
                logger.info("使用预定义的A股股票代码")
                
                # 常见A股股票代码（示例）
                common_stocks = []
                
                # 上证50成分股（部分）
                sh50 = [f'SH600{i:03d}' for i in range(1, 51)]
                common_stocks.extend(sh50)
                
                # 深证100成分股（部分）
                sz100 = [f'SZ000{i:03d}' for i in range(1, 101)]
                common_stocks.extend(sz100)
                
                # 创业板（部分）
                cyb = [f'SZ300{i:03d}' for i in range(1, 101)]
                common_stocks.extend(cyb)
                
                # 限制数量
                if len(common_stocks) > max_stocks:
                    selected_stocks = common_stocks[:max_stocks]
                else:
                    selected_stocks = common_stocks
                
                logger.info(f"使用预定义股票列表: {len(selected_stocks)} 只")
                return selected_stocks
                
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            
            # 方法3: 使用硬编码的股票列表
            logger.info("使用硬编码的股票列表")
            
            # 生成一些示例股票代码
            hardcoded_stocks = []
            
            # 上证股票
            for i in range(1, 501):  # 500只上证股票
                hardcoded_stocks.append(f'SH600{i:03d}')
            
            # 深证股票
            for i in range(1, 501):  # 500只深证股票
                hardcoded_stocks.append(f'SZ000{i:03d}')
            
            # 限制数量
            selected_stocks = hardcoded_stocks[:max_stocks]
            logger.info(f"使用硬编码股票列表: {len(selected_stocks)} 只")
            
            return selected_stocks
    
    def load_stock_data_batch(self, stock_list: List[str], 
                            batch_size: int = 50) -> pd.DataFrame:
        """批量加载股票数据"""
        logger.info(f"批量加载股票数据，每批 {batch_size} 只")
        
        try:
            import qlib
            from qlib.config import REG_CN
            from qlib.data import D
            
            # 初始化Qlib
            provider_uri = Path(self.qlib_config['provider_uri']).expanduser()
            qlib.init(provider_uri=str(provider_uri), region=REG_CN)
            
            all_data = []
            total_stocks = len(stock_list)
            
            # 分批处理
            for batch_start in range(0, total_stocks, batch_size):
                batch_end = min(batch_start + batch_size, total_stocks)
                batch_stocks = stock_list[batch_start:batch_end]
                
                logger.info(f"加载批次 {batch_start//batch_size + 1}/{(total_stocks + batch_size - 1)//batch_size}: "
                          f"股票 {batch_start+1}-{batch_end}")
                
                batch_data = []
                
                # 逐只股票加载（避免内存问题）
                for i, stock in enumerate(batch_stocks, 1):
                    try:
                        # 加载单只股票数据
                        stock_data = D.features(
                            [stock], 
                            ['$open', '$high', '$low', '$close', '$volume'],
                            self.data_start, 
                            self.data_end
                        )
                        
                        if not stock_data.empty:
                            # 转换为标准格式
                            stock_df = self._convert_qlib_format(stock_data, stock)
                            batch_data.append(stock_df)
                            
                            if i % 10 == 0 or i == len(batch_stocks):
                                logger.info(f"    {i}/{len(batch_stocks)}: {stock} ✅")
                        else:
                            logger.debug(f"    {i}/{len(batch_stocks)}: {stock} ⚠️ 数据为空")
                            
                    except Exception as e:
                        logger.debug(f"    {i}/{len(batch_stocks)}: {stock} ❌ {e}")
                        continue
                
                if batch_data:
                    # 合并批次数据
                    batch_combined = pd.concat(batch_data, ignore_index=True)
                    all_data.append(batch_combined)
                    
                    logger.info(f"    批次完成: {len(batch_combined)} 行数据")
                
                # 批次间暂停，避免资源竞争
                time.sleep(1)
            
            if not all_data:
                logger.error("所有批次数据加载失败")
                return pd.DataFrame()
            
            # 合并所有数据
            combined_data = pd.concat(all_data, ignore_index=True)
            
            logger.info(f"✅ 批量数据加载完成")
            logger.info(f"总数据行数: {combined_data.shape[0]}")
            logger.info(f"股票数量: {combined_data['code'].nunique()}")
            logger.info(f"日期范围: {combined_data['date'].min()} 到 {combined_data['date'].max()}")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"批量加载数据失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _convert_qlib_format(self, qlib_data: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """转换Qlib数据格式为标准格式"""
        try:
            if qlib_data.empty:
                return pd.DataFrame()
            
            # 重置索引
            qlib_data = qlib_data.reset_index()
            
            # 重命名列
            qlib_data = qlib_data.rename(columns={
                'datetime': 'date'
            })
            
            # 添加股票代码
            qlib_data['code'] = stock_code
            
            # 设置日期格式
            qlib_data['date'] = pd.to_datetime(qlib_data['date'])
            
            # 重命名字段列
            column_mapping = {}
            for col in qlib_data.columns:
                if col.startswith('$'):
                    # 移除$前缀，转换为小写
                    new_col = col[1:].lower()
                    column_mapping[col] = new_col
            
            qlib_data = qlib_data.rename(columns=column_mapping)
            
            # 选择需要的列
            required_cols = ['date', 'code', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in qlib_data.columns]
            
            return qlib_data[available_cols]
            
        except Exception as e:
            logger.warning(f"转换股票 {stock_code} 数据格式失败: {e}")
            return pd.DataFrame()
    
    def calculate_t1_target(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算T+1目标变量（第二天上涨）"""
        logger.info("计算T+1目标变量...")
        
        try:
            # 按股票分组处理
            data_with_target = []
            
            for code, group in data.groupby('code'):
                group = group.sort_values('date')
                
                # 计算T+1收益率
                group['next_close'] = group['close'].shift(-1)
                group['future_return'] = (group['next_close'] / group['close'] - 1)
                
                # 计算T+1目标：第二天上涨为1，否则为0
                group['target'] = (group['future_return'] > 0).astype(int)
                
                # 移除最后一行（没有未来数据）
                group = group.dropna(subset=['target'])
                
                data_with_target.append(group)
            
            combined = pd.concat(data_with_target, ignore_index=True)
            
            # 统计信息
            logger.info(f"✅ T+1目标变量计算完成")
            logger.info(f"   正类比例（上涨）: {combined['target'].mean():.2%}")
            logger.info(f"   负类比例（下跌）: {1 - combined['target'].mean():.2%}")
            logger.info(f"   有效样本数: {len(combined)}")
            logger.info(f"   股票数量: {combined['code'].nunique()}")
            
            return combined
            
        except Exception as e:
            logger.error(f"计算T+1目标变量失败: {e}")
            raise
    
    def run(self):
        """运行扩展数据流程"""
        print("=" * 100)
        print("📈 扩展股票数据：加载1000只股票")
        print("=" * 100)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        try:
            # 1. 获取股票列表
            print("1. 📋 获取A股股票列表")
            print("-" * 50)
            stock_list = self.get_stock_list(max_stocks=1000)
            
            if not stock_list:
                print("❌ 无法获取股票列表")
                return 1
            
            print(f"✅ 获取到 {len(stock_list)} 只股票")
            print(f"   示例股票: {stock_list[:5]}...")
            print()
            
            # 保存股票列表
            stock_list_path = self.results_dir / "stock_list.txt"
            with open(stock_list_path, 'w', encoding='utf-8') as f:
                for stock in stock_list:
                    f.write(f"{stock}\n")
            print(f"股票列表保存到: {stock_list_path}")
            print()
            
            # 2. 批量加载数据
            print("2. 📦 批量加载股票数据")
            print("-" * 50)
            print(f"注意：加载1000只股票数据可能需要一些时间")
            print(f"数据范围: {self.data_start} 到 {self.data_end}")
            print()
            
            data = self.load_stock_data_batch(stock_list, batch_size=50)
            
            if data.empty:
                print("❌ 数据加载失败")
                return 1
            
            print(f"✅ 数据加载成功")
            print(f"   总数据行数: {data.shape[0]:,}")
            print(f"   股票数量: {data['code'].nunique()}")
            print(f"   日期范围: {data['date'].min()} 到 {data['date'].max()}")
            print()
            
            # 保存原始数据
            raw_data_path = self.results_dir / "expanded_raw_data.csv"
            data.to_csv(raw_data_path, index=False)
            print(f"原始数据保存到: {raw_data_path}")
            print()
            
            # 3. 计算T+1目标
            print("3. 🎯 计算T+1目标变量")
            print("-" * 50)
            data_with_target = self.calculate_t1_target(data)
            
            if data_with_target.empty:
                print("❌ 目标变量计算失败")
                return 1
            
            print(f"✅ T+1目标计算完成")
            print(f"   正类比例: {data_with_target['target'].mean():.2%}")
            print()
            
            # 保存带目标的数据
            target_data_path = self.results_dir / "expanded_data_with_target.csv"
            data_with_target.to_csv(target_data_path, index=False)
            print(f"带目标数据保存到: {target_data_path}")
            print()
            
            # 4. 数据统计
            print("4. 📊 数据统计信息")
            print("-" * 50)
            
            # 按股票统计
            stock_stats = data_with_target.groupby('code').agg({
                'date': ['min', 'max', 'count'],
                'target': 'mean'
            }).round(4)
            
            stock_stats.columns = ['start_date', 'end_date', 'sample_count', 'success_rate']
            stock_stats = stock_stats.reset_index()
            
            # 保存统计信息
            stats_path = self.results_dir / "stock_statistics.csv"
            stock_stats.to_csv(stats_path, index=False)
            
            print(f"数据统计:")
            print(f"   总样本数: {len(data_with_target):,}")
            print(f"   股票数量: {data_with_target['code'].nunique()}")
            print(f"   平均每只股票样本数: {len(data_with_target) // data_with_target['code'].nunique():,}")
            print(f"   整体上涨概率: {data_with_target['target'].mean():.2%}")
            print()
            print(f"统计信息保存到: {stats_path}")
            print()
            
            print("=" * 100)
            print("🎉 股票数据扩展完成！")
            print("=" * 100)
            print(f"\n📁 生成的文件:")
            print(f"   {stock_list_path} - 股票列表")
            print(f"   {raw_data_path} - 原始数据")
            print(f"   {target_data_path} - 带目标数据")
            print(f"   {stats_path} - 统计信息")
            print()
            print("🚀 下一步:")
            print("   1. 使用扩展数据训练FZT模型")
            print("   2. 实现因子发现功能")
            print("   3. 优化模型参数提升成功率")
            
            return 0
            
        except Exception as e:
            logger.error(f"扩展数据流程失败: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    expander = StockDataExpander()
    sys.exit(expander.run())