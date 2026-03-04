#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩展股票数据 - 包含FZT特征计算

修复原扩展脚本的问题：
1. 使用真实的股票列表（不是硬编码）
2. 计算FZT砖型图特征
3. 保存完整特征集

作者：FZT项目组
创建日期：2026年3月2日
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FZTFeatureExpander:
    """FZT特征扩展器"""
    
    def __init__(self):
        """初始化"""
        self.results_dir = Path("results") / "fzt_expanded_features"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 时间范围
        self.start_date = "2005-01-01"
        self.end_date = "2020-12-31"
        
        logger.info("FZT特征扩展器初始化完成")
    
    def get_real_stock_list(self, max_stocks=500):
        """获取真实的股票列表"""
        logger.info(f"📋 获取真实的股票列表（目标: {max_stocks} 只）...")
        
        try:
            # 从优化后的股票列表读取
            optimized_path = Path("results/optimized_stocks/optimized_500_stocks.txt")
            
            if optimized_path.exists():
                with open(optimized_path, 'r', encoding='utf-8') as f:
                    stocks = [line.strip() for line in f if line.strip()]
                
                logger.info(f"✅ 从优化列表读取 {len(stocks)} 只股票")
                
                # 限制数量
                if len(stocks) > max_stocks:
                    stocks = stocks[:max_stocks]
                    logger.info(f"📊 限制为 {max_stocks} 只股票")
                
                return stocks
            else:
                logger.error(f"❌ 优化股票列表不存在: {optimized_path}")
                return []
                
        except Exception as e:
            logger.error(f"❌ 获取股票列表失败: {e}")
            return []
    
    def load_stock_data(self, stock_code):
        """加载单只股票数据"""
        try:
            import qlib
            from qlib.config import REG_CN
            from qlib.data import D
            
            # 初始化Qlib（每次调用都初始化以确保连接）
            qlib_data_dir = Path.home() / ".qlib" / "qlib_data" / "cn_data"
            qlib.init(
                provider_uri=str(qlib_data_dir),
                region=REG_CN,
                redis_port=-1,
                mongo=False,
                auto_mount=False
            )
            
            # 加载基础数据
            fields = ['$open', '$high', '$low', '$close', '$volume']
            df = D.features([stock_code], fields, self.start_date, self.end_date)
            
            if df.empty:
                logger.warning(f"  ⚠️  {stock_code}: 无数据")
                return None
            
            # 重命名列
            df = df.rename(columns={
                '$open': 'open',
                '$high': 'high', 
                '$low': 'low',
                '$close': 'close',
                '$volume': 'volume'
            })
            
            # 添加股票代码
            df['code'] = stock_code
            df['date'] = df.index
            
            # 重置索引
            df = df.reset_index(drop=True)
            
            logger.info(f"  ✅ {stock_code}: {len(df)} 行数据")
            return df
            
        except Exception as e:
            logger.warning(f"  ❌ {stock_code}: 加载失败 - {str(e)[:50]}...")
            return None
    
    def calculate_fzt_features(self, stock_data):
        """计算FZT特征"""
        try:
            from src.fzt_brick_formula import FZTBrickFormula
            
            if stock_data is None or stock_data.empty:
                return None
            
            # 确保数据按日期排序
            stock_data = stock_data.sort_values('date')
            
            # 初始化FZT计算器
            fzt_calculator = FZTBrickFormula()
            
            # 计算FZT特征
            fzt_features = fzt_calculator.generate_features(stock_data)
            
            if fzt_features.empty:
                logger.warning(f"  ⚠️  FZT特征为空")
                return None
            
            # 合并基础数据和FZT特征
            # 确保行数匹配
            if len(fzt_features) != len(stock_data):
                logger.warning(f"  ⚠️  特征行数不匹配: 基础数据={len(stock_data)}, FZT特征={len(fzt_features)}")
                # 对齐数据
                min_len = min(len(stock_data), len(fzt_features))
                stock_data = stock_data.iloc[:min_len]
                fzt_features = fzt_features.iloc[:min_len]
            
            # 合并数据
            combined = pd.concat([stock_data, fzt_features], axis=1)
            
            # 移除重复列
            combined = combined.loc[:, ~combined.columns.duplicated()]
            
            logger.info(f"  🔧 FZT特征: {fzt_features.shape[1]} 个特征")
            return combined
            
        except Exception as e:
            logger.warning(f"  ❌ FZT特征计算失败: {str(e)[:50]}...")
            return None
    
    def calculate_target_variable(self, data):
        """计算目标变量（T+1收益）"""
        if data is None or data.empty:
            return None
        
        try:
            # 确保有close列
            if 'close' not in data.columns:
                logger.warning("  ⚠️  无close列，无法计算目标变量")
                return data
            
            # 按股票分组计算
            data = data.copy()
            
            # 计算未来收盘价（T+1）
            data['next_close'] = data.groupby('code')['close'].shift(-1)
            
            # 计算未来收益率
            data['future_return'] = (data['next_close'] - data['close']) / data['close']
            
            # 创建二元目标变量（上涨=1，下跌=0）
            data['target'] = (data['future_return'] > 0).astype(int)
            
            # 移除最后一行（无未来数据）
            data = data.dropna(subset=['target'])
            
            logger.info(f"  🎯 目标变量计算完成")
            logger.info(f"     正类比例: {(data['target'] == 1).mean():.2%}")
            
            return data
            
        except Exception as e:
            logger.warning(f"  ❌ 目标变量计算失败: {e}")
            return data
    
    def expand_data_with_fzt_features(self, max_stocks=100):
        """扩展数据并计算FZT特征"""
        logger.info(f"🚀 开始扩展数据（包含FZT特征）...")
        
        # 1. 获取股票列表
        stock_list = self.get_real_stock_list(max_stocks=max_stocks)
        
        if not stock_list:
            logger.error("❌ 无股票列表")
            return False
        
        logger.info(f"📊 处理 {len(stock_list)} 只股票")
        
        all_data = []
        success_count = 0
        
        # 2. 分批处理股票
        for i, stock_code in enumerate(stock_list, 1):
            logger.info(f"[{i}/{len(stock_list)}] 处理股票: {stock_code}")
            
            try:
                # 加载基础数据
                stock_data = self.load_stock_data(stock_code)
                if stock_data is None:
                    continue
                
                # 计算FZT特征
                data_with_fzt = self.calculate_fzt_features(stock_data)
                if data_with_fzt is None:
                    # 如果没有FZT特征，使用基础数据
                    data_with_fzt = stock_data
                
                # 计算目标变量
                data_with_target = self.calculate_target_variable(data_with_fzt)
                if data_with_target is None:
                    continue
                
                # 添加到总数据
                all_data.append(data_with_target)
                success_count += 1
                
                logger.info(f"  ✅ 成功处理: {len(data_with_target)} 行, {data_with_target.shape[1]} 列")
                
            except Exception as e:
                logger.error(f"  ❌ 处理失败: {e}")
                continue
        
        if not all_data:
            logger.error("❌ 所有股票处理失败")
            return False
        
        # 3. 合并所有数据
        logger.info(f"📊 合并 {len(all_data)} 只股票的数据...")
        combined_data = pd.concat(all_data, ignore_index=True)
        
        logger.info(f"✅ 数据合并完成")
        logger.info(f"   总行数: {len(combined_data)}")
        logger.info(f"   总列数: {combined_data.shape[1]}")
        logger.info(f"   股票数量: {combined_data['code'].nunique()}")
        logger.info(f"   日期范围: {combined_data['date'].min()} 到 {combined_data['date'].max()}")
        
        # 4. 保存数据
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存完整数据
        full_data_path = self.results_dir / f"fzt_expanded_data_{timestamp}.csv"
        combined_data.to_csv(full_data_path, index=False)
        logger.info(f"💾 完整数据保存到: {full_data_path}")
        
        # 保存特征列表
        feature_cols = [col for col in combined_data.columns 
                       if col not in ['code', 'date', 'target', 'future_return', 'next_close']]
        
        features_path = self.results_dir / f"feature_list_{timestamp}.txt"
        with open(features_path, 'w', encoding='utf-8') as f:
            for feature in feature_cols:
                f.write(f"{feature}\n")
        
        logger.info(f"📋 特征列表保存到: {features_path}")
        logger.info(f"   特征数量: {len(feature_cols)}")
        
        # 5. 生成报告
        report = {
            'generated_at': datetime.now().isoformat(),
            'data_info': {
                'total_rows': len(combined_data),
                'total_columns': combined_data.shape[1],
                'stock_count': combined_data['code'].nunique(),
                'date_range': {
                    'start': str(combined_data['date'].min()),
                    'end': str(combined_data['date'].max())
                },
                'target_distribution': {
                    'positive_ratio': float((combined_data['target'] == 1).mean()),
                    'negative_ratio': float((combined_data['target'] == 0).mean())
                }
            },
            'processing_info': {
                'attempted_stocks': len(stock_list),
                'successful_stocks': success_count,
                'success_rate': success_count / len(stock_list) if stock_list else 0,
                'feature_count': len(feature_cols)
            },
            'files': {
                'full_data': str(full_data_path),
                'feature_list': str(features_path)
            }
        }
        
        import yaml
        report_path = self.results_dir / f"expansion_report_{timestamp}.yaml"
        with open(report_path, 'w', encoding='utf-8') as f:
            yaml.dump(report, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"📋 扩展报告保存到: {report_path}")
        
        return True
    
    def run(self, max_stocks=100):
        """运行扩展流程"""
        logger.info("🎯 FZT特征扩展流程开始")
        logger.info("=" * 60)
        
        success = self.expand_data_with_fzt_features(max_stocks=max_stocks)
        
        if success:
            print("\n" + "="*60)
            print("🎉 FZT特征扩展完成！")
            print("="*60)
            print(f"📊 扩展成果:")
            print(f"  • 包含FZT砖型图特征")
            print(f"  • 包含目标变量（T+1上涨预测）")
            print(f"  • 数据保存在: results/fzt_expanded_features/")
            print(f"\n🚀 下一步:")
            print(f"  1. 使用扩展数据重新训练FZT模型")
            print(f"  2. 预期获得更好的选股成功率")
            print(f"  3. 集成到LambdaRank排序框架")
            print("="*60)
        
        return success


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='扩展股票数据并计算FZT特征')
    parser.add_argument('--max-stocks', type=int, default=100,
                       help='最大股票数量（默认: 100）')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出模式')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    expander = FZTFeatureExpander()
    success = expander.run(max_stocks=args.max_stocks)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())