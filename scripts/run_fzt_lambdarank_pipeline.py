#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行FZT-LambdaRank整合流水线

实现方案3：FZT预筛选 + 特征增强 + LambdaRank排序
完整流程：数据准备 → 训练 → 回测 → 评估

作者：FZT项目组
创建日期：2026年3月2日
"""

import sys
import os
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
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


def load_data(data_path: str) -> pd.DataFrame:
    """加载数据"""
    logger.info(f"📥 加载数据: {data_path}")
    
    try:
        data = pd.read_csv(data_path)
        
        logger.info(f"✅ 数据加载成功")
        logger.info(f"   数据形状: {data.shape}")
        logger.info(f"   数据列: {list(data.columns)}")
        logger.info(f"   日期范围: {data['date'].min()} 到 {data['date'].max()}")
        logger.info(f"   股票数量: {data['code'].nunique()}")
        
        # 确保日期格式
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        
        return data
        
    except Exception as e:
        logger.error(f"❌ 数据加载失败: {e}")
        return pd.DataFrame()


def prepare_training_data(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """准备训练数据"""
    logger.info("🔧 准备训练数据...")
    
    try:
        # 过滤训练期间数据
        train_start = config['data']['train_start']
        train_end = config['data']['train_end']
        
        train_data = data[
            (data['date'] >= train_start) & 
            (data['date'] <= train_end)
        ].copy()
        
        logger.info(f"📊 训练数据:")
        logger.info(f"   期间: {train_start} 到 {train_end}")
        logger.info(f"   样本数: {len(train_data)}")
        logger.info(f"   交易日数: {train_data['date'].nunique()}")
        
        return train_data
        
    except Exception as e:
        logger.error(f"❌ 训练数据准备失败: {e}")
        return pd.DataFrame()


def prepare_test_data(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """准备测试数据"""
    logger.info("🔧 准备测试数据...")
    
    try:
        # 过滤测试期间数据
        test_start = config['data']['test_start']
        test_end = config['data']['test_end']
        
        test_data = data[
            (data['date'] >= test_start) & 
            (data['date'] <= test_end)
        ].copy()
        
        logger.info(f"📊 测试数据:")
        logger.info(f"   期间: {test_start} 到 {test_end}")
        logger.info(f"   样本数: {len(test_data)}")
        logger.info(f"   交易日数: {test_data['date'].nunique()}")
        
        return test_data
        
    except Exception as e:
        logger.error(f"❌ 测试数据准备失败: {e}")
        return pd.DataFrame()


def run_training(pipeline, train_data: pd.DataFrame) -> bool:
    """运行训练"""
    logger.info("🚀 开始训练...")
    
    try:
        success = pipeline.run_training_pipeline(train_data)
        
        if success:
            logger.info("✅ 训练完成")
            return True
        else:
            logger.error("❌ 训练失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ 训练过程失败: {e}")
        return False


def run_backtest(pipeline, test_data: pd.DataFrame, config: dict) -> dict:
    """运行回测"""
    logger.info("📈 开始回测...")
    
    try:
        backtest_result = pipeline.run_backtest_pipeline(
            test_data,
            start_date=config['data']['test_start'],
            end_date=config['data']['test_end']
        )
        
        return backtest_result
        
    except Exception as e:
        logger.error(f"❌ 回测过程失败: {e}")
        return {'success': False, 'error': str(e)}


def generate_report(backtest_result: dict, config: dict) -> dict:
    """生成报告"""
    logger.info("📋 生成报告...")
    
    try:
        if not backtest_result.get('success', False):
            return {
                'success': False,
                'error': backtest_result.get('error', 'Unknown error'),
                'report_generated': False
            }
        
        # 提取关键指标
        period = backtest_result.get('period', {})
        performance = backtest_result.get('performance', {})
        selection_stats = backtest_result.get('selection_stats', {})
        
        # 生成详细报告
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'pipeline_version': '1.0',
                'strategy': 'FZT预筛选 + LambdaRank排序',
                'config_summary': {
                    'fzt_prescreen': config.get('fzt_prescreen', {}).get('enabled', True),
                    'top_k': config.get('selection', {}).get('top_k', 5),
                    'lookahead_days': config.get('data', {}).get('lookahead_days', 1)
                }
            },
            'period_info': {
                'start_date': period.get('start_date'),
                'end_date': period.get('end_date'),
                'trading_days': period.get('trading_days', 0),
                'successful_days': period.get('successful_days', 0),
                'success_rate': period.get('successful_days', 0) / period.get('trading_days', 1) if period.get('trading_days', 0) > 0 else 0
            },
            'performance_metrics': {
                'initial_capital': performance.get('initial_value', 1000000),
                'final_value': performance.get('final_value', 1000000),
                'total_return': performance.get('total_return', 0),
                'annual_return': performance.get('annual_return', 0),
                'daily_success_rate': performance.get('daily_success_rate', 0)
            },
            'selection_metrics': {
                'avg_screened_per_day': selection_stats.get('avg_screened_per_day', 0),
                'avg_selected_per_day': selection_stats.get('avg_selected_per_day', 0),
                'total_selections': selection_stats.get('total_selections', 0),
                'selection_ratio': selection_stats.get('avg_selected_per_day', 0) / selection_stats.get('avg_screened_per_day', 1) if selection_stats.get('avg_screened_per_day', 0) > 0 else 0
            },
            'interpretation': {
                'performance_rating': '优秀' if performance.get('annual_return', 0) > 0.15 else '良好' if performance.get('annual_return', 0) > 0 else '需改进',
                'selection_efficiency': '高效' if selection_stats.get('avg_selected_per_day', 0) >= 4 else '中等' if selection_stats.get('avg_selected_per_day', 0) >= 2 else '较低',
                'recommendations': [
                    "策略表现稳定，可继续使用",
                    "建议定期重新训练模型以适应市场变化",
                    "考虑增加风险控制模块"
                ]
            }
        }
        
        logger.info("✅ 报告生成完成")
        return report
        
    except Exception as e:
        logger.error(f"❌ 报告生成失败: {e}")
        return {'success': False, 'error': str(e), 'report_generated': False}


def save_results(pipeline, backtest_result: dict, report: dict, output_dir: str):
    """保存结果"""
    logger.info("💾 保存结果...")
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 保存回测结果
        backtest_path = output_path / f"backtest_result_{timestamp}.yaml"
        import yaml
        with open(backtest_path, 'w', encoding='utf-8') as f:
            yaml.dump(backtest_result, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"📊 回测结果保存到: {backtest_path}")
        
        # 2. 保存报告
        report_path = output_path / f"pipeline_report_{timestamp}.yaml"
        with open(report_path, 'w', encoding='utf-8') as f:
            yaml.dump(report, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"📋 报告保存到: {report_path}")
        
        # 3. 保存摘要
        summary = {
            'timestamp': timestamp,
            'total_return': report.get('performance_metrics', {}).get('total_return', 0),
            'annual_return': report.get('performance_metrics', {}).get('annual_return', 0),
            'success_rate': report.get('period_info', {}).get('success_rate', 0),
            'avg_selected': report.get('selection_metrics', {}).get('avg_selected_per_day', 0),
            'performance_rating': report.get('interpretation', {}).get('performance_rating', '未知')
        }
        
        summary_path = output_path / f"summary_{timestamp}.yaml"
        with open(summary_path, 'w', encoding='utf-8') as f:
            yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"📝 摘要保存到: {summary_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 结果保存失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行FZT-LambdaRank整合流水线')
    parser.add_argument('--data', type=str, required=True,
                       help='数据文件路径')
    parser.add_argument('--config', type=str, default='config/pipeline_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['train', 'backtest', 'full'],
                       help='运行模式: train(仅训练), backtest(仅回测), full(完整流程)')
    parser.add_argument('--output', type=str, default='results/pipeline_results',
                       help='输出目录')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出模式')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("🎯 FZT-LambdaRank整合流水线启动")
    logger.info("=" * 60)
    logger.info(f"模式: {args.mode}")
    logger.info(f"数据: {args.data}")
    logger.info(f"配置: {args.config}")
    logger.info(f"输出: {args.output}")
    
    try:
        # 1. 加载配置
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("✅ 配置加载成功")
        
        # 2. 加载数据
        data = load_data(args.data)
        if data.empty:
            logger.error("❌ 数据加载失败，退出")
            return 1
        
        # 3. 创建流水线
        from src.integration.fzt_lambdarank_pipeline import create_pipeline
        pipeline = create_pipeline(args.config)
        
        # 4. 根据模式运行
        if args.mode in ['train', 'full']:
            # 准备训练数据
            train_data = prepare_training_data(data, config)
            if train_data.empty:
                logger.error("❌ 训练数据准备失败")
                return 1
            
            # 运行训练
            if not run_training(pipeline, train_data):
                logger.error("❌ 训练失败，退出")
                return 1
        
        if args.mode in ['backtest', 'full']:
            # 准备测试数据
            test_data = prepare_test_data(data, config)
            if test_data.empty:
                logger.error("❌ 测试数据准备失败")
                return 1
            
            # 运行回测
            backtest_result = run_backtest(pipeline, test_data, config)
            
            if not backtest_result.get('success', False):
                logger.error(f"❌ 回测失败: {backtest_result.get('error', 'Unknown error')}")
                return 1
            
            # 生成报告
            report = generate_report(backtest_result, config)
            
            if not report.get('success', False):
                logger.error(f"❌ 报告生成失败: {report.get('error', 'Unknown error')}")
                return 1
            
            # 保存结果
            if not save_results(pipeline, backtest_result, report, args.output):
                logger.error("❌ 结果保存失败")
                return 1
            
            # 打印摘要
            print("\n" + "="*60)
            print("🎉 FZT-LambdaRank整合流水线完成！")
            print("="*60)
            print(f"📊 策略表现摘要:")
            print(f"  总收益率: {report['performance_metrics']['total_return']:.2%}")
            print(f"  年化收益率: {report['performance_metrics']['annual_return']:.2%}")
            print(f"  交易日成功率: {report['period_info']['success_rate']:.2%}")
            print(f"  平均每日选股: {report['selection_metrics']['avg_selected_per_day']:.1f} 只")
            print(f"  表现评级: {report['interpretation']['performance_rating']}")
            print(f"\n📁 结果文件:")
            print(f"  回测结果: {args.output}/backtest_result_*.yaml")
            print(f"  详细报告: {args.output}/pipeline_report_*.yaml")
            print(f"  策略摘要: {args.output}/summary_*.yaml")
            print("="*60)
        
        logger.info("🎉 流水线执行完成")
        return 0
        
    except Exception as e:
        logger.error(f"❌ 流水线执行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())