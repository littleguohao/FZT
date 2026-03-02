"""
风险控制器使用示例
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtest.risk_controller import RiskController


def create_sample_data():
    """创建示例数据"""
    # 创建日期范围
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # 股票列表
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ']
    
    # 生成价格数据
    np.random.seed(42)
    price_data = np.random.randn(len(dates), len(stocks)) * 0.02 + 1.0
    price_data = np.cumprod(price_data, axis=0) * 10
    
    market_data = pd.DataFrame(
        price_data,
        index=dates,
        columns=stocks
    )
    
    # 生成成交量数据
    volume_data = np.random.randint(1000000, 10000000, size=(len(dates), len(stocks)))
    volume_data = pd.DataFrame(
        volume_data,
        index=dates,
        columns=stocks
    )
    
    # 创建投资组合
    portfolio = {
        'positions': {
            '000001.SZ': {
                'weight': 0.25,
                'shares': 2500,
                'cost': 10.0,
                'market_value': 250000
            },
            '000002.SZ': {
                'weight': 0.20,
                'shares': 1667,
                'cost': 12.0,
                'market_value': 200000
            },
            '000003.SZ': {
                'weight': 0.15,
                'shares': 1000,
                'cost': 15.0,
                'market_value': 150000
            },
            '000004.SZ': {
                'weight': 0.10,
                'shares': 556,
                'cost': 18.0,
                'market_value': 100000
            },
            '000005.SZ': {
                'weight': 0.05,
                'shares': 250,
                'cost': 20.0,
                'market_value': 50000
            }
        },
        'total_value': 1000000.0,
        'cash': 250000.0,
        'pnl': 50000.0
    }
    
    # 财务数据
    financial_data = {
        '000001.SZ': {
            'is_st': False,
            'current_ratio': 1.8,
            'debt_ratio': 0.35,
            'credit_rating': 'AA'
        },
        '000002.SZ': {
            'is_st': False,
            'current_ratio': 0.9,
            'debt_ratio': 0.75,
            'credit_rating': 'BB'
        },
        '000003.SZ': {
            'is_st': True,  # ST股票
            'current_ratio': 0.6,
            'debt_ratio': 0.85,
            'credit_rating': 'C'
        },
        '000004.SZ': {
            'is_st': False,
            'current_ratio': 2.2,
            'debt_ratio': 0.25,
            'credit_rating': 'AAA'
        },
        '000005.SZ': {
            'is_st': False,
            'current_ratio': 1.5,
            'debt_ratio': 0.45,
            'credit_rating': 'A'
        }
    }
    
    # 风险因子数据
    risk_factors = {
        'industry': {
            '000001.SZ': '金融',
            '000002.SZ': '房地产',
            '000003.SZ': '金融',
            '000004.SZ': '科技',
            '000005.SZ': '消费'
        },
        'style': {
            '000001.SZ': {'value': 0.4, 'growth': 0.6},
            '000002.SZ': {'value': 0.8, 'growth': 0.2},
            '000003.SZ': {'value': 0.6, 'growth': 0.4},
            '000004.SZ': {'value': 0.2, 'growth': 0.8},
            '000005.SZ': {'value': 0.7, 'growth': 0.3}
        }
    }
    
    return {
        'market_data': market_data,
        'volume_data': volume_data,
        'portfolio': portfolio,
        'financial_data': financial_data,
        'risk_factors': risk_factors
    }


def main():
    """主函数"""
    print("=" * 60)
    print("风险控制器演示")
    print("=" * 60)
    
    # 创建配置
    config = {
        'risk': {
            'market': {
                'max_volatility': 0.3,
                'min_correlation': 0.3,
                'max_beta': 1.5
            },
            'credit': {
                'filter_st': True,
                'min_current_ratio': 1.0,
                'max_debt_ratio': 0.7
            },
            'liquidity': {
                'min_turnover': 10000000,
                'max_bid_ask_spread': 0.02
            },
            'concentration': {
                'max_single_weight': 0.4,
                'max_industry_weight': 0.3,
                'max_style_exposure': 0.5
            },
            'stop_loss': {
                'enabled': True,
                'threshold': -0.1,
                'trailing': True
            }
        }
    }
    
    # 创建风险控制器
    print("\n1. 初始化风险控制器...")
    risk_controller = RiskController(config)
    print("   风险控制器初始化完成")
    
    # 创建示例数据
    print("\n2. 创建示例数据...")
    data = create_sample_data()
    print(f"   市场数据: {len(data['market_data'])} 天 × {len(data['market_data'].columns)} 只股票")
    print(f"   投资组合: {len(data['portfolio']['positions'])} 只股票")
    
    # 执行风险评估
    print("\n3. 执行风险评估...")
    risk_report = risk_controller.assess_risk(
        portfolio=data['portfolio'],
        market_data=data['market_data'],
        volume_data=data['volume_data'],
        financial_data=data['financial_data'],
        risk_factors=data['risk_factors']
    )
    
    print(f"   风险评估完成，生成 {len(risk_report['warnings'])} 个警告")
    
    # 显示风险摘要
    print("\n4. 风险摘要:")
    summary = risk_controller.get_risk_summary()
    print(f"   总警告数: {summary['total_warnings']}")
    print(f"   高风险警告: {summary['high_risk_warnings']}")
    print(f"   中风险警告: {summary['medium_risk_warnings']}")
    print(f"   低风险警告: {summary['low_risk_warnings']}")
    
    # 显示高风险警告
    high_warnings = risk_controller.get_warnings_by_severity('HIGH')
    if high_warnings:
        print("\n5. 高风险警告:")
        for i, warning in enumerate(high_warnings[:3], 1):  # 显示前3个
            print(f"   {i}. [{warning['category']}] {warning['message']}")
    
    # 显示风险指标
    print("\n6. 风险指标:")
    metrics = risk_report['risk_metrics']
    if metrics:
        print(f"   95% VaR: {metrics.get('var_95', 0):.2%}")
        print(f"   95% CVaR: {metrics.get('cvar_95', 0):.2%}")
        print(f"   最大回撤: {metrics.get('max_drawdown', 0):.2%}")
        print(f"   夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   索提诺比率: {metrics.get('sortino_ratio', 0):.2f}")
        print(f"   年化波动率: {metrics.get('annual_volatility', 0):.2%}")
    
    # 显示建议
    print("\n7. 风险控制建议:")
    recommendations = risk_report['recommended_actions']
    if recommendations:
        for i, rec in enumerate(recommendations[:5], 1):  # 显示前5个
            action = rec.get('action', '')
            stock = rec.get('stock', '')
            reason = rec.get('reason', '')
            priority = rec.get('priority', '')
            print(f"   {i}. [{priority}] {action} {stock}: {reason}")
    else:
        print("   无风险控制建议")
    
    # 应用风险控制
    print("\n8. 应用风险控制...")
    if recommendations:
        adjusted_portfolio = risk_controller.apply_risk_controls(
            data['portfolio'],
            recommendations
        )
        
        # 检查ST股票是否被移除
        original_stocks = set(data['portfolio']['positions'].keys())
        adjusted_stocks = set(adjusted_portfolio.get('positions', {}).keys())
        removed_stocks = original_stocks - adjusted_stocks
        
        if removed_stocks:
            print(f"   已移除股票: {', '.join(removed_stocks)}")
        else:
            print("   未移除任何股票")
        
        print(f"   原始现金: {data['portfolio']['cash']:,.2f}")
        print(f"   调整后现金: {adjusted_portfolio.get('cash', 0):,.2f}")
    else:
        print("   无风险控制动作需要执行")
    
    # 保存风险报告
    print("\n9. 保存风险报告...")
    report_file = "risk_report_demo.json"
    risk_controller.save_risk_report(risk_report, report_file)
    print(f"   风险报告已保存到: {report_file}")
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()