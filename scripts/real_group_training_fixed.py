#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实数据分组训练 - 修复版
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def run_real_group_training():
    """运行真实数据分组训练"""
    print("=" * 70)
    print("🔍 真实数据分组训练")
    print("=" * 70)
    print("🚫 100%真实数据，0%模拟数据")
    print("=" * 70)
    
    start_time = datetime.now()
    
    try:
        # 1. 检查QLIB数据
        print("\n🔧 检查QLIB真实数据...")
        
        qlib_data_dir = Path.home() / ".qlib" / "qlib_data" / "cn_data"
        if not qlib_data_dir.exists():
            print(f"❌ QLIB数据目录不存在: {qlib_data_dir}")
            print("请先下载QLIB数据")
            return False
        
        print(f"✅ QLIB真实数据目录: {qlib_data_dir}")
        
        # 2. 初始化QLIB
        print("\n🔧 初始化QLIB...")
        try:
            import qlib
            from qlib.config import REG_CN
            qlib.init(provider_uri=str(qlib_data_dir), region=REG_CN)
            print("✅ QLIB初始化成功")
        except Exception as e:
            print(f"❌ QLIB初始化失败: {e}")
            return False
        
        # 3. 加载真实股票数据
        print("\n📥 加载真实股票数据...")
        
        # 使用20只沪深300成分股（先小规模测试）
        real_stocks = [
            'SH600000', 'SH600036', 'SH600016', 'SH600030', 'SH601318',
            'SH600519', 'SH600887', 'SH600276', 'SH600436', 'SZ000858',
            'SZ000333', 'SZ000651', 'SZ002415', 'SZ300059', 'SZ300750',
            'SH600196', 'SH600085', 'SH600332', 'SZ000002', 'SZ000338'
        ]
        
        start_date = '2020-01-01'
        end_date = '2022-12-31'
        
        print(f"  股票: {len(real_stocks)} 只")
        print(f"  时间: {start_date} 到 {end_date}")
        
        # 4. 使用QLIB加载真实数据
        from qlib.data import D
        
        all_data = []
        success_count = 0
        
        for i, stock in enumerate(real_stocks, 1):
            try:
                # 加载基础数据
                fields = ['$close', '$volume', '$high', '$low', '$open', '$factor']
                stock_data = D.features([stock], fields, start_time=start_date, end_time=end_date)
                
                if stock_data.empty:
                    print(f"  ⚠️  {stock}: 无数据")
                    continue
                
                # 转换为DataFrame
                stock_df = stock_data.reset_index()
                stock_df['code'] = stock
                
                # 重命名列
                stock_df = stock_df.rename(columns={
                    '$close': 'close',
                    '$volume': 'volume',
                    '$high': 'high',
                    '$low': 'low',
                    '$open': 'open',
                    '$factor': 'factor'
                })
                
                stock_df['date'] = pd.to_datetime(stock_df['datetime'])
                
                all_data.append(stock_df)
                success_count += 1
                print(f"  ✅ {stock}: {len(stock_df)} 行数据")
                
            except Exception as e:
                print(f"  ❌ {stock}: 加载失败 - {str(e)[:50]}")
        
        if not all_data:
            print("❌ 没有成功加载任何股票数据")
            return False
        
        # 合并数据
        df = pd.concat(all_data, ignore_index=True)
        print(f"\n✅ 真实数据加载完成: {df.shape}")
        print(f"  成功加载: {success_count}/{len(real_stocks)} 只股票")
        print(f"  总样本: {len(df):,}")
        print(f"  日期范围: {df['date'].min().date()} 到 {df['date'].max().date()}")
        
        # 5. 计算真实技术因子
        print("\n🔧 计算真实技术因子...")
        
        # 按股票分组计算
        df = df.sort_values(['code', 'date'])
        
        # 计算复权价格
        df['adj_close'] = df['close'] * df['factor']
        
        all_stock_data = []
        
        for stock in df['code'].unique():
            stock_mask = df['code'] == stock
            stock_df = df[stock_mask].copy()
            
            if len(stock_df) < 30:
                continue
            
            # 第1组：收益率类因子
            stock_df['return_1d'] = stock_df['adj_close'].pct_change()
            stock_df['return_5d'] = stock_df['adj_close'].pct_change(5)
            stock_df['return_10d'] = stock_df['adj_close'].pct_change(10)
            
            # 第2组：移动平均类因子
            stock_df['ma5'] = stock_df['adj_close'].rolling(5).mean()
            stock_df['ma10'] = stock_df['adj_close'].rolling(10).mean()
            stock_df['ma20'] = stock_df['adj_close'].rolling(20).mean()
            stock_df['ma5_ratio'] = stock_df['adj_close'] / stock_df['ma5'] - 1
            
            # 第3组：成交量类因子
            stock_df['volume_ma5'] = stock_df['volume'].rolling(5).mean()
            stock_df['volume_ma10'] = stock_df['volume'].rolling(10).mean()
            stock_df['volume_ratio_5'] = stock_df['volume'] / stock_df['volume_ma5']
            
            # 第4组：波动率类因子
            stock_df['volatility_5d'] = stock_df['return_1d'].rolling(5).std()
            stock_df['volatility_10d'] = stock_df['return_1d'].rolling(10).std()
            
            # 第5组：价格位置类因子
            stock_df['high_20'] = stock_df['adj_close'].rolling(20).max()
            stock_df['low_20'] = stock_df['adj_close'].rolling(20).min()
            stock_df['price_position'] = (stock_df['adj_close'] - stock_df['low_20']) / (
                stock_df['high_20'] - stock_df['low_20'] + 1e-8)
            
            all_stock_data.append(stock_df)
        
        if not all_stock_data:
            print("❌ 没有足够的股票数据计算因子")
            return False
        
        df = pd.concat(all_stock_data, ignore_index=True)
        
        # 6. 定义因子分组
        print("\n📋 定义因子分组...")
        
        factor_groups = {
            'group1_returns': ['return_1d', 'return_5d', 'return_10d'],
            'group2_moving_averages': ['ma5', 'ma10', 'ma20', 'ma5_ratio'],
            'group3_volume': ['volume_ma5', 'volume_ma10', 'volume_ratio_5'],
            'group4_volatility': ['volatility_5d', 'volatility_10d'],
            'group5_price_position': ['high_20', 'low_20', 'price_position']
        }
        
        all_factors = []
        for group_name, factors in factor_groups.items():
            all_factors.extend(factors)
        
        print(f"✅ 定义 {len(all_factors)} 个真实技术因子")
        print(f"  分组: {len(factor_groups)} 组")
        
        # 7. 计算目标变量
        print("\n🎯 计算目标变量...")
        
        df = df.sort_values(['code', 'date'])
        df['target'] = df.groupby('code')['adj_close'].shift(-1).pct_change()
        
        # 移除无效数据
        df = df.dropna(subset=all_factors + ['target'])
        
        print(f"✅ 数据处理完成: {df.shape}")
        print(f"  有效样本: {len(df):,}")
        
        # 8. 分组计算IC和ICIR
        print("\n📈 分组计算真实IC和ICIR...")
        
        ic_results = []
        group_stats = {}
        
        for group_name, factors in factor_groups.items():
            print(f"\n🎯 分析组: {group_name}")
            
            group_ic = []
            
            for factor in factors:
                if factor not in df.columns:
                    continue
                
                # 按日期计算IC
                dates = df['date'].unique()
                daily_ic = []
                
                for date in dates:
                    date_data = df[df['date'] == date]
                    
                    if len(date_data) >= 5:
                        ic = date_data[factor].corr(date_data['target'])
                        if not np.isnan(ic):
                            daily_ic.append(ic)
                
                if daily_ic and len(daily_ic) >= 10:
                    mean_ic = np.mean(daily_ic)
                    ic_std = np.std(daily_ic)
                    icir = mean_ic / ic_std if ic_std > 0 else 0
                    
                    # 因子方向
                    direction = "正向" if mean_ic > 0 else "负向"
                    
                    # 因子质量评估
                    if abs(mean_ic) > 0.02 and abs(icir) > 0.5:
                        quality = "✅ 优秀"
                    elif abs(mean_ic) > 0.015 and abs(icir) > 0.4:
                        quality = "✅ 良好"
                    elif abs(mean_ic) > 0.01 and abs(icir) > 0.3:
                        quality = "⚠️  一般"
                    elif abs(mean_ic) > 0.005 and abs(icir) > 0.2:
                        quality = "⚠️  较弱"
                    else:
                        quality = "❌ 较差"
                    
                    result = {
                        'factor': factor,
                        'group': group_name,
                        'mean_ic': mean_ic,
                        'icir': icir,
                        'direction': direction,
                        'quality': quality,
                        'n_dates': len(daily_ic)
                    }
                    
                    ic_results.append(result)
                    group_ic.append(result)
                    
                    if quality in ["✅ 优秀", "✅ 良好"]:
                        print(f"  ✅ {factor}: IC={mean_ic:.4f}, ICIR={icir:.2f} ({quality})")
                    elif quality == "⚠️  一般":
                        print(f"  ⚠️  {factor}: IC={mean_ic:.4f}, ICIR={icir:.2f}")
            
            # 统计组内结果
            if group_ic:
                valid_count = len([r for r in group_ic if r['quality'] in ["✅ 优秀", "✅ 良好", "⚠️  一般"]])
                group_stats[group_name] = {
                    'total_factors': len(factors),
                    'valid_factors': valid_count,
                    'mean_ic': np.mean([r['mean_ic'] for r in group_ic]),
                    'mean_icir': np.mean([r['icir'] for r in group_ic])
                }
                print(f"  📊 组统计: {valid_count}/{len(factors)} 个有效因子")
        
        if not ic_results:
            print("❌ 没有计算出有效的IC结果")
            return False
        
        # 9. 筛选有效因子
        print(f"\n🎯 筛选有效因子 (IC>0.01, ICIR>0.3)...")
        
        valid_factors = []
        for result in ic_results:
            if abs(result['mean_ic']) > 0.01 and abs(result['icir']) > 0.3:
                valid_factors.append(result)
        
        print(f"✅ 筛选结果: {len(valid_factors)}/{len(ic_results)} 个有效因子")
        
        if valid_factors:
            # 按IC绝对值排序
            valid_factors.sort(key=lambda x: abs(x['mean_ic']), reverse=True)
            
            print(f"\n🏆 有效因子排名:")
            for i, result in enumerate(valid_factors[:10], 1):
                print(f"  {i:2d}. {result['factor']:20s} [{result['group']:20s}] IC={result['mean_ic']:.4f} ({result['direction']}) ICIR={result['icir']:.2f}")
        
        # 10. 保存结果
        print("\n💾 保存结果...")
        
        output_dir = Path("results/real_group_training")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建报告
        report = {
            "analysis_date": datetime.now().strftime('%Y-%m-%d'),
            "generation_timestamp": timestamp,
            "data_source": "QLIB真实数据",
            "data_statistics": {
                "total_samples": len(df),
                "unique_stocks": df['code'].nunique(),
                "date_range": f"{df['date'].min().date()} 到 {df['date'].max().date()}",
                "analysis_factors": len(ic_results),
                "valid_factors": len(valid_factors)
            },
            "selection_criteria": {
                "min_ic": 0.01,
                "min_icir": 0.3
            },
            "group_statistics": group_stats,
            "ic_results": ic_results,
            "valid_factors": valid_factors
        }
        
        import json
        report_path = output_dir / f"report_{timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 报告保存到: {report_path}")
        
        # 11. 显示总结
        execution_time = datetime.now() - start_time
        
        print("\n" + "=" * 70)
        print("🎉 真实数据分组训练完成！")
        print("=" * 70)
        
        print(f"\n⏱️  执行时间: {execution_time}")
        
        print(f"\n📊 数据统计:")
        print(f"  真实股票: {df['code'].nunique()} 只")
        print(f"  真实样本: {len(df):,}")
        print(f"  分析因子: {len(all_factors)} 个")
        print(f"  有效因子: {len(valid_factors)} 个")
        
        if ic_results:
            mean_ic = np.mean([r['mean_ic'] for r in ic_results])
            mean_icir = np.mean([r['icir'] for r in ic_results])
            
            print(f"\n📈 总体IC统计:")
            print(f"  平均IC: {mean_ic:.4f}")
            print(f"  平均ICIR: {mean_icir:.2f}")
        
        print(f"\n💾 输出文件: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 真实数据分组训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    sys.exit(0 if run_real_group_training() else 1)