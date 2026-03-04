#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FZT筛选 + Alpha158因子 + LightGBM训练工作流

使用：
1. FZT公式筛选符合条件的股票
2. 使用精选的Alpha158因子（与FZT相关性低）
3. LightGBM训练，关注gain得分分析
4. 发掘对选股成功率贡献大的因子

作者：FZT项目组
创建日期：2026年3月2日
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import yaml
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
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


# 精选的Alpha158因子（与FZT相关性较低）
SELECTED_ALPHA158_FACTORS = [
    # 成交量类因子
    'VOLUME0',  # 当日成交量
    'VOLUME1',  # 前1日成交量
    'VOLUME2',  # 前2日成交量
    'VOLUME3',  # 前3日成交量
    'VOLUME4',  # 前4日成交量
    'VOLUME10',  # 成交量比率（当日/前1日）
    'VOLUME11',  # 成交量比率（当日/5日均值）
    
    # 波动率类因子
    'STD0',  # 5日收益率标准差
    'STD1',  # 10日收益率标准差
    'STD2',  # 20日收益率标准差
    'STD3',  # 60日收益率标准差
    'VAR0',  # 5日收益率方差
    
    # 乖离率类因子
    'BIAS0',  # 5日乖离率
    'BIAS1',  # 10日乖离率
    'BIAS2',  # 20日乖离率
    'BIAS3',  # 60日乖离率
    
    # 流动性类因子
    'TURNOVER0',  # 当日换手率
    'TURNOVER1',  # 5日平均换手率
    'TURNOVER2',  # 10日平均换手率
    'TURNOVER_VOLUME0',  # 成交量换手率
    
    # 市值类因子
    'MARKET_CAP0',  # 总市值
    'MARKET_CAP1',  # 流通市值
    'FLOAT_MARKET_CAP0',  # 自由流通市值
    
    # 技术指标类因子
    'RSI0',  # 6日RSI
    'RSI1',  # 12日RSI
    'MACD0',  # MACD指标
    
    # 价格类因子（VWAP，与简单价格形态相关性较低）
    'VWAP0',  # 当日VWAP
    'VWAP1',  # 前1日VWAP
    
    # 收益率类因子（中长期）
    'RETURN5',  # 5日收益率
    'RETURN9',  # 9日收益率
]


class Alpha158FactorCalculator:
    """Alpha158因子计算器（简化版）"""
    
    @staticmethod
    def calculate_volume_factors(df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量类因子"""
        result = df.copy()
        
        # 基础成交量
        result['VOLUME0'] = df['volume']
        result['VOLUME1'] = df.groupby('code')['volume'].shift(1)
        result['VOLUME2'] = df.groupby('code')['volume'].shift(2)
        result['VOLUME3'] = df.groupby('code')['volume'].shift(3)
        result['VOLUME4'] = df.groupby('code')['volume'].shift(4)
        
        # 成交量比率
        result['VOLUME10'] = df['volume'] / df.groupby('code')['volume'].shift(1)
        result['VOLUME11'] = df['volume'] / df.groupby('code')['volume'].rolling(5).mean().reset_index(level=0, drop=True)
        
        return result
    
    @staticmethod
    def calculate_volatility_factors(df: pd.DataFrame) -> pd.DataFrame:
        """计算波动率类因子"""
        result = df.copy()
        
        # 计算收益率
        returns = df.groupby('code')['close'].pct_change(fill_method=None)
        
        # 波动率（标准差）
        result['STD0'] = returns.rolling(5).std()
        result['STD1'] = returns.rolling(10).std()
        result['STD2'] = returns.rolling(20).std()
        result['STD3'] = returns.rolling(60).std()
        
        # 方差
        result['VAR0'] = returns.rolling(5).var()
        
        return result
    
    @staticmethod
    def calculate_bias_factors(df: pd.DataFrame) -> pd.DataFrame:
        """计算乖离率因子"""
        result = df.copy()
        
        # 移动平均
        ma5 = df.groupby('code')['close'].rolling(5).mean().reset_index(level=0, drop=True)
        ma10 = df.groupby('code')['close'].rolling(10).mean().reset_index(level=0, drop=True)
        ma20 = df.groupby('code')['close'].rolling(20).mean().reset_index(level=0, drop=True)
        ma60 = df.groupby('code')['close'].rolling(60).mean().reset_index(level=0, drop=True)
        
        # 乖离率 = (收盘价 - 移动平均) / 移动平均 * 100
        result['BIAS0'] = (df['close'] - ma5) / ma5 * 100
        result['BIAS1'] = (df['close'] - ma10) / ma10 * 100
        result['BIAS2'] = (df['close'] - ma20) / ma20 * 100
        result['BIAS3'] = (df['close'] - ma60) / ma60 * 100
        
        return result
    
    @staticmethod
    def calculate_liquidity_factors(df: pd.DataFrame) -> pd.DataFrame:
        """计算流动性因子"""
        result = df.copy()
        
        # 换手率（简化：用成交量/流通市值，这里用成交量代替）
        result['TURNOVER0'] = df['volume']
        result['TURNOVER1'] = df.groupby('code')['volume'].rolling(5).mean().reset_index(level=0, drop=True)
        result['TURNOVER2'] = df.groupby('code')['volume'].rolling(10).mean().reset_index(level=0, drop=True)
        
        # 成交量换手率（简化）
        result['TURNOVER_VOLUME0'] = df['volume'] / df.groupby('code')['volume'].rolling(20).mean().reset_index(level=0, drop=True)
        
        return result
    
    @staticmethod
    def calculate_market_cap_factors(df: pd.DataFrame) -> pd.DataFrame:
        """计算市值因子（简化：用价格*固定系数）"""
        result = df.copy()
        
        # 简化处理：假设所有股票有相同的股本
        # 实际应用中应从外部数据源获取市值数据
        base_cap = 1e9  # 10亿股本
        
        result['MARKET_CAP0'] = df['close'] * base_cap  # 总市值
        result['MARKET_CAP1'] = df['close'] * base_cap * 0.7  # 流通市值（假设70%流通）
        result['FLOAT_MARKET_CAP0'] = df['close'] * base_cap * 0.5  # 自由流通市值（假设50%）
        
        return result
    
    @staticmethod
    def calculate_technical_factors(df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标因子"""
        result = df.copy()
        
        # RSI计算
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        # RSI
        result['RSI0'] = df.groupby('code')['close'].transform(lambda x: calculate_rsi(x, 6))
        result['RSI1'] = df.groupby('code')['close'].transform(lambda x: calculate_rsi(x, 12))
        
        # MACD简化版
        ema12 = df.groupby('code')['close'].transform(lambda x: x.ewm(span=12).mean())
        ema26 = df.groupby('code')['close'].transform(lambda x: x.ewm(span=26).mean())
        result['MACD0'] = ema12 - ema26
        
        return result
    
    @staticmethod
    def calculate_price_factors(df: pd.DataFrame) -> pd.DataFrame:
        """计算价格因子"""
        result = df.copy()
        
        # VWAP（成交量加权平均价）
        # 简化：用(high+low+close)/3代替
        result['VWAP0'] = (df['high'] + df['low'] + df['close']) / 3
        result['VWAP1'] = result.groupby('code')['VWAP0'].shift(1)
        
        return result
    
    @staticmethod
    def calculate_return_factors(df: pd.DataFrame) -> pd.DataFrame:
        """计算收益率因子"""
        result = df.copy()
        
        # 收益率
        result['RETURN5'] = df.groupby('code')['close'].pct_change(5, fill_method=None)
        result['RETURN9'] = df.groupby('code')['close'].pct_change(9, fill_method=None)
        
        return result
    
    @staticmethod
    def calculate_all_factors(df: pd.DataFrame) -> pd.DataFrame:
        """计算所有Alpha158因子"""
        logger.info("🔧 计算Alpha158因子...")
        
        # 按顺序计算各类因子
        df = Alpha158FactorCalculator.calculate_volume_factors(df)
        df = Alpha158FactorCalculator.calculate_volatility_factors(df)
        df = Alpha158FactorCalculator.calculate_bias_factors(df)
        df = Alpha158FactorCalculator.calculate_liquidity_factors(df)
        df = Alpha158FactorCalculator.calculate_market_cap_factors(df)
        df = Alpha158FactorCalculator.calculate_technical_factors(df)
        df = Alpha158FactorCalculator.calculate_price_factors(df)
        df = Alpha158FactorCalculator.calculate_return_factors(df)
        
        logger.info(f"✅ Alpha158因子计算完成: {len(SELECTED_ALPHA158_FACTORS)} 个因子")
        
        return df


def load_and_prepare_data():
    """加载并准备数据（使用通用DataLoader）"""
    logger.info("📥 加载数据...")
    
    try:
        # 使用通用数据加载器
        from src.data_loader import DataLoader
        loader = DataLoader(config_path="config/data_config.yaml")
        
        # 加载QLIB数据
        df = loader.load_qlib_data()
        if df is None:
            logger.error("❌ QLIB数据加载失败")
            return None
        
        # 计算目标变量
        df = loader.calculate_target_variable(df)
        
        logger.info(f"✅ 数据加载成功: {df.shape[0]} 行, {df.shape[1]} 列")
        
        # 确保数据排序
        df = df.sort_values(['code', 'date'])
        
        return df
        
    except Exception as e:
        logger.error(f"❌ 数据加载失败: {e}")
        return None

def apply_fzt_screening(data: pd.DataFrame):
    """应用FZT公式筛选"""
    logger.info("🔍 应用FZT公式筛选...")
    
    try:
        from src.fzt_brick_formula import FZTBrickFormula
        
        fzt_calculator = FZTBrickFormula()
        
        # 按股票分组处理
        screened_data_list = []
        total_stocks = data['code'].nunique()
        
        for i, (stock_code, stock_data) in enumerate(data.groupby('code')):
            if i % 10 == 0:
                logger.info(f"  处理股票 {i+1}/{total_stocks}: {stock_code}")
            
            if len(stock_data) < 20:  # 需要足够的数据计算FZT
                continue
            
            # 计算FZT指标
            try:
                indicators = fzt_calculator.calculate_all_indicators(stock_data)
                
                # 获取选股条件
                if 'selection_condition' in indicators:
                    selection_signal = indicators['selection_condition']
                    
                    # 筛选符合条件的日期
                    screened_dates = stock_data.index[selection_signal.fillna(False)]
                    
                    if len(screened_dates) > 0:
                        screened_data = stock_data.loc[screened_dates].copy()
                        screened_data_list.append(screened_data)
                        
            except Exception as e:
                logger.warning(f"   股票 {stock_code} FZT计算失败: {str(e)[:50]}")
                continue
        
        if not screened_data_list:
            logger.warning("⚠️  没有股票通过FZT筛选")
            return pd.DataFrame()
        
        # 合并所有筛选后的数据
        screened_df = pd.concat(screened_data_list, ignore_index=True)
        
        logger.info(f"✅ FZT筛选完成")
        logger.info(f"   原始数据: {len(data)} 行")
        logger.info(f"   筛选后数据: {len(screened_df)} 行")
        logger.info(f"   筛选比例: {len(screened_df)/len(data):.2%}")
        
        return screened_df
        
    except Exception as e:
        logger.error(f"❌ FZT筛选失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def prepare_training_features(data: pd.DataFrame):
    """准备训练特征（使用Alpha158因子）"""
    logger.info("🔧 准备训练特征（Alpha158因子）...")
    
    try:
        # 只使用Alpha158因子，不使用FZT特征
        available_features = [col for col in SELECTED_ALPHA158_FACTORS if col in data.columns]
        
        X = data[available_features].fillna(0).values
        y = data['target'].values
        
        logger.info(f"✅ 特征准备完成")
        logger.info(f"   Alpha158因子数量: {len(available_features)}")
        logger.info(f"   样本数量: {len(X)}")
        logger.info(f"   正类比例: {y.mean():.2%}")
        
        return X, y, available_features
        
    except Exception as e:
        logger.error(f"❌ 特征准备失败: {e}")
        return None, None, []


def train_lightgbm_with_gain_analysis(X_train, y_train, X_val, y_val, feature_names):
    """训练LightGBM并分析gain得分"""
    logger.info("🤖 训练LightGBM（关注gain得分）...")
    
    try:
        import lightgbm as lgb
        
        # LightGBM参数（进一步优化）
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.01,      # 降低学习率，更稳定
            'num_leaves': 127,          # 增加，提高模型复杂度
            'max_depth': 10,            # 增加，处理更复杂模式
            'min_child_samples': 5,     # 减少，允许更细粒度分裂
            'min_child_weight': 0.001,  # 减少最小子节点权重
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_estimators': 500,        # 增加，更多树
            'verbose': -1,
            'is_unbalance': True,       # 处理样本不平衡
            'boosting_type': 'gbdt',    # 梯度提升决策树
            'feature_fraction': 0.9,    # 特征采样
            'bagging_fraction': 0.8,    # 数据采样
            'bagging_freq': 5,          # 每5轮进行一次bagging
        }
        
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_names)
        
        # 训练模型
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(20),  # 增加，给模型更多训练机会
                lgb.log_evaluation(50)   # 每50轮输出一次
            ]
        )
        
        # 获取特征重要性（gain得分）
        importance = model.feature_importance(importance_type='gain')
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'gain': importance,
            'split': model.feature_importance(importance_type='split')
        })
        # LightGBM不支持'cover'参数，移除或使用其他方式计算
        
        # 按gain得分排序
        importance_df = importance_df.sort_values('gain', ascending=False)
        
        logger.info("✅ 模型训练完成")
        logger.info(f"   最佳迭代次数: {model.best_iteration}")
        
        print("\n📊 Alpha158因子重要性分析（按gain得分排序）:")
        print("="*90)
        print(f"{'因子':<15} {'gain得分':<12} {'split次数':<12} {'类型':<15} {'实战价值':<30}")
        print("-"*90)
        
        factor_types = {
            'VOLUME': '成交量',
            'STD': '波动率',
            'VAR': '波动率',
            'BIAS': '乖离率',
            'TURNOVER': '流动性',
            'MARKET_CAP': '市值',
            'RSI': '技术指标',
            'MACD': '技术指标',
            'VWAP': '价格(VWAP)',
            'RETURN': '收益率'
        }
        
        for _, row in importance_df.head(15).iterrows():
            # 判断因子类型
            factor_type = '其他'
            for prefix, ftype in factor_types.items():
                if row['feature'].startswith(prefix):
                    factor_type = ftype
                    break
            
            # 实战价值描述
            if factor_type == '成交量':
                value_desc = '反映市场参与度和资金流向'
            elif factor_type == '波动率':
                value_desc = '反映股票风险水平和市场情绪'
            elif factor_type == '乖离率':
                value_desc = '反映价格偏离均线的程度'
            elif factor_type == '流动性':
                value_desc = '反映股票交易活跃度'
            elif factor_type == '市值':
                value_desc = '反映公司规模和市场地位'
            elif factor_type == '技术指标':
                value_desc = '提供额外的技术信号'
            elif factor_type == '价格(VWAP)':
                value_desc = '成交量加权价格，反映真实交易成本'
            elif factor_type == '收益率':
                value_desc = '反映股票价格变动趋势'
            else:
                value_desc = '综合技术因子'
            
            print(f"{row['feature']:<15} {row['gain']:<12.2f} {row['split']:<12} {factor_type:<15} {value_desc:<30}")
        
        print("="*90)
        
        # gain得分分析
        print("\n🎯 gain得分分析（选股场景解读）:")
        print("  gain（增益）: 该因子在所有决策树中带来的误差减少总量（核心！）")
        print("                → 直接反映因子对选股胜率的贡献")
        print("  split（分裂）: 该因子被用作决策树分裂节点的次数")
        print("                → 辅助指标，次数多≠贡献大")
        # cover指标在LightGBM中不可用，使用split作为参考
        print("  split（分裂）: 该因子被用作决策树分裂节点的次数")
        print("                → 辅助指标，反映因子被使用的频率")
        
        return model, importance_df
        
    except Exception as e:
        logger.error(f"❌ 模型训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None


def evaluate_model(model, X_test, y_test, feature_names):
    """评估模型性能"""
    logger.info("📈 评估模型性能...")
    
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # 预测
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, y_pred_prob)
        except:
            roc_auc = 0
        
        # 计算选股成功率（预测上涨且实际上涨的比例）
        success_mask = (y_pred == 1) & (y_test == 1)
        if (y_pred == 1).sum() > 0:
            success_rate = success_mask.sum() / (y_pred == 1).sum()
        else:
            success_rate = 0
        
        logger.info(f"✅ 模型评估完成")
        
        print("\n📊 模型性能指标:")
        print("="*60)
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"选股成功率: {success_rate:.4f}")
        print("="*60)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'success_rate': success_rate
        }
        
    except Exception as e:
        logger.error(f"❌ 模型评估失败: {e}")
        return {}


def discover_effective_factors(importance_df, top_n=15):
    """发掘有效因子"""
    logger.info("🔍 发掘有效Alpha158因子...")
    
    try:
        # 选择gain得分高的因子
        top_factors = importance_df.head(top_n)
        
        print(f"\n🎯 Top-{top_n} 有效Alpha158因子（按gain得分）:")
        print("="*90)
        
        for i, (_, row) in enumerate(top_factors.iterrows(), 1):
            print(f"{i:2d}. {row['feature']}")
            print(f"    gain得分: {row['gain']:.2f} (误差减少总量)")
            print(f"    split次数: {row['split']} (分裂节点次数)")
            
            # 因子类型和价值
            if row['feature'].startswith('VOLUME'):
                print(f"    类型: 成交量因子")
                print(f"    实战价值: 反映市场参与度和资金流向，高成交量通常伴随趋势延续")
            elif row['feature'].startswith('STD') or row['feature'].startswith('VAR'):
                print(f"    类型: 波动率因子")
                print(f"    实战价值: 反映股票风险水平，低波动率可能预示趋势稳定")
            elif row['feature'].startswith('BIAS'):
                print(f"    类型: 乖离率因子")
                print(f"    实战价值: 反映价格偏离均线程度，乖离率回归有交易机会")
            elif row['feature'].startswith('TURNOVER'):
                print(f"    类型: 流动性因子")
                print(f"    实战价值: 反映股票交易活跃度，高流动性降低交易成本")
            elif row['feature'].startswith('MARKET_CAP'):
                print(f"    类型: 市值因子")
                print(f"    实战价值: 反映公司规模，不同市值股票有不同市场行为")
            elif row['feature'].startswith('RSI'):
                print(f"    类型: RSI技术指标")
                print(f"    实战价值: 反映超买超卖状态，提供反转信号")
            elif row['feature'].startswith('MACD'):
                print(f"    类型: MACD技术指标")
                print(f"    实战价值: 反映趋势强度和方向变化")
            elif row['feature'].startswith('VWAP'):
                print(f"    类型: VWAP价格因子")
                print(f"    实战价值: 反映真实交易成本，价格相对VWAP有指导意义")
            elif row['feature'].startswith('RETURN'):
                print(f"    类型: 收益率因子")
                print(f"    实战价值: 反映价格动量，动量延续或反转有交易机会")
            
            print()
        
        print("💡 因子发掘建议:")
        print("   1. 关注gain得分高的因子（直接贡献预测能力）")
        print("   2. 结合FZT筛选，形成多维度选股体系")
        print("   3. 定期验证因子有效性，避免因子失效")
        print("   4. 考虑因子间的相关性，避免多重共线性")
        print("="*90)
        
        return top_factors
        
    except Exception as e:
        logger.error(f"❌ 因子发掘失败: {e}")
        return pd.DataFrame()


def main():
    """主函数"""
    logger.info("🚀 FZT筛选 + Alpha158因子 + LightGBM训练工作流开始")
    logger.info("=" * 70)
    
    try:
        # 1. 加载并准备数据
        data = load_and_prepare_data()
        if data is None or data.empty:
            logger.error("❌ 数据加载失败")
            return 1
        
        print(f"\n📊 原始数据统计:")
        print(f"  总行数: {len(data):,}")
        print(f"  股票数量: {data['code'].nunique()}")
        print(f"  日期范围: {data['date'].min()} 到 {data['date'].max()}")
        print(f"  正类比例: {data['target'].mean():.2%}")
        
        # 2. 应用FZT筛选
        screened_data = apply_fzt_screening(data)
        if screened_data.empty:
            logger.error("❌ FZT筛选后无数据")
            return 1
        
        print(f"\n🔍 FZT筛选结果:")
        print(f"  筛选后行数: {len(screened_data):,}")
        print(f"  筛选比例: {len(screened_data)/len(data):.2%}")
        print(f"  筛选后正类比例: {screened_data['target'].mean():.2%}")
        
        # 3. 准备训练特征（Alpha158因子）
        X, y, feature_names = prepare_training_features(screened_data)
        if X is None or len(X) == 0:
            logger.error("❌ 特征准备失败")
            return 1
        
        # 4. 划分训练集和测试集（时间序列划分）
        train_ratio = 0.7
        split_idx = int(len(X) * train_ratio)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 进一步划分验证集
        val_ratio = 0.2
        val_split_idx = int(len(X_train) * (1 - val_ratio))
        X_train, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
        y_train, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
        
        print(f"\n📈 数据划分:")
        print(f"  训练集: {len(X_train):,} 样本")
        print(f"  验证集: {len(X_val):,} 样本")
        print(f"  测试集: {len(X_test):,} 样本")
        
        # 5. 训练LightGBM并分析gain得分
        model, importance_df = train_lightgbm_with_gain_analysis(
            X_train, y_train, X_val, y_val, feature_names
        )
        
        if model is None:
            logger.error("❌ 模型训练失败")
            return 1
        
        # 6. 评估模型
        metrics = evaluate_model(model, X_test, y_test, feature_names)
        
        # 7. 发掘有效因子
        top_factors = discover_effective_factors(importance_df, top_n=15)
        
        # 8. 保存结果
        logger.info("💾 保存训练结果...")
        
        # 创建输出目录
        output_dir = Path("results") / "fzt_alpha158_training"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存模型
        model_path = output_dir / f"fzt_alpha158_model_{timestamp}.txt"
        model.save_model(str(model_path))
        
        # 保存特征重要性
        importance_path = output_dir / f"feature_importance_{timestamp}.csv"
        importance_df.to_csv(importance_path, index=False)
        
        # 保存训练结果
        result = {
            'training_date': timestamp,
            'data_info': {
                'original_samples': len(data),
                'screened_samples': len(screened_data),
                'screening_ratio': len(screened_data) / len(data),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'positive_ratio': float(y.mean())
            },
            'model_info': {
                'best_iteration': model.best_iteration,
                'feature_count': len(feature_names),
                'alpha158_factors': feature_names
            },
            'performance': metrics,
            'top_factors': top_factors[['feature', 'gain', 'split']].to_dict('records')
        }
        
        result_path = output_dir / f"training_result_{timestamp}.yaml"
        with open(result_path, 'w', encoding='utf-8') as f:
            yaml.dump(result, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n🎉 训练完成！")
        print(f"📁 结果保存到: {output_dir}")
        print(f"  模型文件: {model_path.name}")
        print(f"  特征重要性: {importance_path.name}")
        print(f"  训练结果: {result_path.name}")
        
        print(f"\n🚀 工作流总结:")
        print(f"  1. ✅ FZT筛选: 从 {len(data):,} 行中筛选出 {len(screened_data):,} 行")
        print(f"  2. ✅ Alpha158因子: 使用 {len(feature_names)} 个精选因子")
        print(f"  3. ✅ gain得分分析: 识别出 {len(top_factors)} 个有效因子")
        print(f"  4. ✅ 模型性能: 选股成功率 {metrics.get('success_rate', 0):.4f}")
        
        logger.info("🎯 FZT筛选 + Alpha158因子 + LightGBM训练工作流完成")
        return 0
        
    except Exception as e:
        logger.error(f"❌ 工作流执行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
