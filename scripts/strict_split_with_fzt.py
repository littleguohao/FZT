#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
严格分割工作流 + FZT筛选

训练：QLIB数据（2005-2020） + FZT筛选
回测：CSV数据（2021-2026） + FZT筛选
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.fzt_brick_formula import FZTBrickFormula

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_qlib_training_data():
    """加载QLIB训练数据（2005-2020）"""
    logger.info("📥 加载QLIB训练数据 (2005-2020)...")
    
    try:
        data_file = Path("results/expanded_stock_data/expanded_data_with_target.csv")
        if not data_file.exists():
            logger.error(f"❌ QLIB数据文件不存在: {data_file}")
            return None
        
        # 加载数据
        df = pd.read_csv(data_file)
        
        # 确保日期格式
        df['date'] = pd.to_datetime(df['date'])
        
        # 过滤到2020年及之前
        df = df[df['date'] <= '2020-12-31']
        
        logger.info(f"✅ QLIB训练数据加载成功")
        logger.info(f"   行数: {len(df):,}")
        logger.info(f"   股票数: {df['code'].nunique()}")
        logger.info(f"   时间范围: {df['date'].min().strftime('%Y-%m-%d')} 到 {df['date'].max().strftime('%Y-%m-%d')}")
        logger.info(f"   正类比例: {df['target'].mean():.2%}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ QLIB数据加载失败: {e}")
        return None


def load_csv_backtest_data(sample_size=100):
    """加载CSV回测数据（2021-2026）"""
    logger.info("📥 加载CSV回测数据 (2021-2026)...")
    
    try:
        data_dir = Path("/Users/lucky/Downloads/O_DATA")
        if not data_dir.exists():
            logger.error(f"❌ CSV数据目录不存在: {data_dir}")
            return None
        
        # 获取所有CSV文件
        csv_files = list(data_dir.glob("*.csv"))
        logger.info(f"📋 找到 {len(csv_files):,} 个CSV文件")
        
        # 采样处理
        if sample_size:
            csv_files = csv_files[:sample_size]
            logger.info(f"📊 采样处理前 {sample_size} 个文件")
        
        all_data = []
        
        for csv_file in csv_files:
            try:
                # 读取文件
                df = pd.read_csv(csv_file)
                
                # 重命名列
                column_mapping = {
                    'Date': 'date',
                    'Code': 'code',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }
                
                df = df.rename(columns=column_mapping)
                
                # 只保留需要的列
                keep_cols = ['date', 'code', 'open', 'high', 'low', 'close', 'volume']
                df = df[keep_cols]
                
                # 确保日期格式
                df['date'] = pd.to_datetime(df['date'])
                
                # 过滤到2021年及之后
                df = df[df['date'] >= '2021-01-01']
                
                if len(df) == 0:
                    continue
                
                all_data.append(df)
                
            except Exception:
                continue
        
        if not all_data:
            logger.error("❌ 没有成功加载任何CSV数据")
            return None
        
        # 合并所有数据
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 计算目标变量
        combined_df = calculate_target_variable(combined_df)
        
        logger.info(f"✅ CSV回测数据加载完成")
        logger.info(f"   总行数: {len(combined_df):,}")
        logger.info(f"   股票数: {combined_df['code'].nunique()}")
        logger.info(f"   时间范围: {combined_df['date'].min().strftime('%Y-%m-%d')} 到 {combined_df['date'].max().strftime('%Y-%m-%d')}")
        logger.info(f"   正类比例: {combined_df['target'].mean():.2%}")
        
        return combined_df
        
    except Exception as e:
        logger.error(f"❌ CSV数据加载失败: {e}")
        return None


def calculate_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """计算目标变量"""
    try:
        # 确保按股票和日期排序
        df = df.sort_values(['code', 'date'])
        
        # 计算次日收盘价和未来收益率
        df['next_close'] = df.groupby('code')['close'].shift(-1)
        df['future_return'] = (df['next_close'] - df['close']) / df['close']
        df['target'] = (df['future_return'] > 0).astype(int)
        
        # 移除无效行
        df = df.dropna(subset=['next_close', 'target'])
        
        return df
        
    except Exception as e:
        logger.error(f"❌ 目标变量计算失败: {e}")
        return df


def apply_fzt_screening(df: pd.DataFrame) -> pd.DataFrame:
    """应用FZT筛选"""
    logger.info("🔍 应用FZT公式筛选...")
    
    try:
        # 初始化FZT公式
        fzt_calculator = FZTBrickFormula()
        
        # 按股票分组处理
        screened_data = []
        
        for code, group in df.groupby('code'):
            try:
                # 计算FZT指标
                fzt_features = fzt_calculator.calculate_all_indicators(
                    close_prices=group['close'].values,
                    high_prices=group['high'].values,
                    low_prices=group['low'].values,
                    volumes=group['volume'].values
                )
                
                # 判断是否符合FZT条件
                fzt_condition = fzt_calculator.is_fzt_condition_met(fzt_features)
                
                # 只保留符合条件的行
                if fzt_condition:
                    screened_data.append(group)
                    
            except Exception as e:
                logger.warning(f"   股票 {code} FZT计算失败: {e}")
                continue
        
        if not screened_data:
            logger.warning("⚠️  没有股票通过FZT筛选")
            return pd.DataFrame()
        
        # 合并筛选后的数据
        screened_df = pd.concat(screened_data, ignore_index=True)
        
        logger.info(f"✅ FZT筛选完成")
        logger.info(f"   原始数据: {len(df):,} 行")
        logger.info(f"   筛选后数据: {len(screened_df):,} 行")
        logger.info(f"   筛选比例: {len(screened_df)/len(df):.2%}")
        logger.info(f"   筛选后正类比例: {screened_df['target'].mean():.2%}")
        
        return screened_df
        
    except Exception as e:
        logger.error(f"❌ FZT筛选失败: {e}")
        return pd.DataFrame()


def prepare_features(df: pd.DataFrame):
    """准备特征（不使用FZT特征）"""
    logger.info("🔧 准备特征...")
    
    try:
        # 基础特征
        features_df = df.copy()
        
        # 技术指标（不使用FZT特征）
        # 收益率特征
        for window in [1, 3, 5, 10, 20]:
            features_df[f'return_{window}d'] = df.groupby('code')['close'].pct_change(window)
        
        # 成交量特征
        features_df['volume_ratio'] = df['volume'] / df.groupby('code')['volume'].rolling(20).mean().reset_index(level=0, drop=True)
        features_df['volume_change'] = df.groupby('code')['volume'].pct_change()
        
        # 价格位置特征
        features_df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        features_df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1)
        
        # 波动率特征
        features_df['volatility_5d'] = df.groupby('code')['close'].pct_change().rolling(5).std()
        features_df['volatility_20d'] = df.groupby('code')['close'].pct_change().rolling(20).std()
        
        # 移动平均特征
        features_df['ma5'] = df.groupby('code')['close'].rolling(5).mean().reset_index(level=0, drop=True)
        features_df['ma20'] = df.groupby('code')['close'].rolling(20).mean().reset_index(level=0, drop=True)
        features_df['ma_ratio'] = features_df['ma5'] / features_df['ma20']
        
        # 处理缺失值
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 获取特征列（排除FZT相关列）
        exclude_cols = ['date', 'code', 'target', 'next_close', 'future_return']
        feature_columns = [col for col in features_df.columns if col not in exclude_cols]
        
        logger.info(f"✅ 特征准备完成")
        logger.info(f"   特征数量: {len(feature_columns)}")
        logger.info(f"   样本数量: {len(features_df):,}")
        
        return features_df, feature_columns
        
    except Exception as e:
        logger.error(f"❌ 特征准备失败: {e}")
        return df, []


def train_model(X_train, y_train):
    """训练LightGBM模型"""
    logger.info("🤖 训练LightGBM模型...")
    
    try:
        import lightgbm as lgb
        
        # LightGBM参数
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'num_leaves': 63,
            'max_depth': 7,
            'min_child_samples': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_estimators': 100,
            'verbose': -1,
        }
        
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # 训练模型
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        logger.info("✅ 模型训练完成")
        return model
        
    except Exception as e:
        logger.error(f"❌ 模型训练失败: {e}")
        return None


def evaluate_model(model, X_test, y_test):
    """评估模型"""
    logger.info("📊 评估模型性能...")
    
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        roc_auc = roc_auc_score(y_test, y_pred)
        
        logger.info("✅ 模型评估完成")
        logger.info(f"   准确率: {accuracy:.4f}")
        logger.info(f"   精确率: {precision:.4f}")
        logger.info(f"   召回率: {recall:.4f}")
        logger.info(f"   F1分数: {f1:.4f}")
        logger.info(f"   ROC AUC: {roc_auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
    except Exception as e:
        logger.error(f"❌ 模型评估失败: {e}")
        return None


def run_backtest(features_df, model, feature_columns):
    """运行回测"""
    logger.info("📈 运行回测...")
    
    try:
        # 预测
        X_backtest = features_df[feature_columns]
        y_backtest = features_df['target']
        
        y_pred = model.predict(X_backtest)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # 计算回测指标
        backtest_accuracy = (y_pred_binary == y_backtest).mean()
        
        # 计算策略收益率（修正版本）
        features_df['prediction'] = y_pred_binary
        features_df['prediction_prob'] = y_pred
        
        # 只选择预测为上涨的股票
        selected_stocks = features_df[features_df['prediction'] == 1]
        
        if len(selected_stocks) == 0:
            logger.warning("⚠️  没有股票被选中")
            return backtest_accuracy, 0.0
        
        # 计算平均收益率（避免复利计算问题）
        avg_daily_return = selected_stocks['future_return'].mean()
        
        # 假设每天投资，计算累计收益率
        days = len(selected_stocks)
        total_return = (1 + avg_daily_return) ** days - 1
        
        logger.info("✅ 回测完成")
        logger.info(f"   回测选股成功率: {backtest_accuracy:.2%}")
        logger.info(f"   回测样本数: {len(y_backtest):,}")
        logger.info(f"   选中股票数: {len(selected_stocks):,}")
        logger.info(f"   平均日收益率: {avg_daily_return:.4%}")
        logger.info(f"   策略总收益率: {total_return:.2%}")
        
        return backtest_accuracy, total_return
        
    except Exception as e:
        logger.error(f"❌ 回测失败: {e}")
        return 0.0, 0.0


def main():
    """主函数"""
    logger.info("🚀 严格分割工作流 + FZT筛选开始")
    logger.info("=" * 70)
    logger.info("📊 数据分割策略:")
    logger.info("   训练数据: QLIB (2005-2020) + FZT筛选")
    logger.info("   回测数据: CSV (2021-2026) + FZT筛选")
    logger.info("   严格分割，不混合")
    logger.info("=" * 70)
    
    try:
        # 1. 加载训练数据
        train_df = load_qlib_training_data()
        if train_df is None:
            return 1
        
        # 2. 应用FZT筛选
        train_screened_df = apply_fzt_screening(train_df)
        if len(train_screened_df) == 0:
            logger.error("❌ 训练数据FZT筛选后没有数据")
            return 1
        
        # 3. 准备训练特征
        train_features_df, feature_columns = prepare_features(train_screened_df)
        if len(feature_columns) == 0:
            return 1
        
        # 4. 训练模型
        X_train = train_features_df[feature_columns]
        y