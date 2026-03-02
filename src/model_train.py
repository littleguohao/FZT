#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练模块 - 基于Qlib LGBModel的LightGBM模型训练

使用配置：
model_config = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.model",
        "kwargs": {
            "objective": "binary",  # 二分类（成功/失败）
            "metric": "accuracy",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 5,
            "seed": 42,
            "n_estimators": 100,
            "early_stopping_rounds": 10
        }
    }
}

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
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# 可选依赖
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    logger.warning("seaborn未安装，可视化功能受限")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')


class ModelTrainer:
    """模型训练器（基于Qlib LGBModel）"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        初始化模型训练器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        
        # 提取模型配置
        self.model_config = self.config['model']
        self.model_type = self.model_config.get('type', 'lightgbm')
        
        # 根据模型类型选择参数
        if self.model_type == 'lightgbm':
            self.model_params = self.model_config.get('lightgbm_params', {})
        elif self.model_type == 'random_forest':
            self.model_params = self.model_config.get('random_forest_params', {})
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 输出目录
        self.model_dir = Path("results/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型实例
        self.model = None
        self.feature_importance = None
        
        logger.info("模型训练器初始化完成")
        logger.info(f"模型类型: {self.model_type}")
        logger.info(f"模型参数: {self.model_params}")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"配置文件不存在: {self.config_path}")
            # 使用默认配置
            default_config = {
                'model': {
                    'class': 'LGBModel',
                    'module_path': 'qlib.model',
                    'kwargs': {
                        'objective': 'binary',
                        'metric': 'accuracy',
                        'learning_rate': 0.05,
                        'num_leaves': 31,
                        'max_depth': 5,
                        'seed': 42,
                        'n_estimators': 100,
                        'early_stopping_rounds': 10
                    }
                }
            }
            logger.info("使用默认模型配置")
            return default_config
        except yaml.YAMLError as e:
            logger.error(f"配置文件解析错误: {e}")
            raise
    
    def _validate_config(self) -> None:
        """验证配置的完整性"""
        if 'model' not in self.config:
            raise ValueError("配置文件中缺少model部分")
        
        model_config = self.config['model']
        
        # 检查模型类型
        if 'type' not in model_config:
            raise ValueError("模型配置中缺少type键")
        
        model_type = model_config['type']
        if model_type not in ['lightgbm', 'random_forest']:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 检查对应参数
        if model_type == 'lightgbm' and 'lightgbm_params' not in model_config:
            raise ValueError("LightGBM配置中缺少lightgbm_params")
        
        if model_type == 'random_forest' and 'random_forest_params' not in model_config:
            raise ValueError("随机森林配置中缺少random_forest_params")
        
        logger.info("模型配置验证通过")
    
    def create_model(self):
        """创建模型实例"""
        try:
            if self.model_type == 'lightgbm':
                # 尝试使用LightGBM
                try:
                    import lightgbm as lgb
                    
                    # 从配置中获取参数
                    lgb_params = self.model_params.copy()
                    
                    # 确保必要的参数存在
                    if "objective" not in lgb_params:
                        lgb_params["objective"] = "binary"
                    if "random_state" not in lgb_params:
                        lgb_params["random_state"] = 42
                    if "n_estimators" not in lgb_params:
                        lgb_params["n_estimators"] = 100
                    if "learning_rate" not in lgb_params:
                        lgb_params["learning_rate"] = 0.05
                    
                    # 处理early_stopping_rounds参数
                    # 这个参数在scikit-learn接口中是在fit方法中使用的
                    if "early_stopping_rounds" in lgb_params:
                        # 保存这个参数，在fit方法中使用
                        self.early_stopping_rounds = lgb_params.pop("early_stopping_rounds")
                    else:
                        self.early_stopping_rounds = None
                    
                    # 创建LightGBM模型
                    self.model = lgb.LGBMClassifier(**lgb_params)
                    logger.info("LightGBM模型实例创建成功")
                    
                except (ImportError, OSError) as e:
                    # 如果LightGBM不可用，使用随机森林作为替代
                    logger.warning(f"LightGBM不可用: {e}")
                    logger.info("使用随机森林作为替代模型")
                    
                    from sklearn.ensemble import RandomForestClassifier
                    
                    # 使用随机森林参数
                    rf_params = self.model_config.get('random_forest_params', {
                        'n_estimators': 100,
                        'max_depth': 5,
                        'random_state': 42,
                        'n_jobs': -1
                    })
                    
                    self.model = RandomForestClassifier(**rf_params)
                    logger.info("随机森林模型实例创建成功")
                    
            elif self.model_type == 'random_forest':
                # 使用随机森林
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(**self.model_params)
                logger.info("随机森林模型实例创建成功")
                
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"创建模型失败: {e}")
            raise
    
    def prepare_binary_target(self, target: pd.Series, threshold: float = 0.0) -> pd.Series:
        """
        准备二分类目标变量
        
        Args:
            target: 连续收益率序列
            threshold: 分类阈值（大于阈值为正类1，否则为负类0）
            
        Returns:
            pd.Series: 二分类目标变量
        """
        # 将连续收益率转换为二分类标签
        binary_target = (target > threshold).astype(int)
        
        # 统计类别分布
        class_counts = binary_target.value_counts()
        class_ratio = class_counts.get(1, 0) / len(binary_target) if len(binary_target) > 0 else 0
        
        logger.info(f"二分类目标准备完成:")
        logger.info(f"  正类 (1): {class_counts.get(1, 0)} 个样本")
        logger.info(f"  负类 (0): {class_counts.get(0, 0)} 个样本")
        logger.info(f"  正类比例: {class_ratio:.2%}")
        
        return binary_target
    
    def prepare_data(self, features: pd.DataFrame, 
                    target_col: str = 'target',
                    threshold: float = 0.0) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备训练数据
        
        Args:
            features: 特征数据集
            target_col: 目标列名
            threshold: 分类阈值
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: 特征矩阵和二分类目标
        """
        logger.info("准备训练数据...")
        
        if target_col not in features.columns:
            raise ValueError(f"目标列 '{target_col}' 不存在")
        
        # 分离特征和目标
        target = features[target_col].copy()
        
        # 移除不需要的列
        exclude_cols = ['stock_code', 'date', target_col]
        feature_cols = [col for col in features.columns if col not in exclude_cols]
        
        X = features[feature_cols].copy()
        
        # 准备二分类目标
        y = self.prepare_binary_target(target, threshold)
        
        # 对齐索引
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # 移除目标变量中的NaN
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"数据准备完成: X.shape={X.shape}, y.shape={y.shape}")
        logger.info(f"特征数量: {len(X.columns)}")
        
        return X, y
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: Optional[pd.DataFrame] = None,
                   y_val: Optional[pd.Series] = None) -> Any:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征（可选）
            y_val: 验证目标（可选）
            
        Returns:
            训练好的模型
        """
        logger.info("开始训练模型...")
        
        # 创建模型实例（如果尚未创建）
        if self.model is None:
            self.create_model()
        
        # 准备训练数据 - 确保正确的数据格式
        # scikit-learn模型需要numpy数组或DataFrame，不能是元组
        X_train_data = X_train
        y_train_data = y_train
        
        # 准备验证数据（如果有）
        if X_val is not None and y_val is not None:
            X_val_data = X_val
            y_val_data = y_val
        else:
            X_val_data = None
            y_val_data = None
        
        # 训练模型
        try:
            # 对于LightGBM，需要提供验证集用于早停
            if self.model_type == "lightgbm" and X_val_data is not None and y_val_data is not None:
                # LightGBM的scikit-learn接口使用eval_set参数
                eval_set = [(X_val_data, y_val_data)]
                
                # 设置早停回调（如果配置了早停）
                callbacks = []
                if hasattr(self, 'early_stopping_rounds') and self.early_stopping_rounds:
                    try:
                        from lightgbm import early_stopping
                        callbacks.append(early_stopping(self.early_stopping_rounds))
                        logger.info(f"启用早停，轮数: {self.early_stopping_rounds}")
                    except ImportError:
                        logger.warning("无法导入lightgbm.early_stopping，跳过早停")
                
                # 训练模型
                if callbacks:
                    self.model.fit(
                        X_train_data, y_train_data,
                        eval_set=eval_set,
                        eval_metric="accuracy",
                        callbacks=callbacks
                    )
                    logger.info("LightGBM模型训练完成（使用早停）")
                else:
                    # 不使用早停，只使用验证集
                    self.model.fit(
                        X_train_data, y_train_data,
                        eval_set=eval_set,
                        eval_metric="accuracy"
                    )
                    logger.info("LightGBM模型训练完成（使用验证集）")
            else:
                # 对于其他模型，使用标准的fit方法
                self.model.fit(X_train_data, y_train_data)
                logger.info("模型训练完成")
            
            # 获取特征重要性
            self._extract_feature_importance(X_train.columns)
            
            return self.model
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            raise
    
    def _extract_feature_importance(self, feature_names: List[str]):
        """提取特征重要性"""
        try:
            if hasattr(self.model, 'feature_importance'):
                importance_values = self.model.feature_importance()
                
                self.feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance_values
                }).sort_values('importance', ascending=False)
                
                logger.info(f"特征重要性提取完成: {len(self.feature_importance)} 个特征")
                
                # 打印Top 10特征
                logger.info("Top 10 特征重要性:")
                for i, row in self.feature_importance.head(10).iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']:.4f}")
                    
        except Exception as e:
            logger.warning(f"提取特征重要性失败: {e}")
            self.feature_importance = None
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测
        
        Args:
            X: 特征矩阵
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 预测概率和预测类别
        """
        if self.model is None:
            raise ValueError("模型未训练，请先训练模型")
        
        try:
            # 预测概率
            y_pred_proba = self.model.predict(X)
            
            # 转换为类别（阈值0.5）
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            return y_pred_proba, y_pred
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            raise
    
    def evaluate_model(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X: 特征矩阵
            y_true: 真实标签
            
        Returns:
            Dict[str, float]: 评估指标
        """
        logger.info("评估模型性能...")
        
        # 预测
        y_pred_proba, y_pred = self.predict(X)
        
        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
        }
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # 计算类别分布
        metrics['class_distribution'] = {
            'true_positive': int(cm[1, 1]),
            'true_negative': int(cm[0, 0]),
            'false_positive': int(cm[0, 1]),
            'false_negative': int(cm[1, 0])
        }
        
        # 计算额外指标
        total = len(y_true)
        metrics['sample_count'] = total
        metrics['positive_ratio'] = y_true.mean()
        metrics['predicted_positive_ratio'] = y_pred.mean()
        
        logger.info("模型评估完成:")
        logger.info(f"  准确率: {metrics['accuracy']:.4f}")
        logger.info(f"  精确率: {metrics['precision']:.4f}")
        logger.info(f"  召回率: {metrics['recall']:.4f}")
        logger.info(f"  F1分数: {metrics['f1_score']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  样本数量: {metrics['sample_count']}")
        logger.info(f"  正类比例: {metrics['positive_ratio']:.2%}")
        
        return metrics
    
    def time_series_split(self, X: pd.DataFrame, y: pd.Series, 
                         n_splits: int = 5) -> List[Tuple]:
        """
        时间序列数据划分
        
        Args:
            X: 特征数据
            y: 目标数据
            n_splits: 划分数量
            
        Returns:
            List[Tuple]: 划分结果列表 [(X_train, y_train, X_val, y_val), ...]
        """
        logger.info(f"时间序列数据划分，{n_splits}折")
        
        # 确保数据按时间排序
        if not isinstance(X.index, pd.DatetimeIndex):
            logger.warning("数据索引不是DatetimeIndex，使用简单划分")
            # 简单划分
            split_idx = len(X) // n_splits
            splits = []
            
            for i in range(n_splits - 1):
                val_start = i * split_idx
                val_end = (i + 1) * split_idx
                
                X_train = X.iloc[:val_start]
                y_train = y.iloc[:val_start]
                X_val = X.iloc[val_start:val_end]
                y_val = y.iloc[val_start:val_end]
                
                splits.append((X_train, y_train, X_val, y_val))
            
            return splits
        
        # 时间序列划分
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        
        fold = 1
        for train_idx, val_idx in tscv.split(X):
            logger.info(f"划分第 {fold}/{n_splits} 折...")
            
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            splits.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
            fold += 1
        
        logger.info(f"时间序列划分完成: {len(splits)} 折")
        return splits
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      n_splits: int = 5) -> Dict[str, Any]:
        """
        交叉验证
        
        Args:
            X: 特征数据
            y: 目标数据
            n_splits: 交叉验证折数
            
        Returns:
            Dict[str, Any]: 交叉验证结果
        """
        logger.info(f"开始交叉验证，{n_splits}折")
        
        # 数据划分
        splits = self.time_series_split(X, y, n_splits)
        
        cv_results = {
            'fold_metrics': [],
            'models': [],
            'feature_importances': []
        }
        
        fold = 1
        for X_train, y_train, X_val, y_val in splits:
            logger.info(f"训练第 {fold}/{n_splits} 折...")
            
            # 训练模型
            model = self.train_model(X_train, y_train, X_val, y_val)
            
            # 评估模型
            train_metrics = self.evaluate_model(X_train, y_train)
            val_metrics = self.evaluate_model(X_val, y_val)
            
            # 记录结果
            fold_result = {
                'fold': fold,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            
            cv_results['fold_metrics'].append(fold_result)
            cv_results['models'].append(model)
            
            if self.feature_importance is not None:
                cv_results['feature_importances'].append(self.feature_importance.copy())
            
            logger.info(f"第 {fold} 折 - 验证准确率: {val_metrics['accuracy']:.4f}")
            fold += 1
        
        # 计算平均指标
        if cv_results['fold_metrics']:
            val_accuracies = [m['val_metrics']['accuracy'] for m in cv_results['fold_metrics']]
            val_f1_scores = [m['val_metrics']['f1_score'] for m in cv_results['fold_metrics']]
            
            cv_results['mean_val_accuracy'] = np.mean(val_accuracies)
            cv_results['std_val_accuracy'] = np.std(val_accuracies)
            cv_results['mean_val_f1'] = np.mean(val_f1_scores)
            cv_results['std_val_f1'] = np.std(val_f1_scores)
        
        logger.info(f"交叉验证完成:")
        logger.info(f"  平均验证准确率: {cv_results.get('mean_val_accuracy', 0):.4f} "
                   f"(±{cv_results.get('std_val_accuracy', 0):.4f})")
        logger.info(f"  平均验证F1分数: {cv_results.get('mean_val_f1', 0):.4f} "
                   f"(±{cv_results.get('std_val_f1', 0):.4f})")
        
        return cv_results
    
    def save_model(self, name: str):
        """
        保存模型
        
        Args:
            name: 模型名称
        """
        if self.model is None:
            raise ValueError("模型未训练，无法保存")
        
        # 保存模型
        model_path = self.model_dir / f"{name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # 保存特征重要性
        if self.feature_importance is not None:
            importance_path = self.model_dir / f"{name}_feature_importance.csv"
            self.feature_importance.to_csv(importance_path, index=False)
        
        # 保存配置
        config_path = self.model_dir / f"{name}_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"模型已保存到: {model_path}")
        logger.info(f"特征重要性已保存到: {importance_path}")
        logger.info(f"配置已保存到: {config_path}")
    
    def load_model(self, name: str):
        """
        加载模型
        
        Args:
            name: 模型名称
        """
        model_path = self.model_dir / f"{name}_model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # 加载特征重要性
        importance_path = self.model_dir / f"{name}_feature_importance.csv"
        if importance_path.exists():
            self.feature_importance = pd.read_csv(importance_path)
        
        logger.info(f"模型已从 {model_path} 加载")
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
        """
        绘制特征重要性图
        
        Args:
            top_n: 显示前N个特征
            save_path: 保存路径（可选）
        """
        if self.feature_importance is None:
            logger.warning("特征重要性数据不存在")
            return
        
        # 取前top_n个特征
        top_features = self.feature_importance.head(top_n)
        
        plt.figure(figsize=(12, 8))
        
        if HAS_SEABORN:
            sns.barplot(x='importance', y='feature', data=top_features)
        else:
            # 使用matplotlib的基本绘图
            plt.barh(range(len(top_features)), top_features['importance'].values)
            plt.yticks(range(len(top_features)), top_features['feature'].values)
        
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"特征重要性图已保存到: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None):
        """
        绘制混淆矩阵
        
        Args:
            cm: 混淆矩阵
            save_path: 保存路径（可选）
        """
        plt.figure(figsize=(8, 6))
        
        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Predicted 0', 'Predicted 1'],
                       yticklabels=['Actual 0', 'Actual 1'])
        else:
            # 使用matplotlib的基本绘图
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['Predicted 0', 'Predicted 1'])
            plt.yticks(tick_marks, ['Actual 0', 'Actual 1'])
            
            # 添加文本标注
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵图已保存到: {save_path}")
        
        plt.show()
    
    def generate_report(self, metrics: Dict[str, Any], 
                       cv_results: Optional[Dict[str, Any]] = None) -> str:
        """
        生成模型报告
        
        Args:
            metrics: 模型评估指标
            cv_results: 交叉验证结果（可选）
            
        Returns:
            str: 模型报告
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FZT模型训练报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"模型类型: {self.model_class}")
        report_lines.append("")
        
        # 模型配置
        report_lines.append("模型配置:")
        for key, value in self.model_kwargs.items():
            report_lines.append(f"  {key}: {value}")
        report_lines.append("")
        
        # 评估指标
        report_lines.append("模型性能指标:")
        report_lines.append(f"  准确率 (Accuracy): {metrics.get('accuracy', 0):.4f}")
        report_lines.append(f"  精确率 (Precision): {metrics.get('precision', 0):.4f}")
        report_lines.append(f"  召回率 (Recall): {metrics.get('recall', 0):.4f}")
        report_lines.append(f"  F1分数 (F1-Score): {metrics.get('f1_score', 0):.4f}")
        report_lines.append(f"  ROC AUC: {metrics.get('roc_auc', 0):.4f}")
        report_lines.append(f"  样本数量: {metrics.get('sample_count', 0)}")
        report_lines.append(f"  正类比例: {metrics.get('positive_ratio', 0):.2%}")
        report_lines.append("")
        
        # 混淆矩阵
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            report_lines.append("混淆矩阵:")
            report_lines.append(f"  [[{cm[0][0]:4d}  {cm[0][1]:4d}]")
            report_lines.append(f"   [{cm[1][0]:4d}  {cm[1][1]:4d}]]")
            report_lines.append("")
        
        # 交叉验证结果
        if cv_results:
            report_lines.append("交叉验证结果:")
            report_lines.append(f"  平均验证准确率: {cv_results.get('mean_val_accuracy', 0):.4f} "
                              f"(±{cv_results.get('std_val_accuracy', 0):.4f})")
            report_lines.append(f"  平均验证F1分数: {cv_results.get('mean_val_f1', 0):.4f} "
                              f"(±{cv_results.get('std_val_f1', 0):.4f})")
            report_lines.append("")
        
        # 特征重要性
        if self.feature_importance is not None:
            report_lines.append("Top 10 特征重要性:")
            for i, row in self.feature_importance.head(10).iterrows():
                report_lines.append(f"  {row['feature']}: {row['importance']:.4f}")
        
        report = "\n".join(report_lines)
        return report


def test_model_trainer():
    """测试模型训练器"""
    try:
        print("=" * 80)
        print("模型训练模块测试")
        print("=" * 80)
        
        # 创建测试数据
        print("\n1. 创建测试数据...")
        np.random.seed(42)
        n_samples = 200
        n_features = 20
        
        # 创建特征
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)],
            index=pd.date_range('2023-01-01', periods=n_samples, freq='D')
        )
        
        # 创建目标变量（与某些特征相关）
        y_continuous = (
            0.3 * X['feature_0'] + 
            0.2 * X['feature_1'] + 
            0.1 * X['feature_2'] + 
            np.random.randn(n_samples) * 0.1
        )
        
        # 转换为二分类（大于均值为正类）
        y = (y_continuous > y_continuous.mean()).astype(int)
        y = pd.Series(y, index=X.index, name='target')
        
        print(f"   数据形状: X={X.shape}, y={y.shape}")
        print(f"   正类比例: {y.mean():.2%}")
        
        # 初始化模型训练器
        print("\n2. 初始化模型训练器...")
        trainer = ModelTrainer()
        print("   ✓ 初始化成功")
        
        # 准备数据
        print("\n3. 准备数据...")
        features = X.copy()
        features['target'] = y
        X_prepared, y_prepared = trainer.prepare_data(features, threshold=0.0)
        print(f"   ✓ 数据准备完成: X={X_prepared.shape}, y={y_prepared.shape}")
        
        # 划分训练验证集
        print("\n4. 划分训练验证集...")
        split_idx = int(len(X_prepared) * 0.7)
        X_train = X_prepared.iloc[:split_idx]
        y_train = y_prepared.iloc[:split_idx]
        X_val = X_prepared.iloc[split_idx:]
        y_val = y_prepared.iloc[split_idx:]
        
        print(f"   训练集: {X_train.shape}, 验证集: {X_val.shape}")
        
        # 训练模型
        print("\n5. 训练模型...")
        model = trainer.train_model(X_train, y_train, X_val, y_val)
        print("   ✓ 模型训练完成")
        
        # 评估模型
        print("\n6. 评估模型...")
        train_metrics = trainer.evaluate_model(X_train, y_train)
        val_metrics = trainer.evaluate_model(X_val, y_val)
        
        print(f"   训练准确率: {train_metrics['accuracy']:.4f}")
        print(f"   验证准确率: {val_metrics['accuracy']:.4f}")
        
        # 交叉验证
        print("\n7. 交叉验证...")
        cv_results = trainer.cross_validate(X_prepared, y_prepared, n_splits=3)
        print(f"   ✓ 交叉验证完成")
        print(f"   平均验证准确率: {cv_results.get('mean_val_accuracy', 0):.4f}")
        
        # 生成报告
        print("\n8. 生成模型报告...")
        report = trainer.generate_report(val_metrics, cv_results)
        print(report[:500] + "..." if len(report) > 500 else report)
        
        # 保存模型
        print("\n9. 保存模型...")
        trainer.save_model('test_model')
        print("   ✓ 模型保存完成")
        
        # 绘制特征重要性
        print("\n10. 绘制特征重要性...")
        plot_path = trainer.model_dir / "test_model_feature_importance.png"
        trainer.plot_feature_importance(top_n=10, save_path=str(plot_path))
        print(f"   ✓ 特征重要性图已保存到: {plot_path}")
        
        print("\n" + "=" * 80)
        print("模型训练模块测试完成！所有功能正常。")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_model_trainer())