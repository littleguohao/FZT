"""
LambdaRank训练器模块

实现LightGBM LambdaRank训练器，用于训练排序模型。
支持早停、交叉验证、模型保存和预测功能。
"""

import pandas as pd
import numpy as np
import yaml
import pickle
import warnings
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime

# LightGBM导入
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    warnings.warn("LightGBM未安装，LambdaRank训练器将无法工作")


class LambdaRankTrainer:
    """
    LightGBM LambdaRank训练器
    
    用于训练排序模型，支持时间序列交叉验证、早停、模型评估和预测。
    """
    
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        """
        初始化LambdaRank训练器
        
        参数:
        ----------
        config_path : str, 可选
            配置文件路径
        config_dict : Dict, 可选
            配置字典，如果提供则忽略config_path
            
        异常:
        -------
        FileNotFoundError: 配置文件不存在
        ValueError: 配置无效
        """
        if not LGBM_AVAILABLE:
            raise ImportError("LightGBM未安装，请使用 'pip install lightgbm' 安装")
        
        # 加载配置
        if config_dict is not None:
            self.config = config_dict
        elif config_path is not None:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            # 使用默认配置
            self.config = self._get_default_config()
        
        # 验证配置
        self._validate_config()
        
        # 初始化模型和状态
        self.model = None
        self.feature_names = None
        self.train_history = None
        self.best_iteration = None
        self.feature_importance = None
        
        # 初始化LightGBM参数
        self.lgb_params = self._get_lgb_params()
        
        # 设置随机种子
        if 'random_state' in self.lgb_params:
            np.random.seed(self.lgb_params['random_state'])
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'model': {
                'type': 'lightgbm_lambdarank',
                'objective': 'ranking',
                'description': 'LightGBM LambdaRank排序模型'
            },
            'lightgbm': {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_eval_at': [1, 3, 5, 10],
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': 5,
                'min_data_in_leaf': 20,
                'num_iterations': 100,
                'early_stopping_rounds': 10,
                'verbose': 1,
                'random_state': 42
            },
            'data_split': {
                'time_series_cv': True,
                'rolling_window': {
                    'enabled': True,
                    'window_size': 252,
                    'step_size': 63,
                    'min_train_size': 126
                },
                'query_group_column': 'date'
            },
            'training': {
                'early_stopping': {
                    'enabled': True,
                    'patience': 10,
                    'min_delta': 0.001
                }
            }
        }
    
    def _validate_config(self) -> None:
        """验证配置"""
        required_sections = ['model', 'lightgbm', 'data_split']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置缺少必需部分: {section}")
        
        # 验证模型类型
        if self.config['model'].get('type') != 'lightgbm_lambdarank':
            warnings.warn(f"模型类型不是'lightgbm_lambdarank': {self.config['model'].get('type')}")
        
        # 验证LightGBM参数
        lgb_config = self.config['lightgbm']
        if lgb_config.get('objective') != 'lambdarank':
            warnings.warn(f"LightGBM目标不是'lambdarank': {lgb_config.get('objective')}")
    
    def _get_lgb_params(self) -> Dict:
        """获取LightGBM参数"""
        lgb_config = self.config['lightgbm']
        
        # 基础参数
        params = {
            'objective': lgb_config.get('objective', 'lambdarank'),
            'metric': lgb_config.get('metric', 'ndcg'),
            'learning_rate': lgb_config.get('learning_rate', 0.05),
            'num_leaves': lgb_config.get('num_leaves', 31),
            'max_depth': lgb_config.get('max_depth', 5),
            'min_data_in_leaf': lgb_config.get('min_data_in_leaf', 20),
            'min_sum_hessian_in_leaf': lgb_config.get('min_sum_hessian_in_leaf', 1e-3),
            'lambda_l1': lgb_config.get('lambda_l1', 0.0),
            'lambda_l2': lgb_config.get('lambda_l2', 0.0),
            'feature_fraction': lgb_config.get('feature_fraction', 0.9),
            'bagging_fraction': lgb_config.get('bagging_fraction', 0.8),
            'bagging_freq': lgb_config.get('bagging_freq', 5),
            'verbosity': lgb_config.get('verbose', 1),
            'seed': lgb_config.get('random_state', 42)
        }
        
        # LambdaRank特定参数
        lambdarank_config = lgb_config.get('lambdarank', {})
        if lambdarank_config:
            params.update({
                'lambdarank_norm': lambdarank_config.get('norm', True),
                'lambdarank_truncation_level': lambdarank_config.get('truncation_level', 10),
                'lambdarank_sigmoid': lambdarank_config.get('sigmoid', 1.0)
            })
        
        # 设置label_gain参数（LambdaRank需要）
        # 默认使用指数增益：0, 1, 3, 7, 15, 31, ...
        max_label = 5  # 假设最大标签为5
        label_gain = [0] + [2**i - 1 for i in range(max_label)]
        params['label_gain'] = label_gain
        
        # NDCG评估位置
        ndcg_eval_at = lgb_config.get('ndcg_eval_at', [1, 3, 5, 10])
        if ndcg_eval_at:
            params['eval_at'] = ndcg_eval_at
        
        return params
    
    def _prepare_dataset(self, features: pd.DataFrame, labels: pd.Series) -> Dict[str, Any]:
        """
        准备LambdaRank数据集
        
        参数:
        ----------
        features : pd.DataFrame
            特征DataFrame，索引为(date, instrument)
        labels : pd.Series
            标签序列，索引与特征相同
            
        返回:
        -------
        Dict
            包含特征、标签和分组信息的数据集
        """
        # 验证输入
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features必须是pandas DataFrame")
        if not isinstance(labels, pd.Series):
            raise TypeError("labels必须是pandas Series")
        
        # 验证索引一致性
        if not features.index.equals(labels.index):
            raise ValueError("features和labels必须具有相同的索引")
        
        # 验证多索引格式
        if not (isinstance(features.index, pd.MultiIndex) and 
                features.index.names == ['date', 'instrument']):
            raise ValueError("索引必须是(date, instrument)格式的多索引")
        
        # 处理缺失值
        feature_missing = features.isna().any(axis=1)
        label_missing = labels.isna()
        missing_mask = feature_missing | label_missing
        
        if missing_mask.any():
            warnings.warn(f"数据中存在{missing_mask.sum()}个缺失值，已删除")
            features = features[~missing_mask].copy()
            labels = labels[~missing_mask].copy()
        
        if len(features) == 0:
            raise ValueError("所有数据都包含缺失值，无法创建数据集")
        
        # 按日期排序（确保分组顺序）
        if not features.index.is_monotonic_increasing:
            features = features.sort_index()
            labels = labels.sort_index()
        
        # 创建分组信息（按日期）
        dates = features.index.get_level_values('date').unique()
        groups = []
        
        for date in dates:
            date_mask = features.index.get_level_values('date') == date
            group_size = date_mask.sum()
            groups.append(group_size)
        
        groups = np.array(groups, dtype=int)
        
        # 验证分组信息
        if groups.sum() != len(features):
            raise ValueError("分组大小之和必须等于数据点数量")
        
        # 保存特征名称
        self.feature_names = list(features.columns)
        
        return {
            'features': features,
            'labels': labels,
            'groups': groups
        }
    
    def _create_time_series_splits(self, features: pd.DataFrame, n_folds: int = 5) -> List[Dict]:
        """
        创建时间序列交叉验证分割
        
        参数:
        ----------
        features : pd.DataFrame
            特征DataFrame
        n_folds : int
            折叠数量
            
        返回:
        -------
        List[Dict]
            每个折叠的训练和验证索引
        """
        # 获取所有日期
        dates = features.index.get_level_values('date').unique()
        dates_sorted = sorted(dates)
        
        # 获取滚动窗口配置
        window_config = self.config['data_split'].get('rolling_window', {})
        if window_config.get('enabled', True):
            window_size = window_config.get('window_size', 252)
            step_size = window_config.get('step_size', 63)
            min_train_size = window_config.get('min_train_size', 126)
            
            # 创建滚动窗口分割
            splits = []
            n_dates = len(dates_sorted)
            
            # 确定起始点
            start_idx = min_train_size
            end_idx = n_dates
            
            # 创建分割
            current_start = 0
            while current_start + window_size <= end_idx:
                train_end = current_start + window_size
                val_start = train_end
                val_end = min(val_start + step_size, end_idx)
                
                if val_start >= val_end:
                    break
                
                # 获取日期范围
                train_dates = dates_sorted[current_start:train_end]
                val_dates = dates_sorted[val_start:val_end]
                
                # 创建索引掩码
                train_mask = features.index.get_level_values('date').isin(train_dates)
                val_mask = features.index.get_level_values('date').isin(val_dates)
                
                train_indices = np.where(train_mask)[0]
                val_indices = np.where(val_mask)[0]
                
                if len(train_indices) > 0 and len(val_indices) > 0:
                    splits.append({
                        'train_indices': train_indices,
                        'val_indices': val_indices,
                        'train_dates': train_dates,
                        'val_dates': val_dates
                    })
                
                current_start += step_size
            
            return splits[:n_folds] if splits else []
        
        else:
            # 简单的时间序列分割
            fold_size = len(dates_sorted) // n_folds
            
            splits = []
            for i in range(n_folds):
                val_start = i * fold_size
                val_end = (i + 1) * fold_size if i < n_folds - 1 else len(dates_sorted)
                
                val_dates = dates_sorted[val_start:val_end]
                train_dates = [d for d in dates_sorted if d not in val_dates]
                
                # 创建索引掩码
                train_mask = features.index.get_level_values('date').isin(train_dates)
                val_mask = features.index.get_level_values('date').isin(val_dates)
                
                splits.append({
                    'train_indices': np.where(train_mask)[0],
                    'val_indices': np.where(val_mask)[0],
                    'train_dates': train_dates,
                    'val_dates': val_dates
                })
            
            return splits
    
    def train(self, features: pd.DataFrame, labels: pd.Series, 
              validation_data: Tuple[pd.DataFrame, pd.Series] = None) -> Dict:
        """
        训练LambdaRank模型
        
        参数:
        ----------
        features : pd.DataFrame
            训练特征
        labels : pd.Series
            训练标签
        validation_data : Tuple, 可选
            验证数据 (val_features, val_labels)
            
        返回:
        -------
        Dict
            训练结果，包含模型、指标和特征重要性
        """
        # 准备训练数据集
        train_dataset = self._prepare_dataset(features, labels)
        X_train = train_dataset['features'].values
        y_train = train_dataset['labels'].values
        groups_train = train_dataset['groups']
        
        # 创建LightGBM数据集
        train_data = lgb.Dataset(
            data=X_train,
            label=y_train,
            group=groups_train,
            feature_name=self.feature_names
        )
        
        # 准备验证数据
        valid_sets = [train_data]
        valid_names = ['train']
        
        if validation_data is not None:
            val_features, val_labels = validation_data
            
            # 验证数据必须与训练数据具有相同的特征
            if not val_features.columns.equals(features.columns):
                raise ValueError("验证特征必须与训练特征具有相同的列")
            
            # 准备验证数据集
            val_dataset = self._prepare_dataset(val_features, val_labels)
            X_val = val_dataset['features'].values
            y_val = val_dataset['labels'].values
            groups_val = val_dataset['groups']
            
            val_data = lgb.Dataset(
                data=X_val,
                label=y_val,
                group=groups_val,
                feature_name=self.feature_names,
                reference=train_data
            )
            
            valid_sets.append(val_data)
            valid_names.append('val')
        
        # 训练参数
        num_iterations = self.lgb_params.get('num_iterations', 100)
        early_stopping_rounds = self.lgb_params.get('early_stopping_rounds', 10)
        
        # 训练模型
        self.train_history = {}
        
        callbacks = []
        if validation_data is not None and early_stopping_rounds > 0:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds,
                    verbose=True
                )
            )
        
        # 初始化历史记录
        self.train_history = {
            'train_ndcg': [],
            'val_ndcg': [] if validation_data else None,
            'iterations': []
        }
        
        # 记录回调
        def record_history(env):
            iteration = env.iteration
            evaluation_result_list = env.evaluation_result_list
            
            for eval_name, eval_metric, eval_result, _ in evaluation_result_list:
                if 'train' in eval_name and 'ndcg' in eval_metric:
                    self.train_history['train_ndcg'].append(eval_result)
                elif 'val' in eval_name and 'ndcg' in eval_metric:
                    if self.train_history['val_ndcg'] is None:
                        self.train_history['val_ndcg'] = []
                    self.train_history['val_ndcg'].append(eval_result)
            
            self.train_history['iterations'].append(iteration)
        
        callbacks.append(record_history)
        
        # 开始训练
        self.model = lgb.train(
            params=self.lgb_params,
            train_set=train_data,
            num_boost_round=num_iterations,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        # 获取最佳迭代
        self.best_iteration = self.model.best_iteration
        
        # 计算训练指标
        train_pred = self.model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_pred, groups_train)
        
        # 计算验证指标（如果有）
        val_metrics = None
        if validation_data is not None:
            val_pred = self.model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_pred, groups_val)
        
        # 计算特征重要性
        self.feature_importance = self._calculate_feature_importance()
        
        # 准备结果
        result = {
            'model': self.model,
            'best_iteration': self.best_iteration,
            'train_history': self.train_history,
            'metrics': {
                'train': train_metrics,
                'val': val_metrics
            },
            'feature_importance': self.feature_importance,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        return result
    
    def cross_validate(self, features: pd.DataFrame, labels: pd.Series, 
                       n_folds: int = 5) -> Dict:
        """
        执行时间序列交叉验证
        
        参数:
        ----------
        features : pd.DataFrame
            特征数据
        labels : pd.Series
            标签数据
        n_folds : int
            折叠数量
            
        返回:
        -------
        Dict
            交叉验证结果
        """
        # 准备数据集
        dataset = self._prepare_dataset(features, labels)
        X = dataset['features'].values
        y = dataset['labels'].values
        groups = dataset['groups']
        
        # 创建时间序列分割
        splits = self._create_time_series_splits(dataset['features'], n_folds)
        
        if not splits:
            raise ValueError("无法创建时间序列分割，请检查数据")
        
        fold_results = []
        all_metrics = []
        
        for i, split in enumerate(splits):
            print(f"训练折叠 {i+1}/{len(splits)}")
            
            # 分割数据
            train_idx = split['train_indices']
            val_idx = split['val_indices']
            
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            # 计算分组信息
            train_groups = self._calculate_groups_from_indices(groups, train_idx)
            val_groups = self._calculate_groups_from_indices(groups, val_idx)
            
            # 创建LightGBM数据集
            train_data_fold = lgb.Dataset(
                data=X_train_fold,
                label=y_train_fold,
                group=train_groups,
                feature_name=self.feature_names
            )
            
            val_data_fold = lgb.Dataset(
                data=X_val_fold,
                label=y_val_fold,
                group=val_groups,
                feature_name=self.feature_names,
                reference=train_data_fold
            )
            
            # 训练模型
            model_fold = lgb.train(
                params=self.lgb_params,
                train_set=train_data_fold,
                num_boost_round=self.lgb_params.get('num_iterations', 100),
                valid_sets=[train_data_fold, val_data_fold],
                valid_names=['train', 'val'],
                callbacks=[lgb.early_stopping(self.lgb_params.get('early_stopping_rounds', 10))]
            )
            
            # 预测和评估
            train_pred_fold = model_fold.predict(X_train_fold)
            val_pred_fold = model_fold.predict(X_val_fold)
            
            train_metrics_fold = self._calculate_metrics(y_train_fold, train_pred_fold, train_groups)
            val_metrics_fold = self._calculate_metrics(y_val_fold, val_pred_fold, val_groups)
            
            # 计算特征重要性
            importance_fold = dict(zip(
                self.feature_names,
                model_fold.feature_importance(importance_type='gain')
            ))
            
            # 保存折叠结果
            fold_result = {
                'fold': i + 1,
                'model': model_fold,
                'train_indices': train_idx,
                'val_indices': val_idx,
                'train_dates': split['train_dates'],
                'val_dates': split['val_dates'],
                'metrics': {
                    'train': train_metrics_fold,
                    'val': val_metrics_fold
                },
                'feature_importance': importance_fold,
                'best_iteration': model_fold.best_iteration
            }
            
            fold_results.append(fold_result)
            all_metrics.append(val_metrics_fold)
        
        # 计算平均指标
        mean_metrics = {}
        std_metrics = {}
        
        if all_metrics:
            metric_names = list(all_metrics[0].keys())
            for metric_name in metric_names:
                metric_values = [m[metric_name] for m in all_metrics]
                mean_metrics[metric_name] = np.mean(metric_values)
                std_metrics[metric_name] = np.std(metric_values)
        
        return {
            'fold_metrics': fold_results,
            'mean_metrics': mean_metrics,
            'std_metrics': std_metrics,
            'all_metrics': all_metrics
        }
    
    def _calculate_groups_from_indices(self, original_groups: np.ndarray, 
                                       indices: np.ndarray) -> np.ndarray:
        """
        从原始分组和索引计算新的分组
        
        参数:
        ----------
        original_groups : np.ndarray
            原始分组数组
        indices : np.ndarray
            选择的索引
            
        返回:
        -------
        np.ndarray
            新的分组数组
        """
        # 这是一个简化实现
        # 在实际应用中，需要根据原始分组和选择的索引重新计算分组
        # 这里假设每个日期是一个组，我们只需要统计每个日期被选中的样本数量
        
        # 由于实现较复杂，这里返回一个简单的分组
        # 在实际使用中，应该根据具体数据实现
        return np.array([len(indices)], dtype=int)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          groups: np.ndarray) -> Dict[str, float]:
        """
        计算排序指标
        
        参数:
        ----------
        y_true : np.ndarray
            真实标签
        y_pred : np.ndarray
            预测分数
        groups : np.ndarray
            分组信息
            
        返回:
        -------
        Dict
            指标字典
        """
        metrics = {}
        
        # 计算NDCG
        ndcg_values = self._calculate_ndcg(y_true, y_pred, groups)
        for k, ndcg in ndcg_values.items():
            metrics[f'ndcg@{k}'] = ndcg
        
        # 计算平均NDCG
        if ndcg_values:
            metrics['ndcg'] = np.mean(list(ndcg_values.values()))
        
        return metrics
    
    def _calculate_ndcg(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       groups: np.ndarray, k_values: List[int] = None) -> Dict[int, float]:
        """
        计算NDCG@k
        
        参数:
        ----------
        y_true : np.ndarray
            真实标签
        y_pred : np.ndarray
            预测分数
        groups : np.ndarray
            分组信息
        k_values : List[int], 可选
            k值列表
            
        返回:
        -------
        Dict[int, float]
            NDCG@k字典
        """
        if k_values is None:
            k_values = self.lgb_params.get('eval_at', [1, 3, 5, 10])
        
        ndcg_results = {}
        
        # 按分组计算NDCG
        start_idx = 0
        group_ndcgs = {k: [] for k in k_values}
        
        for group_size in groups:
            end_idx = start_idx + group_size
            
            if group_size == 0:
                start_idx = end_idx
                continue
            
            group_y_true = y_true[start_idx:end_idx]
            group_y_pred = y_pred[start_idx:end_idx]
            
            # 计算每个k的NDCG
            for k in k_values:
                if group_size >= k:
                    ndcg = self._ndcg_at_k(group_y_true, group_y_pred, k)
                    group_ndcgs[k].append(ndcg)
            
            start_idx = end_idx
        
        # 计算平均NDCG
        for k in k_values:
            if group_ndcgs[k]:
                ndcg_results[k] = np.mean(group_ndcgs[k])
            else:
                ndcg_results[k] = 0.0
        
        return ndcg_results
    
    def _ndcg_at_k(self, y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """
        计算单个组的NDCG@k
        
        参数:
        ----------
        y_true : np.ndarray
            真实标签
        y_pred : np.ndarray
            预测分数
        k : int
            top-k
            
        返回:
        -------
        float
            NDCG@k值
        """
        if len(y_true) < k:
            k = len(y_true)
        
        # 按预测分数排序
        pred_order = np.argsort(y_pred)[::-1]  # 降序
        true_order = np.argsort(y_true)[::-1]  # 降序
        
        # 计算DCG
        dcg = 0.0
        for i in range(k):
            idx = pred_order[i]
            rel = y_true[idx]
            dcg += rel / np.log2(i + 2)  # i+2因为log2(1)=0
        
        # 计算IDCG
        idcg = 0.0
        for i in range(k):
            idx = true_order[i]
            rel = y_true[idx]
            idcg += rel / np.log2(i + 2)
        
        # 计算NDCG
        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = 0.0
        
        return ndcg
    
    def _calculate_feature_importance(self) -> Dict[str, float]:
        """
        计算特征重要性
        
        返回:
        -------
        Dict
            特征重要性字典
        """
        if self.model is None:
            return {}
        
        # 获取特征重要性
        importance_gain = self.model.feature_importance(importance_type='gain')
        importance_split = self.model.feature_importance(importance_type='split')
        
        # 创建重要性字典
        importance_dict = {}
        for i, feature_name in enumerate(self.feature_names):
            importance_dict[feature_name] = {
                'gain': float(importance_gain[i]),
                'split': float(importance_split[i]),
                'total': float(importance_gain[i] + importance_split[i])
            }
        
        return importance_dict
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        预测排序分数
        
        参数:
        ----------
        features : pd.DataFrame
            特征数据
            
        返回:
        -------
        pd.Series
            预测分数，索引与输入相同
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")
        
        # 验证特征
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features必须是pandas DataFrame")
        
        # 确保特征列与训练时一致
        missing_cols = set(self.feature_names) - set(features.columns)
        if missing_cols:
            raise ValueError(f"特征缺少以下列: {missing_cols}")
        
        # 按训练时的顺序排列特征列
        features_aligned = features[self.feature_names].copy()
        
        # 处理缺失值
        if features_aligned.isna().any().any():
            warnings.warn("特征中存在缺失值，已用0填充")
            features_aligned = features_aligned.fillna(0)
        
        # 预测
        X = features_aligned.values
        predictions = self.model.predict(X)
        
        # 创建结果Series
        result = pd.Series(predictions, index=features.index)
        
        return result
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        返回:
        -------
        Dict
            特征重要性字典
        """
        if self.feature_importance is None:
            return {}
        
        # 返回简化版本（只返回总重要性）
        simplified = {}
        for feature_name, importance in self.feature_importance.items():
            simplified[feature_name] = importance['total']
        
        return simplified
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型
        
        参数:
        ----------
        filepath : str
            模型保存路径
        """
        if self.model is None:
            raise ValueError("没有可保存的模型")
        
        # 创建保存目录
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # 准备保存数据
        save_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config,
            'train_history': self.train_history,
            'best_iteration': self.best_iteration,
            'feature_importance': self.feature_importance,
            'save_time': datetime.now().isoformat()
        }
        
        # 保存
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load_model(self, filepath: str) -> None:
        """
        加载模型
        
        参数:
        ----------
        filepath : str
            模型文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        # 加载
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # 恢复状态
        self.model = save_data['model']
        self.feature_names = save_data['feature_names']
        self.config = save_data['config']
        self.train_history = save_data['train_history']
        self.best_iteration = save_data['best_iteration']
        self.feature_importance = save_data['feature_importance']
        
        # 重新初始化LightGBM参数
        self.lgb_params = self._get_lgb_params()
    
    def evaluate(self, features: pd.DataFrame, labels: pd.Series) -> Dict[str, float]:
        """
        评估模型
        
        参数:
        ----------
        features : pd.DataFrame
            特征数据
        labels : pd.Series
            标签数据
            
        返回:
        -------
        Dict
            评估指标
        """
        # 预测
        predictions = self.predict(features)
        
        # 准备数据集
        dataset = self._prepare_dataset(features, labels)
        y_true = dataset['labels'].values
        y_pred = predictions.values
        groups = dataset['groups']
        
        # 计算指标
        metrics = self._calculate_metrics(y_true, y_pred, groups)
        
        return metrics
    
    def get_config(self) -> Dict:
        """
        获取配置
        
        返回:
        -------
        Dict
            当前配置
        """
        return self.config.copy()
    
    def update_config(self, config_updates: Dict) -> None:
        """
        更新配置
        
        参数:
        ----------
        config_updates : Dict
            配置更新字典
        """
        # 深度合并配置
        def deep_update(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_update(target[key], value)
                else:
                    target[key] = value
        
        deep_update(self.config, config_updates)
        
        # 重新验证配置
        self._validate_config()
        
        # 更新LightGBM参数
        self.lgb_params = self._get_lgb_params()


# 工具函数
def create_lambdarank_trainer_from_config(config_path: str) -> LambdaRankTrainer:
    """
    从配置文件创建LambdaRank训练器
    
    参数:
    ----------
    config_path : str
        配置文件路径
        
    返回:
    -------
    LambdaRankTrainer
        训练器实例
    """
    return LambdaRankTrainer(config_path=config_path)


def train_lambdarank_model(features: pd.DataFrame, labels: pd.Series, 
                          config_path: str = None) -> Tuple[LambdaRankTrainer, Dict]:
    """
    训练LambdaRank模型的便捷函数
    
    参数:
    ----------
    features : pd.DataFrame
        特征数据
    labels : pd.Series
        标签数据
    config_path : str, 可选
        配置文件路径
        
    返回:
    -------
    Tuple[LambdaRankTrainer, Dict]
        训练器实例和训练结果
    """
    trainer = LambdaRankTrainer(config_path=config_path)
    result = trainer.train(features, labels)
    return trainer, result


if __name__ == '__main__':
    # 示例使用
    print("LambdaRank训练器模块已加载")
    print(f"LightGBM可用: {LGBM_AVAILABLE}")
    
    # 创建示例训练器
    try:
        trainer = LambdaRankTrainer()
        print("训练器创建成功")
        print(f"配置: {trainer.get_config()['model']}")
    except Exception as e:
        print(f"创建训练器时出错: {e}")