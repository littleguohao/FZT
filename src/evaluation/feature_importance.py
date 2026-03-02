"""
特征重要性分析模块

提供多种特征重要性计算方法：
1. LightGBM原生重要性 (gain, split, cover)
2. Permutation重要性
3. SHAP值
4. 相关性分析

支持特征重要性可视化、报告生成和特征选择建议。
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ndcg_score
from sklearn.utils import check_random_state
import lightgbm as lgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")


class FeatureImportanceAnalyzer:
    """
    特征重要性分析器
    
    提供多种特征重要性计算方法，支持可视化、报告生成和特征选择建议。
    """
    
    def __init__(self, random_state: int = 42):
        """
        初始化特征重要性分析器
        
        Parameters
        ----------
        random_state : int, default=42
            随机种子，用于可重复性
        """
        self.random_state = random_state
        self.rng = check_random_state(random_state)
        
    def calculate_importance(
        self,
        model: Optional[lgb.Booster],
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'gain',
        **kwargs
    ) -> pd.DataFrame:
        """
        计算特征重要性
        
        Parameters
        ----------
        model : lgb.Booster or None
            训练好的LightGBM模型。对于相关性分析，可以为None
        X : pd.DataFrame
            特征数据，形状为 (n_samples, n_features)
        y : pd.Series
            目标变量，形状为 (n_samples,)
        method : str, default='gain'
            重要性计算方法，可选：
            - 'gain': LightGBM gain重要性
            - 'split': LightGBM split重要性  
            - 'cover': LightGBM cover重要性
            - 'permutation': Permutation重要性
            - 'shap': SHAP值（需要安装shap）
            - 'correlation': 相关性分析
            
        **kwargs : dict
            额外参数，具体取决于method：
            - 对于'permutation': n_repeats, scoring, sample_fraction
            - 对于'shap': sample_fraction, check_additivity
            
        Returns
        -------
        importance_df : pd.DataFrame
            特征重要性DataFrame，包含列：
            - feature: 特征名称
            - importance: 重要性值（或importance_mean对于permutation）
            - 其他方法特定的列
        """
        # 验证输入
        self._validate_inputs(model, X, y, method)
        
        # 根据方法计算重要性
        if method in ['gain', 'split']:  # LightGBM 4.6.0只支持gain和split
            return self._calculate_lgb_importance(model, X, y, method)
        elif method == 'permutation':
            return self._calculate_permutation_importance(model, X, y, **kwargs)
        elif method == 'shap':
            return self._calculate_shap_importance(model, X, y, **kwargs)
        elif method == 'correlation':
            return self._calculate_correlation_importance(X, y)
        else:
            raise ValueError(f"未知的重要性计算方法: {method}. "
                           f"可用方法: {['gain', 'split', 'permutation', 'shap', 'correlation']}")
    
    def _validate_inputs(
        self,
        model: Optional[lgb.Booster],
        X: pd.DataFrame,
        y: pd.Series,
        method: str
    ):
        """验证输入参数"""
        if X.empty or len(X) == 0:
            raise ValueError("特征数据X不能为空")
        
        if len(y) == 0:
            raise ValueError("目标变量y不能为空")
        
        if len(X) != len(y):
            raise ValueError(f"X和y的长度不匹配: X={len(X)}, y={len(y)}")
        
        if method != 'correlation' and model is None:
            raise ValueError(f"方法'{method}'需要模型参数")
        
        if method == 'shap' and not SHAP_AVAILABLE:
            raise ImportError("SHAP不可用。请安装: pip install shap")
    
    def _calculate_lgb_importance(
        self,
        model: lgb.Booster,
        X: pd.DataFrame,
        y: pd.Series,
        importance_type: str
    ) -> pd.DataFrame:
        """
        计算LightGBM原生重要性
        
        Parameters
        ----------
        model : lgb.Booster
            LightGBM模型
        X : pd.DataFrame
            特征数据
        y : pd.Series
            目标变量
        importance_type : str
            重要性类型：'gain', 'split', 'cover'
            
        Returns
        -------
        importance_df : pd.DataFrame
            特征重要性DataFrame
        """
        # 获取特征重要性
        importance_dict = model.feature_importance(importance_type=importance_type)
        feature_names = model.feature_name()
        
        # 创建DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_dict,
            'importance_type': importance_type
        })
        
        # 按重要性降序排序
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def _calculate_permutation_importance(
        self,
        model: lgb.Booster,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 5,
        scoring: str = 'mse',
        sample_fraction: float = 0.3
    ) -> pd.DataFrame:
        """
        计算Permutation重要性
        
        Parameters
        ----------
        model : lgb.Booster
            LightGBM模型
        X : pd.DataFrame
            特征数据
        y : pd.Series
            目标变量
        n_repeats : int, default=5
            重复次数，减少随机性
        scoring : str, default='ndcg'
            评分指标，支持'ndcg'或自定义函数
        sample_fraction : float, default=0.3
            采样比例，用于加速计算
            
        Returns
        -------
        importance_df : pd.DataFrame
            Permutation重要性DataFrame
        """
        # 获取基线分数
        y_pred = model.predict(X)
        
        if scoring == 'ndcg':
            # 对于排序问题，使用NDCG
            baseline_score = ndcg_score([y], [y_pred], k=10)
            score_func = lambda y_true, y_pred: ndcg_score([y_true], [y_pred], k=10)
        elif scoring == 'mse':
            # 均方误差
            from sklearn.metrics import mean_squared_error
            baseline_score = -mean_squared_error(y, y_pred)  # 负值，因为我们要最大化
            score_func = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
        elif scoring == 'l2':
            # L2损失（同MSE）
            from sklearn.metrics import mean_squared_error
            baseline_score = -mean_squared_error(y, y_pred)
            score_func = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
        else:
            # 自定义评分函数
            score_func = scoring
            baseline_score = score_func(y, y_pred)
        
        # 采样以加速计算
        n_samples = int(len(X) * sample_fraction)
        if n_samples < 10:
            n_samples = min(10, len(X))
        
        sample_indices = self.rng.choice(len(X), size=n_samples, replace=False)
        X_sample = X.iloc[sample_indices].copy()
        y_sample = y.iloc[sample_indices].copy()
        
        # 计算每个特征的permutation重要性
        feature_names = X.columns.tolist()
        importance_scores = []
        
        for feature in feature_names:
            feature_scores = []
            
            for _ in range(n_repeats):
                # 复制数据
                X_permuted = X_sample.copy()
                
                # 打乱特征值
                permuted_values = X_permuted[feature].values.copy()
                self.rng.shuffle(permuted_values)
                X_permuted[feature] = permuted_values
                
                # 计算打乱后的预测
                y_pred_permuted = model.predict(X_permuted)
                
                # 计算分数
                permuted_score = score_func(y_sample, y_pred_permuted)
                
                # 计算重要性（性能下降）
                importance = baseline_score - permuted_score
                feature_scores.append(importance)
            
            # 计算均值和标准差
            importance_scores.append({
                'feature': feature,
                'importance_mean': np.mean(feature_scores),
                'importance_std': np.std(feature_scores),
                'importance_min': np.min(feature_scores),
                'importance_max': np.max(feature_scores)
            })
        
        # 创建DataFrame
        importance_df = pd.DataFrame(importance_scores)
        
        # 按重要性均值降序排序
        importance_df = importance_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def _calculate_shap_importance(
        self,
        model: lgb.Booster,
        X: pd.DataFrame,
        y: pd.Series,
        sample_fraction: float = 0.3,
        check_additivity: bool = True
    ) -> pd.DataFrame:
        """
        计算SHAP重要性
        
        Parameters
        ----------
        model : lgb.Booster
            LightGBM模型
        X : pd.DataFrame
            特征数据
        y : pd.Series
            目标变量
        sample_fraction : float, default=0.3
            采样比例，用于加速计算
        check_additivity : bool, default=True
            是否检查SHAP值的可加性
            
        Returns
        -------
        importance_df : pd.DataFrame
            SHAP重要性DataFrame
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP不可用。请安装: pip install shap")
        
        # 采样以加速计算
        n_samples = int(len(X) * sample_fraction)
        if n_samples < 10:
            n_samples = min(10, len(X))
        
        sample_indices = self.rng.choice(len(X), size=n_samples, replace=False)
        X_sample = X.iloc[sample_indices].copy()
        
        # 计算SHAP值
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample, check_additivity=check_additivity)
        
        # 如果是多分类，取第一个类的SHAP值（对于回归或二分类，shap_values是二维数组）
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # 计算每个特征的SHAP重要性（绝对值的均值）
        shap_importance = np.abs(shap_values).mean(axis=0)
        
        # 创建DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns.tolist(),
            'shap_importance': shap_importance,
            'shap_mean': shap_values.mean(axis=0),
            'shap_std': shap_values.std(axis=0)
        })
        
        # 按SHAP重要性降序排序
        importance_df = importance_df.sort_values('shap_importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def _calculate_correlation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """
        计算相关性重要性
        
        Parameters
        ----------
        X : pd.DataFrame
            特征数据
        y : pd.Series
            目标变量
            
        Returns
        -------
        importance_df : pd.DataFrame
            相关性重要性DataFrame
        """
        # 计算每个特征与目标的相关性
        correlations = []
        
        for feature in X.columns:
            # 计算Pearson相关系数
            corr = X[feature].corr(y)
            correlations.append({
                'feature': feature,
                'correlation': corr,
                'abs_correlation': abs(corr)
            })
        
        # 创建DataFrame
        importance_df = pd.DataFrame(correlations)
        
        # 按绝对相关性降序排序
        importance_df = importance_df.sort_values('abs_correlation', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def get_feature_ranking(
        self,
        importance_df: pd.DataFrame,
        importance_col: str = 'importance'
    ) -> pd.DataFrame:
        """
        获取特征排名
        
        Parameters
        ----------
        importance_df : pd.DataFrame
            特征重要性DataFrame
        importance_col : str, default='importance'
            重要性列名
            
        Returns
        -------
        ranking_df : pd.DataFrame
            特征排名DataFrame，包含：
            - feature: 特征名称
            - rank: 排名（1为最重要）
            - importance: 重要性值
            - normalized_importance: 归一化重要性（0-1）
            - cumulative_importance: 累积重要性
        """
        # 复制数据
        ranking_df = importance_df.copy()
        
        # 确保重要性值非负
        if importance_col in ranking_df.columns:
            # 对于可能为负的重要性（如permutation），取绝对值
            if (ranking_df[importance_col] < 0).any():
                ranking_df[importance_col] = ranking_df[importance_col].abs()
        else:
            # 尝试找到重要性列
            importance_candidates = ['importance', 'importance_mean', 'shap_importance', 'abs_correlation']
            for col in importance_candidates:
                if col in ranking_df.columns:
                    importance_col = col
                    break
            else:
                raise ValueError("未找到重要性列")
        
        # 归一化重要性
        total_importance = ranking_df[importance_col].sum()
        if total_importance > 0:
            ranking_df['normalized_importance'] = ranking_df[importance_col] / total_importance
        else:
            ranking_df['normalized_importance'] = 0
        
        # 计算累积重要性
        ranking_df['cumulative_importance'] = ranking_df['normalized_importance'].cumsum()
        
        # 添加排名
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        
        # 重新排序列
        columns_order = ['rank', 'feature', importance_col, 'normalized_importance', 'cumulative_importance']
        # 添加其他列
        other_cols = [col for col in ranking_df.columns if col not in columns_order]
        columns_order.extend(other_cols)
        
        return ranking_df[columns_order]
    
    def generate_report(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 10
    ) -> Dict:
        """
        生成特征重要性报告
        
        Parameters
        ----------
        importance_df : pd.DataFrame
            特征重要性DataFrame
        top_n : int, default=10
            显示前N个重要特征
            
        Returns
        -------
        report : dict
            特征重要性报告，包含：
            - summary: 摘要信息
            - top_features: 前N个重要特征
            - statistics: 统计信息
        """
        # 获取排名
        ranking_df = self.get_feature_ranking(importance_df)
        
        # 摘要信息
        summary = {
            'total_features': len(ranking_df),
            'top_feature': ranking_df.iloc[0]['feature'] if len(ranking_df) > 0 else None,
            'top_importance': ranking_df.iloc[0]['importance'] if len(ranking_df) > 0 else None,
            'analysis_method': self._detect_analysis_method(importance_df)
        }
        
        # 前N个重要特征
        top_features = ranking_df.head(top_n).to_dict('records')
        
        # 统计信息
        importance_values = ranking_df['importance'].values
        statistics = {
            'mean_importance': float(np.mean(importance_values)),
            'std_importance': float(np.std(importance_values)),
            'max_importance': float(np.max(importance_values)),
            'min_importance': float(np.min(importance_values)),
            'median_importance': float(np.median(importance_values)),
            'iqr_importance': float(np.percentile(importance_values, 75) - np.percentile(importance_values, 25))
        }
        
        # 累积重要性分析
        cumulative_stats = {
            'features_for_50pct': len(ranking_df[ranking_df['cumulative_importance'] <= 0.5]),
            'features_for_80pct': len(ranking_df[ranking_df['cumulative_importance'] <= 0.8]),
            'features_for_90pct': len(ranking_df[ranking_df['cumulative_importance'] <= 0.9]),
            'cumulative_50pct': ranking_df[ranking_df['cumulative_importance'] <= 0.5]['cumulative_importance'].max() 
                               if any(ranking_df['cumulative_importance'] <= 0.5) else 0,
            'cumulative_80pct': ranking_df[ranking_df['cumulative_importance'] <= 0.8]['cumulative_importance'].max() 
                               if any(ranking_df['cumulative_importance'] <= 0.8) else 0,
            'cumulative_90pct': ranking_df[ranking_df['cumulative_importance'] <= 0.9]['cumulative_importance'].max() 
                               if any(ranking_df['cumulative_importance'] <= 0.9) else 0
        }
        
        report = {
            'summary': summary,
            'top_features': top_features,
            'statistics': statistics,
            'cumulative_stats': cumulative_stats,
            'ranking_df': ranking_df.to_dict('records')
        }
        
        return report
    
    def _detect_analysis_method(self, importance_df: pd.DataFrame) -> str:
        """检测分析方法的类型"""
        if 'importance_type' in importance_df.columns:
            return f"lgb_{importance_df['importance_type'].iloc[0]}"
        elif 'importance_mean' in importance_df.columns:
            return 'permutation'
        elif 'shap_importance' in importance_df.columns:
            return 'shap'
        elif 'correlation' in importance_df.columns:
            return 'correlation'
        else:
            return 'unknown'
    
    def visualize_importance(
        self,
        importance_df: pd.DataFrame,
        plot_type: str = 'bar',
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8),
        title: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """
        可视化特征重要性
        
        Parameters
        ----------
        importance_df : pd.DataFrame
            特征重要性DataFrame
        plot_type : str, default='bar'
            图表类型，可选：
            - 'bar': 重要性条形图
            - 'cumulative': 累积重要性图
            - 'heatmap': 特征相关性热力图（需要原始数据）
            - 'shap_summary': SHAP摘要图（需要SHAP值）
        top_n : int, default=20
            显示前N个特征
        figsize : tuple, default=(12, 8)
            图表大小
        title : str, optional
            图表标题
        **kwargs : dict
            额外参数，具体取决于plot_type
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            图表对象
        """
        # 设置样式
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 根据图表类型创建图表
        if plot_type == 'bar':
            fig = self._plot_bar_importance(importance_df, top_n, figsize, title, **kwargs)
        elif plot_type == 'cumulative':
            fig = self._plot_cumulative_importance(importance_df, figsize, title, **kwargs)
        elif plot_type == 'heatmap':
            fig = self._plot_correlation_heatmap(importance_df, figsize, title, **kwargs)
        elif plot_type == 'shap_summary':
            fig = self._plot_shap_summary(importance_df, figsize, title, **kwargs)
        else:
            raise ValueError(f"未知的图表类型: {plot_type}. "
                           f"可用类型: {['bar', 'cumulative', 'heatmap', 'shap_summary']}")
        
        plt.tight_layout()
        return fig
    
    def _plot_bar_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int,
        figsize: Tuple[int, int],
        title: Optional[str],
        **kwargs
    ) -> plt.Figure:
        """绘制重要性条形图"""
        # 获取重要性列
        importance_col = self._get_importance_column(importance_df)
        
        # 选择前N个特征
        plot_df = importance_df.head(top_n).copy()
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制条形图
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(plot_df)))
        bars = ax.barh(range(len(plot_df)), plot_df[importance_col], color=colors)
        
        # 设置y轴标签
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df['feature'])
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, plot_df[importance_col])):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', va='center', fontsize=9)
        
        # 设置标题和标签
        if title is None:
            method = self._detect_analysis_method(importance_df)
            title = f'特征重要性排名 (方法: {method})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('重要性', fontsize=12)
        ax.set_ylabel('特征', fontsize=12)
        
        # 美化图表
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.invert_yaxis()  # 最重要的在顶部
        
        return fig
    
    def _plot_cumulative_importance(
        self,
        importance_df: pd.DataFrame,
        figsize: Tuple[int, int],
        title: Optional[str],
        **kwargs
    ) -> plt.Figure:
        """绘制累积重要性图"""
        # 获取排名
        ranking_df = self.get_feature_ranking(importance_df)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 子图1：累积重要性曲线
        ax1.plot(range(1, len(ranking_df) + 1), ranking_df['cumulative_importance'], 
                'b-', linewidth=2, marker='o', markersize=4)
        
        # 添加参考线
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50%')
        ax1.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='80%')
        ax1.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90%')
        
        # 设置标题和标签
        if title is None:
            title = '累积特征重要性'
        ax1.set_title(f'{title} - 曲线', fontsize=12, fontweight='bold')
        ax1.set_xlabel('特征数量', fontsize=10)
        ax1.set_ylabel('累积重要性', fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2：帕累托图
        # 计算达到不同阈值所需的特征数量
        thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
        features_needed = []
        
        for threshold in thresholds:
            idx = np.where(ranking_df['cumulative_importance'] >= threshold)[0]
            if len(idx) > 0:
                features_needed.append(idx[0] + 1)  # +1因为索引从0开始
            else:
                features_needed.append(len(ranking_df))
        
        # 绘制帕累托图
        bars = ax2.bar(range(len(thresholds)), features_needed, color='skyblue', alpha=0.7)
        
        # 添加数值标签
        for bar, value in zip(bars, features_needed):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    str(value), ha='center', va='bottom', fontsize=9)
        
        # 设置x轴标签
        ax2.set_xticks(range(len(thresholds)))
        ax2.set_xticklabels([f'{t*100:.0f}%' for t in thresholds])
        
        ax2.set_title(f'{title} - 帕累托分析', fontsize=12, fontweight='bold')
        ax2.set_xlabel('累积重要性阈值', fontsize=10)
        ax2.set_ylabel('所需特征数量', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        return fig
    
    def _plot_correlation_heatmap(
        self,
        importance_df: pd.DataFrame,
        figsize: Tuple[int, int],
        title: Optional[str],
        **kwargs
    ) -> plt.Figure:
        """绘制相关性热力图"""
        # 注意：这个方法需要原始数据，但importance_df可能不包含
        # 这里我们假设importance_df是相关性分析的结果
        if 'correlation' not in importance_df.columns:
            raise ValueError("相关性热力图需要相关性分析的结果")
        
        # 获取前N个特征的相关性矩阵（这里简化处理）
        # 在实际应用中，需要原始数据来计算特征间的相关性
        top_features = importance_df.head(20)['feature'].tolist()
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 这里简化处理：只显示与目标的相关性
        # 在实际应用中，应该计算特征间的相关性矩阵
        correlations = importance_df.set_index('feature')['correlation'].loc[top_features]
        
        # 创建热力图数据（简化版）
        heatmap_data = pd.DataFrame({
            '特征': top_features,
            '与目标相关性': correlations.values
        })
        
        # 使用条形图代替热力图（简化）
        colors = ['red' if x < 0 else 'green' for x in correlations.values]
        bars = ax.barh(range(len(correlations)), correlations.values, color=colors, alpha=0.7)
        
        # 设置y轴标签
        ax.set_yticks(range(len(correlations)))
        ax.set_yticklabels(top_features)
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, correlations.values)):
            color = 'darkred' if value < 0 else 'darkgreen'
            ax.text(bar.get_width() * (1.01 if value >= 0 else -0.01), 
                   bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', 
                   va='center', 
                   ha='left' if value >= 0 else 'right',
                   color=color, fontsize=9)
        
        # 设置标题和标签
        if title is None:
            title = '特征与目标相关性'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('相关系数', fontsize=12)
        ax.set_ylabel('特征', fontsize=12)
        
        # 添加零线
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()  # 最重要的在顶部
        
        return fig
    
    def _plot_shap_summary(
        self,
        importance_df: pd.DataFrame,
        figsize: Tuple[int, int],
        title: Optional[str],
        **kwargs
    ) -> plt.Figure:
        """绘制SHAP摘要图"""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP不可用。请安装: pip install shap")
        
        if 'shap_importance' not in importance_df.columns:
            raise ValueError("SHAP摘要图需要SHAP分析的结果")
        
        # 获取前N个特征
        top_n = kwargs.get('top_n', 20)
        top_features = importance_df.head(top_n)['feature'].tolist()
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 子图1：SHAP重要性条形图
        top_df = importance_df.head(top_n).copy()
        colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(top_df)))
        
        bars1 = ax1.barh(range(len(top_df)), top_df['shap_importance'], color=colors)
        ax1.set_yticks(range(len(top_df)))
        ax1.set_yticklabels(top_df['feature'])
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars1, top_df['shap_importance'])):
            ax1.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.4f}', va='center', fontsize=8)
        
        ax1.set_title('SHAP特征重要性', fontsize=12, fontweight='bold')
        ax1.set_xlabel('SHAP重要性 (绝对值均值)', fontsize=10)
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 子图2：SHAP值分布（简化版）
        # 这里我们使用SHAP均值和标准差来创建箱线图
        if 'shap_mean' in top_df.columns and 'shap_std' in top_df.columns:
            # 创建模拟的SHAP值分布
            positions = range(len(top_df))
            means = top_df['shap_mean'].values
            stds = top_df['shap_std'].values
            
            # 创建箱线图数据
            boxplot_data = []
            for mean, std in zip(means, stds):
                # 生成模拟数据点
                simulated_data = np.random.normal(mean, std, 100)
                boxplot_data.append(simulated_data)
            
            # 绘制箱线图
            bp = ax2.boxplot(boxplot_data, positions=positions, vert=False, 
                           widths=0.6, patch_artist=True)
            
            # 设置箱线图颜色
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            ax2.set_yticks(positions)
            ax2.set_yticklabels(top_df['feature'])
            ax2.set_title('SHAP值分布', fontsize=12, fontweight='bold')
            ax2.set_xlabel('SHAP值', fontsize=10)
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def _get_importance_column(self, importance_df: pd.DataFrame) -> str:
        """获取重要性列名"""
        importance_columns = ['importance', 'importance_mean', 'shap_importance', 'abs_correlation']
        
        for col in importance_columns:
            if col in importance_df.columns:
                return col
        
        # 如果没有找到标准列，尝试其他列
        numeric_cols = importance_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return numeric_cols[0]
        
        raise ValueError("未找到重要性数值列")
    
    def get_feature_selection_suggestions(
        self,
        importance_df: pd.DataFrame,
        threshold: float = 0.8,
        min_features: int = 5,
        max_features: Optional[int] = None
    ) -> Dict:
        """
        获取特征选择建议
        
        Parameters
        ----------
        importance_df : pd.DataFrame
            特征重要性DataFrame
        threshold : float, default=0.8
            累积重要性阈值，选择达到此阈值的特征
        min_features : int, default=5
            最小特征数量
        max_features : Optional[int], default=None
            最大特征数量
            
        Returns
        -------
        suggestions : dict
            特征选择建议，包含：
            - selected_features: 选择的特征列表
            - num_selected: 选择的特征数量
            - cumulative_importance: 累积重要性
            - threshold: 使用的阈值
            - efficiency_ratio: 效率比（累积重要性/特征数量）
        """
        # 获取排名
        ranking_df = self.get_feature_ranking(importance_df)
        
        # 找到达到阈值的特征
        threshold_idx = np.where(ranking_df['cumulative_importance'] >= threshold)[0]
        
        if len(threshold_idx) > 0:
            n_selected = threshold_idx[0] + 1  # +1因为索引从0开始
        else:
            n_selected = len(ranking_df)
        
        # 应用最小和最大限制
        n_selected = max(min_features, n_selected)
        if max_features is not None:
            n_selected = min(max_features, n_selected)
        
        # 获取选择的特征
        selected_features = ranking_df.head(n_selected)['feature'].tolist()
        cumulative_importance = ranking_df.head(n_selected)['cumulative_importance'].iloc[-1]
        
        # 计算效率比
        efficiency_ratio = cumulative_importance / n_selected if n_selected > 0 else 0
        
        suggestions = {
            'selected_features': selected_features,
            'num_selected': n_selected,
            'cumulative_importance': float(cumulative_importance),
            'threshold': threshold,
            'efficiency_ratio': float(efficiency_ratio),
            'coverage_per_feature': float(cumulative_importance / n_selected) if n_selected > 0 else 0
        }
        
        return suggestions
    
    def compare_importance_methods(
        self,
        model: lgb.Booster,
        X: pd.DataFrame,
        y: pd.Series,
        methods: List[str] = None,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        比较不同重要性方法的结果
        
        Parameters
        ----------
        model : lgb.Booster
            LightGBM模型
        X : pd.DataFrame
            特征数据
        y : pd.Series
            目标变量
        methods : List[str], optional
            要比较的方法列表，如果为None则使用所有可用方法
        **kwargs : dict
            传递给calculate_importance的额外参数
            
        Returns
        -------
        results : dict
            字典，键为方法名，值为重要性DataFrame
        """
        if methods is None:
            methods = ['gain', 'split', 'cover', 'permutation', 'correlation']
            if SHAP_AVAILABLE:
                methods.append('shap')
        
        results = {}
        
        for method in methods:
            try:
                importance_df = self.calculate_importance(
                    model=model if method != 'correlation' else None,
                    X=X,
                    y=y,
                    method=method,
                    **kwargs.get(method, {})
                )
                results[method] = importance_df
            except Exception as e:
                warnings.warn(f"方法 {method} 失败: {str(e)}")
                continue
        
        return results
    
    def save_report(
        self,
        report: Dict,
        filepath: str,
        format: str = 'json'
    ):
        """
        保存报告到文件
        
        Parameters
        ----------
        report : dict
            特征重要性报告
        filepath : str
            文件路径
        format : str, default='json'
            文件格式，支持'json'或'csv'
        """
        import json
        
        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        elif format == 'csv':
            # 保存排名数据
            if 'ranking_df' in report:
                ranking_data = report['ranking_df']
                df = pd.DataFrame(ranking_data)
                df.to_csv(filepath, index=False, encoding='utf-8')
            else:
                raise ValueError("报告中不包含ranking_df，无法保存为CSV")
        else:
            raise ValueError(f"不支持的格式: {format}. 支持: 'json', 'csv'")
    
    def plot_comparison(
        self,
        results: Dict[str, pd.DataFrame],
        top_n: int = 15,
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        绘制不同重要性方法的比较图
        
        Parameters
        ----------
        results : dict
            不同方法的结果，键为方法名，值为重要性DataFrame
        top_n : int, default=15
            显示前N个特征
        figsize : tuple, default=(14, 10)
            图表大小
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            比较图
        """
        # 获取所有方法共有的前N个特征（基于第一个方法）
        first_method = list(results.keys())[0]
        first_df = results[first_method]
        common_features = first_df.head(top_n)['feature'].tolist()
        
        # 创建比较数据
        comparison_data = []
        
        for method, df in results.items():
            # 获取每个特征的重要性
            for feature in common_features:
                if feature in df['feature'].values:
                    importance = df.loc[df['feature'] == feature, self._get_importance_column(df)].values[0]
                    comparison_data.append({
                        'method': method,
                        'feature': feature,
                        'importance': importance
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # 子图1：不同方法的特征重要性热力图
        pivot_df = comparison_df.pivot(index='feature', columns='method', values='importance')
        
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                   ax=axes[0], cbar_kws={'label': '重要性'})
        axes[0].set_title('不同方法的特征重要性比较', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('方法', fontsize=10)
        axes[0].set_ylabel('特征', fontsize=10)
        
        # 子图2：不同方法的排名相关性
        # 计算排名
        rank_dfs = []
        for method, df in results.items():
            rank_df = df.copy()
            rank_df['rank'] = range(1, len(rank_df) + 1)
            rank_df = rank_df[['feature', 'rank']]
            rank_df.columns = ['feature', f'{method}_rank']
            rank_dfs.append(rank_df)
        
        # 合并排名
        merged_ranks = rank_dfs[0]
        for rank_df in rank_dfs[1:]:
            merged_ranks = pd.merge(merged_ranks, rank_df, on='feature', how='inner')
        
        # 计算排名相关性矩阵
        rank_cols = [col for col in merged_ranks.columns if '_rank' in col]
        rank_corr = merged_ranks[rank_cols].corr()
        
        # 重命名列
        rank_corr.columns = [col.replace('_rank', '') for col in rank_corr.columns]
        rank_corr.index = [idx.replace('_rank', '') for idx in rank_corr.index]
        
        sns.heatmap(rank_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   vmin=-1, vmax=1, center=0, ax=axes[1],
                   cbar_kws={'label': '排名相关系数'})
        axes[1].set_title('不同方法排名相关性', fontsize=12, fontweight='bold')
        
        # 子图3：不同方法的前N个特征重叠情况
        top_features_by_method = {}
        for method, df in results.items():
            top_features = df.head(top_n)['feature'].tolist()
            top_features_by_method[method] = set(top_features)
        
        # 计算重叠数量
        methods = list(results.keys())
        overlap_matrix = np.zeros((len(methods), len(methods)))
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i == j:
                    overlap_matrix[i, j] = top_n
                else:
                    overlap = len(top_features_by_method[method1] & top_features_by_method[method2])
                    overlap_matrix[i, j] = overlap
        
        overlap_df = pd.DataFrame(overlap_matrix, index=methods, columns=methods)
        
        sns.heatmap(overlap_df, annot=True, fmt='.0f', cmap='Blues', 
                   ax=axes[2], cbar_kws={'label': '重叠特征数量'})
        axes[2].set_title(f'前{top_n}个特征重叠情况', fontsize=12, fontweight='bold')
        
        # 子图4：不同方法的一致性分析
        # 计算每个特征在不同方法中的平均排名
        all_features = set()
        for df in results.values():
            all_features.update(df['feature'].tolist())
        
        consistency_data = []
        for feature in all_features:
            ranks = []
            for method, df in results.items():
                if feature in df['feature'].values:
                    rank = df.loc[df['feature'] == feature].index[0] + 1
                    ranks.append(rank)
            
            if ranks:  # 只考虑在所有方法中都出现的特征
                consistency_data.append({
                    'feature': feature,
                    'mean_rank': np.mean(ranks),
                    'std_rank': np.std(ranks),
                    'cv_rank': np.std(ranks) / np.mean(ranks) if np.mean(ranks) > 0 else 0
                })
        
        consistency_df = pd.DataFrame(consistency_data)
        consistency_df = consistency_df.sort_values('mean_rank').head(top_n)
        
        # 绘制误差条形图
        x_pos = range(len(consistency_df))
        axes[3].errorbar(x_pos, consistency_df['mean_rank'], 
                        yerr=consistency_df['std_rank'],
                        fmt='o', capsize=5, capthick=2, 
                        color='steelblue', alpha=0.7)
        
        axes[3].set_xticks(x_pos)
        axes[3].set_xticklabels(consistency_df['feature'], rotation=45, ha='right')
        axes[3].set_title('特征排名一致性分析', fontsize=12, fontweight='bold')
        axes[3].set_xlabel('特征', fontsize=10)
        axes[3].set_ylabel('平均排名 ± 标准差', fontsize=10)
        axes[3].grid(True, alpha=0.3)
        axes[3].invert_yaxis()  # 排名越小越好，所以在顶部
        
        plt.tight_layout()
        return fig


# 工具函数
def analyze_feature_importance(
    model: lgb.Booster,
    X: pd.DataFrame,
    y: pd.Series,
    methods: List[str] = None,
    save_path: Optional[str] = None,
    **kwargs
) -> Dict:
    """
    特征重要性分析工具函数
    
    Parameters
    ----------
    model : lgb.Booster
        LightGBM模型
    X : pd.DataFrame
        特征数据
    y : pd.Series
        目标变量
    methods : List[str], optional
        要使用的方法列表
    save_path : str, optional
        保存报告的路径
    **kwargs : dict
        额外参数
        
    Returns
    -------
    results : dict
        分析结果
    """
    analyzer = FeatureImportanceAnalyzer()
    
    # 计算重要性
    results = analyzer.compare_importance_methods(model, X, y, methods, **kwargs)
    
    # 生成报告（使用第一个方法）
    if results:
        first_method = list(results.keys())[0]
        report = analyzer.generate_report(results[first_method])
        
        # 保存报告
        if save_path:
            analyzer.save_report(report, save_path)
        
        return {
            'analyzer': analyzer,
            'results': results,
            'report': report,
            'comparison_plot': analyzer.plot_comparison(results)
        }
    
    return {}


if __name__ == '__main__':
    # 示例用法
    print("特征重要性分析模块已加载")
    print("使用方法:")
    print("1. 创建分析器: analyzer = FeatureImportanceAnalyzer()")
    print("2. 计算重要性: importance_df = analyzer.calculate_importance(model, X, y, method='gain')")
    print("3. 生成报告: report = analyzer.generate_report(importance_df)")
    print("4. 可视化: fig = analyzer.visualize_importance(importance_df, plot_type='bar')")
