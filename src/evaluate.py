#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估模块 - 评估原始FZT公式和Qlib模型优化效果

评估指标：
1. 原始FZT公式成功率（基准）
2. Qlib模型优化后成功率
3. 模型精准率（选中即成功）
4. 特征重要性分析

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
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        初始化模型评估器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # 输出目录
        self.results_dir = Path("results/evaluation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("模型评估器初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"配置文件不存在: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"配置文件解析错误: {e}")
            raise
    
    def calculate_original_fzt_success_rate(self, 
                                          features: pd.DataFrame,
                                          target_col: str = 'target',
                                          fzt_signal_col: str = 'fzt_selection_condition') -> float:
        """
        计算原始FZT公式成功率
        
        Args:
            features: 特征数据集
            target_col: 目标列名
            fzt_signal_col: FZT选股信号列名
            
        Returns:
            float: 原始FZT公式成功率
        """
        logger.info("计算原始FZT公式成功率...")
        
        if target_col not in features.columns:
            raise ValueError(f"目标列 '{target_col}' 不存在")
        
        if fzt_signal_col not in features.columns:
            raise ValueError(f"FZT选股信号列 '{fzt_signal_col}' 不存在")
        
        # 获取FZT选股信号和目标变量
        fzt_signals = features[fzt_signal_col]
        targets = features[target_col]
        
        # 对齐索引
        common_idx = fzt_signals.index.intersection(targets.index)
        fzt_signals = fzt_signals.loc[common_idx]
        targets = targets.loc[common_idx]
        
        # 移除NaN值
        valid_mask = fzt_signals.notna() & targets.notna()
        fzt_signals = fzt_signals[valid_mask]
        targets = targets[valid_mask]
        
        if len(fzt_signals) == 0:
            logger.warning("没有有效数据计算原始FZT成功率")
            return 0.0
        
        # 计算成功率：FZT信号为True且目标为True的比例
        # 注意：这里假设FZT信号已经是二分类（True/False）
        if fzt_signals.dtype == bool:
            # 布尔类型直接使用
            fzt_binary = fzt_signals.astype(int)
        else:
            # 数值类型，大于0为True
            fzt_binary = (fzt_signals > 0).astype(int)
        
        # 目标变量也需要是二分类
        if targets.dtype != bool and targets.dtype != int:
            # 假设连续值，大于0为成功
            target_binary = (targets > 0).astype(int)
        else:
            target_binary = targets.astype(int)
        
        # 计算原始FZT成功率
        original_accuracy = (fzt_binary == target_binary).mean()
        
        # 计算FZT信号的精准率
        fzt_precision = precision_score(target_binary, fzt_binary, zero_division=0)
        
        # 计算FZT信号的召回率
        fzt_recall = recall_score(target_binary, fzt_binary, zero_division=0)
        
        logger.info(f"原始FZT公式评估结果:")
        logger.info(f"  样本数量: {len(fzt_signals)}")
        logger.info(f"  FZT信号比例: {fzt_binary.mean():.2%}")
        logger.info(f"  正类比例: {target_binary.mean():.2%}")
        logger.info(f"  原始成功率: {original_accuracy:.2%}")
        logger.info(f"  原始精准率: {fzt_precision:.2%}")
        logger.info(f"  原始召回率: {fzt_recall:.2%}")
        
        return original_accuracy
    
    def evaluate_model_performance(self,
                                 model: Any,
                                 X_test: pd.DataFrame,
                                 y_test: pd.Series,
                                 original_accuracy: float) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试目标
            original_accuracy: 原始FZT成功率
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        logger.info("评估模型性能...")
        
        # 模型预测
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # 正类的概率
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # 计算各项指标
        model_accuracy = accuracy_score(y_test, y_pred)
        model_precision = precision_score(y_test, y_pred, zero_division=0)
        model_recall = recall_score(y_test, y_pred, zero_division=0)
        model_f1 = f1_score(y_test, y_pred, zero_division=0)
        model_roc_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 计算提升比例
        accuracy_improvement = (model_accuracy - original_accuracy) / original_accuracy if original_accuracy > 0 else 0
        
        # 精准率（选中即成功）
        # 这里使用模型预测为正类且实际为正类的比例
        selected_success_rate = model_precision
        
        results = {
            'original_accuracy': original_accuracy,
            'model_accuracy': model_accuracy,
            'model_precision': model_precision,
            'model_recall': model_recall,
            'model_f1': model_f1,
            'model_roc_auc': model_roc_auc,
            'accuracy_improvement': accuracy_improvement,
            'selected_success_rate': selected_success_rate,
            'confusion_matrix': cm.tolist(),
            'sample_count': len(y_test),
            'positive_ratio': y_test.mean()
        }
        
        logger.info("模型评估结果:")
        logger.info(f"  原始FZT成功率: {original_accuracy:.2%}")
        logger.info(f"  模型优化后成功率: {model_accuracy:.2%}")
        logger.info(f"  成功率提升: {accuracy_improvement:.2%}")
        logger.info(f"  模型精准率（选中即成功）: {selected_success_rate:.2%}")
        logger.info(f"  模型F1分数: {model_f1:.2%}")
        logger.info(f"  模型ROC AUC: {model_roc_auc:.4f}")
        
        return results
    
    def analyze_feature_importance(self,
                                 model: Any,
                                 feature_names: List[str],
                                 top_n: int = 20) -> pd.DataFrame:
        """
        分析特征重要性
        
        Args:
            model: 训练好的模型
            feature_names: 特征名称列表
            top_n: 显示前N个特征
            
        Returns:
            pd.DataFrame: 特征重要性DataFrame
        """
        logger.info("分析特征重要性...")
        
        try:
            # 获取特征重要性
            if hasattr(model, 'feature_importances_'):
                # scikit-learn风格
                importance_values = model.feature_importances_
            elif hasattr(model, 'feature_importance'):
                # LightGBM风格
                importance_values = model.feature_importance()
            else:
                logger.warning("模型不支持特征重要性分析")
                return pd.DataFrame()
            
            # 创建特征重要性DataFrame
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=False)
            
            # 计算重要性百分比
            total_importance = feature_importance['importance'].sum()
            if total_importance > 0:
                feature_importance['importance_pct'] = feature_importance['importance'] / total_importance * 100
            
            logger.info(f"特征重要性分析完成: {len(feature_importance)} 个特征")
            
            # 打印Top N特征
            logger.info(f"核心特征重要性（提升胜率的关键因子）:")
            for i, row in feature_importance.head(top_n).iterrows():
                importance_pct = row.get('importance_pct', 0)
                logger.info(f"  {row['feature']}: {row['importance']:.4f} ({importance_pct:.1f}%)")
            
            return feature_importance
            
        except Exception as e:
            logger.warning(f"特征重要性分析失败: {e}")
            return pd.DataFrame()
    
    def compare_strategies(self,
                          features: pd.DataFrame,
                          model: Any,
                          X_test: pd.DataFrame,
                          y_test: pd.Series,
                          fzt_signal_col: str = 'fzt_selection_condition',
                          target_col: str = 'target') -> Dict[str, Any]:
        """
        比较原始FZT策略和模型优化策略
        
        Args:
            features: 完整特征数据集
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试目标
            fzt_signal_col: FZT选股信号列名
            target_col: 目标列名
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        logger.info("比较原始FZT策略和模型优化策略...")
        
        # 1. 计算原始FZT成功率
        original_accuracy = self.calculate_original_fzt_success_rate(
            features, target_col, fzt_signal_col
        )
        
        # 2. 评估模型性能
        model_results = self.evaluate_model_performance(
            model, X_test, y_test, original_accuracy
        )
        
        # 3. 分析特征重要性
        feature_names = X_test.columns.tolist()
        feature_importance = self.analyze_feature_importance(model, feature_names)
        
        # 4. 生成详细比较报告
        comparison = {
            'original_strategy': {
                'accuracy': original_accuracy,
                'description': '原始FZT选股公式'
            },
            'model_strategy': model_results,
            'improvement_summary': {
                'accuracy_improvement_abs': model_results['model_accuracy'] - original_accuracy,
                'accuracy_improvement_pct': model_results['accuracy_improvement'],
                'key_improvement_factors': feature_importance.head(10).to_dict('records') if not feature_importance.empty else []
            },
            'feature_importance': feature_importance.to_dict('records') if not feature_importance.empty else [],
            'evaluation_time': datetime.now().isoformat()
        }
        
        return comparison
    
    def generate_evaluation_report(self, comparison: Dict[str, Any]) -> str:
        """
        生成评估报告
        
        Args:
            comparison: 比较结果
            
        Returns:
            str: 评估报告
        """
        logger.info("生成评估报告...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FZT选股策略评估报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 原始策略结果
        original = comparison['original_strategy']
        report_lines.append("📊 原始FZT选股策略:")
        report_lines.append(f"  策略描述: {original['description']}")
        report_lines.append(f"  测试集成功率: {original['accuracy']:.2%}")
        report_lines.append("")
        
        # 模型优化策略结果
        model_results = comparison['model_strategy']
        report_lines.append("🤖 Qlib模型优化策略:")
        report_lines.append(f"  模型优化后成功率: {model_results['model_accuracy']:.2%}")
        report_lines.append(f"  模型精准率（选中即成功）: {model_results['selected_success_rate']:.2%}")
        report_lines.append(f"  模型F1分数: {model_results['model_f1']:.2%}")
        report_lines.append(f"  模型ROC AUC: {model_results['model_roc_auc']:.4f}")
        report_lines.append("")
        
        # 改进总结
        improvement = comparison['improvement_summary']
        report_lines.append("🚀 策略改进效果:")
        report_lines.append(f"  成功率绝对提升: {improvement['accuracy_improvement_abs']:.2%}")
        report_lines.append(f"  成功率相对提升: {improvement['accuracy_improvement_pct']:.2%}")
        report_lines.append("")
        
        # 样本统计
        report_lines.append("📈 样本统计:")
        report_lines.append(f"  测试集样本数量: {model_results['sample_count']}")
        report_lines.append(f"  正类比例: {model_results['positive_ratio']:.2%}")
        report_lines.append("")
        
        # 特征重要性
        if comparison['feature_importance']:
            report_lines.append("🔍 核心特征重要性（提升胜率的关键因子）:")
            for i, feature in enumerate(comparison['feature_importance'][:10], 1):
                feature_name = feature.get('feature', 'Unknown')
                importance = feature.get('importance', 0)
                importance_pct = feature.get('importance_pct', 0)
                report_lines.append(f"  {i:2d}. {feature_name}: {importance:.4f} ({importance_pct:.1f}%)")
        
        report = "\n".join(report_lines)
        
        # 保存报告到文件
        report_path = self.results_dir / "evaluation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"评估报告已保存到: {report_path}")
        
        return report
    
    def plot_comparison_chart(self, comparison: Dict[str, Any], save_path: Optional[str] = None):
        """
        绘制策略比较图
        
        Args:
            comparison: 比较结果
            save_path: 保存路径（可选）
        """
        logger.info("绘制策略比较图...")
        
        # 准备数据
        labels = ['原始FZT策略', '模型优化策略']
        accuracy_values = [
            comparison['original_strategy']['accuracy'],
            comparison['model_strategy']['model_accuracy']
        ]
        
        precision_values = [
            0,  # 原始策略没有精准率数据
            comparison['model_strategy']['selected_success_rate']
        ]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 成功率比较
        bars1 = ax1.bar(x - width/2, accuracy_values, width, label='成功率', color='skyblue')
        ax1.set_xlabel('策略类型')
        ax1.set_ylabel('成功率')
        ax1.set_title('策略成功率比较')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend()
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.2%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # 精准率展示
        bars2 = ax2.bar(['模型精准率'], [precision_values[1]], color='lightgreen')
        ax2.set_xlabel('指标')
        ax2.set_ylabel('精准率')
        ax2.set_title('模型精准率（选中即成功）')
        ax2.set_ylim(0, 1)
        
        # 添加数值标签
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{height:.2%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"策略比较图已保存到: {save_path}")
        
        plt.show()
    
    def save_comparison_results(self, comparison: Dict[str, Any], name: str = "comparison"):
        """
        保存比较结果
        
        Args:
            comparison: 比较结果
            name: 结果名称
        """
        # 保存为JSON
        import json
        json_path = self.results_dir / f"{name}_results.json"
        
        # 转换numpy类型为Python原生类型
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            else:
                return obj
        
        serializable_comparison = {}
        for key, value in comparison.items():
            if isinstance(value, dict):
                serializable_comparison[key] = {k: convert_to_serializable(v) for k, v in value.items()}
            else:
                serializable_comparison[key] = convert_to_serializable(value)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_comparison, f, indent=2, ensure_ascii=False)
        
        logger.info(f"比较结果已保存到: {json_path}")
        
        # 保存特征重要性为CSV
        if 'feature_importance' in comparison and comparison['feature_importance']:
            importance_df = pd.DataFrame(comparison['feature_importance'])
            csv_path = self.results_dir / f"{name}_feature_importance.csv"
            importance_df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"特征重要性已保存到: {csv_path}")


def test_evaluator():
    """测试评估器"""
    try:
        print("=" * 80)
        print("模型评估模块测试")
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
        
        # 创建目标变量
        y_continuous = (
            0.3 * X['feature_0'] + 
            0.2 * X['feature_1'] + 
            0.1 * X['feature_2'] + 
            np.random.randn(n_samples) * 0.1
        )
        
        y = (y_continuous > y_continuous.mean()).astype(int)
        y = pd.Series(y, index=X.index, name='target')
        
        # 创建模拟的FZT选股信号（有一定准确率）
        fzt_signal = ((X['feature_0'] > 0) & (X['feature_1'] > 0)).astype(int)
        
        # 创建完整特征DataFrame
        features = X.copy()
        features['target'] = y
        features['fzt_selection_condition'] = fzt_signal
        
        print(f"   数据形状: {features.shape}")
        print(f"   正类比例: {y.mean():.2%}")
        print(f"   FZT信号比例: {fzt_signal.mean():.2%}")
        
        # 初始化评估器
        print("\n2. 初始化评估器...")
        evaluator = ModelEvaluator()
        print("   ✓ 评估器初始化成功")
        
        # 计算原始FZT成功率
        print("\n3. 计算原始FZT成功率...")
        original_accuracy = evaluator.calculate_original_fzt_success_rate(
            features, 'target', 'fzt_selection_condition'
        )
        print(f"   ✓ 原始FZT成功率: {original_accuracy:.2%}")
        
        # 创建模拟模型
        print("\n4. 创建模拟模型...")
        from sklearn.ensemble import RandomForestClassifier
        
        # 划分训练测试集
        split_idx = int(len(X) * 0.7)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        # 训练简单模型
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        print("   ✓ 模拟模型训练完成")
        
        # 评估模型性能
        print("\n5. 评估模型性能...")
        model_results = evaluator.evaluate_model_performance(
            model, X_test, y_test, original_accuracy
        )
        print(f"   ✓ 模型评估完成")
        print(f"   模型成功率: {model_results['model_accuracy']:.2%}")
        
        # 分析特征重要性
        print("\n6. 分析特征重要性...")
        feature_names = X.columns.tolist()
        feature_importance = evaluator.analyze_feature_importance(model, feature_names, top_n=5)
        print(f"   ✓ 特征重要性分析完成")
        
        # 比较策略
        print("\n7. 比较原始FZT策略和模型优化策略...")
        comparison = evaluator.compare_strategies(
            features, model, X_test, y_test, 'fzt_selection_condition', 'target'
        )
        print(f"   ✓ 策略比较完成")
        
        # 生成评估报告
        print("\n8. 生成评估报告...")
        report = evaluator.generate_evaluation_report(comparison)
        print(report[:400] + "..." if len(report) > 400 else report)
        
        # 保存结果
        print("\n9. 保存比较结果...")
        evaluator.save_comparison_results(comparison, 'test_evaluation')
        print("   ✓ 结果保存完成")
        
        print("\n" + "=" * 80)
        print("模型评估模块测试完成！所有功能正常。")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_evaluator())