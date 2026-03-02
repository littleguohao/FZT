"""
NDCG评估器 - 用于评估排序模型的质量

实现多种排序评估指标：
1. NDCG@K: 归一化折扣累积增益
2. MAP@K: 平均精度
3. MRR: 平均倒数排名
4. Precision@K: 精度
5. Recall@K: 召回率
6. F1@K: F1分数

支持分组评估（按日期/查询）
"""
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
import warnings


class NDCGEvaluator:
    """
    NDCG评估器类
    
    用于评估排序模型的质量，支持多种排序评估指标和分组评估。
    
    Attributes:
        k_values (List[int]): 要计算的K值列表
        metrics (List[str]): 要计算的指标列表
    """
    
    def __init__(self, k_values: List[int] = None, metrics: List[str] = None):
        """
        初始化NDCG评估器
        
        Args:
            k_values: 要计算的K值列表，默认[1, 3, 5, 10]
            metrics: 要计算的指标列表，默认所有指标
        """
        self.k_values = k_values or [1, 3, 5, 10]
        self.metrics = metrics or ['ndcg', 'map', 'mrr', 'precision', 'recall', 'f1']
        
        # 验证参数
        if not self.k_values:
            raise ValueError("k_values不能为空")
        if any(k <= 0 for k in self.k_values):
            raise ValueError("k_values必须为正整数")
            
    def _compute_dcg(self, relevance: List[float], k: int = None) -> float:
        """
        计算折扣累积增益(DCG)
        
        Args:
            relevance: 相关性分数列表（按预测排序）
            k: 只考虑前K个结果，None表示考虑所有
            
        Returns:
            DCG@K值
        """
        if k is None:
            k = len(relevance)
        else:
            k = min(k, len(relevance))
            
        # 计算DCG: Σ(i=1 to K) (rel_i / log2(i + 1))
        dcg = 0.0
        for i in range(k):
            # 避免log(1) = 0的情况，使用log2(i + 2)
            dcg += relevance[i] / np.log2(i + 2)
        return dcg
    
    def _compute_ndcg(self, y_true: List[float], y_pred: List[float], k: int = None) -> float:
        """
        计算归一化折扣累积增益(NDCG)
        
        Args:
            y_true: 真实相关性分数
            y_pred: 预测排序分数
            k: 只考虑前K个结果
            
        Returns:
            NDCG@K值，范围[0, 1]
        """
        if len(y_true) != len(y_pred):
            raise ValueError(f"y_true和y_pred长度不匹配: {len(y_true)} != {len(y_pred)}")
            
        if k is None:
            k = len(y_true)
            
        # 根据预测分数排序
        pred_order = np.argsort(y_pred)[::-1]  # 降序
        sorted_relevance = [y_true[i] for i in pred_order[:k]]
        
        # 计算DCG
        dcg = self._compute_dcg(sorted_relevance, k)
        
        # 计算理想DCG（按真实相关性降序排序）
        ideal_order = np.argsort(y_true)[::-1]
        ideal_relevance = [y_true[i] for i in ideal_order[:k]]
        idcg = self._compute_dcg(ideal_relevance, k)
        
        # 避免除以0
        if idcg == 0:
            return 0.0
            
        return dcg / idcg
    
    def _compute_map(self, y_true: List[float], y_pred: List[float], k: int = None) -> float:
        """
        计算平均精度(MAP)
        
        Args:
            y_true: 真实相关性分数（二值化：>0为相关）
            y_pred: 预测排序分数
            k: 只考虑前K个结果
            
        Returns:
            MAP@K值，范围[0, 1]
        """
        if len(y_true) != len(y_pred):
            raise ValueError(f"y_true和y_pred长度不匹配: {len(y_true)} != {len(y_pred)}")
            
        if k is None:
            k = len(y_true)
            
        # 二值化相关性（>0为相关）
        binary_relevance = [1 if rel > 0 else 0 for rel in y_true]
        
        # 如果没有相关文档，返回0
        if sum(binary_relevance) == 0:
            return 0.0
            
        # 根据预测分数排序
        pred_order = np.argsort(y_pred)[::-1]
        
        # 计算平均精度
        ap = 0.0
        relevant_count = 0
        
        for i in range(min(k, len(pred_order))):
            idx = pred_order[i]
            if binary_relevance[idx] == 1:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                ap += precision_at_i
                
        # 归一化
        total_relevant = sum(binary_relevance)
        return ap / min(total_relevant, k)
    
    def _compute_mrr(self, y_true: List[float], y_pred: List[float], k: int = None) -> float:
        """
        计算平均倒数排名(MRR)
        
        Args:
            y_true: 真实相关性分数（二值化：>0为相关）
            y_pred: 预测排序分数
            k: 只考虑前K个结果
            
        Returns:
            MRR@K值，范围[0, 1]
        """
        if len(y_true) != len(y_pred):
            raise ValueError(f"y_true和y_pred长度不匹配: {len(y_true)} != {len(y_pred)}")
            
        if k is None:
            k = len(y_true)
            
        # 二值化相关性
        binary_relevance = [1 if rel > 0 else 0 for rel in y_true]
        
        # 如果没有相关文档，返回0
        if sum(binary_relevance) == 0:
            return 0.0
            
        # 根据预测分数排序
        pred_order = np.argsort(y_pred)[::-1]
        
        # 找到第一个相关文档的位置
        for i in range(min(k, len(pred_order))):
            idx = pred_order[i]
            if binary_relevance[idx] == 1:
                return 1.0 / (i + 1)
                
        return 0.0
    
    def _compute_precision_recall(self, y_true: List[float], y_pred: List[float], k: int = None) -> Tuple[float, float]:
        """
        计算精度和召回率
        
        Args:
            y_true: 真实相关性分数（二值化：>0为相关）
            y_pred: 预测排序分数
            k: 只考虑前K个结果
            
        Returns:
            (precision@K, recall@K)
        """
        if len(y_true) != len(y_pred):
            raise ValueError(f"y_true和y_pred长度不匹配: {len(y_true)} != {len(y_pred)}")
            
        if k is None:
            k = len(y_true)
            
        # 二值化相关性
        binary_relevance = [1 if rel > 0 else 0 for rel in y_true]
        
        # 根据预测分数排序
        pred_order = np.argsort(y_pred)[::-1]
        
        # 计算前K个中的相关文档数
        relevant_in_top_k = 0
        for i in range(min(k, len(pred_order))):
            idx = pred_order[i]
            if binary_relevance[idx] == 1:
                relevant_in_top_k += 1
                
        # 计算精度和召回率
        precision = relevant_in_top_k / min(k, len(pred_order)) if min(k, len(pred_order)) > 0 else 0.0
        total_relevant = sum(binary_relevance)
        recall = relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0
        
        return precision, recall
    
    def _compute_f1(self, precision: float, recall: float) -> float:
        """
        计算F1分数
        
        Args:
            precision: 精度
            recall: 召回率
            
        Returns:
            F1分数
        """
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def evaluate_single_query(self, y_true: List[float], y_pred: List[float]) -> Dict[str, Union[Dict[int, float], float]]:
        """
        评估单个查询/日期
        
        Args:
            y_true: 真实相关性分数列表
            y_pred: 预测排序分数列表
            
        Returns:
            包含所有指标结果的字典
        """
        if len(y_true) != len(y_pred):
            raise ValueError(f"y_true和y_pred长度不匹配: {len(y_true)} != {len(y_pred)}")
            
        if len(y_true) == 0:
            raise ValueError("输入列表不能为空")
            
        results = {}
        
        # 计算每个指标
        for metric in self.metrics:
            if metric == 'ndcg':
                results[metric] = {k: self._compute_ndcg(y_true, y_pred, k) for k in self.k_values}
            elif metric == 'map':
                results[metric] = {k: self._compute_map(y_true, y_pred, k) for k in self.k_values}
            elif metric == 'mrr':
                results[metric] = {k: self._compute_mrr(y_true, y_pred, k) for k in self.k_values}
            elif metric == 'precision':
                precision_dict = {}
                for k in self.k_values:
                    precision, _ = self._compute_precision_recall(y_true, y_pred, k)
                    precision_dict[k] = precision
                results[metric] = precision_dict
            elif metric == 'recall':
                recall_dict = {}
                for k in self.k_values:
                    _, recall = self._compute_precision_recall(y_true, y_pred, k)
                    recall_dict[k] = recall
                results[metric] = recall_dict
            elif metric == 'f1':
                f1_dict = {}
                for k in self.k_values:
                    precision, recall = self._compute_precision_recall(y_true, y_pred, k)
                    f1_dict[k] = self._compute_f1(precision, recall)
                results[metric] = f1_dict
                
        return results
    
    def evaluate_batch(self, y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> Dict[str, Dict]:
        """
        批量评估多个查询/日期
        
        Args:
            y_true: 真实相关性分数数组
            y_pred: 预测排序分数数组
            groups: 分组信息数组，相同值表示同一组
            
        Returns:
            包含汇总统计和每个查询详细结果的字典
        """
        if len(y_true) != len(y_pred) or len(y_true) != len(groups):
            raise ValueError(f"输入数组长度不匹配: y_true={len(y_true)}, y_pred={len(y_pred)}, groups={len(groups)}")
            
        # 获取唯一分组
        unique_groups = np.unique(groups)
        
        # 为每个分组评估
        per_query_results = []
        for group_id in unique_groups:
            # 获取该组的索引
            mask = groups == group_id
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            
            # 评估该组
            if len(group_y_true) > 0:
                query_result = self.evaluate_single_query(group_y_true.tolist(), group_y_pred.tolist())
                per_query_results.append({
                    'group_id': group_id,
                    'results': query_result,
                    'n_samples': len(group_y_true)
                })
        
        # 计算汇总统计
        summary = self._compute_summary(per_query_results)
        
        return {
            'summary': summary,
            'per_query': per_query_results,
            'n_queries': len(per_query_results),
            'total_samples': len(y_true)
        }
    
    def _compute_summary(self, per_query_results: List[Dict]) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        计算汇总统计
        
        Args:
            per_query_results: 每个查询的结果列表
            
        Returns:
            汇总统计字典
        """
        if not per_query_results:
            return {}
            
        summary = {}
        
        # 初始化汇总结构
        for metric in self.metrics:
            summary[metric] = {}
            for k in self.k_values:
                summary[metric][k] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 1.0,
                    'max': 0.0,
                    'median': 0.0
                }
        
        # 收集所有查询的指标值
        metric_values = {}
        for metric in self.metrics:
            metric_values[metric] = {}
            for k in self.k_values:
                metric_values[metric][k] = []
        
        # 填充值
        for query_result in per_query_results:
            results = query_result['results']
            for metric in self.metrics:
                for k in self.k_values:
                    if metric in results and k in results[metric]:
                        metric_values[metric][k].append(results[metric][k])
        
        # 计算统计量
        for metric in self.metrics:
            for k in self.k_values:
                values = metric_values[metric][k]
                if values:
                    summary[metric][k]['mean'] = float(np.mean(values))
                    summary[metric][k]['std'] = float(np.std(values))
                    summary[metric][k]['min'] = float(np.min(values))
                    summary[metric][k]['max'] = float(np.max(values))
                    summary[metric][k]['median'] = float(np.median(values))
                else:
                    # 如果没有值，设为0
                    summary[metric][k]['mean'] = 0.0
                    summary[metric][k]['std'] = 0.0
                    summary[metric][k]['min'] = 0.0
                    summary[metric][k]['max'] = 0.0
                    summary[metric][k]['median'] = 0.0
        
        return summary
    
    def get_report(self, evaluation_results: Dict) -> str:
        """
        生成评估报告
        
        Args:
            evaluation_results: evaluate_batch返回的结果
            
        Returns:
            格式化的评估报告字符串
        """
        if not evaluation_results:
            return "无评估结果"
            
        summary = evaluation_results.get('summary', {})
        n_queries = evaluation_results.get('n_queries', 0)
        total_samples = evaluation_results.get('total_samples', 0)
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("排序模型评估报告")
        report_lines.append("=" * 60)
        report_lines.append(f"查询数量: {n_queries}")
        report_lines.append(f"总样本数: {total_samples}")
        report_lines.append("")
        
        # 为每个指标生成报告
        for metric in self.metrics:
            if metric in summary:
                report_lines.append(f"{metric.upper()}指标:")
                report_lines.append("-" * 40)
                
                for k in self.k_values:
                    if k in summary[metric]:
                        stats = summary[metric][k]
                        report_lines.append(
                            f"  K={k:2d}: 均值={stats['mean']:.4f} ± {stats['std']:.4f} "
                            f"(范围: {stats['min']:.4f}-{stats['max']:.4f}, 中位数: {stats['median']:.4f})"
                        )
                report_lines.append("")
        
        return "\n".join(report_lines)
    
    def evaluate_with_threshold(self, y_true: List[float], y_pred: List[float], 
                               threshold: float = 0.0) -> Dict[str, float]:
        """
        使用阈值进行二值化评估
        
        Args:
            y_true: 真实相关性分数
            y_pred: 预测排序分数
            threshold: 相关性阈值，大于该值为相关
            
        Returns:
            二值化评估结果
        """
        # 二值化真实相关性
        binary_y_true = [1 if rel > threshold else 0 for rel in y_true]
        
        # 使用二值化相关性重新计算指标
        results = {}
        
        for k in self.k_values:
            # 精度和召回率
            precision, recall = self._compute_precision_recall(binary_y_true, y_pred, k)
            f1 = self._compute_f1(precision, recall)
            
            results[f'precision@{k}'] = precision
            results[f'recall@{k}'] = recall
            results[f'f1@{k}'] = f1
            
            # MAP和MRR
            results[f'map@{k}'] = self._compute_map(binary_y_true, y_pred, k)
            results[f'mrr@{k}'] = self._compute_mrr(binary_y_true, y_pred, k)
        
        return results


# 便捷函数
def compute_ndcg(y_true: List[float], y_pred: List[float], k: int = 10) -> float:
    """
    便捷函数：计算NDCG@K
    
    Args:
        y_true: 真实相关性分数
        y_pred: 预测排序分数
        k: K值
        
    Returns:
        NDCG@K值
    """
    evaluator = NDCGEvaluator(k_values=[k])
    return evaluator._compute_ndcg(y_true, y_pred, k)


def compute_map(y_true: List[float], y_pred: List[float], k: int = 10) -> float:
    """
    便捷函数：计算MAP@K
    
    Args:
        y_true: 真实相关性分数
        y_pred: 预测排序分数
        k: K值
        
    Returns:
        MAP@K值
    """
    evaluator = NDCGEvaluator(k_values=[k])
    return evaluator._compute_map(y_true, y_pred, k)


def compute_mrr(y_true: List[float], y_pred: List[float], k: int = 10) -> float:
    """
    便捷函数：计算MRR@K
    
    Args:
        y_true: 真实相关性分数
        y_pred: 预测排序分数
        k: K值
        
    Returns:
        MRR@K值
    """
    evaluator = NDCGEvaluator(k_values=[k])
    return evaluator._compute_mrr(y_true, y_pred, k)


def compute_precision_recall(y_true: List[float], y_pred: List[float], k: int = 10) -> Tuple[float, float]:
    """
    便捷函数：计算精度和召回率
    
    Args:
        y_true: 真实相关性分数
        y_pred: 预测排序分数
        k: K值
        
    Returns:
        (precision@K, recall@K)
    """
    evaluator = NDCGEvaluator(k_values=[k])
    return evaluator._compute_precision_recall(y_true, y_pred, k)


if __name__ == "__main__":
    # 示例用法
    print("NDCG评估器示例")
    print("=" * 40)
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 20
    
    # 生成真实相关性（未来收益率）
    y_true = np.random.randn(n_samples) * 0.1
    
    # 生成预测分数（加入噪声）
    y_pred = y_true + np.random.randn(n_samples) * 0.05
    
    # 创建评估器
    evaluator = NDCGEvaluator(k_values=[5, 10, 20])
    
    # 评估单个查询
    print("单个查询评估:")
    results = evaluator.evaluate_single_query(y_true.tolist(), y_pred.tolist())
    
    for metric in ['ndcg', 'map', 'mrr']:
        print(f"\n{metric.upper()}:")
        for k in [5, 10, 20]:
            print(f"  @{k}: {results[metric][k]:.4f}")
    
    # 批量评估示例
    print("\n" + "=" * 40)
    print("批量评估示例:")
    
    # 创建分组数据（3个日期，每个日期20只股票）
    n_dates = 3
    n_stocks = 20
    
    y_true_batch = []
    y_pred_batch = []
    groups_batch = []
    
    for date_idx in range(n_dates):
        # 每个日期生成不同的相关性
        date_true = np.random.randn(n_stocks) * 0.1
        date_pred = date_true + np.random.randn(n_stocks) * 0.05
        
        y_true_batch.extend(date_true)
        y_pred_batch.extend(date_pred)
        groups_batch.extend([date_idx] * n_stocks)
    
    y_true_batch = np.array(y_true_batch)
    y_pred_batch = np.array(y_pred_batch)
    groups_batch = np.array(groups_batch)
    
    # 批量评估
    batch_results = evaluator.evaluate_batch(y_true_batch, y_pred_batch, groups_batch)
    
    # 生成报告
    report = evaluator.get_report(batch_results)
    print(report)