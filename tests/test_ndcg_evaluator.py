"""
测试NDCG评估器
"""
import numpy as np
import pytest
from src.evaluation.ndcg_evaluator import NDCGEvaluator


class TestNDCGEvaluator:
    """测试NDCGEvaluator类"""
    
    def test_initialization(self):
        """测试初始化"""
        evaluator = NDCGEvaluator(k_values=[1, 3, 5, 10])
        assert evaluator.k_values == [1, 3, 5, 10]
        assert evaluator.metrics == ['ndcg', 'map', 'mrr', 'precision', 'recall', 'f1']
        
    def test_compute_dcg(self):
        """测试DCG计算"""
        evaluator = NDCGEvaluator()
        
        # 测试简单DCG计算
        relevance = [3, 2, 1, 0]
        dcg = evaluator._compute_dcg(relevance, k=4)
        expected = 3 / np.log2(2) + 2 / np.log2(3) + 1 / np.log2(4) + 0 / np.log2(5)
        assert np.isclose(dcg, expected)
        
        # 测试K值限制
        dcg_k2 = evaluator._compute_dcg(relevance, k=2)
        expected_k2 = 3 / np.log2(2) + 2 / np.log2(3)
        assert np.isclose(dcg_k2, expected_k2)
        
    def test_compute_ndcg(self):
        """测试NDCG计算"""
        evaluator = NDCGEvaluator()
        
        # 完美排序情况
        relevance = [3, 2, 1, 0]
        ndcg = evaluator._compute_ndcg(relevance, relevance, k=4)
        assert np.isclose(ndcg, 1.0)
        
        # 最差排序情况
        worst_order = [0, 1, 2, 3]
        ndcg_worst = evaluator._compute_ndcg(relevance, worst_order, k=4)
        assert ndcg_worst < 1.0
        
        # 部分排序情况
        partial_order = [2, 3, 1, 0]
        ndcg_partial = evaluator._compute_ndcg(relevance, partial_order, k=4)
        assert 0 < ndcg_partial < 1.0
        
    def test_compute_map(self):
        """测试MAP计算"""
        evaluator = NDCGEvaluator()
        
        # 完美排序情况
        relevance = [1, 1, 0, 0]  # 前两个相关
        map_score = evaluator._compute_map(relevance, relevance, k=4)
        assert np.isclose(map_score, 1.0)
        
        # 部分排序情况
        partial_order = [0, 1, 1, 0]
        map_partial = evaluator._compute_map(relevance, partial_order, k=4)
        assert 0 < map_partial < 1.0
        
    def test_compute_mrr(self):
        """测试MRR计算"""
        evaluator = NDCGEvaluator()
        
        # 第一个相关（预测分数完美匹配）
        relevance = [1, 0, 0, 0]
        mrr = evaluator._compute_mrr(relevance, relevance, k=4)
        assert np.isclose(mrr, 1.0)
        
        # 相关文档在第二个位置（预测分数把相关文档排第二）
        # 真实相关性：[1, 0, 0, 0]
        # 预测分数：[0.5, 1.0, 0.0, 0.0] - 第二个分数最高
        y_pred = [0.5, 1.0, 0.0, 0.0]
        mrr_second = evaluator._compute_mrr(relevance, y_pred, k=4)
        assert np.isclose(mrr_second, 0.5)  # 1/2 = 0.5
        
        # 相关文档在第三个位置
        y_pred_third = [0.5, 0.6, 1.0, 0.0]
        mrr_third = evaluator._compute_mrr(relevance, y_pred_third, k=4)
        assert np.isclose(mrr_third, 1/3)  # 1/3 ≈ 0.333
        
        # 没有相关
        no_relevant = [0, 0, 0, 0]
        mrr_none = evaluator._compute_mrr(no_relevant, no_relevant, k=4)
        assert np.isclose(mrr_none, 0.0)
        
    def test_compute_precision_recall(self):
        """测试精度和召回率计算"""
        evaluator = NDCGEvaluator()
        
        # 完美情况
        relevance = [1, 1, 0, 0]
        precision, recall = evaluator._compute_precision_recall(relevance, relevance, k=2)
        assert np.isclose(precision, 1.0)
        assert np.isclose(recall, 1.0)
        
        # 部分情况
        order = [1, 0, 1, 0]
        precision_partial, recall_partial = evaluator._compute_precision_recall(relevance, order, k=2)
        assert precision_partial == 0.5
        assert recall_partial == 0.5
        
    def test_evaluate_single_query(self):
        """测试单个查询评估"""
        evaluator = NDCGEvaluator(k_values=[1, 2, 4])
        
        # 测试数据
        y_true = [3, 2, 1, 0]
        y_pred = [3, 2, 1, 0]  # 完美排序
        
        results = evaluator.evaluate_single_query(y_true, y_pred)
        
        # 检查结果格式
        assert 'ndcg' in results
        assert 'map' in results
        assert 'mrr' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        
        # 检查K值
        for metric in results:
            if metric != 'f1':  # f1不是字典
                assert 1 in results[metric]
                assert 2 in results[metric]
                assert 4 in results[metric]
                
        # 完美排序应该得到高分数
        assert np.isclose(results['ndcg'][1], 1.0)
        assert np.isclose(results['ndcg'][2], 1.0)
        assert np.isclose(results['ndcg'][4], 1.0)
        
    def test_evaluate_batch(self):
        """测试批量评估"""
        evaluator = NDCGEvaluator(k_values=[1, 2])
        
        # 测试数据 - 3个查询
        y_true = [
            [3, 2, 1, 0],  # 查询1
            [2, 1, 0, 0],  # 查询2
            [1, 0, 0, 0]   # 查询3
        ]
        y_pred = [
            [3, 2, 1, 0],  # 完美排序
            [2, 1, 0, 0],  # 完美排序
            [1, 0, 0, 0]   # 完美排序
        ]
        groups = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]  # 每个查询4个样本
        
        # 展平数据
        y_true_flat = np.concatenate(y_true)
        y_pred_flat = np.concatenate(y_pred)
        
        results = evaluator.evaluate_batch(y_true_flat, y_pred_flat, groups)
        
        # 检查结果格式
        assert 'summary' in results
        assert 'per_query' in results
        
        # 检查汇总统计
        summary = results['summary']
        assert 'ndcg' in summary
        assert 'map' in summary
        assert 'mrr' in summary
        
        # 完美排序应该得到高平均分数
        assert np.isclose(summary['ndcg'][1]['mean'], 1.0)
        assert np.isclose(summary['ndcg'][2]['mean'], 1.0)
        
    def test_evaluate_with_groups(self):
        """测试分组评估"""
        evaluator = NDCGEvaluator(k_values=[1, 2])
        
        # 创建分组数据（按日期）
        n_dates = 3
        n_stocks_per_date = 4
        
        y_true = []
        y_pred = []
        groups = []
        
        for date_idx in range(n_dates):
            # 每个日期生成不同的相关性分数
            base_score = date_idx + 1
            date_true = [base_score * (n_stocks_per_date - i) for i in range(n_stocks_per_date)]
            date_pred = date_true.copy()  # 完美排序
            
            y_true.extend(date_true)
            y_pred.extend(date_pred)
            groups.extend([date_idx] * n_stocks_per_date)
            
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        groups = np.array(groups)
        
        results = evaluator.evaluate_batch(y_true, y_pred, groups)
        
        # 检查每个查询都有结果
        assert len(results['per_query']) == n_dates
        
        # 检查汇总统计
        summary = results['summary']
        assert 'ndcg' in summary
        assert summary['ndcg'][1]['mean'] > 0.9  # 应该接近完美
        
    def test_edge_cases(self):
        """测试边界情况"""
        evaluator = NDCGEvaluator()
        
        # 空列表
        with pytest.raises(ValueError):
            evaluator.evaluate_single_query([], [])
            
        # 长度不匹配
        with pytest.raises(ValueError):
            evaluator.evaluate_single_query([1, 2], [1, 2, 3])
            
        # 所有相关性为0
        y_true = [0, 0, 0, 0]
        y_pred = [1, 2, 3, 4]
        results = evaluator.evaluate_single_query(y_true, y_pred)
        
        # NDCG应该为0（避免除以0）
        assert np.isclose(results['ndcg'][1], 0.0)
        
    def test_f1_score_calculation(self):
        """测试F1分数计算"""
        evaluator = NDCGEvaluator()
        
        # 完美情况
        precision = 1.0
        recall = 1.0
        f1 = evaluator._compute_f1(precision, recall)
        assert np.isclose(f1, 1.0)
        
        # 部分情况
        precision = 0.5
        recall = 0.5
        f1 = evaluator._compute_f1(precision, recall)
        assert np.isclose(f1, 0.5)
        
        # 边界情况
        precision = 0.0
        recall = 0.0
        f1 = evaluator._compute_f1(precision, recall)
        assert np.isclose(f1, 0.0)
        
    def test_realistic_stock_scenario(self):
        """测试真实股票排序场景"""
        evaluator = NDCGEvaluator(k_values=[5, 10, 20])
        
        # 模拟股票数据：20只股票，相关性为未来收益率
        n_stocks = 20
        np.random.seed(42)
        
        # 生成真实相关性（未来收益率，有正有负）
        y_true = np.random.randn(n_stocks) * 0.1
        
        # 生成预测分数（与真实相关性相关但加入噪声）
        y_pred = y_true + np.random.randn(n_stocks) * 0.05
        
        results = evaluator.evaluate_single_query(y_true, y_pred)
        
        # 检查所有指标都有值
        for metric in ['ndcg', 'map', 'mrr', 'precision', 'recall', 'f1']:
            if metric != 'f1':
                for k in [5, 10, 20]:
                    assert k in results[metric]
                    assert 0 <= results[metric][k] <= 1.0
            else:
                for k in [5, 10, 20]:
                    assert 0 <= results['f1'][k] <= 1.0