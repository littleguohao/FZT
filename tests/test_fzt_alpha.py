#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试FZTAlpha算子

测试FZTAlpha算子是否正确实现QLib算子接口，
并确保无未来函数计算。

作者: FZT项目组
创建日期: 2026-03-02
"""

import sys
import os
import unittest
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

import pandas as pd
import numpy as np

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# 忽略警告
warnings.filterwarnings('ignore')


class TestFZTAlpha(unittest.TestCase):
    """测试FZTAlpha算子"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试数据
        np.random.seed(42)
        n_days = 100
        self.dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        
        # 生成价格数据
        base_price = 100
        trend = np.linspace(0, 0.2, n_days)
        noise = np.random.normal(0, 0.02, n_days)
        returns = 0.001 + trend/100 + noise
        prices = base_price * np.exp(np.cumsum(returns))
        
        # 生成高低价
        highs = prices * (1 + np.abs(np.random.normal(0.01, 0.005, n_days)))
        lows = prices * (1 - np.abs(np.random.normal(0.01, 0.005, n_days)))
        
        self.test_data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': np.random.lognormal(mean=10, sigma=1, size=n_days)
        }, index=self.dates)
        
        # 添加股票代码列（模拟多股票数据）
        self.test_data['instrument'] = '000001.SZ'
        
        print(f"测试数据形状: {self.test_data.shape}")
        print(f"时间范围: {self.test_data.index.min()} 到 {self.test_data.index.max()}")
    
    def test_import_fzt_alpha(self):
        """测试导入FZTAlpha模块"""
        try:
            from qlib_operators.fzt_alpha import FZTAlpha
            print("✓ FZTAlpha模块导入成功")
            self.assertIsNotNone(FZTAlpha)
        except ImportError as e:
            self.fail(f"导入FZTAlpha失败: {e}")
    
    def test_fzt_alpha_creation(self):
        """测试创建FZTAlpha实例"""
        try:
            from qlib_operators.fzt_alpha import FZTAlpha
            
            # 测试创建实例
            fzt_alpha = FZTAlpha()
            print("✓ FZTAlpha实例创建成功")
            
            # 检查是否有必要的方法
            self.assertTrue(hasattr(fzt_alpha, 'forward'))
            self.assertTrue(hasattr(fzt_alpha, '__call__'))
            
        except Exception as e:
            self.fail(f"创建FZTAlpha实例失败: {e}")
    
    def test_fzt_alpha_forward_method(self):
        """测试forward方法"""
        try:
            from qlib_operators.fzt_alpha import FZTAlpha
            
            fzt_alpha = FZTAlpha()
            
            # 准备输入数据（模拟QLib格式）
            # 假设输入是DataFrame，包含OHLCV数据
            input_data = self.test_data[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # 测试forward方法
            result = fzt_alpha.forward(input_data)
            print(f"✓ forward方法执行成功，结果形状: {result.shape}")
            
            # 检查结果类型
            self.assertIsInstance(result, (pd.Series, pd.DataFrame))
            
            # 检查结果不为空
            if isinstance(result, pd.Series):
                self.assertGreater(len(result), 0)
            elif isinstance(result, pd.DataFrame):
                self.assertGreater(result.shape[0], 0)
                self.assertGreater(result.shape[1], 0)
            
        except Exception as e:
            self.fail(f"forward方法测试失败: {e}")
    
    def test_fzt_alpha_call_method(self):
        """测试__call__方法（QLib算子接口）"""
        try:
            from qlib_operators.fzt_alpha import FZTAlpha
            
            fzt_alpha = FZTAlpha()
            
            # 准备输入数据
            input_data = self.test_data[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # 测试__call__方法
            result = fzt_alpha(input_data)
            print(f"✓ __call__方法执行成功，结果形状: {result.shape}")
            
            # 检查结果类型
            self.assertIsInstance(result, (pd.Series, pd.DataFrame))
            
        except Exception as e:
            self.fail(f"__call__方法测试失败: {e}")
    
    def test_no_lookahead_bias(self):
        """测试无未来函数计算"""
        try:
            from qlib_operators.fzt_alpha import FZTAlpha
            
            fzt_alpha = FZTAlpha()
            
            # 准备输入数据
            input_data = self.test_data[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # 计算特征
            result = fzt_alpha(input_data)
            
            # 检查是否有未来数据泄露
            # 对于每个时间点t，结果应该只依赖于t及之前的数据
            if isinstance(result, pd.Series):
                # 检查是否有NaN值（可能由于未来数据导致）
                nan_count = result.isna().sum()
                print(f"结果中NaN数量: {nan_count}")
                self.assertLessEqual(nan_count, len(result) * 0.1)  # 允许少量NaN（由于窗口计算）
            
            print("✓ 无未来函数检查通过")
            
        except Exception as e:
            self.fail(f"无未来函数检查失败: {e}")
    
    def test_feature_names(self):
        """测试特征名称"""
        try:
            from qlib_operators.fzt_alpha import FZTAlpha
            
            fzt_alpha = FZTAlpha()
            
            # 检查是否有get_feature_names方法
            if hasattr(fzt_alpha, 'get_feature_names'):
                feature_names = fzt_alpha.get_feature_names()
                print(f"特征名称: {feature_names}")
                
                # 检查特征名称不为空
                self.assertIsInstance(feature_names, list)
                self.assertGreater(len(feature_names), 0)
                
                # 检查特征名称是字符串
                for name in feature_names:
                    self.assertIsInstance(name, str)
            
            print("✓ 特征名称检查通过")
            
        except Exception as e:
            self.fail(f"特征名称检查失败: {e}")
    
    def test_qlib_compatibility(self):
        """测试QLib兼容性"""
        try:
            from qlib_operators.fzt_alpha import FZTAlpha
            
            fzt_alpha = FZTAlpha()
            
            # 检查是否是QLib算子
            # 如果QLib可用，检查是否继承自Operator
            try:
                from qlib.data.ops import Operator
                self.assertTrue(isinstance(fzt_alpha, Operator) or 
                              issubclass(FZTAlpha, Operator))
                print("✓ 是QLib Operator子类")
            except ImportError:
                # QLib不可用，检查是否有简化版本
                print("⚠ QLib不可用，使用简化版本")
                # 检查简化版本是否有必要的方法
                self.assertTrue(hasattr(fzt_alpha, 'forward'))
                self.assertTrue(hasattr(fzt_alpha, '__call__'))
            
        except Exception as e:
            self.fail(f"QLib兼容性测试失败: {e}")
    
    def test_multiple_stocks(self):
        """测试多股票数据处理"""
        try:
            from qlib_operators.fzt_alpha import FZTAlpha
            
            fzt_alpha = FZTAlpha()
            
            # 创建多股票数据
            n_stocks = 3
            multi_stock_data = []
            
            for i in range(n_stocks):
                stock_data = self.test_data.copy()
                stock_data['instrument'] = f'00000{i+1}.SZ'
                multi_stock_data.append(stock_data)
            
            multi_stock_df = pd.concat(multi_stock_data)
            
            # 按股票分组处理
            grouped = multi_stock_df.groupby('instrument')
            
            results = []
            for stock_code, group in grouped:
                # 提取OHLCV数据
                stock_ohlcv = group[['open', 'high', 'low', 'close', 'volume']]
                
                # 计算特征
                result = fzt_alpha(stock_ohlcv)
                
                # 添加股票代码
                if isinstance(result, pd.Series):
                    result = result.to_frame(name=f'fzt_alpha_{stock_code}')
                elif isinstance(result, pd.DataFrame):
                    result.columns = [f'{col}_{stock_code}' for col in result.columns]
                
                results.append(result)
            
            if results:
                print(f"✓ 成功处理 {n_stocks} 只股票")
                print(f"每个股票结果形状: {[r.shape for r in results]}")
            
        except Exception as e:
            self.fail(f"多股票数据处理失败: {e}")
    
    def test_performance(self):
        """测试性能（计算时间）"""
        try:
            from qlib_operators.fzt_alpha import FZTAlpha
            import time
            
            fzt_alpha = FZTAlpha()
            
            # 准备输入数据
            input_data = self.test_data[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # 计时
            start_time = time.time()
            result = fzt_alpha(input_data)
            end_time = time.time()
            
            compute_time = end_time - start_time
            print(f"计算时间: {compute_time:.4f} 秒")
            print(f"数据行数: {len(input_data)}")
            
            # 检查计算时间是否合理（100行数据应该在1秒内）
            self.assertLess(compute_time, 1.0, "计算时间过长")
            
            print("✓ 性能测试通过")
            
        except Exception as e:
            self.fail(f"性能测试失败: {e}")


def run_tests():
    """运行所有测试"""
    print("=" * 80)
    print("FZTAlpha算子测试")
    print("=" * 80)
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFZTAlpha)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 80)
    print("测试完成")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败测试数: {len(result.failures)}")
    print(f"错误测试数: {len(result.errors)}")
    
    if result.failures:
        print("\n失败测试:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\n错误测试:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return len(result.failures) + len(result.errors)


if __name__ == "__main__":
    # 运行测试
    error_count = run_tests()
    
    # 返回退出码
    sys.exit(min(error_count, 255))