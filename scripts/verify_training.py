#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证训练脚本功能
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_files_exist():
    """验证必要文件是否存在"""
    print("验证必要文件是否存在...")
    
    required_files = [
        'config/training_pipeline.yaml',
        'scripts/train_ranking_model.py',
        'docs/training_pipeline_usage.md'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (缺失)")
            all_exist = False
    
    return all_exist

def verify_imports():
    """验证导入是否正常"""
    print("\n验证导入是否正常...")
    
    try:
        # 尝试导入训练脚本
        from scripts.train_ranking_model import FZTRankingTrainer, parse_arguments
        print("✓ 成功导入FZTRankingTrainer和parse_arguments")
        
        # 尝试导入必要的库
        import pandas as pd
        import numpy as np
        import yaml
        import joblib
        print("✓ 成功导入必要库")
        
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def verify_config_structure():
    """验证配置文件结构"""
    print("\n验证配置文件结构...")
    
    try:
        import yaml
        
        with open('config/training_pipeline.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查必要部分
        if 'training' not in config:
            print("✗ 配置文件中缺少'training'部分")
            return False
        
        training_config = config['training']
        
        required_sections = ['data', 'features', 'model', 'mode', 'output']
        missing_sections = []
        
        for section in required_sections:
            if section not in training_config:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"✗ 配置文件中缺少以下部分: {missing_sections}")
            return False
        
        print("✓ 配置文件结构正确")
        return True
        
    except Exception as e:
        print(f"✗ 验证配置文件失败: {e}")
        return False

def verify_script_syntax():
    """验证脚本语法"""
    print("\n验证脚本语法...")
    
    try:
        import py_compile
        py_compile.compile('scripts/train_ranking_model.py', doraise=True)
        print("✓ 训练脚本语法正确")
        return True
    except py_compile.PyCompileError as e:
        print(f"✗ 训练脚本语法错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 验证脚本语法失败: {e}")
        return False

def main():
    """主验证函数"""
    print("="*60)
    print("FZT排序增强策略训练脚本验证")
    print("="*60)
    
    tests = [
        ("文件存在性", verify_files_exist),
        ("导入检查", verify_imports),
        ("配置文件结构", verify_config_structure),
        ("脚本语法", verify_script_syntax)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed_tests += 1
    
    print("\n" + "="*60)
    print(f"验证结果: {passed_tests}/{total_tests} 通过")
    
    if passed_tests == total_tests:
        print("✓ 所有验证通过！训练脚本准备就绪。")
        print("\n使用说明:")
        print("1. 基本训练: python scripts/train_ranking_model.py")
        print("2. 查看帮助: python scripts/train_ranking_model.py --help")
        print("3. 干运行测试: python scripts/train_ranking_model.py --dry-run")
        print("4. 详细文档: docs/training_pipeline_usage.md")
        return 0
    else:
        print("⚠ 部分验证失败，请检查问题。")
        return 1

if __name__ == "__main__":
    sys.exit(main())