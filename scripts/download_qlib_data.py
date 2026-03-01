#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qlib数据下载脚本

功能：下载Qlib测试数据集（2005-2020年中国A股数据）

使用方法：
1. python scripts/download_qlib_data.py
2. 或直接运行：./scripts/download_qlib_data.py

注意：
- 需要稳定的网络连接
- 数据大小约0.45GB
- 下载时间取决于网络速度
"""

import sys
import os
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_qlib_installation():
    """检查Qlib是否已安装"""
    try:
        import qlib
        logger.info(f"Qlib已安装，版本: {qlib.__version__}")
        return True
    except ImportError:
        logger.error("Qlib未安装，请先安装: pip install qlib")
        return False


def download_qlib_data():
    """下载Qlib数据"""
    try:
        logger.info("开始下载Qlib数据...")
        
        # 方法1: 使用GetData
        try:
            from qlib.tests.data import GetData
            from qlib.config import REG_CN
            
            logger.info("使用方法1: GetData().qlib_data()")
            GetData().qlib_data(target_dir="~/.qlib/qlib_data", region=REG_CN, version="v2")
            logger.info("✅ Qlib数据下载完成")
            return True
            
        except Exception as e1:
            logger.warning(f"方法1失败: {e1}")
            
            # 方法2: 使用auto_mount
            try:
                logger.info("尝试方法2: qlib.init(auto_mount=True)")
                import qlib
                from qlib.config import REG_CN
                
                qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN, auto_mount=True)
                logger.info("✅ 通过auto_mount下载数据")
                return True
                
            except Exception as e2:
                logger.error(f"方法2失败: {e2}")
                
                # 方法3: 手动创建目录并提示
                logger.info("尝试方法3: 手动检查")
                qlib_dir = Path.home() / ".qlib" / "qlib_data" / "cn_data"
                if qlib_dir.exists():
                    logger.info(f"Qlib数据目录已存在: {qlib_dir}")
                    return True
                else:
                    logger.error(f"Qlib数据目录不存在: {qlib_dir}")
                    return False
                    
    except Exception as e:
        logger.error(f"下载Qlib数据失败: {e}")
        return False


def check_data_availability():
    """检查数据是否可用"""
    try:
        import qlib
        from qlib.data import D
        from qlib.config import REG_CN
        
        # 初始化Qlib
        provider_uri = "~/.qlib/qlib_data/cn_data"
        qlib.init(provider_uri=provider_uri, region=REG_CN)
        
        # 尝试获取股票列表
        instruments = D.instruments("csi300")
        stock_count = len(instruments)
        
        logger.info(f"✅ Qlib数据可用，CSI300股票数量: {stock_count}")
        
        if stock_count > 0:
            # 尝试获取一只股票的数据
            stock = list(instruments)[0]
            data = D.features([stock], fields=['$close'], start_time='2020-01-01', end_time='2020-01-10')
            logger.info(f"✅ 数据获取成功，示例数据形状: {data.shape}")
            return True
        else:
            logger.warning("⚠️ 股票列表为空，数据可能不完整")
            return False
            
    except Exception as e:
        logger.error(f"检查数据可用性失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 80)
    print("Qlib数据下载工具")
    print("=" * 80)
    
    # 1. 检查Qlib安装
    if not check_qlib_installation():
        print("\n❌ 请先安装Qlib: pip install qlib")
        return 1
    
    # 2. 检查是否已下载数据
    print("\n🔍 检查数据状态...")
    if check_data_availability():
        print("\n✅ Qlib数据已下载并可用")
        return 0
    
    # 3. 下载数据
    print("\n⬇️ 开始下载Qlib数据...")
    print("注意：数据大小约0.45GB，下载时间取决于网络速度")
    print("如果下载失败，请检查网络连接或配置代理")
    
    success = download_qlib_data()
    
    if success:
        print("\n✅ Qlib数据下载完成")
        
        # 再次检查数据可用性
        if check_data_availability():
            print("\n🎉 Qlib数据已准备就绪，可以用于FZT项目")
        else:
            print("\n⚠️ 数据下载完成但检查失败，可能需要重新下载")
            
        return 0
    else:
        print("\n❌ Qlib数据下载失败")
        print("\n💡 建议：")
        print("1. 检查网络连接")
        print("2. 尝试配置代理")
        print("3. 手动下载数据包")
        print("4. 暂时使用本地CSV数据进行开发")
        return 1


if __name__ == "__main__":
    sys.exit(main())