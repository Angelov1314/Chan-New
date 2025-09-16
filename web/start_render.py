#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render部署启动脚本
包含错误处理和资源优化
"""

import os
import sys
import traceback
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """检查环境配置"""
    logger.info("检查环境配置...")
    
    # 检查必要的环境变量
    port = os.environ.get('PORT', '8080')
    logger.info(f"PORT: {port}")
    
    # 检查Python版本
    logger.info(f"Python版本: {sys.version}")
    
    # 检查工作目录
    logger.info(f"当前工作目录: {os.getcwd()}")
    
    return True

def test_imports():
    """测试关键模块导入"""
    logger.info("测试模块导入...")
    
    try:
        import flask
        logger.info("✓ Flask导入成功")
    except ImportError as e:
        logger.error(f"✗ Flask导入失败: {e}")
        return False
    
    try:
        import pandas
        logger.info("✓ Pandas导入成功")
    except ImportError as e:
        logger.error(f"✗ Pandas导入失败: {e}")
        return False
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # 设置非交互式后端
        logger.info("✓ Matplotlib导入成功")
    except ImportError as e:
        logger.error(f"✗ Matplotlib导入失败: {e}")
        return False
    
    try:
        import yfinance
        logger.info("✓ YFinance导入成功")
    except ImportError as e:
        logger.error(f"✗ YFinance导入失败: {e}")
        return False
    
    return True

def test_chan_analysis():
    """测试缠论分析模块"""
    logger.info("测试缠论分析模块...")
    
    try:
        # 添加项目根目录到Python路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        
        from scripts.run_fixed import (
            load_ohlc, find_fractals, build_strokes, build_segments, 
            detect_zhongshu, detect_divergence, resolve_inclusion
        )
        logger.info("✓ 缠论分析模块导入成功")
        return True
    except ImportError as e:
        logger.warning(f"⚠️ 缠论分析模块导入失败: {e}")
        logger.info("应用将使用fallback函数运行")
        return False

def create_app():
    """创建Flask应用"""
    logger.info("创建Flask应用...")
    
    try:
        from app import app
        logger.info("✓ Flask应用创建成功")
        return app
    except Exception as e:
        logger.error(f"✗ Flask应用创建失败: {e}")
        traceback.print_exc()
        return None

def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("缠论技术分析系统 - Render部署启动")
    logger.info("=" * 50)
    
    # 检查环境
    if not check_environment():
        logger.error("环境检查失败")
        return False
    
    # 测试基本导入
    if not test_imports():
        logger.error("基本模块导入失败")
        return False
    
    # 测试缠论分析模块
    test_chan_analysis()
    
    # 创建应用
    app = create_app()
    if not app:
        logger.error("应用创建失败")
        return False
    
    logger.info("✅ 所有检查通过，应用准备就绪")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
