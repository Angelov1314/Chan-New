#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动脚本，用于调试部署问题
"""

import os
import sys
import traceback

def test_imports():
    """测试所有必要的导入"""
    print("Testing imports...")
    
    try:
        import flask
        print("✓ Flask imported successfully")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        import pandas
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import numpy
        print("✓ Numpy imported successfully")
    except ImportError as e:
        print(f"✗ Numpy import failed: {e}")
        return False
    
    try:
        import matplotlib
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import yfinance
        print("✓ YFinance imported successfully")
    except ImportError as e:
        print(f"✗ YFinance import failed: {e}")
        return False
    
    return True

def test_chan_imports():
    """测试缠论分析模块导入"""
    print("\nTesting chan analysis imports...")
    
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
        print("✓ Chan analysis modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Chan analysis modules import failed: {e}")
        traceback.print_exc()
        return False

def test_app_creation():
    """测试应用创建"""
    print("\nTesting app creation...")
    
    try:
        from app import app
        print("✓ App created successfully")
        return True
    except Exception as e:
        print(f"✗ App creation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("Chan Analysis Web App - Deployment Test")
    print("=" * 50)
    
    # 测试基本导入
    if not test_imports():
        print("\n❌ Basic imports failed!")
        return False
    
    # 测试缠论分析模块导入
    if not test_chan_imports():
        print("\n⚠️  Chan analysis imports failed, but app may still work with fallback functions")
    
    # 测试应用创建
    if not test_app_creation():
        print("\n❌ App creation failed!")
        return False
    
    print("\n✅ All tests passed! App should be ready to run.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
