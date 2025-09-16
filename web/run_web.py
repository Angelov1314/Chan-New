#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动缠论技术分析Web应用
"""

import os
import sys
import subprocess

def install_requirements():
    """安装依赖包"""
    print("正在安装依赖包...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("依赖包安装完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"依赖包安装失败: {e}")
        return False

def start_app():
    """启动应用"""
    print("正在启动缠论技术分析Web应用...")
    print("=" * 50)
    print("应用地址: http://localhost:5000")
    print("按 Ctrl+C 停止应用")
    print("=" * 50)
    
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖包已正确安装")
    except Exception as e:
        print(f"启动失败: {e}")

if __name__ == "__main__":
    # 检查是否在正确的目录
    if not os.path.exists("app.py"):
        print("错误: 请在web目录下运行此脚本")
        sys.exit(1)
    
    # 安装依赖
    if install_requirements():
        # 启动应用
        start_app()
    else:
        print("无法启动应用，请手动安装依赖包")
        sys.exit(1)
