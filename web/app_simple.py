#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版缠论分析Web应用 - 用于Render调试
"""

import os
import sys
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['SECRET_KEY'] = 'chan_analysis_secret_key'

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 尝试导入缠论分析模块，失败时使用占位符
try:
    from scripts.run_fixed import (
        load_ohlc, find_fractals, build_strokes, build_segments, 
        detect_zhongshu, detect_divergence, resolve_inclusion
    )
    CHAN_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: 缠论分析模块导入失败: {e}")
    CHAN_MODULES_AVAILABLE = False
    
    # 创建占位符函数
    def load_ohlc(*args, **kwargs): return None
    def find_fractals(*args, **kwargs): return []
    def build_strokes(*args, **kwargs): return []
    def build_segments(*args, **kwargs): return []
    def detect_zhongshu(*args, **kwargs): return []
    def detect_divergence(*args, **kwargs): return []
    def resolve_inclusion(*args, **kwargs): return args[0] if args else None

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'chan_modules': CHAN_MODULES_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """执行缠论分析"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        start_date = data.get('start_date', '')
        end_date = data.get('end_date', '')
        
        if not symbol or not start_date or not end_date:
            return jsonify({'error': '请填写完整的股票代码和日期范围'}), 400
        
        # 检查缠论分析模块是否可用
        if not CHAN_MODULES_AVAILABLE:
            return jsonify({
                'success': False,
                'error': '缠论分析模块不可用，请检查部署配置',
                'debug_info': {
                    'python_path': sys.path,
                    'current_dir': current_dir,
                    'parent_dir': parent_dir
                }
            }), 500
        
        # 这里可以添加实际的分析逻辑
        # 暂时返回模拟数据
        return jsonify({
            'success': True,
            'message': f'分析 {symbol} 从 {start_date} 到 {end_date}',
            'debug_info': {
                'chan_modules_available': CHAN_MODULES_AVAILABLE,
                'python_path': sys.path[:3]  # 只显示前3个路径
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'分析失败: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
