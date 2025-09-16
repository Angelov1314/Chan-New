#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的测试应用，用于验证Render部署
"""

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    """健康检查端点"""
    return jsonify({
        'status': 'success',
        'message': 'Chan Analysis Web App is running!',
        'version': '1.0.0'
    })

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
