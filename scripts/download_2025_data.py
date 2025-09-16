#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载2025年1月到6月的AAPL数据
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def download_2025_data():
    """下载2025年1月到6月的AAPL数据"""
    
    print("正在下载2025年1月到6月的AAPL数据...")
    
    try:
        # 创建ticker对象
        ticker = yf.Ticker("AAPL")
        
        # 下载2025年1月到6月的数据
        data = yf.download("AAPL", start="2025-01-01", end="2025-06-30", interval="1d")
        
        if data.empty:
            print("警告: 2025年数据为空，可能还未到2025年或数据不可用")
            return None
        
        print(f"成功下载 {len(data)} 天的2025年数据")
        print(f"日期范围: {data.index[0].strftime('%Y-%m-%d')} 到 {data.index[-1].strftime('%Y-%m-%d')}")
        
        # 数据清理和格式化
        data_clean = data.copy()
        # 修复列名处理
        if isinstance(data_clean.columns, pd.MultiIndex):
            data_clean.columns = [col[0] if isinstance(col, tuple) else col for col in data_clean.columns]
        else:
            data_clean.columns = [col.replace(' ', '_') for col in data_clean.columns]
        
        # 添加时间相关列
        data_clean['Date'] = data_clean.index.strftime('%Y-%m-%d')
        data_clean['Time'] = data_clean.index.strftime('%H:%M:%S')
        data_clean['Weekday'] = data_clean.index.weekday
        data_clean['Level'] = '1d'
        
        # 计算技术指标
        data_clean = add_technical_indicators(data_clean)
        
        # 重新排列列顺序
        base_columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Level']
        other_columns = [col for col in data_clean.columns if col not in base_columns]
        data_clean = data_clean[base_columns + other_columns]
        
        # 保存数据
        csv_path = "data/AAPL_2025_data.csv"
        data_clean.to_csv(csv_path, index=True)
        
        print(f"2025年数据已保存到: {csv_path}")
        print(f"价格范围: ${data_clean['Low'].min():.2f} - ${data_clean['High'].max():.2f}")
        
        return csv_path
        
    except Exception as e:
        print(f"下载2025年数据失败: {e}")
        return None

def add_technical_indicators(data):
    """添加技术指标"""
    # 移动平均线
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    # 布林带
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    
    # RSI
    data['RSI'] = calculate_rsi(data['Close'])
    
    # MACD
    macd_data = calculate_macd(data['Close'])
    data['MACD'] = macd_data['MACD']
    data['MACD_Signal'] = macd_data['Signal']
    data['MACD_Histogram'] = macd_data['Histogram']
    
    # 成交量指标
    data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
    
    # 价格变化
    data['Price_Change'] = data['Close'].pct_change()
    data['High_Low_Range'] = (data['High'] - data['Low']) / data['Close']
    
    return data

def calculate_rsi(prices, window=14):
    """计算RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """计算MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    
    return {
        'MACD': macd,
        'Signal': signal_line,
        'Histogram': histogram
    }

if __name__ == "__main__":
    download_2025_data()
