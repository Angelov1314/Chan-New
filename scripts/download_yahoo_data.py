#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从Yahoo Finance下载缠论分析所需的完整数据
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

class YahooDataDownloader:
    def __init__(self, symbol="AAPL"):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        
    def download_multi_timeframe_data(self, start_date="2023-01-01", end_date="2024-12-31"):
        """下载多时间周期数据"""
        
        # 定义时间周期
        timeframes = {
            '1m': '1m',      # 1分钟 - 用于精确分型识别
            '5m': '5m',      # 5分钟 - 短期笔和线段
            '15m': '15m',    # 15分钟 - 中期分析
            '30m': '30m',    # 30分钟 - 重要级别
            '1h': '1h',      # 1小时 - 日线以下重要级别
            '1d': '1d',      # 日线 - 主要分析级别
            '1wk': '1wk',    # 周线 - 长期趋势
            '1mo': '1mo'     # 月线 - 超长期趋势
        }
        
        print(f"开始下载 {self.symbol} 的多时间周期数据...")
        print(f"时间范围: {start_date} 到 {end_date}")
        
        all_data = {}
        
        for level, interval in timeframes.items():
            print(f"\n下载 {level} 数据...")
            
            try:
                # 下载数据
                data = self.ticker.history(
                    start=start_date, 
                    end=end_date, 
                    interval=interval,
                    auto_adjust=True,  # 自动调整价格
                    prepost=True,      # 包含盘前盘后数据
                    actions=True       # 包含股息和分割信息
                )
                
                if data.empty:
                    print(f"  警告: {level} 数据为空")
                    continue
                
                # 数据清理和格式化
                data_clean = self._clean_data(data, level)
                
                # 保存数据
                filename = f"{self.symbol}_{level}_data.csv"
                data_clean.to_csv(filename, index=True)
                
                all_data[level] = data_clean
                
                print(f"  ✓ 成功下载 {len(data_clean)} 条 {level} 数据")
                print(f"  ✓ 价格范围: ${data_clean['Low'].min():.2f} - ${data_clean['High'].max():.2f}")
                print(f"  ✓ 保存到: {filename}")
                
            except Exception as e:
                print(f"  ✗ 下载 {level} 数据失败: {e}")
                continue
        
        return all_data
    
    def _clean_data(self, data, level):
        """清理和格式化数据"""
        # 移除NaN值
        data = data.dropna()
        
        # 确保列名标准化
        data.columns = [col.replace(' ', '_') for col in data.columns]
        
        # 添加时间相关列
        data['Date'] = data.index.strftime('%Y-%m-%d')
        data['Time'] = data.index.strftime('%H:%M:%S')
        data['Weekday'] = data.index.weekday
        data['Level'] = level
        
        # 计算技术指标
        data = self._add_technical_indicators(data)
        
        # 重新排列列顺序
        base_columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Level']
        other_columns = [col for col in data.columns if col not in base_columns]
        data = data[base_columns + other_columns]
        
        return data
    
    def _add_technical_indicators(self, data):
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
        data['RSI'] = self._calculate_rsi(data['Close'])
        
        # MACD
        macd_data = self._calculate_macd(data['Close'])
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
    
    def _calculate_rsi(self, prices, window=14):
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
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
    
    def download_company_info(self):
        """下载公司基本信息"""
        print(f"\n下载 {self.symbol} 公司信息...")
        
        try:
            info = self.ticker.info
            
            # 保存基本信息
            basic_info = {
                'symbol': info.get('symbol', ''),
                'longName': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'marketCap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', ''),
                'country': info.get('country', ''),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', '')
            }
            
            # 保存到JSON文件
            with open(f"{self.symbol}_company_info.json", 'w', encoding='utf-8') as f:
                json.dump(basic_info, f, ensure_ascii=False, indent=2)
            
            print(f"✓ 公司信息已保存到 {self.symbol}_company_info.json")
            return basic_info
            
        except Exception as e:
            print(f"✗ 下载公司信息失败: {e}")
            return None
    
    def download_news_data(self, max_news=50):
        """下载新闻数据"""
        print(f"\n下载 {self.symbol} 新闻数据...")
        
        try:
            news = self.ticker.news
            
            if not news:
                print("  没有找到新闻数据")
                return None
            
            # 限制新闻数量
            news = news[:max_news]
            
            # 格式化新闻数据
            news_data = []
            for item in news:
                news_item = {
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', ''),
                    'link': item.get('link', ''),
                    'published': item.get('providerPublishTime', 0),
                    'summary': item.get('summary', '')
                }
                news_data.append(news_item)
            
            # 保存到CSV
            news_df = pd.DataFrame(news_data)
            news_df.to_csv(f"{self.symbol}_news.csv", index=False, encoding='utf-8')
            
            print(f"✓ 下载了 {len(news_data)} 条新闻")
            print(f"✓ 新闻数据已保存到 {self.symbol}_news.csv")
            
            return news_data
            
        except Exception as e:
            print(f"✗ 下载新闻数据失败: {e}")
            return None
    
    def create_data_summary(self, all_data):
        """创建数据摘要"""
        print(f"\n创建数据摘要...")
        
        summary = {
            'symbol': self.symbol,
            'download_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'timeframes': {}
        }
        
        for level, data in all_data.items():
            summary['timeframes'][level] = {
                'records': len(data),
                'start_date': data.index[0].strftime('%Y-%m-%d') if len(data) > 0 else None,
                'end_date': data.index[-1].strftime('%Y-%m-%d') if len(data) > 0 else None,
                'price_range': {
                    'min': float(data['Low'].min()) if len(data) > 0 else None,
                    'max': float(data['High'].max()) if len(data) > 0 else None
                },
                'avg_volume': float(data['Volume'].mean()) if len(data) > 0 else None
            }
        
        # 保存摘要
        with open(f"{self.symbol}_data_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 数据摘要已保存到 {self.symbol}_data_summary.json")
        return summary

def main():
    """主函数"""
    print("=" * 60)
    print("Yahoo Finance 缠论数据下载器")
    print("=" * 60)
    
    # 创建下载器
    downloader = YahooDataDownloader("AAPL")
    
    # 下载多时间周期数据
    all_data = downloader.download_multi_timeframe_data(
        start_date="2023-01-01",
        end_date="2024-12-31"
    )
    
    # 下载公司信息
    company_info = downloader.download_company_info()
    
    # 下载新闻数据
    news_data = downloader.download_news_data(max_news=30)
    
    # 创建数据摘要
    summary = downloader.create_data_summary(all_data)
    
    print("\n" + "=" * 60)
    print("下载完成！")
    print("=" * 60)
    print(f"已下载 {len(all_data)} 个时间周期的数据")
    print("文件列表:")
    
    # 列出生成的文件
    files = [f for f in os.listdir('.') if f.startswith('AAPL_') and f.endswith(('.csv', '.json'))]
    for file in sorted(files):
        print(f"  - {file}")

if __name__ == "__main__":
    main()
