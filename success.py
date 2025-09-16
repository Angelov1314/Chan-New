#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缠论技术分析Web应用
提供股票代码输入、时间段选择、缠论图表生成和准确性评估功能
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import io
import base64

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入缠论分析模块
from scripts.run_fixed import (
    load_ohlc, find_fractals, build_strokes, build_segments, 
    detect_zhongshu, detect_divergence, resolve_inclusion
)

# 导入用户管理和股票工具
from user_manager import user_manager
from stock_utils import stock_utils

app = Flask(__name__)
app.config['SECRET_KEY'] = 'chan_analysis_secret_key'

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

class ChanWebAnalyzer:
    def __init__(self):
        self.analysis_cache = {}
        
    def download_stock_data(self, symbol, start_date, end_date, timeframe="1d"):
        """下载股票数据"""
        try:
            ticker = yf.Ticker(symbol)
            data = yf.download(symbol, start=start_date, end=end_date, interval=timeframe)
            
            if data.empty:
                # 根据时间框架提供更具体的错误信息
                if timeframe in ['1m', '5m', '15m', '30m', '1h']:
                    max_days = 7 if timeframe == '1m' else 60
                    return None, f"无法获取{timeframe}数据。日内数据最多只能获取最近{max_days}天的历史数据，请调整日期范围。"
                else:
                    return None, "无法获取股票数据，请检查股票代码或日期范围"
            
            # 数据清理和格式化
            data_clean = data.copy()
            if isinstance(data_clean.columns, pd.MultiIndex):
                data_clean.columns = [col[0] if isinstance(col, tuple) else col for col in data_clean.columns]
            else:
                data_clean.columns = [col.replace(' ', '_') for col in data_clean.columns]
            
            # 添加时间相关列
            data_clean['Date'] = data_clean.index.strftime('%Y-%m-%d')
            data_clean['Time'] = data_clean.index.strftime('%H:%M:%S')
            data_clean['Weekday'] = data_clean.index.weekday
            data_clean['Level'] = timeframe
            
            # 计算技术指标
            data_clean = self._add_technical_indicators(data_clean)
            
            return data_clean, None
            
        except Exception as e:
            return None, f"数据下载失败: {str(e)}"
    
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
    
    def analyze_chan(self, df):
        """执行缠论分析"""
        try:
            # 检查数据格式
            if df.empty:
                return None, "数据为空"
            
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return None, f"缺少必要的列: {missing_columns}"
            
            # 数据预处理
            df_s = resolve_inclusion(df)
            df_use = df.copy()
            if "High_smooth" in df_s.columns:
                df_use["High"] = df_s["High_smooth"]
                df_use["Low"] = df_s["Low_smooth"]
            
            # 缠论分析
            frs = find_fractals(df_use)
            strokes = build_strokes(df_use, frs)
            segs = build_segments(strokes)
            zses = detect_zhongshu(strokes)
            divs = detect_divergence(df_use, strokes)
            
            return {
                'fractals': frs,
                'strokes': strokes,
                'segments': segs,
                'zhongshus': zses,
                'divergences': divs,
                'data': df_use
            }, None
            
        except Exception as e:
            return None, f"分析失败: {str(e)}"
    
    def generate_chart(self, df, analysis_results, start_date, end_date, timeframe="1d"):
        """Generate Chan Theory chart with English labels and detailed legend"""
        try:
            # Date handling - use actual search dates
            if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
                dates = df["Date"].dt.strftime("%Y-%m-%d").values
            elif hasattr(df.index, 'strftime'):  # 如果索引是datetime
                dates = df.index.strftime("%Y-%m-%d %H:%M").values
            else:
                # Generate dates based on search period
                from datetime import datetime, timedelta
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                
                # 根据时间框架调整频率
                if timeframe in ['1m', '5m', '15m', '30m', '1h']:
                    freq = timeframe
                else:
                    freq = 'D'
                
                date_range = pd.date_range(start=start_dt, end=end_dt, freq=freq)
                if timeframe in ['1m', '5m', '15m', '30m', '1h']:
                    dates = [d.strftime("%m-%d %H:%M") for d in date_range[:len(df)]]
                else:
                    dates = [d.strftime("%Y-%m-%d") for d in date_range[:len(df)]]
            
            O, H, L, C = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
            
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Candlesticks
            width = 0.6
            for i in range(len(df)):
                lower = min(O[i], C[i])
                height = abs(C[i] - O[i])
                color = 'red' if C[i] > O[i] else 'green'
                ax.add_patch(Rectangle((i - width/2, lower), width, max(height, 1e-8), 
                                     fill=True, alpha=0.6, color=color))
                ax.plot([i, i], [L[i], H[i]], color='black', linewidth=0.5)
            
            # Fractals
            for f in analysis_results['fractals']:
                if f.kind == "top":
                    ax.scatter(f.idx, f.price, marker="^", s=80, color='red', zorder=5)
                else:
                    ax.scatter(f.idx, f.price, marker="v", s=80, color='green', zorder=5)
            
            # Strokes
            for s in analysis_results['strokes']:
                color = 'blue' if s.direction == 'up' else 'orange'
                ax.plot([s.start_idx, s.end_idx], [s.start_price, s.end_price], 
                       linewidth=2, color=color, alpha=0.8)
            
            # Segments (bounds)
            for seg in analysis_results['segments']:
                ax.plot([seg.start_idx, seg.end_idx], [seg.low, seg.low], 
                       linestyle="--", linewidth=2, color='purple', alpha=0.7)
                ax.plot([seg.start_idx, seg.end_idx], [seg.high, seg.high], 
                       linestyle="--", linewidth=2, color='purple', alpha=0.7)
            
            # Zhongshu (range boxes)
            for z in analysis_results['zhongshus']:
                ax.add_patch(Rectangle((z.start_idx, z.lower),
                                     z.end_idx - z.start_idx,
                                     z.upper - z.lower,
                                     fill=True, alpha=0.2, color='yellow'))
            
            # Divergences
            for idx, kind in analysis_results['divergences']:
                y = C[idx]
                txt = "Bearish div" if kind == "bear_div" else "Bullish div"
                ax.text(idx, y, txt, fontsize=10, ha="center", va="bottom", 
                       color='red', weight='bold')
            
            # Axes labels and title
            ax.set_xlim(-1, len(df))
            ax.set_title("Chan Theory Analysis", fontsize=16, weight='bold')
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Price", fontsize=12)
            
            # X ticks - 修复分线数据刻度问题
            if len(df) > 100:  # 对于大量数据点（如分线数据）
                step = max(1, len(df)//20)  # 减少刻度数量
                ax.set_xticks(range(0, len(df), step))
                ax.set_xticklabels(dates[::step], rotation=45, ha="right")
            else:  # 对于少量数据点（如日线数据）
                step = max(1, len(df)//15)
                ax.set_xticks(range(0, len(df), step))
                ax.set_xticklabels(dates[::step], rotation=45, ha="right")
            
            # Legend (explicit handles)
            legend_handles = [
                Patch(facecolor='red', alpha=0.6, label='Candlestick up (close>open)'),
                Patch(facecolor='green', alpha=0.6, label='Candlestick down (close<open)'),
                Line2D([0], [0], marker='^', color='w', label='Top fractal (swing high)',
                       markerfacecolor='red', markersize=8),
                Line2D([0], [0], marker='v', color='w', label='Bottom fractal (swing low)',
                       markerfacecolor='green', markersize=8),
                Line2D([0], [0], color='blue', lw=2, label='Stroke (up)'),
                Line2D([0], [0], color='orange', lw=2, label='Stroke (down)'),
                Line2D([0], [0], color='purple', lw=2, ls='--', label='Segment bounds (high/low)'),
                Patch(facecolor='yellow', alpha=0.2, label='Zhongshu (overlap range)'),
                Line2D([0], [0], color='red', lw=0, label='Divergence label')
            ]
            ax.legend(handles=legend_handles, loc='upper left', fontsize=10, framealpha=0.9)
            
            plt.tight_layout()
            
            # 转换为base64字符串
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            raise Exception(f"图表生成失败: {str(e)}")
    
    def generate_evaluation_report(self, analysis_results, df):
        """生成评估报告（含交易报告）"""
        try:
            frs = analysis_results['fractals']
            strokes = analysis_results['strokes']
            segs = analysis_results['segments']
            zses = analysis_results['zhongshus']
            divs = analysis_results['divergences']
            
            # 计算统计数据
            current_price = df['Close'].iloc[-1]
            price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
            price_change_pct = (price_change / df['Close'].iloc[-2]) * 100
            
            # 分型统计
            top_fractals = len([f for f in frs if f.kind == 'top'])
            bottom_fractals = len([f for f in frs if f.kind == 'bottom'])
            
            # 笔统计
            up_strokes = len([s for s in strokes if s.direction == 'up'])
            down_strokes = len([s for s in strokes if s.direction == 'down'])
            
            # 线段统计
            up_segments = len([s for s in segs if s.direction == 'up'])
            down_segments = len([s for s in segs if s.direction == 'down'])
            
            # 背驰统计
            bear_divs = len([d for d in divs if d[1] == 'bear_div'])
            bull_divs = len([d for d in divs if d[1] == 'bull_div'])
            
            # 生成报告
            report = {
                'market_overview': {
                    'current_price': round(current_price, 2),
                    'price_change': round(price_change, 2),
                    'price_change_pct': round(price_change_pct, 2),
                    'total_days': len(df),
                    'date_range': f"{df['Date'].iloc[0]} 至 {df['Date'].iloc[-1]}"
                },
                'fractal_analysis': {
                    'total_fractals': len(frs),
                    'top_fractals': top_fractals,
                    'bottom_fractals': bottom_fractals,
                    'latest_fractal': {
                        'kind': frs[-1].kind if frs else 'N/A',
                        'price': round(frs[-1].price, 2) if frs else 'N/A',
                        'idx': frs[-1].idx if frs else 'N/A'
                    }
                },
                'stroke_analysis': {
                    'total_strokes': len(strokes),
                    'up_strokes': up_strokes,
                    'down_strokes': down_strokes,
                    'current_stroke': {
                        'direction': strokes[-1].direction if strokes else 'N/A',
                        'length': strokes[-1].end_idx - strokes[-1].start_idx if strokes else 'N/A',
                        'swing': round(strokes[-1].swing, 4) if strokes else 'N/A'
                    }
                },
                'segment_analysis': {
                    'total_segments': len(segs),
                    'up_segments': up_segments,
                    'down_segments': down_segments,
                    'current_segment': {
                        'direction': segs[-1].direction if segs else 'N/A',
                        'high': round(segs[-1].high, 2) if segs else 'N/A',
                        'low': round(segs[-1].low, 2) if segs else 'N/A'
                    }
                },
                'zhongshu_analysis': {
                    'total_zhongshus': len(zses),
                    'current_zhongshu': {
                        'upper': round(zses[-1].upper, 2) if zses else 'N/A',
                        'lower': round(zses[-1].lower, 2) if zses else 'N/A',
                        'width': round(zses[-1].upper - zses[-1].lower, 2) if zses else 'N/A'
                    }
                },
                'divergence_analysis': {
                    'total_divergences': len(divs),
                    'bear_divergences': bear_divs,
                    'bull_divergences': bull_divs,
                    'latest_divergence': {
                        'type': '顶背驰' if divs[-1][1] == 'bear_div' else '底背驰' if divs else 'N/A',
                        'price': round(df['Close'].iloc[divs[-1][0]], 2) if divs else 'N/A'
                    }
                },
                'trading_report': self._build_trading_report(df, segs, divs)
            }
            
            return report, None
            
        except Exception as e:
            return None, f"报告生成失败: {str(e)}"

    def _build_trading_report(self, df, segments, divergences):
        """基于背驰与线段生成交易报告：短/中/长线、仓位建议、风险与操作建议"""
        total_len = len(df)
        # 分类时间尺度的简单启发式：
        # 短线: 最近10%K线内的背驰；中线: 最近30%K线内；长线: 结合最后线段与中枢方向
        short_threshold = max(1, int(total_len * 0.1))
        mid_threshold = max(1, int(total_len * 0.3))

        now_idx = total_len - 1
        def categorize(idx: int) -> str:
            distance = max(0, now_idx - idx)
            if distance <= short_threshold:
                return 'short'
            if distance <= mid_threshold:
                return 'mid'
            return 'long'

        categorized = {
            'short': {'buy': [], 'sell': []},
            'mid': {'buy': [], 'sell': []},
            'long': {'buy': [], 'sell': []}
        }

        for idx, kind in divergences:
            price = float(df['Close'].iloc[idx]) if 0 <= idx < len(df) else None
            if price is None:
                continue
            bucket = categorize(idx)
            if kind == 'bull_div':
                categorized[bucket]['buy'].append({
                    'position': round(price, 2),
                    'reason': '底背驰（Bullish divergence）',
                    'stop': round(price * 0.98, 2),
                    'target': round(price * 1.05, 2)
                })
            elif kind == 'bear_div':
                categorized[bucket]['sell'].append({
                    'position': round(price, 2),
                    'reason': '顶背驰（Bearish divergence）',
                    'stop': round(price * 1.02, 2),
                    'target': round(price * 0.95, 2)
                })

        # 趋势与仓位建议
        trend_hint = '观望'
        pos_pct = {'short': 20, 'mid': 30, 'long': 40}  # 默认总仓位建议拆分
        if segments:
            last_seg = segments[-1]
            if getattr(last_seg, 'direction', '') == 'up':
                trend_hint = '顺势做多（Prefer long bias）'
                pos_pct = {'short': 30, 'mid': 40, 'long': 50}
            else:
                trend_hint = '逢高减仓（Reduce on strength）'
                pos_pct = {'short': 10, 'mid': 20, 'long': 30}

        # 风险等级：背驰数量、近端波动
        recent_divs = [d for d in divergences if now_idx - d[0] <= mid_threshold]
        risk_level = '低'
        if len(recent_divs) >= 3:
            risk_level = '高'
            pos_scale = 0.4
        elif len(recent_divs) == 2:
            risk_level = '中'
            pos_scale = 0.7
        else:
            pos_scale = 1.0

        # 依风险调整仓位百分比（上限不超过100，总体建议为相对权重）
        for k in pos_pct:
            pos_pct[k] = int(min(100, round(pos_pct[k] * pos_scale)))

        return {
            'horizon': {
                'short': {
                    'buy_signals': categorized['short']['buy'][:3],
                    'sell_signals': categorized['short']['sell'][:3],
                    'position_pct': pos_pct['short']
                },
                'mid': {
                    'buy_signals': categorized['mid']['buy'][:3],
                    'sell_signals': categorized['mid']['sell'][:3],
                    'position_pct': pos_pct['mid']
                },
                'long': {
                    'buy_signals': categorized['long']['buy'][:3],
                    'sell_signals': categorized['long']['sell'][:3],
                    'position_pct': pos_pct['long']
                }
            },
            'risk_level': risk_level,
            'advice': [
                trend_hint,
                '关注中枢突破方向（Zhongshu breakout），顺势操作',
                '分批建仓与止损管理（Position scaling & stops）',
                '背驰附近谨慎加减仓（Manage around divergences）'
            ]
        }
    
    def validate_accuracy(self, symbol, start_date, end_date, validation_date):
        """验证分析准确性"""
        try:
            # 下载历史数据（到验证日期）
            hist_data, error = self.download_stock_data(symbol, start_date, validation_date, '1d')
            if error:
                return None, error
            
            # 下载完整数据（到结束日期）
            full_data, error = self.download_stock_data(symbol, start_date, end_date, '1d')
            if error:
                return None, error
                
            # 对历史数据进行分析
            hist_analysis, error = self.analyze_chan(hist_data)
            if error:
                return None, error
            
            # 对完整数据进行分析
            full_analysis, error = self.analyze_chan(full_data)
            if error:
                return None, error
            
            # 计算准确性指标
            split_point = len(hist_data)
            
            # 分型准确性
            hist_fractals = [f for f in hist_analysis['fractals'] if f.idx < split_point]
            full_fractals = [f for f in full_analysis['fractals'] if f.idx < split_point]
            fractal_accuracy = len(set(f.idx for f in hist_fractals).intersection(
                set(f.idx for f in full_fractals))) / max(len(hist_fractals), len(full_fractals)) if max(len(hist_fractals), len(full_fractals)) > 0 else 0
            
            # 笔准确性
            hist_strokes = [s for s in hist_analysis['strokes'] if s.end_idx < split_point]
            full_strokes = [s for s in full_analysis['strokes'] if s.end_idx < split_point]
            stroke_accuracy = len(hist_strokes) / len(full_strokes) if len(full_strokes) > 0 else 0
            
            # 价格预测准确性
            hist_price = hist_data['Close'].iloc[-1]
            actual_price = full_data['Close'].iloc[split_point-1] if split_point < len(full_data) else full_data['Close'].iloc[-1]
            price_accuracy = 1 - abs(hist_price - actual_price) / hist_price if hist_price > 0 else 0
            
            # 趋势预测准确性
            hist_trend = self._calculate_trend(hist_data['Close'].tail(20))
            future_data = full_data.iloc[split_point:]
            if len(future_data) > 0:
                future_trend = self._calculate_trend(future_data['Close'])
                trend_accuracy = 1 if hist_trend * future_trend > 0 else 0
            else:
                trend_accuracy = 0
            
            accuracy_report = {
                'fractal_accuracy': round(fractal_accuracy * 100, 1),
                'stroke_accuracy': round(stroke_accuracy * 100, 1),
                'price_accuracy': round(price_accuracy * 100, 1),
                'trend_accuracy': round(trend_accuracy * 100, 1),
                'overall_accuracy': round((fractal_accuracy + stroke_accuracy + price_accuracy + trend_accuracy) / 4 * 100, 1),
                'validation_info': {
                    'historical_period': f"{start_date} 至 {validation_date}",
                    'validation_period': f"{validation_date} 至 {end_date}",
                    'historical_price': round(hist_price, 2),
                    'actual_price': round(actual_price, 2),
                    'price_change': round(actual_price - hist_price, 2)
                }
            }
            
            # 生成对比图
            validation_chart = self.generate_validation_chart(hist_data, hist_analysis, full_data, split_point)
            
            return { 'metrics': accuracy_report, 'chart': validation_chart }, None
            
        except Exception as e:
            return None, f"准确性验证失败: {str(e)}"
    
    def _calculate_trend(self, prices):
        """计算价格趋势"""
        if len(prices) < 2:
            return 0
        
        x = np.arange(len(prices))
        y = prices.values
        
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        else:
            slope = 0
        
        return slope

    def generate_validation_chart(self, hist_df, hist_analysis, full_df, split_idx):
        """生成训练期与未来期对比图，并标注预测方向"""
        try:
            fig, ax = plt.subplots(figsize=(16, 8))
            
            # 时间轴
            dates = None
            if 'Date' in full_df.columns and pd.api.types.is_datetime64_any_dtype(full_df['Date']):
                dates = full_df['Date'].dt.strftime('%Y-%m-%d').values
            else:
                dates = [f'Day {i}' for i in range(len(full_df))]
            
            # 价格曲线
            ax.plot(range(len(full_df)), full_df['Close'].values, color='#6c757d', lw=1.5, ls='--', label='Future close (actual)')
            ax.plot(range(split_idx), hist_df['Close'].values, color='#007bff', lw=2.0, label='Training close (with Chan)')
            
            # 分割线
            ax.axvline(split_idx-0.5, color='black', lw=1, ls=':', alpha=0.6)
            ax.text(split_idx, ax.get_ylim()[1], 'Validation start', va='top', ha='left', fontsize=9)
            
            # 训练期简单预测线（基于线性斜率）
            tail = hist_df['Close'].tail(20)
            slope = self._calculate_trend(tail)
            start_y = float(hist_df['Close'].iloc[-1])
            steps = max(1, len(full_df) - split_idx)
            proj_x = np.arange(steps)
            proj_y = start_y + slope * proj_x
            ax.plot(range(split_idx, len(full_df)), proj_y, color='#28a745', lw=2, label='Predicted trend (linear)')
            
            # 训练期叠加关键Chan要素（仅笔）
            for s in hist_analysis['strokes']:
                if s.end_idx < split_idx:
                    color = '#0dcaf0' if s.direction == 'up' else '#fd7e14'
                    ax.plot([s.start_idx, s.end_idx], [s.start_price, s.end_price], color=color, lw=1.5, alpha=0.8)
            
            # X 轴刻度
            step = max(1, len(full_df)//15)
            ax.set_xticks(range(0, len(full_df), step))
            ax.set_xticklabels(dates[::step], rotation=45, ha='right')
            
            ax.set_title('Training vs Future: Predicted vs Actual', fontsize=16, weight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.legend(loc='upper left')
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode()
            plt.close()
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            print(f"图表生成失败: {str(e)}")
            return None

# 创建分析器实例
analyzer = ChanWebAnalyzer()

# 用户认证装饰器
def require_auth(f):
    """需要认证的装饰器"""
    def decorated_function(*args, **kwargs):
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '需要登录'}), 401
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

# 用户认证相关路由
@app.route('/api/auth/register', methods=['POST'])
def register():
    """用户注册"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        if not all([username, email, password]):
            return jsonify({'error': '请填写完整信息'}), 400
        
        success, message = user_manager.register_user(username, email, password)
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'error': message}), 400
            
    except Exception as e:
        return jsonify({'error': f'注册失败: {str(e)}'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """用户登录"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'error': '请填写用户名和密码'}), 400
        
        user_id, message = user_manager.authenticate_user(username, password)
        if user_id:
            session['user_id'] = user_id
            user_info = user_manager.get_user_info(user_id)
            return jsonify({
                'success': True, 
                'message': message,
                'user': user_info
            })
        else:
            return jsonify({'error': message}), 401
            
    except Exception as e:
        return jsonify({'error': f'登录失败: {str(e)}'}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """用户登出"""
    session.pop('user_id', None)
    return jsonify({'success': True, 'message': '已登出'})

@app.route('/api/auth/me', methods=['GET'])
@require_auth
def get_current_user():
    """获取当前用户信息"""
    user_id = session.get('user_id')
    user_info = user_manager.get_user_info(user_id)
    if user_info:
        return jsonify({'success': True, 'user': user_info})
    else:
        return jsonify({'error': '用户不存在'}), 404

# 股票工具相关路由
@app.route('/api/stock/format', methods=['POST'])
def format_stock_symbol():
    """格式化股票代码"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').strip()
        
        if not symbol:
            return jsonify({'error': '股票代码不能为空'}), 400
        
        # 验证和格式化股票代码
        is_valid, message = stock_utils.validate_symbol(symbol)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        stock_info = stock_utils.get_stock_info(symbol)
        return jsonify({'success': True, 'stock_info': stock_info})
        
    except Exception as e:
        return jsonify({'error': f'格式化失败: {str(e)}'}), 500

@app.route('/api/stock/search', methods=['POST'])
def search_stocks():
    """搜索股票"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': '搜索关键词不能为空'}), 400
        
        results = stock_utils.search_stocks(query)
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'error': f'搜索失败: {str(e)}'}), 500

# 关注列表相关路由
@app.route('/api/watchlist', methods=['GET'])
@require_auth
def get_watchlist():
    """获取关注列表"""
    try:
        user_id = session.get('user_id')
        watchlist = user_manager.get_watchlist(user_id)
        return jsonify({'success': True, 'watchlist': watchlist})
    except Exception as e:
        return jsonify({'error': f'获取关注列表失败: {str(e)}'}), 500

@app.route('/api/watchlist', methods=['POST'])
@require_auth
def add_to_watchlist():
    """添加到关注列表"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').strip()
        display_name = data.get('display_name', '').strip()
        
        if not symbol:
            return jsonify({'error': '股票代码不能为空'}), 400
        
        # 格式化股票代码
        formatted_symbol = stock_utils.format_chinese_stock(symbol)
        if not display_name:
            display_name = formatted_symbol
        
        user_id = session.get('user_id')
        success, message = user_manager.add_to_watchlist(user_id, formatted_symbol, display_name)
        
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'error': message}), 400
            
    except Exception as e:
        return jsonify({'error': f'添加失败: {str(e)}'}), 500

@app.route('/api/watchlist/<symbol>', methods=['DELETE'])
@require_auth
def remove_from_watchlist(symbol):
    """从关注列表移除"""
    try:
        user_id = session.get('user_id')
        success, message = user_manager.remove_from_watchlist(user_id, symbol)
        
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'error': message}), 400
            
    except Exception as e:
        return jsonify({'error': f'移除失败: {str(e)}'}), 500

# 研究历史相关路由
@app.route('/api/history', methods=['GET'])
@require_auth
def get_research_history():
    """获取研究历史"""
    try:
        user_id = session.get('user_id')
        limit = request.args.get('limit', 50, type=int)
        history = user_manager.get_research_history(user_id, limit)
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        return jsonify({'error': f'获取历史记录失败: {str(e)}'}), 500

@app.route('/api/history/<history_id>', methods=['DELETE'])
@require_auth
def delete_research_history(history_id):
    """删除研究历史"""
    try:
        user_id = session.get('user_id')
        success, message = user_manager.delete_research_history(user_id, history_id)
        
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'error': message}), 400
            
    except Exception as e:
        return jsonify({'error': f'删除失败: {str(e)}'}), 500

@app.route('/handbook')
def handbook():
    """新手手册页面"""
    return render_template('handbook.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """执行缠论分析"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').strip()
        start_date = data.get('start_date', '')
        end_date = data.get('end_date', '')
        
        if not symbol or not start_date or not end_date:
            return jsonify({'error': '请填写完整的股票代码和日期范围'}), 400
        
        # 格式化股票代码（自动添加中文股票后缀）
        formatted_symbol = stock_utils.format_chinese_stock(symbol)
        
        # 获取时间框架参数
        timeframe = data.get('timeframe', '1d')
        
        # 下载数据
        df, error = analyzer.download_stock_data(formatted_symbol, start_date, end_date, timeframe)
        if error:
            return jsonify({'error': error}), 400
        
        # 执行分析
        try:
            analysis_results, error = analyzer.analyze_chan(df)
            if error:
                return jsonify({'error': f'分析失败: {error}'}), 400
        except Exception as e:
            return jsonify({'error': f'分析过程中出现错误: {str(e)}'}), 400
        
        # 生成图表
        try:
            chart_data = analyzer.generate_chart(df, analysis_results, start_date, end_date, timeframe)
        except Exception as e:
            return jsonify({'error': f'图表生成失败: {str(e)}'}), 400
        
        # 生成评估报告（含交易报告）
        report, error = analyzer.generate_evaluation_report(analysis_results, df)
        if error:
            return jsonify({'error': error}), 400
        
        # 如果用户已登录，保存研究历史
        user_id = session.get('user_id')
        if user_id:
            try:
                user_manager.save_research_history(
                    user_id, formatted_symbol, start_date, end_date, 
                    timeframe, report, chart_data
                )
            except Exception as e:
                print(f"保存研究历史失败: {str(e)}")
        
        return jsonify({
            'success': True,
            'chart': chart_data,
            'report': report,
            'symbol': formatted_symbol,
            'original_symbol': symbol,
            'period': f"{start_date} 至 {end_date}",
            'stock_info': stock_utils.get_stock_info(symbol)
        })
        
    except Exception as e:
        return jsonify({'error': f'分析失败: {str(e)}'}), 500

@app.route('/validate', methods=['POST'])
def validate():
    """验证分析准确性"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        start_date = data.get('start_date', '')
        end_date = data.get('end_date', '')
        validation_date = data.get('validation_date', '')
        
        if not all([symbol, start_date, end_date, validation_date]):
            return jsonify({'error': '请填写完整的参数'}), 400
        
        # 验证准确性
        accuracy_report, error = analyzer.validate_accuracy(symbol, start_date, end_date, validation_date)
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify({
            'success': True,
            'accuracy': accuracy_report
        })
        
    except Exception as e:
        return jsonify({'error': f'验证失败: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for API status"""
    return jsonify({
        'status': 'healthy',
        'message': 'Chan Theory API is running',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)
