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
from flask import Flask, render_template, request, jsonify, send_file
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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'chan_analysis_secret_key'

# 简单的内存存储（生产环境建议使用数据库）
watchlist_storage = {
    'stocks': [],  # 存储股票代码列表
    'last_updated': None
}

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

    def _calculate_investment_index(self, df, segments, divergences, fractals, strokes):
        """计算基于风险和预测确定性的投资指数"""
        total_len = len(df)
        now_idx = total_len - 1
        
        # 1. 基础风险评分 (0-100)
        risk_score = self._calculate_risk_score(df, divergences, segments)
        
        # 2. 预测确定性评分 (0-100)
        certainty_score = self._calculate_certainty_score(df, fractals, strokes, segments, divergences)
        
        # 3. 市场趋势强度 (0-100)
        trend_strength = self._calculate_trend_strength(df, segments, strokes)
        
        # 4. 信号质量评分 (0-100)
        signal_quality = self._calculate_signal_quality(divergences, fractals, now_idx)
        
        # 5. 综合投资指数计算
        # 权重分配：风险30%，确定性25%，趋势25%，信号质量20%
        investment_index = (
            (100 - risk_score) * 0.30 +  # 风险越低，指数越高
            certainty_score * 0.25 +     # 确定性越高，指数越高
            trend_strength * 0.25 +      # 趋势越强，指数越高
            signal_quality * 0.20        # 信号质量越高，指数越高
        )
        
        # 6. 按时间框架调整（更均衡，避免长线长期最高）
        short_threshold = max(1, int(total_len * 0.1))
        mid_threshold = max(1, int(total_len * 0.3))

        # 统计不同时间窗口内的信号
        recent_divs = [d for d in divergences if now_idx - d[0] <= short_threshold]
        mid_divs = [d for d in divergences if short_threshold < now_idx - d[0] <= mid_threshold]
        long_divs = [d for d in divergences if now_idx - d[0] > mid_threshold]

        # 波动率与震荡度（线段方向切换频率）
        price_volatility = float(df['Close'].pct_change().std() or 0)  # 0-? 小数
        if len(segments) > 1:
            segment_changes = sum(1 for i in range(1, len(segments)) if segments[i].direction != segments[i-1].direction)
            choppiness = segment_changes / max(1, len(segments) - 1)  # 0-1
        else:
            choppiness = 0.5

        # 当前线段强度与时长
        last_seg_len = (segments[-1].end_idx - segments[-1].start_idx) if segments else 0
        last_seg_len_norm = min(1.0, last_seg_len / max(1, int(total_len * 0.2)))

        # 基于环境对不同周期的偏好系数（1 为中性）
        short_bias = 1.0
        mid_bias = 1.0
        long_bias = 1.0

        # 高波动与高震荡 → 倾向短线
        short_bias += min(0.3, price_volatility * 2.0) + min(0.3, choppiness * 0.6)
        # 趋势强、线段长 → 倾向长线
        long_bias += min(0.4, trend_strength / 250.0) + min(0.3, last_seg_len_norm * 0.6)
        # 中线随市场中性度微调
        mid_bias += 0.1 * (len(mid_divs) > 0) - 0.05 * abs(len(recent_divs) - len(long_divs))

        # 信号驱动的偏好
        short_bias += min(0.3, len(recent_divs) * 0.12)
        mid_bias += min(0.25, len(mid_divs) * 0.12)
        long_bias += min(0.2, len(long_divs) * 0.10)

        # 风险越高，抑制长线并略微提升短线
        long_bias *= (1.0 - min(0.35, risk_score / 300.0))
        short_bias *= (1.0 + min(0.25, risk_score / 400.0))

        # 将偏好应用到基础指数
        short_index = investment_index * short_bias
        mid_index = investment_index * mid_bias
        long_index = investment_index * long_bias

        # 7. 归一化与相对排序增强：确保三者有区分度
        def normalize_common(idx):
            # 统一范围，去除对长线的天生上限优势
            min_val, max_val = 15, 85
            normalized = min_val + (idx / 100) * (max_val - min_val)
            return max(0, min(100, int(normalized)))

        short_n = normalize_common(short_index)
        mid_n = normalize_common(mid_index)
        long_n = normalize_common(long_index)

        # 拉开差距：将最高的指数再+5，最低的-5（边界保护）
        values = [('short', short_n), ('mid', mid_n), ('long', long_n)]
        values_sorted = sorted(values, key=lambda x: x[1], reverse=True)
        if values_sorted[0][1] - values_sorted[-1][1] < 8:
            # 若差距过小，轻微强调当前最优周期
            values_sorted[0] = (values_sorted[0][0], min(95, values_sorted[0][1] + 5))
            values_sorted[-1] = (values_sorted[-1][0], max(5, values_sorted[-1][1] - 5))
        short_n = next(v for k, v in values_sorted if k == 'short')
        mid_n = next(v for k, v in values_sorted if k == 'mid')
        long_n = next(v for k, v in values_sorted if k == 'long')

        return {
            'short': short_n,
            'mid': mid_n,
            'long': long_n,
            'components': {
                'risk_score': round(risk_score, 1),
                'certainty_score': round(certainty_score, 1),
                'trend_strength': round(trend_strength, 1),
                'signal_quality': round(signal_quality, 1),
                'base_index': round(investment_index, 1)
            }
        }
    
    def _calculate_risk_score(self, df, divergences, segments):
        """计算风险评分 (0-100，越高风险越大)"""
        total_len = len(df)
        now_idx = total_len - 1
        
        # 1. 价格波动性风险 (30%)
        price_volatility = df['Close'].pct_change().std() * 100
        volatility_risk = min(100, price_volatility * 10)  # 标准化到0-100
        
        # 2. 背驰密度风险 (25%)
        recent_divs = [d for d in divergences if now_idx - d[0] <= int(total_len * 0.2)]
        divergence_risk = min(100, len(recent_divs) * 20)
        
        # 3. 线段变化频率风险 (25%)
        if len(segments) > 1:
            segment_changes = sum(1 for i in range(1, len(segments)) 
                                if segments[i].direction != segments[i-1].direction)
            change_frequency = segment_changes / max(1, len(segments) - 1)
            segment_risk = min(100, change_frequency * 100)
        else:
            segment_risk = 50  # 中性风险
        
        # 4. 成交量异常风险 (20%)
        if 'Volume' in df.columns:
            volume_ma = df['Volume'].rolling(20).mean()
            recent_volume = df['Volume'].tail(5).mean()
            avg_volume = volume_ma.tail(5).mean()
            volume_ratio = recent_volume / max(avg_volume, 1)
            volume_risk = min(100, abs(volume_ratio - 1) * 50)
        else:
            volume_risk = 50
        
        # 综合风险评分
        risk_score = (
            volatility_risk * 0.30 +
            divergence_risk * 0.25 +
            segment_risk * 0.25 +
            volume_risk * 0.20
        )
        
        return min(100, max(0, risk_score))
    
    def _calculate_certainty_score(self, df, fractals, strokes, segments, divergences):
        """计算预测确定性评分 (0-100)"""
        total_len = len(df)
        
        # 1. 分型质量 (25%)
        if len(fractals) > 0:
            # 分型密度和分布均匀性
            fractal_density = len(fractals) / total_len * 100
            fractal_quality = min(100, fractal_density * 2)
        else:
            fractal_quality = 0
        
        # 2. 笔结构稳定性 (25%)
        if len(strokes) > 1:
            # 笔的长度一致性
            stroke_lengths = [s.end_idx - s.start_idx for s in strokes]
            if stroke_lengths:
                length_std = np.std(stroke_lengths)
                length_mean = np.mean(stroke_lengths)
                consistency = max(0, 100 - (length_std / max(length_mean, 1)) * 100)
            else:
                consistency = 50
        else:
            consistency = 50
        
        # 3. 线段趋势一致性 (25%)
        if len(segments) > 1:
            # 检查线段方向的一致性
            up_segments = sum(1 for s in segments if s.direction == 'up')
            down_segments = sum(1 for s in segments if s.direction == 'down')
            total_segments = len(segments)
            direction_consistency = 100 - abs(up_segments - down_segments) / max(total_segments, 1) * 100
        else:
            direction_consistency = 50
        
        # 4. 背驰信号质量 (25%)
        if len(divergences) > 0:
            # 背驰信号的分布和强度
            recent_divs = [d for d in divergences if total_len - 1 - d[0] <= int(total_len * 0.3)]
            signal_strength = min(100, len(recent_divs) * 15)
        else:
            signal_strength = 30  # 无信号时给予基础分
        
        # 综合确定性评分
        certainty_score = (
            fractal_quality * 0.25 +
            consistency * 0.25 +
            direction_consistency * 0.25 +
            signal_strength * 0.25
        )
        
        return min(100, max(0, certainty_score))
    
    def _calculate_trend_strength(self, df, segments, strokes):
        """计算趋势强度 (0-100)"""
        if not segments:
            return 50  # 中性强度
        
        # 1. 当前线段强度 (40%)
        last_segment = segments[-1]
        segment_length = last_segment.end_idx - last_segment.start_idx
        segment_swing = abs(last_segment.high - last_segment.low)
        current_strength = min(100, (segment_swing / df['Close'].iloc[-1]) * 1000)
        
        # 2. 笔的连续性 (30%)
        if len(strokes) > 1:
            recent_strokes = strokes[-5:] if len(strokes) >= 5 else strokes
            same_direction = sum(1 for i in range(1, len(recent_strokes)) 
                               if recent_strokes[i].direction == recent_strokes[i-1].direction)
            continuity = (same_direction / max(len(recent_strokes) - 1, 1)) * 100
        else:
            continuity = 50
        
        # 3. 价格动量 (30%)
        if len(df) >= 20:
            recent_prices = df['Close'].tail(20)
            price_momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0] * 100
            momentum_strength = min(100, abs(price_momentum) * 2)
        else:
            momentum_strength = 50
        
        # 综合趋势强度
        trend_strength = (
            current_strength * 0.40 +
            continuity * 0.30 +
            momentum_strength * 0.30
        )
        
        return min(100, max(0, trend_strength))
    
    def _calculate_signal_quality(self, divergences, fractals, now_idx):
        """计算信号质量 (0-100)"""
        if not divergences:
            return 30  # 无信号时给予基础分
        
        # 1. 信号密度 (30%)
        recent_divs = [d for d in divergences if now_idx - d[0] <= 20]  # 最近20个周期
        signal_density = min(100, len(recent_divs) * 20)
        
        # 2. 信号分布均匀性 (25%)
        if len(divergences) > 1:
            div_indices = [d[0] for d in divergences]
            intervals = [div_indices[i+1] - div_indices[i] for i in range(len(div_indices)-1)]
            if intervals:
                interval_std = np.std(intervals)
                interval_mean = np.mean(intervals)
                distribution_quality = max(0, 100 - (interval_std / max(interval_mean, 1)) * 50)
            else:
                distribution_quality = 50
        else:
            distribution_quality = 50
        
        # 3. 分型与背驰的匹配度 (25%)
        if fractals and divergences:
            fractal_indices = [f.idx for f in fractals]
            div_indices = [d[0] for d in divergences]
            matches = sum(1 for div_idx in div_indices 
                         if any(abs(div_idx - f_idx) <= 3 for f_idx in fractal_indices))
            match_ratio = matches / max(len(divergences), 1)
            match_quality = match_ratio * 100
        else:
            match_quality = 50
        
        # 4. 信号强度 (20%)
        signal_strength = min(100, len(divergences) * 10)
        
        # 综合信号质量
        signal_quality = (
            signal_density * 0.30 +
            distribution_quality * 0.25 +
            match_quality * 0.25 +
            signal_strength * 0.20
        )
        
        return min(100, max(0, signal_quality))

    def _build_trading_report(self, df, segments, divergences):
        """基于背驰与线段生成交易报告：短/中/长线、投资指数、风险与操作建议"""
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

        # 获取分析结果用于计算投资指数
        analysis_results, _ = self.analyze_chan(df)
        if analysis_results:
            fractals = analysis_results['fractals']
            strokes = analysis_results['strokes']
        else:
            fractals = []
            strokes = []

        # 计算智能投资指数
        investment_indices = self._calculate_investment_index(df, segments, divergences, fractals, strokes)

        # 趋势与操作建议
        trend_hint = '观望'
        if segments:
            last_seg = segments[-1]
            if getattr(last_seg, 'direction', '') == 'up':
                trend_hint = '顺势做多（Prefer long bias）'
            else:
                trend_hint = '逢高减仓（Reduce on strength）'

        # 风险等级：基于投资指数组件
        risk_score = investment_indices['components']['risk_score']
        if risk_score >= 70:
            risk_level = '高'
        elif risk_score >= 40:
            risk_level = '中'
        else:
            risk_level = '低'

        return {
            'horizon': {
                'short': {
                    'buy_signals': categorized['short']['buy'][:3],
                    'sell_signals': categorized['short']['sell'][:3],
                    'position_pct': investment_indices['short']
                },
                'mid': {
                    'buy_signals': categorized['mid']['buy'][:3],
                    'sell_signals': categorized['mid']['sell'][:3],
                    'position_pct': investment_indices['mid']
                },
                'long': {
                    'buy_signals': categorized['long']['buy'][:3],
                    'sell_signals': categorized['long']['sell'][:3],
                    'position_pct': investment_indices['long']
                }
            },
            'risk_level': risk_level,
            'investment_analysis': investment_indices['components'],
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
            from datetime import datetime, timedelta
            
            # 检查是否为未来预测模式（验证日期等于或接近结束日期）
            validation_dt = datetime.strptime(validation_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            is_future_prediction = (end_dt - validation_dt).days <= 1
            
            # 下载历史数据（到验证日期）
            hist_data, error = self.download_stock_data(symbol, start_date, validation_date, '1d')
            if error:
                return None, error
            
            # 对历史数据进行分析
            hist_analysis, error = self.analyze_chan(hist_data)
            if error:
                return None, error
            
            if is_future_prediction:
                # 未来预测模式：基于历史数据预测未来
                return self._generate_future_prediction(hist_data, hist_analysis, start_date, validation_date, end_date)
            else:
                # 历史验证模式：下载完整数据进行对比
                full_data, error = self.download_stock_data(symbol, start_date, end_date, '1d')
                if error:
                    return None, error
                    
                # 对完整数据进行分析
                full_analysis, error = self.analyze_chan(full_data)
                if error:
                    return None, error
                
                # 计算准确性指标
                split_point = len(hist_data)
                
                # 使用优化的准确性计算方法
                accuracy_metrics = self._calculate_enhanced_accuracy(
                    hist_data, hist_analysis, full_data, full_analysis, split_point
                )
                
                # 生成对比图
                validation_chart = self.generate_validation_chart(hist_data, hist_analysis, full_data, split_point)
                
                return { 'metrics': accuracy_metrics, 'chart': validation_chart }, None
            
        except Exception as e:
            return None, f"准确性验证失败: {str(e)}"
    
    def _generate_future_prediction(self, hist_data, hist_analysis, start_date, validation_date, end_date):
        """生成未来预测报告"""
        try:
            # 基于历史数据生成预测
            predictions = self._generate_predictions(hist_data, hist_analysis, validation_date, end_date)
            
            # 计算预测质量评分
            prediction_quality = self._calculate_prediction_quality(hist_data, hist_analysis)
            
            # 生成预测图表
            prediction_chart = self.generate_prediction_chart(hist_data, hist_analysis, predictions)
            
            return {
                'metrics': {
                    'prediction_quality': round(prediction_quality, 1),
                    'confidence_level': self._calculate_confidence_level(hist_analysis),
                    'risk_assessment': self._assess_prediction_risk(hist_data, hist_analysis),
                    'prediction_type': 'future_prediction',
                    'validation_info': {
                        'historical_period': f"{start_date} 至 {validation_date}",
                        'prediction_period': f"{validation_date} 至 {end_date}",
                        'current_price': round(hist_data['Close'].iloc[-1], 2),
                        'prediction_days': (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(validation_date, '%Y-%m-%d')).days
                    }
                },
                'chart': prediction_chart,
                'predictions': predictions
            }, None
            
        except Exception as e:
            return None, f"未来预测生成失败: {str(e)}"

    def _generate_predictions(self, hist_data, hist_analysis, validation_date, end_date):
        """使用简单线性外推生成未来预测值"""
        try:
            start_y = float(hist_data['Close'].iloc[-1])
            tail = hist_data['Close'].tail(20)
            slope = float(self._calculate_trend(tail))

            v_dt = datetime.strptime(validation_date, '%Y-%m-%d')
            e_dt = datetime.strptime(end_date, '%Y-%m-%d')
            steps = max(1, (e_dt - v_dt).days)

            preds = []
            for i in range(steps):
                preds.append(max(0.0, start_y + slope * i))
            return preds
        except Exception:
            # 兜底返回1步预测
            return [float(hist_data['Close'].iloc[-1])]

    def generate_prediction_chart(self, hist_df, hist_analysis, predictions):
        """仅展示训练期与未来预测曲线（无实际未来数据）"""
        try:
            fig, ax = plt.subplots(figsize=(16, 8))

            # 时间轴（训练期）
            if 'Date' in hist_df.columns:
                hist_dates = hist_df['Date'].tolist()
            else:
                hist_dates = [f'Day {i}' for i in range(len(hist_df))]

            split_idx = len(hist_df)

            # 绘制训练期收盘
            ax.plot(range(split_idx), hist_df['Close'].values, color='#007bff', lw=2.0, label='Training close (with Chan)')

            # 绘制预测曲线
            proj_x = np.arange(len(predictions))
            proj_y = np.array(predictions, dtype=float)
            ax.plot(range(split_idx, split_idx + len(predictions)), proj_y, color='#28a745', lw=2, label='Predicted trend (linear)')

            # 分割线
            ax.axvline(split_idx-0.5, color='black', lw=1, ls=':', alpha=0.6)
            ax.text(split_idx, ax.get_ylim()[1], 'Prediction start', va='top', ha='left', fontsize=9)

            # 生成未来日期标签
            try:
                if len(hist_dates) == 0:
                    hist_dates = []
                need = split_idx + len(predictions) - len(hist_dates)
                if need > 0:
                    # 使用日频扩展
                    if 'Date' in hist_df.columns:
                        last_date = pd.to_datetime(hist_df['Date'].iloc[-1])
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=need, freq='D').strftime('%Y-%m-%d').tolist()
                    else:
                        start_i = len(hist_dates)
                        future_dates = [f'Day {i}' for i in range(start_i, start_i + need)]
                    dates = hist_dates + future_dates
                else:
                    dates = hist_dates
            except Exception:
                dates = hist_dates

            # X轴刻度
            total_len = split_idx + len(predictions)
            step = max(1, total_len // 15)
            ax.set_xticks(range(0, total_len, step))
            safe_labels = dates[:total_len] if dates else []
            if len(safe_labels) >= total_len:
                ax.set_xticklabels(safe_labels[::step], rotation=45, ha='right')
            else:
                ax.set_xticklabels([str(i) for i in range(0, total_len, step)], rotation=45, ha='right')

            ax.set_title('Training vs Future: Predicted (no actual future data)', fontsize=16, weight='bold')
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
        except Exception:
            return None

    def _calculate_prediction_quality(self, hist_data, hist_analysis):
        """简单预测质量评分：基于近期趋势稳定性与信号质量"""
        try:
            tail = hist_data['Close'].tail(20)
            slope = abs(float(self._calculate_trend(tail)))
            trend_score = max(0.0, min(100.0, slope * 50.0))

            # 信号质量：使用分型密度与笔数量
            frs = hist_analysis.get('fractals', [])
            strokes = hist_analysis.get('strokes', [])
            density = (len(frs) / max(1, len(hist_data))) * 100
            signal_score = max(0.0, min(100.0, density + min(40, len(strokes) * 2)))

            return (trend_score * 0.5 + signal_score * 0.5)
        except Exception:
            return 50.0

    def _calculate_confidence_level(self, hist_analysis):
        """返回置信度标签"""
        try:
            strokes = hist_analysis.get('strokes', [])
            if len(strokes) >= 6:
                return 'high'
            if len(strokes) >= 3:
                return 'medium'
            return 'low'
        except Exception:
            return 'low'

    def _assess_prediction_risk(self, hist_data, hist_analysis):
        """返回风险等级标签"""
        try:
            vol = float(hist_data['Close'].pct_change().std() or 0)
            if vol >= 0.03:
                return '高'
            if vol >= 0.015:
                return '中'
            return '低'
        except Exception:
            return '中'
    
    def _calculate_enhanced_accuracy(self, hist_data, hist_analysis, full_data, full_analysis, split_point):
        """计算增强的准确性指标"""
        # 1. 分型准确性 - 改进算法
        hist_fractals = [f for f in hist_analysis['fractals'] if f.idx < split_point]
        full_fractals = [f for f in full_analysis['fractals'] if f.idx < split_point]
        
        if len(hist_fractals) == 0 and len(full_fractals) == 0:
            fractal_accuracy = 1.0  # 都没有分型，认为准确
        elif len(hist_fractals) == 0 or len(full_fractals) == 0:
            fractal_accuracy = 0.0  # 一个有分型一个没有，认为不准确
        else:
            # 计算分型位置的重叠度
            hist_positions = set(f.idx for f in hist_fractals)
            full_positions = set(f.idx for f in full_fractals)
            intersection = len(hist_positions.intersection(full_positions))
            union = len(hist_positions.union(full_positions))
            fractal_accuracy = intersection / union if union > 0 else 0
        
        # 2. 笔准确性 - 改进算法
        hist_strokes = [s for s in hist_analysis['strokes'] if s.end_idx < split_point]
        full_strokes = [s for s in full_analysis['strokes'] if s.end_idx < split_point]
        
        if len(full_strokes) == 0:
            stroke_accuracy = 1.0 if len(hist_strokes) == 0 else 0.0
        else:
            # 计算笔的方向一致性
            if len(hist_strokes) == 0:
                stroke_accuracy = 0.0
            else:
                # 比较笔的方向和长度
                direction_matches = 0
                length_accuracy = 0
                
                for i, hist_stroke in enumerate(hist_strokes):
                    if i < len(full_strokes):
                        # 方向匹配
                        if hist_stroke.direction == full_strokes[i].direction:
                            direction_matches += 1
                        
                        # 长度准确性（允许一定误差）
                        hist_length = hist_stroke.end_idx - hist_stroke.start_idx
                        full_length = full_strokes[i].end_idx - full_strokes[i].start_idx
                        if hist_length > 0:
                            length_error = abs(hist_length - full_length) / hist_length
                            length_accuracy += max(0, 1 - length_error)
                
                direction_accuracy = direction_matches / len(hist_strokes) if len(hist_strokes) > 0 else 0
                length_accuracy = length_accuracy / len(hist_strokes) if len(hist_strokes) > 0 else 0
                stroke_accuracy = (direction_accuracy * 0.7 + length_accuracy * 0.3)
        
        # 3. 价格预测准确性 - 改进算法
        hist_price = hist_data['Close'].iloc[-1]
        actual_price = full_data['Close'].iloc[split_point-1] if split_point < len(full_data) else full_data['Close'].iloc[-1]
        
        # 使用相对误差和绝对误差的组合
        relative_error = abs(hist_price - actual_price) / hist_price if hist_price > 0 else 1
        absolute_error = abs(hist_price - actual_price)
        price_volatility = hist_data['Close'].pct_change().std()
        
        # 考虑价格波动性的价格准确性
        if price_volatility > 0:
            normalized_error = relative_error / (price_volatility * 2)  # 标准化到波动性
            price_accuracy = max(0, 1 - min(1, normalized_error))
        else:
            price_accuracy = max(0, 1 - relative_error)
        
        # 4. 趋势预测准确性 - 改进算法
        hist_trend = self._calculate_trend(hist_data['Close'].tail(20))
        future_data = full_data.iloc[split_point:]
        
        if len(future_data) > 0:
            future_trend = self._calculate_trend(future_data['Close'])
            
            # 不仅看方向，还看趋势强度
            if hist_trend == 0 and future_trend == 0:
                trend_accuracy = 1.0  # 都无趋势，认为准确
            elif hist_trend == 0 or future_trend == 0:
                trend_accuracy = 0.0  # 一个有趋势一个没有，认为不准确
            else:
                # 方向一致性
                direction_match = 1 if hist_trend * future_trend > 0 else 0
                
                # 强度相似性
                strength_ratio = min(abs(hist_trend), abs(future_trend)) / max(abs(hist_trend), abs(future_trend))
                
                trend_accuracy = direction_match * 0.7 + strength_ratio * 0.3
        else:
            trend_accuracy = 0
        
        # 5. 结构稳定性评分
        structure_stability = self._calculate_structure_stability(hist_analysis, full_analysis, split_point)
        
        # 6. 综合准确性评分（加权平均）
        overall_accuracy = (
            fractal_accuracy * 0.25 +
            stroke_accuracy * 0.25 +
            price_accuracy * 0.20 +
            trend_accuracy * 0.20 +
            structure_stability * 0.10
        )
        
        return {
            'fractal_accuracy': round(fractal_accuracy * 100, 1),
            'stroke_accuracy': round(stroke_accuracy * 100, 1),
            'price_accuracy': round(price_accuracy * 100, 1),
            'trend_accuracy': round(trend_accuracy * 100, 1),
            'structure_stability': round(structure_stability * 100, 1),
            'overall_accuracy': round(overall_accuracy * 100, 1),
            'validation_info': {
                'historical_period': f"{hist_data['Date'].iloc[0]} 至 {hist_data['Date'].iloc[-1]}",
                'validation_period': f"{full_data['Date'].iloc[split_point-1] if split_point < len(full_data) else 'N/A'} 至 {full_data['Date'].iloc[-1]}",
                'historical_price': round(hist_price, 2),
                'actual_price': round(actual_price, 2),
                'price_change': round(actual_price - hist_price, 2),
                'price_change_pct': round((actual_price - hist_price) / hist_price * 100, 2)
            }
        }
    
    def _calculate_structure_stability(self, hist_analysis, full_analysis, split_point):
        """计算结构稳定性评分"""
        try:
            # 比较分型、笔、线段的数量变化
            hist_fractals = len([f for f in hist_analysis['fractals'] if f.idx < split_point])
            full_fractals = len([f for f in full_analysis['fractals'] if f.idx < split_point])
            
            hist_strokes = len([s for s in hist_analysis['strokes'] if s.end_idx < split_point])
            full_strokes = len([s for s in full_analysis['strokes'] if s.end_idx < split_point])
            
            hist_segments = len([s for s in hist_analysis['segments'] if s.end_idx < split_point])
            full_segments = len([s for s in full_analysis['segments'] if s.end_idx < split_point])
            
            # 计算结构数量的一致性
            fractal_consistency = 1 - abs(hist_fractals - full_fractals) / max(hist_fractals, full_fractals, 1)
            stroke_consistency = 1 - abs(hist_strokes - full_strokes) / max(hist_strokes, full_strokes, 1)
            segment_consistency = 1 - abs(hist_segments - full_segments) / max(hist_segments, full_segments, 1)
            
            return (fractal_consistency + stroke_consistency + segment_consistency) / 3
            
        except Exception:
            return 0.5  # 默认中性评分
    
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
                dates = full_df['Date'].dt.strftime('%Y-%m-%d').tolist()
            else:
                dates = [f'Day {i}' for i in range(len(full_df))]
            
            # 是否存在验证期实际数据
            has_future_actual = len(full_df) > split_idx
            steps_actual = max(0, len(full_df) - split_idx)

            # 价格曲线
            if has_future_actual:
                ax.plot(range(len(full_df)), full_df['Close'].values, color='#6c757d', lw=1.5, ls='--', label='Future close (actual)')
            ax.plot(range(split_idx), hist_df['Close'].values, color='#007bff', lw=2.0, label='Training close (with Chan)')
            
            # 分割线
            ax.axvline(split_idx-0.5, color='black', lw=1, ls=':', alpha=0.6)
            ax.text(split_idx, ax.get_ylim()[1], 'Validation start', va='top', ha='left', fontsize=9)
            
            # 训练期简单预测线（基于线性斜率）
            tail = hist_df['Close'].tail(20)
            slope = float(self._calculate_trend(tail))
            if not np.isfinite(slope):
                slope = 0.0
            start_y_val = hist_df['Close'].iloc[-1]
            try:
                start_y = float(start_y_val)
            except Exception:
                start_y = float(hist_df['Close'].dropna().iloc[-1]) if hist_df['Close'].dropna().shape[0] else 0.0
            # 预测步数：若没有未来实际数据，至少展示10步；
            # 若有部分实际数据，也继续延伸预测线以便在实际数据尽头之后可见
            steps = max(1, steps_actual)
            min_pred_steps = 10
            if not has_future_actual:
                steps = max(min_pred_steps, steps)
            else:
                # 存在实际数据时，也延长预测到至少 min_pred_steps
                steps = max(min_pred_steps, steps)
            steps = max(2, int(steps))
            proj_x = np.arange(steps)
            proj_y = start_y + slope * proj_x
            # 若出现NaN，回退为水平线
            if not np.isfinite(np.nanmean(proj_y)):
                proj_y = np.full(steps, start_y)
            ax.plot(range(split_idx, split_idx + steps), proj_y, color='#28a745', lw=2.2, alpha=0.95, label='Predicted trend (linear)', zorder=10)

            # 若需要，扩展日期标签供显示（无未来数据或需要延长预测长度时）
            if (not has_future_actual) or (split_idx + steps > len(dates)):
                try:
                    # 依据最后一个日期推演
                    if len(dates) == 0:
                        dates = []
                    total_needed = split_idx + steps
                    if len(dates) < total_needed:
                        need = total_needed - len(dates)
                        if 'Date' in full_df.columns and pd.api.types.is_datetime64_any_dtype(full_df['Date']):
                            base_date = full_df['Date'].iloc[-1] if len(full_df) > 0 else pd.to_datetime('today')
                            last_date = pd.to_datetime(base_date)
                            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=need, freq='D').strftime('%Y-%m-%d').tolist()
                        else:
                            start_i = len(dates)
                            future_dates = [f'Day {i}' for i in range(start_i, start_i + need)]
                        dates = dates + future_dates
                except Exception:
                    # 即使扩展日期失败，也不影响图表绘制
                    pass
            
            # 训练期叠加关键Chan要素（仅笔）
            for s in hist_analysis['strokes']:
                if s.end_idx < split_idx:
                    color = '#0dcaf0' if s.direction == 'up' else '#fd7e14'
                    ax.plot([s.start_idx, s.end_idx], [s.start_price, s.end_price], color=color, lw=1.5, alpha=0.8)
            
            # X 轴刻度
            total_len_for_ticks = max(split_idx + steps, len(full_df))
            step = max(1, total_len_for_ticks//15)
            tick_positions = list(range(0, total_len_for_ticks, step))
            ax.set_xticks(tick_positions)
            # 生成与tick数量一致的标签
            labels = []
            for pos in tick_positions:
                if dates and pos < len(dates):
                    labels.append(dates[pos])
                else:
                    labels.append(str(pos))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            # 重新计算边界，确保预测线纳入可视范围
            ax.relim()
            ax.autoscale_view()

            ax.set_title('Training vs Future: Predicted vs Actual', fontsize=16, weight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.legend(loc='upper left')
            # 确保X轴范围包含预测段
            ax.set_xlim(0, max(1, total_len_for_ticks - 1))
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode()
            plt.close()
            return f"data:image/png;base64,{img_str}"
        except Exception:
            return None

# 创建分析器实例
analyzer = ChanWebAnalyzer()

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/handbook')
def handbook():
    """新手手册页面"""
    return render_template('handbook.html')

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
        
        # 获取时间框架参数
        timeframe = data.get('timeframe', '1d')
        
        # 下载数据
        df, error = analyzer.download_stock_data(symbol, start_date, end_date, timeframe)
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
        
        return jsonify({
            'success': True,
            'chart': chart_data,
            'report': report,
            'symbol': symbol,
            'period': f"{start_date} 至 {end_date}"
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

@app.route('/watchlist', methods=['GET'])
def get_watchlist():
    """获取收藏列表"""
    try:
        return jsonify({
            'success': True,
            'stocks': watchlist_storage['stocks'],
            'last_updated': watchlist_storage['last_updated']
        })
    except Exception as e:
        return jsonify({'error': f'获取收藏列表失败: {str(e)}'}), 500

@app.route('/watchlist', methods=['POST'])
def add_to_watchlist():
    """添加股票到收藏列表"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        
        if not symbol:
            return jsonify({'error': '请输入股票代码'}), 400
        
        # 检查是否已存在
        if symbol in watchlist_storage['stocks']:
            return jsonify({'error': '该股票已在收藏列表中'}), 400
        
        # 添加到收藏列表
        watchlist_storage['stocks'].append(symbol)
        watchlist_storage['last_updated'] = datetime.now().isoformat()
        
        return jsonify({
            'success': True,
            'message': f'已添加 {symbol} 到收藏列表',
            'stocks': watchlist_storage['stocks']
        })
        
    except Exception as e:
        return jsonify({'error': f'添加失败: {str(e)}'}), 500

@app.route('/watchlist', methods=['DELETE'])
def remove_from_watchlist():
    """从收藏列表移除股票"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        
        if not symbol:
            return jsonify({'error': '请输入股票代码'}), 400
        
        # 检查是否存在
        if symbol not in watchlist_storage['stocks']:
            return jsonify({'error': '该股票不在收藏列表中'}), 400
        
        # 从收藏列表移除
        watchlist_storage['stocks'].remove(symbol)
        watchlist_storage['last_updated'] = datetime.now().isoformat()
        
        return jsonify({
            'success': True,
            'message': f'已从收藏列表移除 {symbol}',
            'stocks': watchlist_storage['stocks']
        })
        
    except Exception as e:
        return jsonify({'error': f'移除失败: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
