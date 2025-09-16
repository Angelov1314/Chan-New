#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缠论分析准确性评价系统
使用2025年实际数据验证分析准确性
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_fixed import load_ohlc, find_fractals, build_strokes, build_segments, detect_zhongshu, detect_divergence, resolve_inclusion

class AccuracyEvaluator:
    def __init__(self):
        self.accuracy_scores = {}
        self.total_accuracy = 0
        
    def evaluate_accuracy(self, historical_data_path, future_data_path):
        """评估分析准确性"""
        
        print("=" * 80)
        print("缠论分析准确性评价报告")
        print("=" * 80)
        
        # 加载历史数据（2023-2024）
        print("加载历史数据...")
        df_historical = load_ohlc(historical_data_path)
        
        # 加载未来数据（2025年1-6月）
        print("加载未来数据...")
        df_future = load_ohlc(future_data_path)
        
        # 对历史数据进行缠论分析
        print("对历史数据进行缠论分析...")
        historical_analysis = self._analyze_data(df_historical)
        
        # 对完整数据（历史+未来）进行缠论分析
        print("对完整数据进行缠论分析...")
        df_combined = pd.concat([df_historical, df_future], ignore_index=True)
        combined_analysis = self._analyze_data(df_combined)
        
        # 评估各项准确性
        print("\n开始准确性评估...")
        
        # 1. 分型预测准确性
        fractal_accuracy = self._evaluate_fractal_accuracy(historical_analysis, combined_analysis, len(df_historical))
        
        # 2. 笔预测准确性
        stroke_accuracy = self._evaluate_stroke_accuracy(historical_analysis, combined_analysis, len(df_historical))
        
        # 3. 线段预测准确性
        segment_accuracy = self._evaluate_segment_accuracy(historical_analysis, combined_analysis, len(df_historical))
        
        # 4. 中枢预测准确性
        zhongshu_accuracy = self._evaluate_zhongshu_accuracy(historical_analysis, combined_analysis, len(df_historical))
        
        # 5. 背驰信号准确性
        divergence_accuracy = self._evaluate_divergence_accuracy(historical_analysis, combined_analysis, len(df_historical))
        
        # 6. 价格预测准确性
        price_accuracy = self._evaluate_price_accuracy(df_historical, df_future)
        
        # 7. 趋势预测准确性
        trend_accuracy = self._evaluate_trend_accuracy(df_historical, df_future)
        
        # 计算总准确性
        self.total_accuracy = (fractal_accuracy + stroke_accuracy + segment_accuracy + 
                              zhongshu_accuracy + divergence_accuracy + price_accuracy + trend_accuracy) / 7
        
        # 生成准确性报告
        self._generate_accuracy_report()
        
        return self.total_accuracy
    
    def _analyze_data(self, df):
        """对数据进行缠论分析"""
        df_s = resolve_inclusion(df)
        df_use = df.copy()
        if "High_smooth" in df_s.columns:
            df_use["High"] = df_s["High_smooth"]
            df_use["Low"] = df_s["Low_smooth"]
        
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
        }
    
    def _evaluate_fractal_accuracy(self, historical, combined, split_point):
        """评估分型预测准确性"""
        print("\n【分型预测准确性评估】")
        print("-" * 40)
        
        # 历史分型
        hist_fractals = [f for f in historical['fractals'] if f.idx < split_point]
        # 完整数据分型
        full_fractals = [f for f in combined['fractals'] if f.idx < split_point]
        
        # 计算分型识别准确性
        if len(hist_fractals) > 0 and len(full_fractals) > 0:
            # 简单的匹配度计算
            hist_positions = set(f.idx for f in hist_fractals)
            full_positions = set(f.idx for f in full_fractals)
            
            common_positions = len(hist_positions.intersection(full_positions))
            total_positions = len(hist_positions.union(full_positions))
            
            if total_positions > 0:
                accuracy = common_positions / total_positions
            else:
                accuracy = 0
        else:
            accuracy = 0
        
        print(f"历史分型数量: {len(hist_fractals)}")
        print(f"完整数据分型数量: {len(full_fractals)}")
        print(f"分型识别准确性: {accuracy:.3f}")
        
        self.accuracy_scores['fractal'] = accuracy
        return accuracy
    
    def _evaluate_stroke_accuracy(self, historical, combined, split_point):
        """评估笔预测准确性"""
        print("\n【笔预测准确性评估】")
        print("-" * 40)
        
        # 历史笔
        hist_strokes = [s for s in historical['strokes'] if s.end_idx < split_point]
        # 完整数据笔
        full_strokes = [s for s in combined['strokes'] if s.end_idx < split_point]
        
        # 计算笔的准确性
        if len(hist_strokes) > 0 and len(full_strokes) > 0:
            # 比较笔的方向和位置
            hist_directions = [s.direction for s in hist_strokes]
            full_directions = [s.direction for s in full_strokes]
            
            # 计算方向匹配度
            min_len = min(len(hist_directions), len(full_directions))
            if min_len > 0:
                direction_matches = sum(1 for i in range(min_len) if hist_directions[i] == full_directions[i])
                accuracy = direction_matches / min_len
            else:
                accuracy = 0
        else:
            accuracy = 0
        
        print(f"历史笔数量: {len(hist_strokes)}")
        print(f"完整数据笔数量: {len(full_strokes)}")
        print(f"笔方向准确性: {accuracy:.3f}")
        
        self.accuracy_scores['stroke'] = accuracy
        return accuracy
    
    def _evaluate_segment_accuracy(self, historical, combined, split_point):
        """评估线段预测准确性"""
        print("\n【线段预测准确性评估】")
        print("-" * 40)
        
        # 历史线段
        hist_segments = [s for s in historical['segments'] if s.end_idx < split_point]
        # 完整数据线段
        full_segments = [s for s in combined['segments'] if s.end_idx < split_point]
        
        # 计算线段准确性
        if len(hist_segments) > 0 and len(full_segments) > 0:
            # 比较线段方向
            hist_directions = [s.direction for s in hist_segments]
            full_directions = [s.direction for s in full_segments]
            
            min_len = min(len(hist_directions), len(full_directions))
            if min_len > 0:
                direction_matches = sum(1 for i in range(min_len) if hist_directions[i] == full_directions[i])
                accuracy = direction_matches / min_len
            else:
                accuracy = 0
        else:
            accuracy = 0
        
        print(f"历史线段数量: {len(hist_segments)}")
        print(f"完整数据线段数量: {len(full_segments)}")
        print(f"线段方向准确性: {accuracy:.3f}")
        
        self.accuracy_scores['segment'] = accuracy
        return accuracy
    
    def _evaluate_zhongshu_accuracy(self, historical, combined, split_point):
        """评估中枢预测准确性"""
        print("\n【中枢预测准确性评估】")
        print("-" * 40)
        
        # 历史中枢
        hist_zhongshus = [z for z in historical['zhongshus'] if z.end_idx < split_point]
        # 完整数据中枢
        full_zhongshus = [z for z in combined['zhongshus'] if z.end_idx < split_point]
        
        # 计算中枢准确性
        if len(hist_zhongshus) > 0 and len(full_zhongshus) > 0:
            # 比较中枢位置和范围
            accuracy = min(len(hist_zhongshus), len(full_zhongshus)) / max(len(hist_zhongshus), len(full_zhongshus))
        else:
            accuracy = 0
        
        print(f"历史中枢数量: {len(hist_zhongshus)}")
        print(f"完整数据中枢数量: {len(full_zhongshus)}")
        print(f"中枢识别准确性: {accuracy:.3f}")
        
        self.accuracy_scores['zhongshu'] = accuracy
        return accuracy
    
    def _evaluate_divergence_accuracy(self, historical, combined, split_point):
        """评估背驰信号准确性"""
        print("\n【背驰信号准确性评估】")
        print("-" * 40)
        
        # 历史背驰
        hist_divergences = [d for d in historical['divergences'] if d[0] < split_point]
        # 完整数据背驰
        full_divergences = [d for d in combined['divergences'] if d[0] < split_point]
        
        # 计算背驰准确性
        if len(hist_divergences) > 0 and len(full_divergences) > 0:
            # 比较背驰类型
            hist_types = [d[1] for d in hist_divergences]
            full_types = [d[1] for d in full_divergences]
            
            min_len = min(len(hist_types), len(full_types))
            if min_len > 0:
                type_matches = sum(1 for i in range(min_len) if hist_types[i] == full_types[i])
                accuracy = type_matches / min_len
            else:
                accuracy = 0
        else:
            accuracy = 0
        
        print(f"历史背驰数量: {len(hist_divergences)}")
        print(f"完整数据背驰数量: {len(full_divergences)}")
        print(f"背驰信号准确性: {accuracy:.3f}")
        
        self.accuracy_scores['divergence'] = accuracy
        return accuracy
    
    def _evaluate_price_accuracy(self, df_historical, df_future):
        """评估价格预测准确性"""
        print("\n【价格预测准确性评估】")
        print("-" * 40)
        
        # 获取历史数据最后的价格
        last_historical_price = df_historical['Close'].iloc[-1]
        
        # 获取未来数据的价格范围
        future_high = df_future['High'].max()
        future_low = df_future['Low'].min()
        future_close = df_future['Close'].iloc[-1]
        
        # 计算价格变化
        price_change = future_close - last_historical_price
        price_change_pct = price_change / last_historical_price
        
        # 评估价格预测准确性（基于历史趋势的延续性）
        historical_trend = self._calculate_trend(df_historical['Close'].tail(20))
        future_trend = self._calculate_trend(df_future['Close'])
        
        # 趋势一致性
        if historical_trend * future_trend > 0:  # 同方向
            trend_consistency = 1.0
        else:  # 不同方向
            trend_consistency = 0.0
        
        # 价格波动合理性
        price_volatility = (future_high - future_low) / last_historical_price
        if 0.05 <= price_volatility <= 0.30:  # 合理的波动范围
            volatility_score = 1.0
        elif 0.02 <= price_volatility <= 0.50:
            volatility_score = 0.8
        else:
            volatility_score = 0.6
        
        accuracy = (trend_consistency + volatility_score) / 2
        
        print(f"历史最后价格: ${last_historical_price:.2f}")
        print(f"未来最高价格: ${future_high:.2f}")
        print(f"未来最低价格: ${future_low:.2f}")
        print(f"未来最后价格: ${future_close:.2f}")
        print(f"价格变化: {price_change:+.2f} ({price_change_pct:+.2%})")
        print(f"趋势一致性: {trend_consistency:.3f}")
        print(f"波动合理性: {volatility_score:.3f}")
        print(f"价格预测准确性: {accuracy:.3f}")
        
        self.accuracy_scores['price'] = accuracy
        return accuracy
    
    def _evaluate_trend_accuracy(self, df_historical, df_future):
        """评估趋势预测准确性"""
        print("\n【趋势预测准确性评估】")
        print("-" * 40)
        
        # 计算历史趋势
        hist_trend = self._calculate_trend(df_historical['Close'].tail(50))
        
        # 计算未来趋势
        future_trend = self._calculate_trend(df_future['Close'])
        
        # 趋势方向一致性
        if hist_trend * future_trend > 0:
            trend_accuracy = 1.0
        else:
            trend_accuracy = 0.0
        
        # 趋势强度比较
        hist_strength = abs(hist_trend)
        future_strength = abs(future_trend)
        
        if hist_strength > 0 and future_strength > 0:
            strength_ratio = min(hist_strength, future_strength) / max(hist_strength, future_strength)
            strength_score = strength_ratio
        else:
            strength_score = 0.5
        
        accuracy = (trend_accuracy + strength_score) / 2
        
        print(f"历史趋势: {hist_trend:.4f}")
        print(f"未来趋势: {future_trend:.4f}")
        print(f"趋势方向一致性: {trend_accuracy:.3f}")
        print(f"趋势强度匹配: {strength_score:.3f}")
        print(f"趋势预测准确性: {accuracy:.3f}")
        
        self.accuracy_scores['trend'] = accuracy
        return accuracy
    
    def _calculate_trend(self, prices):
        """计算价格趋势"""
        if len(prices) < 2:
            return 0
        
        # 使用线性回归计算趋势
        x = np.arange(len(prices))
        y = prices.values
        
        # 简单线性回归
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
    
    def _generate_accuracy_report(self):
        """生成准确性报告"""
        print("\n" + "=" * 80)
        print("缠论分析准确性评价总结")
        print("=" * 80)
        
        # 各维度准确性
        print("\n【各维度准确性得分】")
        print("-" * 50)
        for category, score in self.accuracy_scores.items():
            category_name = {
                'fractal': '分型预测',
                'stroke': '笔预测',
                'segment': '线段预测',
                'zhongshu': '中枢预测',
                'divergence': '背驰信号',
                'price': '价格预测',
                'trend': '趋势预测'
            }.get(category, category)
            
            percentage = score * 100
            print(f"{category_name}: {score:.3f} ({percentage:.1f}%)")
        
        # 总体准确性
        print(f"\n【总体准确性评估】")
        print("-" * 50)
        print(f"总体准确性: {self.total_accuracy:.3f}")
        
        percentage = self.total_accuracy * 100
        print(f"准确性得分率: {percentage:.1f}%")
        
        # 准确性评级
        if percentage >= 90:
            grade = "优秀 (A+)"
            comment = "分析准确性极高，各项预测表现优异"
        elif percentage >= 80:
            grade = "良好 (A)"
            comment = "分析准确性良好，大部分预测准确"
        elif percentage >= 70:
            grade = "中等 (B)"
            comment = "分析准确性中等，部分预测需要改进"
        elif percentage >= 60:
            grade = "及格 (C)"
            comment = "分析准确性及格，多项预测需要优化"
        else:
            grade = "不及格 (D)"
            comment = "分析准确性较差，需要全面改进"
        
        print(f"准确性评级: {grade}")
        print(f"评价: {comment}")
        
        # 改进建议
        print(f"\n【改进建议】")
        print("-" * 50)
        
        suggestions = []
        for category, score in self.accuracy_scores.items():
            if score < 0.7:
                category_name = {
                    'fractal': '分型预测',
                    'stroke': '笔预测',
                    'segment': '线段预测',
                    'zhongshu': '中枢预测',
                    'divergence': '背驰信号',
                    'price': '价格预测',
                    'trend': '趋势预测'
                }.get(category, category)
                suggestions.append(f"- 改进{category_name}算法")
        
        if suggestions:
            for suggestion in suggestions:
                print(suggestion)
        else:
            print("- 整体分析准确性良好，继续保持")
        
        print("\n" + "=" * 80)

def main():
    """主函数"""
    evaluator = AccuracyEvaluator()
    
    # 评估准确性
    accuracy = evaluator.evaluate_accuracy(
        "data/AAPL_1d_data.csv",  # 历史数据
        "data/AAPL_2025_data.csv"  # 未来数据
    )
    
    return accuracy

if __name__ == "__main__":
    main()
