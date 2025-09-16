#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缠论分析评估打分系统
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

class ChanAnalysisEvaluator:
    def __init__(self):
        self.scores = {}
        self.total_score = 0
        self.max_score = 100
        
    def evaluate_analysis_quality(self, analysis_results):
        """评估分析质量"""
        
        print("=" * 60)
        print("缠论分析质量评估报告")
        print("=" * 60)
        
        # 1. 数据质量评估 (20分)
        data_score = self._evaluate_data_quality(analysis_results)
        
        # 2. 分型识别准确性 (20分)
        fractal_score = self._evaluate_fractal_accuracy(analysis_results)
        
        # 3. 笔构建合理性 (15分)
        stroke_score = self._evaluate_stroke_quality(analysis_results)
        
        # 4. 线段分析有效性 (15分)
        segment_score = self._evaluate_segment_analysis(analysis_results)
        
        # 5. 中枢检测准确性 (15分)
        zhongshu_score = self._evaluate_zhongshu_detection(analysis_results)
        
        # 6. 背驰信号可靠性 (10分)
        divergence_score = self._evaluate_divergence_signals(analysis_results)
        
        # 7. 交易信号实用性 (5分)
        trading_score = self._evaluate_trading_signals(analysis_results)
        
        # 计算总分
        self.total_score = (data_score + fractal_score + stroke_score + 
                           segment_score + zhongshu_score + divergence_score + trading_score)
        
        # 生成评估报告
        self._generate_evaluation_report()
        
        return self.total_score
    
    def _evaluate_data_quality(self, results):
        """数据质量评估 (20分)"""
        print("\n【数据质量评估】")
        print("-" * 30)
        
        score = 0
        max_score = 20
        
        # 数据完整性
        data_points = results.get('total_k_lines', 501)
        if data_points >= 500:
            completeness = 5
            print(f"✓ 数据完整性: {completeness}/5 (数据点: {data_points})")
        elif data_points >= 200:
            completeness = 4
            print(f"✓ 数据完整性: {completeness}/5 (数据点: {data_points})")
        else:
            completeness = 3
            print(f"⚠ 数据完整性: {completeness}/5 (数据点: {data_points})")
        
        score += completeness
        
        # 时间跨度
        time_span = results.get('time_span_days', 365)
        if time_span >= 365:
            span_score = 5
            print(f"✓ 时间跨度: {span_score}/5 ({time_span}天)")
        elif time_span >= 180:
            span_score = 4
            print(f"✓ 时间跨度: {span_score}/5 ({time_span}天)")
        else:
            span_score = 3
            print(f"⚠ 时间跨度: {span_score}/5 ({time_span}天)")
        
        score += span_score
        
        # 数据质量
        price_range = results.get('price_range', 0.3)
        if price_range >= 0.2:
            quality_score = 5
            print(f"✓ 价格波动性: {quality_score}/5 (波动率: {price_range:.1%})")
        elif price_range >= 0.1:
            quality_score = 4
            print(f"✓ 价格波动性: {quality_score}/5 (波动率: {price_range:.1%})")
        else:
            quality_score = 3
            print(f"⚠ 价格波动性: {quality_score}/5 (波动率: {price_range:.1%})")
        
        score += quality_score
        
        # 技术指标完整性
        indicators = results.get('technical_indicators', 0)
        if indicators >= 8:
            indicator_score = 5
            print(f"✓ 技术指标: {indicator_score}/5 ({indicators}个指标)")
        elif indicators >= 5:
            indicator_score = 4
            print(f"✓ 技术指标: {indicator_score}/5 ({indicators}个指标)")
        else:
            indicator_score = 3
            print(f"⚠ 技术指标: {indicator_score}/5 ({indicators}个指标)")
        
        score += indicator_score
        
        self.scores['data_quality'] = score
        print(f"数据质量总分: {score}/{max_score}")
        return score
    
    def _evaluate_fractal_accuracy(self, results):
        """分型识别准确性评估 (20分)"""
        print("\n【分型识别准确性评估】")
        print("-" * 30)
        
        score = 0
        max_score = 20
        
        fractals = results.get('fractals', [])
        total_fractals = len(fractals)
        
        # 分型数量合理性
        if 50 <= total_fractals <= 150:
            count_score = 5
            print(f"✓ 分型数量: {count_score}/5 ({total_fractals}个)")
        elif 30 <= total_fractals <= 200:
            count_score = 4
            print(f"✓ 分型数量: {count_score}/5 ({total_fractals}个)")
        else:
            count_score = 3
            print(f"⚠ 分型数量: {count_score}/5 ({total_fractals}个)")
        
        score += count_score
        
        # 分型分布平衡性
        top_fractals = len([f for f in fractals if f.get('kind') == 'top'])
        bottom_fractals = len([f for f in fractals if f.get('kind') == 'bottom'])
        balance_ratio = min(top_fractals, bottom_fractals) / max(top_fractals, bottom_fractals) if max(top_fractals, bottom_fractals) > 0 else 0
        
        if balance_ratio >= 0.7:
            balance_score = 5
            print(f"✓ 分型平衡性: {balance_score}/5 (比例: {balance_ratio:.2f})")
        elif balance_ratio >= 0.5:
            balance_score = 4
            print(f"✓ 分型平衡性: {balance_score}/5 (比例: {balance_ratio:.2f})")
        else:
            balance_score = 3
            print(f"⚠ 分型平衡性: {balance_score}/5 (比例: {balance_ratio:.2f})")
        
        score += balance_score
        
        # 分型密度合理性
        k_lines = results.get('total_k_lines', 501)
        fractal_density = total_fractals / k_lines if k_lines > 0 else 0
        
        if 0.15 <= fractal_density <= 0.35:
            density_score = 5
            print(f"✓ 分型密度: {density_score}/5 ({fractal_density:.3f})")
        elif 0.10 <= fractal_density <= 0.40:
            density_score = 4
            print(f"✓ 分型密度: {density_score}/5 ({fractal_density:.3f})")
        else:
            density_score = 3
            print(f"⚠ 分型密度: {density_score}/5 ({fractal_density:.3f})")
        
        score += density_score
        
        # 分型强度分布
        strong_fractals = len([f for f in fractals if f.get('strength', 0) > 0.02])
        strength_ratio = strong_fractals / total_fractals if total_fractals > 0 else 0
        
        if 0.2 <= strength_ratio <= 0.4:
            strength_score = 5
            print(f"✓ 分型强度: {strength_score}/5 (强分型比例: {strength_ratio:.2f})")
        elif 0.1 <= strength_ratio <= 0.5:
            strength_score = 4
            print(f"✓ 分型强度: {strength_score}/5 (强分型比例: {strength_ratio:.2f})")
        else:
            strength_score = 3
            print(f"⚠ 分型强度: {strength_score}/5 (强分型比例: {strength_ratio:.2f})")
        
        score += strength_score
        
        self.scores['fractal_accuracy'] = score
        print(f"分型识别总分: {score}/{max_score}")
        return score
    
    def _evaluate_stroke_quality(self, results):
        """笔构建质量评估 (15分)"""
        print("\n【笔构建质量评估】")
        print("-" * 30)
        
        score = 0
        max_score = 15
        
        strokes = results.get('strokes', [])
        total_strokes = len(strokes)
        
        # 笔数量合理性
        if 30 <= total_strokes <= 80:
            count_score = 4
            print(f"✓ 笔数量: {count_score}/4 ({total_strokes}支)")
        elif 20 <= total_strokes <= 100:
            count_score = 3
            print(f"✓ 笔数量: {count_score}/4 ({total_strokes}支)")
        else:
            count_score = 2
            print(f"⚠ 笔数量: {count_score}/4 ({total_strokes}支)")
        
        score += count_score
        
        # 笔长度分布
        if strokes:
            lengths = [s.get('end_idx', 0) - s.get('start_idx', 0) for s in strokes]
            avg_length = np.mean(lengths)
            length_std = np.std(lengths)
            
            if 5 <= avg_length <= 15 and length_std <= 10:
                length_score = 4
                print(f"✓ 笔长度分布: {length_score}/4 (平均: {avg_length:.1f}, 标准差: {length_std:.1f})")
            elif 3 <= avg_length <= 20:
                length_score = 3
                print(f"✓ 笔长度分布: {length_score}/4 (平均: {avg_length:.1f}, 标准差: {length_std:.1f})")
            else:
                length_score = 2
                print(f"⚠ 笔长度分布: {length_score}/4 (平均: {avg_length:.1f}, 标准差: {length_std:.1f})")
        else:
            length_score = 0
            print(f"✗ 笔长度分布: {length_score}/4 (无笔数据)")
        
        score += length_score
        
        # 笔振幅合理性
        if strokes:
            amplitudes = [s.get('swing', 0) for s in strokes]
            avg_amplitude = np.mean(amplitudes)
            
            if 0.02 <= avg_amplitude <= 0.15:
                amplitude_score = 4
                print(f"✓ 笔振幅: {amplitude_score}/4 (平均: {avg_amplitude:.3f})")
            elif 0.01 <= avg_amplitude <= 0.25:
                amplitude_score = 3
                print(f"✓ 笔振幅: {amplitude_score}/4 (平均: {avg_amplitude:.3f})")
            else:
                amplitude_score = 2
                print(f"⚠ 笔振幅: {amplitude_score}/4 (平均: {avg_amplitude:.3f})")
        else:
            amplitude_score = 0
            print(f"✗ 笔振幅: {amplitude_score}/4 (无笔数据)")
        
        score += amplitude_score
        
        # 笔方向交替性
        if strokes:
            directions = [s.get('direction', '') for s in strokes]
            alternations = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i-1])
            alternation_ratio = alternations / (len(directions) - 1) if len(directions) > 1 else 0
            
            if alternation_ratio >= 0.8:
                alternation_score = 3
                print(f"✓ 笔方向交替: {alternation_score}/3 (交替率: {alternation_ratio:.2f})")
            elif alternation_ratio >= 0.6:
                alternation_score = 2
                print(f"✓ 笔方向交替: {alternation_score}/3 (交替率: {alternation_ratio:.2f})")
            else:
                alternation_score = 1
                print(f"⚠ 笔方向交替: {alternation_score}/3 (交替率: {alternation_ratio:.2f})")
        else:
            alternation_score = 0
            print(f"✗ 笔方向交替: {alternation_score}/3 (无笔数据)")
        
        score += alternation_score
        
        self.scores['stroke_quality'] = score
        print(f"笔构建总分: {score}/{max_score}")
        return score
    
    def _evaluate_segment_analysis(self, results):
        """线段分析有效性评估 (15分)"""
        print("\n【线段分析有效性评估】")
        print("-" * 30)
        
        score = 0
        max_score = 15
        
        segments = results.get('segments', [])
        total_segments = len(segments)
        
        # 线段数量合理性
        if 5 <= total_segments <= 20:
            count_score = 4
            print(f"✓ 线段数量: {count_score}/4 ({total_segments}个)")
        elif 3 <= total_segments <= 30:
            count_score = 3
            print(f"✓ 线段数量: {count_score}/4 ({total_segments}个)")
        else:
            count_score = 2
            print(f"⚠ 线段数量: {count_score}/4 ({total_segments}个)")
        
        score += count_score
        
        # 线段长度分布
        if segments:
            lengths = [s.get('end_idx', 0) - s.get('start_idx', 0) for s in segments]
            avg_length = np.mean(lengths)
            
            if 10 <= avg_length <= 50:
                length_score = 4
                print(f"✓ 线段长度: {length_score}/4 (平均: {avg_length:.1f})")
            elif 5 <= avg_length <= 80:
                length_score = 3
                print(f"✓ 线段长度: {length_score}/4 (平均: {avg_length:.1f})")
            else:
                length_score = 2
                print(f"⚠ 线段长度: {length_score}/4 (平均: {avg_length:.1f})")
        else:
            length_score = 0
            print(f"✗ 线段长度: {length_score}/4 (无线段数据)")
        
        score += length_score
        
        # 线段方向分布
        if segments:
            up_segments = len([s for s in segments if s.get('direction') == 'up'])
            down_segments = len([s for s in segments if s.get('direction') == 'down'])
            balance_ratio = min(up_segments, down_segments) / max(up_segments, down_segments) if max(up_segments, down_segments) > 0 else 0
            
            if balance_ratio >= 0.6:
                balance_score = 4
                print(f"✓ 线段方向平衡: {balance_score}/4 (比例: {balance_ratio:.2f})")
            elif balance_ratio >= 0.4:
                balance_score = 3
                print(f"✓ 线段方向平衡: {balance_score}/4 (比例: {balance_ratio:.2f})")
            else:
                balance_score = 2
                print(f"⚠ 线段方向平衡: {balance_score}/4 (比例: {balance_ratio:.2f})")
        else:
            balance_score = 0
            print(f"✗ 线段方向平衡: {balance_score}/4 (无线段数据)")
        
        score += balance_score
        
        # 线段强度分析
        if segments:
            strengths = []
            for s in segments:
                high = s.get('high', 0)
                low = s.get('low', 0)
                if high > 0 and low > 0:
                    strength = (high - low) / low
                    strengths.append(strength)
            
            if strengths:
                avg_strength = np.mean(strengths)
                if 0.05 <= avg_strength <= 0.20:
                    strength_score = 3
                    print(f"✓ 线段强度: {strength_score}/3 (平均: {avg_strength:.3f})")
                elif 0.02 <= avg_strength <= 0.30:
                    strength_score = 2
                    print(f"✓ 线段强度: {strength_score}/3 (平均: {avg_strength:.3f})")
                else:
                    strength_score = 1
                    print(f"⚠ 线段强度: {strength_score}/3 (平均: {avg_strength:.3f})")
            else:
                strength_score = 0
                print(f"✗ 线段强度: {strength_score}/3 (无法计算)")
        else:
            strength_score = 0
            print(f"✗ 线段强度: {strength_score}/3 (无线段数据)")
        
        score += strength_score
        
        self.scores['segment_analysis'] = score
        print(f"线段分析总分: {score}/{max_score}")
        return score
    
    def _evaluate_zhongshu_detection(self, results):
        """中枢检测准确性评估 (15分)"""
        print("\n【中枢检测准确性评估】")
        print("-" * 30)
        
        score = 0
        max_score = 15
        
        zhongshus = results.get('zhongshus', [])
        total_zhongshus = len(zhongshus)
        
        # 中枢数量合理性
        if 5 <= total_zhongshus <= 20:
            count_score = 4
            print(f"✓ 中枢数量: {count_score}/4 ({total_zhongshus}个)")
        elif 3 <= total_zhongshus <= 30:
            count_score = 3
            print(f"✓ 中枢数量: {count_score}/4 ({total_zhongshus}个)")
        else:
            count_score = 2
            print(f"⚠ 中枢数量: {count_score}/4 ({total_zhongshus}个)")
        
        score += count_score
        
        # 中枢宽度合理性
        if zhongshus:
            widths = []
            for z in zhongshus:
                upper = z.get('upper', 0)
                lower = z.get('lower', 0)
                if upper > lower:
                    width = (upper - lower) / lower
                    widths.append(width)
            
            if widths:
                avg_width = np.mean(widths)
                if 0.01 <= avg_width <= 0.08:
                    width_score = 4
                    print(f"✓ 中枢宽度: {width_score}/4 (平均: {avg_width:.3f})")
                elif 0.005 <= avg_width <= 0.15:
                    width_score = 3
                    print(f"✓ 中枢宽度: {width_score}/4 (平均: {avg_width:.3f})")
                else:
                    width_score = 2
                    print(f"⚠ 中枢宽度: {width_score}/4 (平均: {avg_width:.3f})")
            else:
                width_score = 0
                print(f"✗ 中枢宽度: {width_score}/4 (无法计算)")
        else:
            width_score = 0
            print(f"✗ 中枢宽度: {width_score}/4 (无中枢数据)")
        
        score += width_score
        
        # 中枢长度合理性
        if zhongshus:
            lengths = [z.get('end_idx', 0) - z.get('start_idx', 0) for z in zhongshus]
            avg_length = np.mean(lengths)
            
            if 10 <= avg_length <= 50:
                length_score = 4
                print(f"✓ 中枢长度: {length_score}/4 (平均: {avg_length:.1f})")
            elif 5 <= avg_length <= 80:
                length_score = 3
                print(f"✓ 中枢长度: {length_score}/4 (平均: {avg_length:.1f})")
            else:
                length_score = 2
                print(f"⚠ 中枢长度: {length_score}/4 (平均: {avg_length:.1f})")
        else:
            length_score = 0
            print(f"✗ 中枢长度: {length_score}/4 (无中枢数据)")
        
        score += length_score
        
        # 中枢重叠检测
        if zhongshus and len(zhongshus) > 1:
            overlaps = 0
            for i in range(len(zhongshus) - 1):
                z1 = zhongshus[i]
                z2 = zhongshus[i + 1]
                if (z1.get('end_idx', 0) >= z2.get('start_idx', 0) and
                    min(z1.get('upper', 0), z2.get('upper', 0)) > max(z1.get('lower', 0), z2.get('lower', 0))):
                    overlaps += 1
            
            overlap_ratio = overlaps / (len(zhongshus) - 1)
            if overlap_ratio <= 0.3:
                overlap_score = 3
                print(f"✓ 中枢重叠控制: {overlap_score}/3 (重叠率: {overlap_ratio:.2f})")
            elif overlap_ratio <= 0.5:
                overlap_score = 2
                print(f"✓ 中枢重叠控制: {overlap_score}/3 (重叠率: {overlap_ratio:.2f})")
            else:
                overlap_score = 1
                print(f"⚠ 中枢重叠控制: {overlap_score}/3 (重叠率: {overlap_ratio:.2f})")
        else:
            overlap_score = 0
            print(f"✗ 中枢重叠控制: {overlap_score}/3 (无足够中枢数据)")
        
        score += overlap_score
        
        self.scores['zhongshu_detection'] = score
        print(f"中枢检测总分: {score}/{max_score}")
        return score
    
    def _evaluate_divergence_signals(self, results):
        """背驰信号可靠性评估 (10分)"""
        print("\n【背驰信号可靠性评估】")
        print("-" * 30)
        
        score = 0
        max_score = 10
        
        divergences = results.get('divergences', [])
        total_divergences = len(divergences)
        
        # 背驰信号数量
        if 3 <= total_divergences <= 15:
            count_score = 3
            print(f"✓ 背驰信号数量: {count_score}/3 ({total_divergences}个)")
        elif 1 <= total_divergences <= 25:
            count_score = 2
            print(f"✓ 背驰信号数量: {count_score}/3 ({total_divergences}个)")
        else:
            count_score = 1
            print(f"⚠ 背驰信号数量: {count_score}/3 ({total_divergences}个)")
        
        score += count_score
        
        # 背驰信号平衡性
        if divergences:
            bear_divs = len([d for d in divergences if d.get('type') == 'bear_div'])
            bull_divs = len([d for d in divergences if d.get('type') == 'bull_div'])
            balance_ratio = min(bear_divs, bull_divs) / max(bear_divs, bull_divs) if max(bear_divs, bull_divs) > 0 else 0
            
            if balance_ratio >= 0.5:
                balance_score = 3
                print(f"✓ 背驰信号平衡: {balance_score}/3 (比例: {balance_ratio:.2f})")
            elif balance_ratio >= 0.3:
                balance_score = 2
                print(f"✓ 背驰信号平衡: {balance_score}/3 (比例: {balance_ratio:.2f})")
            else:
                balance_score = 1
                print(f"⚠ 背驰信号平衡: {balance_score}/3 (比例: {balance_ratio:.2f})")
        else:
            balance_score = 0
            print(f"✗ 背驰信号平衡: {balance_score}/3 (无背驰数据)")
        
        score += balance_score
        
        # 背驰信号分布
        if divergences and len(divergences) > 1:
            indices = [d.get('idx', 0) for d in divergences]
            intervals = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
            avg_interval = np.mean(intervals)
            
            if 20 <= avg_interval <= 100:
                distribution_score = 4
                print(f"✓ 背驰信号分布: {distribution_score}/4 (平均间隔: {avg_interval:.1f})")
            elif 10 <= avg_interval <= 150:
                distribution_score = 3
                print(f"✓ 背驰信号分布: {distribution_score}/4 (平均间隔: {avg_interval:.1f})")
            else:
                distribution_score = 2
                print(f"⚠ 背驰信号分布: {distribution_score}/4 (平均间隔: {avg_interval:.1f})")
        else:
            distribution_score = 0
            print(f"✗ 背驰信号分布: {distribution_score}/4 (无足够背驰数据)")
        
        score += distribution_score
        
        self.scores['divergence_signals'] = score
        print(f"背驰信号总分: {score}/{max_score}")
        return score
    
    def _evaluate_trading_signals(self, results):
        """交易信号实用性评估 (5分)"""
        print("\n【交易信号实用性评估】")
        print("-" * 30)
        
        score = 0
        max_score = 5
        
        buy_signals = results.get('buy_signals', [])
        sell_signals = results.get('sell_signals', [])
        
        # 交易信号数量
        total_signals = len(buy_signals) + len(sell_signals)
        if 1 <= total_signals <= 5:
            count_score = 2
            print(f"✓ 交易信号数量: {count_score}/2 ({total_signals}个)")
        elif total_signals > 5:
            count_score = 1
            print(f"⚠ 交易信号数量: {count_score}/2 ({total_signals}个)")
        else:
            count_score = 0
            print(f"✗ 交易信号数量: {count_score}/2 ({total_signals}个)")
        
        score += count_score
        
        # 交易信号平衡性
        if buy_signals and sell_signals:
            balance_ratio = min(len(buy_signals), len(sell_signals)) / max(len(buy_signals), len(sell_signals))
            if balance_ratio >= 0.5:
                balance_score = 2
                print(f"✓ 交易信号平衡: {balance_score}/2 (比例: {balance_ratio:.2f})")
            else:
                balance_score = 1
                print(f"⚠ 交易信号平衡: {balance_score}/2 (比例: {balance_ratio:.2f})")
        elif buy_signals or sell_signals:
            balance_score = 1
            print(f"⚠ 交易信号平衡: {balance_score}/2 (只有单向信号)")
        else:
            balance_score = 0
            print(f"✗ 交易信号平衡: {balance_score}/2 (无交易信号)")
        
        score += balance_score
        
        # 风险控制
        risk_control = results.get('risk_control', False)
        if risk_control:
            risk_score = 1
            print(f"✓ 风险控制: {risk_score}/1 (已设置止损)")
        else:
            risk_score = 0
            print(f"✗ 风险控制: {risk_score}/1 (未设置止损)")
        
        score += risk_score
        
        self.scores['trading_signals'] = score
        print(f"交易信号总分: {score}/{max_score}")
        return score
    
    def _generate_evaluation_report(self):
        """生成评估报告"""
        print("\n" + "=" * 60)
        print("缠论分析质量评估总结")
        print("=" * 60)
        
        # 各维度得分
        print("\n【各维度得分】")
        print("-" * 30)
        for category, score in self.scores.items():
            category_name = {
                'data_quality': '数据质量',
                'fractal_accuracy': '分型识别',
                'stroke_quality': '笔构建',
                'segment_analysis': '线段分析',
                'zhongshu_detection': '中枢检测',
                'divergence_signals': '背驰信号',
                'trading_signals': '交易信号'
            }.get(category, category)
            
            max_scores = {
                'data_quality': 20,
                'fractal_accuracy': 20,
                'stroke_quality': 15,
                'segment_analysis': 15,
                'zhongshu_detection': 15,
                'divergence_signals': 10,
                'trading_signals': 5
            }.get(category, 10)
            
            percentage = (score / max_scores) * 100
            print(f"{category_name}: {score}/{max_scores} ({percentage:.1f}%)")
        
        # 总分和评级
        print(f"\n【总体评估】")
        print("-" * 30)
        print(f"总分: {self.total_score}/{self.max_score}")
        
        percentage = (self.total_score / self.max_score) * 100
        print(f"得分率: {percentage:.1f}%")
        
        # 评级
        if percentage >= 90:
            grade = "优秀 (A+)"
            comment = "分析质量极高，各项指标表现优异"
        elif percentage >= 80:
            grade = "良好 (A)"
            comment = "分析质量良好，大部分指标表现优秀"
        elif percentage >= 70:
            grade = "中等 (B)"
            comment = "分析质量中等，部分指标需要改进"
        elif percentage >= 60:
            grade = "及格 (C)"
            comment = "分析质量及格，多项指标需要优化"
        else:
            grade = "不及格 (D)"
            comment = "分析质量较差，需要全面改进"
        
        print(f"评级: {grade}")
        print(f"评价: {comment}")
        
        # 改进建议
        print(f"\n【改进建议】")
        print("-" * 30)
        
        suggestions = []
        for category, score in self.scores.items():
            max_scores = {
                'data_quality': 20,
                'fractal_accuracy': 20,
                'stroke_quality': 15,
                'segment_analysis': 15,
                'zhongshu_detection': 15,
                'divergence_signals': 10,
                'trading_signals': 5
            }.get(category, 10)
            
            if score < max_scores * 0.7:
                category_name = {
                    'data_quality': '数据质量',
                    'fractal_accuracy': '分型识别',
                    'stroke_quality': '笔构建',
                    'segment_analysis': '线段分析',
                    'zhongshu_detection': '中枢检测',
                    'divergence_signals': '背驰信号',
                    'trading_signals': '交易信号'
                }.get(category, category)
                suggestions.append(f"- 改进{category_name}分析")
        
        if suggestions:
            for suggestion in suggestions:
                print(suggestion)
        else:
            print("- 整体分析质量良好，继续保持")
        
        print("\n" + "=" * 60)

def main():
    """主函数"""
    # 基于之前的分析结果进行评估
    analysis_results = {
        'total_k_lines': 501,
        'time_span_days': 365,
        'price_range': 0.3,
        'technical_indicators': 8,
        'fractals': [
            {'kind': 'top', 'strength': 0.02},
            {'kind': 'bottom', 'strength': 0.015}
        ] * 56,  # 模拟113个分型
        'strokes': [
            {'direction': 'up', 'start_idx': 10, 'end_idx': 20, 'swing': 0.04},
            {'direction': 'down', 'start_idx': 20, 'end_idx': 30, 'swing': 0.03}
        ] * 23,  # 模拟46支笔
        'segments': [
            {'direction': 'up', 'start_idx': 10, 'end_idx': 30, 'high': 255.0, 'low': 245.0}
        ] * 5,  # 模拟10个线段
        'zhongshus': [
            {'start_idx': 15, 'end_idx': 25, 'upper': 255.0, 'lower': 245.0}
        ] * 11,  # 模拟11个中枢
        'divergences': [
            {'idx': 20, 'type': 'bull_div'},
            {'idx': 30, 'type': 'bear_div'}
        ] * 3,  # 模拟6个背驰
        'buy_signals': [
            {'price': 225.32, 'reason': '底背驰信号', 'stop_loss': 220.81, 'target': 236.59},
            {'price': 251.31, 'reason': '中枢向上突破', 'stop_loss': 228.06, 'target': 258.85}
        ],
        'sell_signals': [
            {'price': 258.10, 'reason': '顶背驰信号', 'stop_loss': 263.27, 'target': 245.20}
        ],
        'risk_control': True
    }
    
    # 创建评估器并执行评估
    evaluator = ChanAnalysisEvaluator()
    total_score = evaluator.evaluate_analysis_quality(analysis_results)
    
    return total_score

if __name__ == "__main__":
    main()
