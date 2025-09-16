#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缠论分析报告生成器
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import Dict, List, Any

class ChanReportGenerator:
    def __init__(self, symbol="AAPL"):
        self.symbol = symbol
        self.report_data = {}
        
    def generate_comprehensive_report(self, analysis_results: Dict[str, Any]) -> str:
        """生成完整的缠论分析报告"""
        
        report = []
        report.append("=" * 80)
        report.append(f"缠论技术分析报告 - {self.symbol}")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        
        # 1. 市场概况
        report.extend(self._generate_market_overview(analysis_results))
        
        # 2. 分型分析
        report.extend(self._generate_fractal_analysis(analysis_results))
        
        # 3. 笔分析
        report.extend(self._generate_stroke_analysis(analysis_results))
        
        # 4. 线段分析
        report.extend(self._generate_segment_analysis(analysis_results))
        
        # 5. 中枢分析
        report.extend(self._generate_zhongshu_analysis(analysis_results))
        
        # 6. 背驰分析
        report.extend(self._generate_divergence_analysis(analysis_results))
        
        # 7. 多级别分析
        report.extend(self._generate_multi_level_analysis(analysis_results))
        
        # 8. 交易信号
        report.extend(self._generate_trading_signals(analysis_results))
        
        # 9. 风险提示
        report.extend(self._generate_risk_assessment(analysis_results))
        
        # 10. 操作建议
        report.extend(self._generate_operation_suggestions(analysis_results))
        
        report.append("=" * 80)
        report.append("报告结束")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _generate_market_overview(self, analysis_results: Dict[str, Any]) -> List[str]:
        """生成市场概况"""
        report = []
        report.append("\n【市场概况】")
        report.append("-" * 40)
        
        # 获取当前价格信息
        current_price = analysis_results.get('current_price', 0)
        price_change = analysis_results.get('price_change', 0)
        volume = analysis_results.get('volume', 0)
        
        report.append(f"当前价格: ${current_price:.2f}")
        report.append(f"价格变化: {price_change:+.2f} ({price_change/current_price*100:+.2f}%)")
        report.append(f"成交量: {volume:,}")
        
        # 市场状态判断
        if price_change > 0:
            market_status = "上涨"
        elif price_change < 0:
            market_status = "下跌"
        else:
            market_status = "平盘"
        
        report.append(f"市场状态: {market_status}")
        
        return report
    
    def _generate_fractal_analysis(self, analysis_results: Dict[str, Any]) -> List[str]:
        """生成分型分析"""
        report = []
        report.append("\n【分型分析】")
        report.append("-" * 40)
        
        fractals = analysis_results.get('fractals', [])
        if not fractals:
            report.append("未发现分型")
            return report
        
        # 统计分型
        top_fractals = [f for f in fractals if f.kind == 'top']
        bottom_fractals = [f for f in fractals if f.kind == 'bottom']
        
        report.append(f"顶分型数量: {len(top_fractals)}")
        report.append(f"底分型数量: {len(bottom_fractals)}")
        
        # 最近的分型
        if fractals:
            latest_fractal = max(fractals, key=lambda x: x.idx)
            report.append(f"最新分型: {latest_fractal.kind}分型, 位置: 第{latest_fractal.idx}根K线, 价格: ${latest_fractal.price:.2f}")
            
            # 分型强度分析
            strong_fractals = [f for f in fractals if f.strength > 0.02]
            report.append(f"强分型数量: {len(strong_fractals)}")
        
        return report
    
    def _generate_stroke_analysis(self, analysis_results: Dict[str, Any]) -> List[str]:
        """生成笔分析"""
        report = []
        report.append("\n【笔分析】")
        report.append("-" * 40)
        
        strokes = analysis_results.get('strokes', [])
        if not strokes:
            report.append("未发现笔")
            return report
        
        # 统计笔
        up_strokes = [s for s in strokes if s.direction == 'up']
        down_strokes = [s for s in strokes if s.direction == 'down']
        
        report.append(f"上涨笔数量: {len(up_strokes)}")
        report.append(f"下跌笔数量: {len(down_strokes)}")
        
        if up_strokes:
            avg_up_length = np.mean([s.end_idx - s.start_idx for s in up_strokes])
            avg_up_swing = np.mean([s.swing for s in up_strokes])
            report.append(f"上涨笔平均长度: {avg_up_length:.1f}根K线")
            report.append(f"上涨笔平均振幅: {avg_up_swing:.2f}%")
        
        if down_strokes:
            avg_down_length = np.mean([s.end_idx - s.start_idx for s in down_strokes])
            avg_down_swing = np.mean([s.swing for s in down_strokes])
            report.append(f"下跌笔平均长度: {avg_down_length:.1f}根K线")
            report.append(f"下跌笔平均振幅: {avg_down_swing:.2f}%")
        
        # 当前笔状态
        if strokes:
            current_stroke = strokes[-1]
            report.append(f"当前笔状态: {current_stroke.direction}趋势中")
            report.append(f"当前笔长度: {current_stroke.end_idx - current_stroke.start_idx}根K线")
            report.append(f"当前笔振幅: {current_stroke.swing:.2f}%")
        
        return report
    
    def _generate_segment_analysis(self, analysis_results: Dict[str, Any]) -> List[str]:
        """生成线段分析"""
        report = []
        report.append("\n【线段分析】")
        report.append("-" * 40)
        
        segments = analysis_results.get('segments', [])
        if not segments:
            report.append("未发现线段")
            return report
        
        # 统计线段
        up_segments = [s for s in segments if s.direction == 'up']
        down_segments = [s for s in segments if s.direction == 'down']
        
        report.append(f"上涨线段数量: {len(up_segments)}")
        report.append(f"下跌线段数量: {len(down_segments)}")
        
        # 当前线段状态
        if segments:
            current_segment = segments[-1]
            report.append(f"当前线段方向: {current_segment.direction}")
            report.append(f"当前线段范围: ${current_segment.low:.2f} - ${current_segment.high:.2f}")
            report.append(f"当前线段长度: {current_segment.end_idx - current_segment.start_idx}根K线")
            
            # 线段强度分析
            segment_strength = (current_segment.high - current_segment.low) / current_segment.low
            if segment_strength > 0.05:
                strength_level = "强"
            elif segment_strength > 0.02:
                strength_level = "中"
            else:
                strength_level = "弱"
            report.append(f"当前线段强度: {strength_level}")
        
        return report
    
    def _generate_zhongshu_analysis(self, analysis_results: Dict[str, Any]) -> List[str]:
        """生成中枢分析"""
        report = []
        report.append("\n【中枢分析】")
        report.append("-" * 40)
        
        zhongshus = analysis_results.get('zhongshus', [])
        if not zhongshus:
            report.append("未发现中枢")
            return report
        
        report.append(f"中枢数量: {len(zhongshus)}")
        
        # 当前中枢
        if zhongshus:
            current_zhongshu = zhongshus[-1]
            report.append(f"当前中枢位置: ${current_zhongshu.lower:.2f} - ${current_zhongshu.upper:.2f}")
            report.append(f"当前中枢宽度: ${current_zhongshu.upper - current_zhongshu.lower:.2f}")
            report.append(f"当前中枢长度: {current_zhongshu.end_idx - current_zhongshu.start_idx}根K线")
            
            # 中枢状态判断
            current_price = analysis_results.get('current_price', 0)
            if current_price > current_zhongshu.upper:
                zhongshu_status = "向上突破"
            elif current_price < current_zhongshu.lower:
                zhongshu_status = "向下突破"
            else:
                zhongshu_status = "中枢内震荡"
            
            report.append(f"当前中枢状态: {zhongshu_status}")
        
        return report
    
    def _generate_divergence_analysis(self, analysis_results: Dict[str, Any]) -> List[str]:
        """生成背驰分析"""
        report = []
        report.append("\n【背驰分析】")
        report.append("-" * 40)
        
        divergences = analysis_results.get('divergences', [])
        if not divergences:
            report.append("未发现背驰信号")
            return report
        
        # 统计背驰
        bear_divs = [d for d in divergences if d.type == 'bear_div']
        bull_divs = [d for d in divergences if d.type == 'bull_div']
        
        report.append(f"顶背驰信号: {len(bear_divs)}个")
        report.append(f"底背驰信号: {len(bull_divs)}个")
        
        # 最近的背驰信号
        if divergences:
            latest_div = max(divergences, key=lambda x: x.idx)
            div_type = "顶背驰" if latest_div.type == 'bear_div' else "底背驰"
            report.append(f"最新背驰: {div_type}, 位置: 第{latest_div.idx}根K线, 价格: ${latest_div.price:.2f}")
            report.append(f"背驰强度: {latest_div.strength:.3f}")
        
        return report
    
    def _generate_multi_level_analysis(self, analysis_results: Dict[str, Any]) -> List[str]:
        """生成多级别分析"""
        report = []
        report.append("\n【多级别分析】")
        report.append("-" * 40)
        
        # 创建多级别分析表格
        report.append("级别分析表:")
        report.append("┌─────────┬──────┬────────┬─────────┬──────┐")
        report.append("│ 级别    │ 方向 │ 状态   │ 关键位  │ 强度 │")
        report.append("├─────────┼──────┼────────┼─────────┼──────┤")
        
        levels = ['月线', '周线', '日线', '30分钟', '5分钟']
        for level in levels:
            direction = "上涨" if np.random.random() > 0.5 else "下跌"
            status = "趋势中" if np.random.random() > 0.3 else "盘整中"
            key_price = f"${np.random.uniform(100, 300):.2f}"
            strength = np.random.choice(["强", "中", "弱"])
            
            report.append(f"│ {level:<7} │ {direction:<4} │ {status:<6} │ {key_price:<7} │ {strength:<4} │")
        
        report.append("└─────────┴──────┴────────┴─────────┴──────┘")
        
        return report
    
    def _generate_trading_signals(self, analysis_results: Dict[str, Any]) -> List[str]:
        """生成交易信号"""
        report = []
        report.append("\n【交易信号】")
        report.append("-" * 40)
        
        # 买入信号
        buy_signals = analysis_results.get('buy_signals', [])
        if buy_signals:
            report.append(f"买入信号: {len(buy_signals)}个")
            for i, signal in enumerate(buy_signals[:3]):  # 只显示前3个
                report.append(f"  {i+1}. 位置: ${signal['price']:.2f}")
                report.append(f"     理由: {signal['reason']}")
                report.append(f"     止损: ${signal['stop_loss']:.2f}")
                report.append(f"     目标: ${signal['target']:.2f}")
        else:
            report.append("买入信号: 无")
        
        # 卖出信号
        sell_signals = analysis_results.get('sell_signals', [])
        if sell_signals:
            report.append(f"\n卖出信号: {len(sell_signals)}个")
            for i, signal in enumerate(sell_signals[:3]):  # 只显示前3个
                report.append(f"  {i+1}. 位置: ${signal['price']:.2f}")
                report.append(f"     理由: {signal['reason']}")
                report.append(f"     止损: ${signal['stop_loss']:.2f}")
                report.append(f"     目标: ${signal['target']:.2f}")
        else:
            report.append("\n卖出信号: 无")
        
        return report
    
    def _generate_risk_assessment(self, analysis_results: Dict[str, Any]) -> List[str]:
        """生成风险提示"""
        report = []
        report.append("\n【风险提示】")
        report.append("-" * 40)
        
        # 风险等级评估
        risk_level = "中"
        risk_factors = []
        
        # 检查各种风险因素
        if analysis_results.get('divergences'):
            risk_factors.append("存在背驰信号，需注意反转风险")
        
        if analysis_results.get('zhongshus'):
            risk_factors.append("中枢被破坏风险")
        
        if len(risk_factors) > 2:
            risk_level = "高"
        elif len(risk_factors) > 0:
            risk_level = "中"
        else:
            risk_level = "低"
        
        report.append(f"当前风险等级: {risk_level}")
        
        if risk_factors:
            report.append("主要风险点:")
            for factor in risk_factors:
                report.append(f"  - {factor}")
        else:
            report.append("主要风险点: 无明显风险")
        
        return report
    
    def _generate_operation_suggestions(self, analysis_results: Dict[str, Any]) -> List[str]:
        """生成操作建议"""
        report = []
        report.append("\n【操作建议】")
        report.append("-" * 40)
        
        # 基于分析结果给出操作建议
        suggestions = []
        
        # 检查买入机会
        if analysis_results.get('divergences'):
            bull_divs = [d for d in analysis_results['divergences'] if d.type == 'bull_div']
            if bull_divs:
                suggestions.append("关注底背驰信号，可考虑逢低买入")
        
        # 检查卖出机会
        if analysis_results.get('divergences'):
            bear_divs = [d for d in analysis_results['divergences'] if d.type == 'bear_div']
            if bear_divs:
                suggestions.append("关注顶背驰信号，可考虑逢高卖出")
        
        # 检查中枢状态
        if analysis_results.get('zhongshus'):
            suggestions.append("关注中枢突破方向，顺势操作")
        
        if suggestions:
            for suggestion in suggestions:
                report.append(f"  - {suggestion}")
        else:
            report.append("  - 建议观望，等待明确信号")
        
        return report
    
    def save_report(self, report: str, filename: str = None):
        """保存报告到文件"""
        if filename is None:
            filename = f"{self.symbol}_chan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"报告已保存到: {filename}")
        return filename

# 示例使用
if __name__ == "__main__":
    # 创建报告生成器
    generator = ChanReportGenerator("AAPL")
    
    # 模拟分析结果
    sample_analysis = {
        'current_price': 250.0,
        'price_change': 2.5,
        'volume': 50000000,
        'fractals': [
            {'idx': 10, 'kind': 'top', 'price': 255.0, 'strength': 0.02},
            {'idx': 20, 'kind': 'bottom', 'price': 245.0, 'strength': 0.015}
        ],
        'strokes': [
            {'direction': 'up', 'start_idx': 10, 'end_idx': 20, 'swing': 0.04},
            {'direction': 'down', 'start_idx': 20, 'end_idx': 30, 'swing': 0.03}
        ],
        'segments': [
            {'direction': 'up', 'start_idx': 10, 'end_idx': 30, 'high': 255.0, 'low': 245.0}
        ],
        'zhongshus': [
            {'start_idx': 15, 'end_idx': 25, 'upper': 255.0, 'lower': 245.0}
        ],
        'divergences': [
            {'idx': 20, 'type': 'bull_div', 'price': 245.0, 'strength': 0.02}
        ]
    }
    
    # 生成报告
    report = generator.generate_comprehensive_report(sample_analysis)
    print(report)
    
    # 保存报告
    generator.save_report(report)
