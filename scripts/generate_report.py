#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成缠论分析报告
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_fixed import load_ohlc, find_fractals, build_strokes, build_segments, detect_zhongshu, detect_divergence, resolve_inclusion
import pandas as pd
import numpy as np
from datetime import datetime

def generate_chan_report(csv_path: str, output_path: str = None):
    """生成缠论分析报告"""
    
    if output_path is None:
        output_path = f"reports/chan_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    print("正在生成缠论分析报告...")
    
    # 加载数据
    df = load_ohlc(csv_path)
    df_s = resolve_inclusion(df)
    df_use = df.copy()
    if "High_smooth" in df_s.columns:
        df_use["High"] = df_s["High_smooth"]
        df_use["Low"] = df_s["Low_smooth"]
    
    # 执行分析
    frs = find_fractals(df_use)
    strokes = build_strokes(df_use, frs)
    segs = build_segments(strokes)
    zses = detect_zhongshu(strokes)
    divs = detect_divergence(df_use, strokes)
    
    # 生成报告
    report = []
    report.append("=" * 80)
    report.append("缠论技术分析报告 - AAPL")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    
    # 市场概况
    current_price = df["Close"].iloc[-1]
    price_change = df["Close"].iloc[-1] - df["Close"].iloc[-2]
    price_change_pct = (price_change / df["Close"].iloc[-2]) * 100
    volume = df["Volume"].iloc[-1] if "Volume" in df.columns else 0
    
    report.append("\n【市场概况】")
    report.append("-" * 40)
    report.append(f"当前价格: ${current_price:.2f}")
    report.append(f"价格变化: {price_change:+.2f} ({price_change_pct:+.2f}%)")
    report.append(f"成交量: {volume:,}")
    report.append(f"数据范围: {df['Date'].iloc[0].strftime('%Y-%m-%d')} 至 {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    report.append(f"总K线数: {len(df)}")
    
    # 分型分析
    report.append("\n【分型分析】")
    report.append("-" * 40)
    top_fractals = [f for f in frs if f.kind == 'top']
    bottom_fractals = [f for f in frs if f.kind == 'bottom']
    
    report.append(f"顶分型数量: {len(top_fractals)}")
    report.append(f"底分型数量: {len(bottom_fractals)}")
    report.append(f"总分型数量: {len(frs)}")
    
    if frs:
        latest_fractal = max(frs, key=lambda x: x.idx)
        report.append(f"最新分型: {latest_fractal.kind}分型")
        report.append(f"最新分型位置: 第{latest_fractal.idx}根K线")
        report.append(f"最新分型价格: ${latest_fractal.price:.2f}")
        
        # 最近5个分型
        recent_fractals = sorted(frs, key=lambda x: x.idx)[-5:]
        report.append("\n最近5个分型:")
        for i, f in enumerate(recent_fractals, 1):
            report.append(f"  {i}. {f.kind}分型 - 第{f.idx}根K线 - ${f.price:.2f}")
    
    # 笔分析
    report.append("\n【笔分析】")
    report.append("-" * 40)
    up_strokes = [s for s in strokes if s.direction == 'up']
    down_strokes = [s for s in strokes if s.direction == 'down']
    
    report.append(f"上涨笔数量: {len(up_strokes)}")
    report.append(f"下跌笔数量: {len(down_strokes)}")
    report.append(f"总笔数: {len(strokes)}")
    
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
    
    if strokes:
        current_stroke = strokes[-1]
        report.append(f"\n当前笔状态: {current_stroke.direction}趋势中")
        report.append(f"当前笔长度: {current_stroke.end_idx - current_stroke.start_idx}根K线")
        report.append(f"当前笔振幅: {current_stroke.swing:.2f}%")
        report.append(f"当前笔价格范围: ${min(current_stroke.start_price, current_stroke.end_price):.2f} - ${max(current_stroke.start_price, current_stroke.end_price):.2f}")
    
    # 线段分析
    report.append("\n【线段分析】")
    report.append("-" * 40)
    up_segments = [s for s in segs if s.direction == 'up']
    down_segments = [s for s in segs if s.direction == 'down']
    
    report.append(f"上涨线段数量: {len(up_segments)}")
    report.append(f"下跌线段数量: {len(down_segments)}")
    report.append(f"总线段数: {len(segs)}")
    
    if segs:
        current_segment = segs[-1]
        report.append(f"\n当前线段方向: {current_segment.direction}")
        report.append(f"当前线段范围: ${current_segment.low:.2f} - ${current_segment.high:.2f}")
        report.append(f"当前线段长度: {current_segment.end_idx - current_segment.start_idx}根K线")
        
        segment_strength = (current_segment.high - current_segment.low) / current_segment.low
        if segment_strength > 0.05:
            strength_level = "强"
        elif segment_strength > 0.02:
            strength_level = "中"
        else:
            strength_level = "弱"
        report.append(f"当前线段强度: {strength_level}")
    
    # 中枢分析
    report.append("\n【中枢分析】")
    report.append("-" * 40)
    report.append(f"中枢数量: {len(zses)}")
    
    if zses:
        current_zhongshu = zses[-1]
        report.append(f"当前中枢位置: ${current_zhongshu.lower:.2f} - ${current_zhongshu.upper:.2f}")
        report.append(f"当前中枢宽度: ${current_zhongshu.upper - current_zhongshu.lower:.2f}")
        report.append(f"当前中枢长度: {current_zhongshu.end_idx - current_zhongshu.start_idx}根K线")
        
        # 中枢状态判断
        if current_price > current_zhongshu.upper:
            zhongshu_status = "向上突破"
        elif current_price < current_zhongshu.lower:
            zhongshu_status = "向下突破"
        else:
            zhongshu_status = "中枢内震荡"
        
        report.append(f"当前中枢状态: {zhongshu_status}")
        
        # 最近3个中枢
        recent_zhongshus = zses[-3:] if len(zses) >= 3 else zses
        report.append(f"\n最近{len(recent_zhongshus)}个中枢:")
        for i, z in enumerate(recent_zhongshus, 1):
            report.append(f"  {i}. ${z.lower:.2f} - ${z.upper:.2f} (长度: {z.end_idx - z.start_idx}根K线)")
    
    # 背驰分析
    report.append("\n【背驰分析】")
    report.append("-" * 40)
    bear_divs = [d for d in divs if d[1] == 'bear_div']
    bull_divs = [d for d in divs if d[1] == 'bull_div']
    
    report.append(f"顶背驰信号: {len(bear_divs)}个")
    report.append(f"底背驰信号: {len(bull_divs)}个")
    report.append(f"总背驰信号: {len(divs)}个")
    
    if divs:
        latest_div = max(divs, key=lambda x: x[0])
        div_type = "顶背驰" if latest_div[1] == 'bear_div' else "底背驰"
        div_price = df["Close"].iloc[latest_div[0]]
        report.append(f"\n最新背驰: {div_type}")
        report.append(f"背驰位置: 第{latest_div[0]}根K线")
        report.append(f"背驰价格: ${div_price:.2f}")
        
        # 最近3个背驰
        recent_divs = sorted(divs, key=lambda x: x[0])[-3:]
        report.append(f"\n最近{len(recent_divs)}个背驰信号:")
        for i, (idx, div_type) in enumerate(recent_divs, 1):
            price = df["Close"].iloc[idx]
            type_name = "顶背驰" if div_type == 'bear_div' else "底背驰"
            report.append(f"  {i}. {type_name} - 第{idx}根K线 - ${price:.2f}")
    
    # 交易信号
    report.append("\n【交易信号】")
    report.append("-" * 40)
    
    # 基于分析结果生成交易信号
    buy_signals = []
    sell_signals = []
    
    # 检查底背驰买入信号
    if bull_divs:
        latest_bull_div = max(bull_divs, key=lambda x: x[0])
        buy_signals.append({
            'price': df["Close"].iloc[latest_bull_div[0]],
            'reason': '底背驰信号',
            'stop_loss': df["Close"].iloc[latest_bull_div[0]] * 0.98,
            'target': df["Close"].iloc[latest_bull_div[0]] * 1.05
        })
    
    # 检查中枢突破信号
    if zses and current_price > current_zhongshu.upper:
        buy_signals.append({
            'price': current_price,
            'reason': '中枢向上突破',
            'stop_loss': current_zhongshu.upper,
            'target': current_price * 1.03
        })
    
    # 检查顶背驰卖出信号
    if bear_divs:
        latest_bear_div = max(bear_divs, key=lambda x: x[0])
        sell_signals.append({
            'price': df["Close"].iloc[latest_bear_div[0]],
            'reason': '顶背驰信号',
            'stop_loss': df["Close"].iloc[latest_bear_div[0]] * 1.02,
            'target': df["Close"].iloc[latest_bear_div[0]] * 0.95
        })
    
    if buy_signals:
        report.append(f"买入信号: {len(buy_signals)}个")
        for i, signal in enumerate(buy_signals, 1):
            report.append(f"  {i}. 位置: ${signal['price']:.2f}")
            report.append(f"     理由: {signal['reason']}")
            report.append(f"     止损: ${signal['stop_loss']:.2f}")
            report.append(f"     目标: ${signal['target']:.2f}")
    else:
        report.append("买入信号: 无")
    
    if sell_signals:
        report.append(f"\n卖出信号: {len(sell_signals)}个")
        for i, signal in enumerate(sell_signals, 1):
            report.append(f"  {i}. 位置: ${signal['price']:.2f}")
            report.append(f"     理由: {signal['reason']}")
            report.append(f"     止损: ${signal['stop_loss']:.2f}")
            report.append(f"     目标: ${signal['target']:.2f}")
    else:
        report.append("\n卖出信号: 无")
    
    # 风险提示
    report.append("\n【风险提示】")
    report.append("-" * 40)
    
    risk_factors = []
    risk_level = "低"
    
    if len(divs) > 5:
        risk_factors.append("存在多个背驰信号，需注意反转风险")
        risk_level = "中"
    
    if zses and current_price < current_zhongshu.lower:
        risk_factors.append("价格跌破中枢下沿，存在继续下跌风险")
        risk_level = "高"
    
    if len(strokes) > 0 and strokes[-1].direction == "down":
        risk_factors.append("当前处于下跌笔中，需注意趋势延续")
        risk_level = "中"
    
    report.append(f"当前风险等级: {risk_level}")
    
    if risk_factors:
        report.append("主要风险点:")
        for factor in risk_factors:
            report.append(f"  - {factor}")
    else:
        report.append("主要风险点: 无明显风险")
    
    # 操作建议
    report.append("\n【操作建议】")
    report.append("-" * 40)
    
    suggestions = []
    
    if buy_signals:
        suggestions.append("关注买入信号，可考虑逢低买入")
    
    if sell_signals:
        suggestions.append("关注卖出信号，可考虑逢高卖出")
    
    if zses:
        suggestions.append("关注中枢突破方向，顺势操作")
    
    if len(divs) > 0:
        suggestions.append("注意背驰信号，做好风险控制")
    
    if not suggestions:
        suggestions.append("建议观望，等待明确信号")
    
    for suggestion in suggestions:
        report.append(f"  - {suggestion}")
    
    report.append("\n" + "=" * 80)
    report.append("报告结束")
    report.append("=" * 80)
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"分析报告已生成: {output_path}")
    return output_path

if __name__ == "__main__":
    # 生成报告
    report_path = generate_chan_report("data/AAPL_1d_data.csv")
    print(f"报告已保存到: {report_path}")
