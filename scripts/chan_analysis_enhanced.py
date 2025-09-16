# -*- coding: utf-8 -*-
"""
增强版缠论分析 - 包含更多必要参数
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import argparse

# ========== 缠论参数配置 ==========
@dataclass
class ChanConfig:
    """缠论分析参数配置"""
    
    # 分型参数
    fractal_window: int = 3  # 分型识别窗口（默认3根K线）
    min_fractal_gap: int = 2  # 分型间最小间隔
    
    # 笔参数
    min_stroke_bars: int = 3  # 笔的最小跨越K线数
    min_stroke_pct: float = 0.002  # 笔的最小振幅占比
    stroke_merge_threshold: float = 0.001  # 笔合并阈值
    
    # 线段参数
    min_segment_strokes: int = 3  # 线段最少笔数
    segment_break_threshold: float = 0.005  # 线段突破阈值
    
    # 中枢参数
    min_zhongshu_strokes: int = 3  # 中枢最少笔数
    zhongshu_overlap_threshold: float = 0.001  # 中枢重叠阈值
    zhongshu_merge_threshold: int = 5  # 中枢合并时间阈值
    
    # MACD参数
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # 背驰参数
    divergence_lookback: int = 2  # 背驰回看笔数
    divergence_threshold: float = 0.001  # 背驰确认阈值
    
    # 包含关系参数
    inclusion_threshold: float = 0.0005  # 包含关系阈值
    inclusion_merge: bool = True  # 是否启用包含关系合并
    
    # 多级别参数
    levels: List[str] = None  # 分析级别列表
    
    def __post_init__(self):
        if self.levels is None:
            self.levels = ['1m', '5m', '30m', '1d', '1w']

# ========== 数据结构 ==========
@dataclass
class Fractal:
    idx: int
    kind: str  # 'top' or 'bottom'
    price: float
    high: float
    low: float
    strength: float = 0.0  # 分型强度
    confirmed: bool = False  # 是否确认

@dataclass
class Stroke:
    start_idx: int
    end_idx: int
    start_price: float
    end_price: float
    direction: str  # 'up' or 'down'
    swing: float
    strength: float = 0.0  # 笔的强度
    level: str = '1d'  # 级别

@dataclass
class Segment:
    start_idx: int
    end_idx: int
    high: float
    low: float
    direction: str
    strokes: List[Stroke] = None
    level: str = '1d'
    
    def __post_init__(self):
        if self.strokes is None:
            self.strokes = []

@dataclass
class ZhongShu:
    start_idx: int
    end_idx: int
    upper: float
    lower: float
    strokes: List[Stroke] = None
    level: str = '1d'
    strength: float = 0.0
    
    def __post_init__(self):
        if self.strokes is None:
            self.strokes = []

@dataclass
class Divergence:
    idx: int
    type: str  # 'bull_div' or 'bear_div'
    price: float
    macd_value: float
    strength: float = 0.0
    level: str = '1d'

# ========== 增强版分析函数 ==========

def find_fractals_enhanced(df: pd.DataFrame, config: ChanConfig) -> List[Fractal]:
    """增强版分型识别"""
    frs: List[Fractal] = []
    H, L = df["High"].values, df["Low"].values
    
    for i in range(config.fractal_window//2, len(df) - config.fractal_window//2):
        # 顶分型检查
        if H[i] > H[i-1] and H[i] > H[i+1]:
            # 计算分型强度
            strength = (H[i] - max(H[i-1], H[i+1])) / H[i]
            frs.append(Fractal(
                idx=i, kind="top", price=H[i], 
                high=H[i], low=L[i], strength=strength
            ))
        
        # 底分型检查
        if L[i] < L[i-1] and L[i] < L[i+1]:
            strength = (min(L[i-1], L[i+1]) - L[i]) / L[i]
            frs.append(Fractal(
                idx=i, kind="bottom", price=L[i], 
                high=H[i], low=L[i], strength=strength
            ))
    
    # 过滤太近的分型
    filtered_frs = []
    for f in frs:
        if not filtered_frs or f.idx - filtered_frs[-1].idx >= config.min_fractal_gap:
            filtered_frs.append(f)
        else:
            # 保留强度更大的分型
            if f.strength > filtered_frs[-1].strength:
                filtered_frs[-1] = f
    
    return filtered_frs

def build_strokes_enhanced(df: pd.DataFrame, frs: List[Fractal], config: ChanConfig) -> List[Stroke]:
    """增强版笔构建"""
    strokes: List[Stroke] = []
    if not frs:
        return strokes
    
    # 确保分型交替
    ordered = []
    for f in frs:
        if not ordered or ordered[-1].kind != f.kind:
            ordered.append(f)
        else:
            # 同类分型，保留更极端者
            if f.kind == "top" and f.price > ordered[-1].price:
                ordered[-1] = f
            elif f.kind == "bottom" and f.price < ordered[-1].price:
                ordered[-1] = f
    
    # 构建笔
    closes = df["Close"].values
    px_ref = np.nanmean(closes[-200:]) if len(closes) >= 10 else np.nanmean(closes)
    
    for a, b in zip(ordered[:-1], ordered[1:]):
        bars = b.idx - a.idx
        swing = abs(b.price - a.price)
        swing_pct = swing / px_ref
        
        if bars >= config.min_stroke_bars and swing_pct >= config.min_stroke_pct:
            direction = "up" if b.price > a.price else "down"
            strength = swing_pct  # 笔的强度基于振幅
            
            strokes.append(Stroke(
                start_idx=a.idx, end_idx=b.idx,
                start_price=a.price, end_price=b.price,
                direction=direction, swing=swing, strength=strength
            ))
    
    return strokes

def detect_zhongshu_enhanced(strokes: List[Stroke], config: ChanConfig) -> List[ZhongShu]:
    """增强版中枢检测"""
    zs: List[ZhongShu] = []
    if len(strokes) < config.min_zhongshu_strokes:
        return zs
    
    for i in range(len(strokes) - config.min_zhongshu_strokes + 1):
        # 取连续的最少笔数
        stroke_group = strokes[i:i + config.min_zhongshu_strokes]
        
        # 计算重叠区间
        highs = [max(s.start_price, s.end_price) for s in stroke_group]
        lows = [min(s.start_price, s.end_price) for s in stroke_group]
        
        upper = min(highs)
        lower = max(lows)
        
        # 检查是否有有效重叠
        if upper > lower and (upper - lower) / lower >= config.zhongshu_overlap_threshold:
            strength = (upper - lower) / lower
            zs.append(ZhongShu(
                start_idx=stroke_group[0].start_idx,
                end_idx=stroke_group[-1].end_idx,
                upper=upper, lower=lower,
                strokes=stroke_group, strength=strength
            ))
    
    # 合并重叠的中枢
    merged = []
    for z in zs:
        if not merged:
            merged.append(z)
        else:
            last = merged[-1]
            # 检查时间重叠和价格重叠
            time_overlap = z.start_idx <= last.end_idx + config.zhongshu_merge_threshold
            price_overlap = min(z.upper, last.upper) > max(z.lower, last.lower)
            
            if time_overlap and price_overlap:
                # 合并中枢
                merged[-1] = ZhongShu(
                    start_idx=min(last.start_idx, z.start_idx),
                    end_idx=max(last.end_idx, z.end_idx),
                    upper=min(last.upper, z.upper),
                    lower=max(last.lower, z.lower),
                    strokes=last.strokes + z.strokes,
                    strength=max(last.strength, z.strength)
                )
            else:
                merged.append(z)
    
    return merged

def detect_divergence_enhanced(df: pd.DataFrame, strokes: List[Stroke], config: ChanConfig) -> List[Divergence]:
    """增强版背驰检测"""
    closes = df["Close"].values
    _, _, hist = macd(closes, config.macd_fast, config.macd_slow, config.macd_signal)
    
    divergences = []
    
    for i in range(config.divergence_lookback, len(strokes)):
        current_stroke = strokes[i]
        prev_strokes = strokes[i-config.divergence_lookback:i]
        
        if not prev_strokes:
            continue
            
        # 检查价格和MACD的背离
        current_price = current_stroke.end_price
        current_macd = hist[current_stroke.end_idx]
        
        for prev_stroke in prev_strokes:
            prev_price = prev_stroke.end_price
            prev_macd = hist[prev_stroke.end_idx]
            
            # 顶背驰：价格创新高，MACD未创新高
            if (current_stroke.direction == "up" and 
                current_price > prev_price and 
                current_macd < prev_macd and
                abs(current_macd - prev_macd) / abs(prev_macd) > config.divergence_threshold):
                
                strength = abs(current_macd - prev_macd) / abs(prev_macd)
                divergences.append(Divergence(
                    idx=current_stroke.end_idx,
                    type="bear_div",
                    price=current_price,
                    macd_value=current_macd,
                    strength=strength
                ))
            
            # 底背驰：价格创新低，MACD未创新低
            elif (current_stroke.direction == "down" and 
                  current_price < prev_price and 
                  current_macd > prev_macd and
                  abs(current_macd - prev_macd) / abs(prev_macd) > config.divergence_threshold):
                
                strength = abs(current_macd - prev_macd) / abs(prev_macd)
                divergences.append(Divergence(
                    idx=current_stroke.end_idx,
                    type="bull_div",
                    price=current_price,
                    macd_value=current_macd,
                    strength=strength
                ))
    
    return divergences

def macd(close: np.ndarray, fast=12, slow=26, signal=9):
    """MACD计算"""
    def ema(arr, n):
        alpha = 2/(n+1)
        res = np.zeros_like(arr, dtype=float)
        res[0] = arr[0]
        for i in range(1, len(arr)):
            res[i] = alpha*arr[i] + (1-alpha)*res[i-1]
        return res
    
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    dif = ema_fast - ema_slow
    dea = ema(dif, signal)
    macd_hist = (dif - dea) * 2
    return dif, dea, macd_hist

# ========== 主分析函数 ==========
def analyze_chan_enhanced(df: pd.DataFrame, config: ChanConfig = None) -> dict:
    """增强版缠论分析"""
    if config is None:
        config = ChanConfig()
    
    print("开始增强版缠论分析...")
    print(f"数据量: {len(df)} 根K线")
    
    # 1. 分型识别
    print("1. 识别分型...")
    fractals = find_fractals_enhanced(df, config)
    print(f"   找到 {len(fractals)} 个分型")
    
    # 2. 笔构建
    print("2. 构建笔...")
    strokes = build_strokes_enhanced(df, fractals, config)
    print(f"   构建 {len(strokes)} 支笔")
    
    # 3. 线段构建
    print("3. 构建线段...")
    segments = build_segments_enhanced(strokes, config)
    print(f"   构建 {len(segments)} 个线段")
    
    # 4. 中枢检测
    print("4. 检测中枢...")
    zhongshus = detect_zhongshu_enhanced(strokes, config)
    print(f"   检测到 {len(zhongshus)} 个中枢")
    
    # 5. 背驰检测
    print("5. 检测背驰...")
    divergences = detect_divergence_enhanced(df, strokes, config)
    print(f"   检测到 {len(divergences)} 个背驰信号")
    
    return {
        'fractals': fractals,
        'strokes': strokes,
        'segments': segments,
        'zhongshus': zhongshus,
        'divergences': divergences,
        'config': config
    }

def build_segments_enhanced(strokes: List[Stroke], config: ChanConfig) -> List[Segment]:
    """增强版线段构建"""
    segments = []
    if not strokes:
        return segments
    
    current_direction = strokes[0].direction
    current_strokes = [strokes[0]]
    
    for stroke in strokes[1:]:
        if stroke.direction == current_direction:
            current_strokes.append(stroke)
        else:
            # 检查是否满足线段条件
            if len(current_strokes) >= config.min_segment_strokes:
                segment = Segment(
                    start_idx=current_strokes[0].start_idx,
                    end_idx=current_strokes[-1].end_idx,
                    high=max(s.end_price for s in current_strokes),
                    low=min(s.end_price for s in current_strokes),
                    direction=current_direction,
                    strokes=current_strokes.copy()
                )
                segments.append(segment)
            
            # 开始新的线段
            current_direction = stroke.direction
            current_strokes = [stroke]
    
    # 处理最后一个线段
    if len(current_strokes) >= config.min_segment_strokes:
        segment = Segment(
            start_idx=current_strokes[0].start_idx,
            end_idx=current_strokes[-1].end_idx,
            high=max(s.end_price for s in current_strokes),
            low=min(s.end_price for s in current_strokes),
            direction=current_direction,
            strokes=current_strokes.copy()
        )
        segments.append(segment)
    
    return segments

if __name__ == "__main__":
    # 示例用法
    print("增强版缠论分析脚本")
    print("包含更多必要的缠论参数和功能")
