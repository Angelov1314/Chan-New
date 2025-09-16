# chan_annotator.py
# -*- coding: utf-8 -*-
"""
根据缠论思路对K线自动标注：分型 -> 笔 -> 线段 -> 中枢 + MACD背驰
使用方法：
  python run_fixed.py --csv your_data.csv --out out.png
CSV列要求（大小写不敏感）：Date, Open, High, Low, Close[, Volume]
Date 可为 "YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM"
"""

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ========== 数据结构 ==========
@dataclass
class Fractal:
    idx: int           # 在DataFrame中的索引位置
    kind: str          # 'top' or 'bottom'
    price: float       # 顶/底对应的极值价
    high: float        # 当根K的高
    low: float         # 当根K的低

@dataclass
class Stroke:          # 笔（连接相邻分型）
    start_idx: int
    end_idx: int
    start_price: float
    end_price: float
    direction: str     # 'up' or 'down'
    swing: float       # 振幅（绝对值）

@dataclass
class Segment:         # 线段（由多笔构成）
    start_idx: int
    end_idx: int
    high: float
    low: float
    direction: str

@dataclass
class ZhongShu:        # 中枢（由至少三笔形成的重叠区间）
    start_idx: int
    end_idx: int
    upper: float
    lower: float

# ========== 工具函数 ==========
def load_ohlc(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    for need in ["date", "open", "high", "low", "close"]:
        assert need in cols, f"CSV缺少列：{need}"
    # 规范列名
    df = df.rename(columns={cols["date"]: "Date",
                            cols["open"]: "Open",
                            cols["high"]: "High",
                            cols["low"]: "Low",
                            cols["close"]: "Close"})
    
    # 修复日期处理
    if 'Date' in df.columns:
        # 如果Date列已经是datetime类型，直接使用
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df["Date"] = pd.to_datetime(df["Date"], utc=True)
        # 转换为无时区的datetime
        df["Date"] = df["Date"].dt.tz_localize(None)
    
    df = df.sort_values("Date").reset_index(drop=True)
    return df

# 分型（简化：2根bar窗口；顶分型: H[i]>H[i-1]且H[i]>H[i+1]；底分型同理）
def find_fractals(df: pd.DataFrame) -> List[Fractal]:
    frs: List[Fractal] = []
    H, L = df["High"].values, df["Low"].values
    for i in range(1, len(df)-1):
        if H[i] > H[i-1] and H[i] > H[i+1]:
            frs.append(Fractal(idx=i, kind="top", price=H[i], high=H[i], low=L[i]))
        if L[i] < L[i-1] and L[i] < L[i+1]:
            frs.append(Fractal(idx=i, kind="bottom", price=L[i], high=H[i], low=L[i]))
    # 同一位置若同时出现顶/底，取极端（很少见；按更极端优先）
    frs.sort(key=lambda x: x.idx)
    return frs

# 包含关系处理（简化版）：若相邻K线存在"实体完全包含"，则合并成更极端的高/低
def resolve_inclusion(df: pd.DataFrame) -> pd.DataFrame:
    H = df["High"].copy().values
    L = df["Low"].copy().values
    for i in range(1, len(df)):
        # i-1 包含 i
        if H[i-1] >= H[i] and L[i-1] <= L[i]:
            H[i] = max(H[i], H[i-1])
            L[i] = min(L[i], L[i-1])
        # i 包含 i-1
        elif H[i] >= H[i-1] and L[i] <= L[i-1]:
            H[i-1] = max(H[i], H[i-1])
            L[i-1] = min(L[i], L[i-1])
    out = df.copy()
    out["High_smooth"] = H
    out["Low_smooth"] = L
    return out

# 根据分型生成笔（相邻不同类型分型相连；加筛选阈值：最小跨越bar数 & 最小振幅）
def build_strokes(df: pd.DataFrame, frs: List[Fractal],
                  min_bars: int = 3, min_swing_pct: float = 0.002) -> List[Stroke]:
    strokes: List[Stroke] = []
    if not frs:
        return strokes
    # 保证分型交替（top/bottom/top/...）
    ordered = []
    for f in frs:
        if not ordered:
            ordered.append(f)
        else:
            if ordered[-1].kind != f.kind:
                ordered.append(f)
            else:
                # 同类分型，保留更极端者（顶保留更高，底保留更低）
                if f.kind == "top":
                    if f.price > ordered[-1].price:
                        ordered[-1] = f
                else:
                    if f.price < ordered[-1].price:
                        ordered[-1] = f
    # 生成笔，并做最小跨度/振幅过滤
    closes = df["Close"].values
    px_ref = np.nanmean(closes[-200:]) if len(closes) >= 10 else np.nanmean(closes)
    for a, b in zip(ordered[:-1], ordered[1:]):
        bars = b.idx - a.idx
        swing = abs(b.price - a.price)
        if bars >= min_bars and swing / px_ref >= min_swing_pct:
            direction = "up" if b.price > a.price else "down"
            strokes.append(Stroke(start_idx=a.idx, end_idx=b.idx,
                                  start_price=a.price, end_price=b.price,
                                  direction=direction, swing=swing))
    return strokes

# 合并笔为线段（简化：方向一致的若干笔合并，方向拐点处结束）
def build_segments(strokes: List[Stroke], min_strokes_per_seg: int = 3) -> List[Segment]:
    segs: List[Segment] = []
    if not strokes:
        return segs
    buf = [strokes[0]]
    for s in strokes[1:]:
        if s.direction == buf[-1].direction:
            buf.append(s)
        else:
            # 输出上一个方向的线段
            if len(buf) >= max(2, min_strokes_per_seg - 1):
                segs.append(_strokes_to_segment(buf))
            buf = [s]
    # 末尾
    if len(buf) >= max(2, min_strokes_per_seg - 1):
        segs.append(_strokes_to_segment(buf))
    return segs

def _strokes_to_segment(buf: List[Stroke]) -> Segment:
    start_idx = buf[0].start_idx
    end_idx = buf[-1].end_idx
    highs = [max(x.start_price, x.end_price) for x in buf]
    lows = [min(x.start_price, x.end_price) for x in buf]
    direction = buf[0].direction
    return Segment(start_idx=start_idx, end_idx=end_idx,
                   high=max(highs), low=min(lows), direction=direction)

# 检测中枢（简化：任意相邻3笔的价格区间若存在交集，则定义为一个中枢）
def detect_zhongshu(strokes: List[Stroke]) -> List[ZhongShu]:
    zs: List[ZhongShu] = []
    if len(strokes) < 3:
        return zs
    for i in range(len(strokes) - 2):
        a, b, c = strokes[i], strokes[i+1], strokes[i+2]
        # 三笔的重叠区间：上界为三者low中最大，下界为三者high中最小（注意上下界定义）
        hi_a, lo_a = max(a.start_price, a.end_price), min(a.start_price, a.end_price)
        hi_b, lo_b = max(b.start_price, b.end_price), min(b.start_price, b.end_price)
        hi_c, lo_c = max(c.start_price, c.end_price), min(c.start_price, c.end_price)
        upper = min(hi_a, hi_b, hi_c)   # 上界
        lower = max(lo_a, lo_b, lo_c)   # 下界
        if upper > lower:
            zs.append(ZhongShu(start_idx=a.start_idx, end_idx=c.end_idx,
                               upper=upper, lower=lower))
    # 合并重叠中枢
    merged: List[ZhongShu] = []
    for z in zs:
        if not merged:
            merged.append(z)
        else:
            m = merged[-1]
            # 若时间重叠或相邻，且价格区间也有交集，则合并
            if z.start_idx <= m.end_idx and min(z.upper, m.upper) > max(z.lower, m.lower):
                merged[-1] = ZhongShu(start_idx=min(m.start_idx, z.start_idx),
                                      end_idx=max(m.end_idx, z.end_idx),
                                      upper=min(m.upper, z.upper),
                                      lower=max(m.lower, z.lower))
            else:
                merged.append(z)
    return merged

# 计算MACD（12,26,9）
def macd(close: np.ndarray, fast=12, slow=26, signal=9):
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

# 顶/底背驰（简化）：相邻两个同方向笔端点价格创新高/新低，但MACD柱能量未创新高/新低
def detect_divergence(df: pd.DataFrame, strokes: List[Stroke]) -> List[Tuple[int, str]]:
    closes = df["Close"].values
    _, _, hist = macd(closes)
    signals = []
    # 仅在笔的端点检查
    for i in range(2, len(strokes)):
        s1, s2, s3 = strokes[i-2], strokes[i-1], strokes[i]
        # 上涨背驰：价格抬高（最后两笔终点价上升），但MACD柱未同步抬高
        if s3.direction == "up":
            p_prev = df["Close"].iloc[s2.end_idx]
            p_curr = df["Close"].iloc[s3.end_idx]
            h_prev = hist[s2.end_idx]
            h_curr = hist[s3.end_idx]
            if p_curr > p_prev and h_curr < h_prev:
                signals.append((s3.end_idx, "bear_div"))  # 顶背驰
        # 下跌背驰：价格创新低，但MACD柱未同步创新低（绝对值变小）
        if s3.direction == "down":
            p_prev = df["Close"].iloc[s2.end_idx]
            p_curr = df["Close"].iloc[s3.end_idx]
            h_prev = hist[s2.end_idx]
            h_curr = hist[s3.end_idx]
            if p_curr < p_prev and h_curr > h_prev:
                signals.append((s3.end_idx, "bull_div"))  # 底背驰
    return signals

# ========== 绘图 ==========
def plot_candles_with_annotations(df: pd.DataFrame,
                                  frs: List[Fractal],
                                  strokes: List[Stroke],
                                  segs: List[Segment],
                                  zses: List[ZhongShu],
                                  divs: List[Tuple[int, str]],
                                  out_path: str):
    # 修复日期处理
    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
        dates = df["Date"].dt.strftime("%Y-%m-%d").values
    else:
        dates = [f"Day {i}" for i in range(len(df))]
    
    O, H, L, C = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values

    fig, ax = plt.subplots(figsize=(14, 7))
    # 画K线（无第三方：用矩形 + 竖线模拟）
    width = 0.6
    for i in range(len(df)):
        # 蜡烛实心方向：涨/跌
        lower = min(O[i], C[i])
        height = abs(C[i] - O[i])
        ax.add_patch(Rectangle((i - width/2, lower), width, max(height, 1e-8), fill=True, alpha=0.6))
        ax.plot([i, i], [L[i], H[i]])

    # 分型标注（顶用^，底用v）
    for f in frs:
        if f.kind == "top":
            ax.scatter(f.idx, f.price, marker="^", s=60, color='red')
        else:
            ax.scatter(f.idx, f.price, marker="v", s=60, color='green')

    # 笔（连线）
    for s in strokes:
        ax.plot([s.start_idx, s.end_idx], [s.start_price, s.end_price], linewidth=2, color='blue')

    # 线段（更粗）
    for seg in segs:
        ax.plot([seg.start_idx, seg.end_idx], [seg.low, seg.low], linestyle="--", linewidth=1.5, color='orange')
        ax.plot([seg.start_idx, seg.end_idx], [seg.high, seg.high], linestyle="--", linewidth=1.5, color='orange')

    # 中枢（画成半透明矩形带）
    for z in zses:
        ax.add_patch(Rectangle((z.start_idx, z.lower),
                               z.end_idx - z.start_idx,
                               z.upper - z.lower,
                               fill=True, alpha=0.15, color='purple'))

    # 背驰点（文字标注）
    for idx, kind in divs:
        y = C[idx]
        txt = "顶背驰" if kind == "bear_div" else "底背驰"
        ax.text(idx, y, txt, fontsize=9, ha="center", va="bottom", color='red')

    ax.set_xlim(-1, len(df))
    ax.set_title("缠论自动标注：分型 / 笔 / 线段 / 中枢 / 背驰")
    ax.set_xlabel("Index (按日期顺序)")
    ax.set_ylabel("Price")
    # x轴少量刻度（避免太挤）
    step = max(1, len(df)//10)
    ax.set_xticks(range(0, len(df), step))
    ax.set_xticklabels(dates[::step], rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)

# ========== 主流程 ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="OHLC CSV文件路径")
    ap.add_argument("--out", default="chan_out.png", help="输出图片路径")
    ap.add_argument("--min_bars", type=int, default=3, help="笔的最小跨越K线数")
    ap.add_argument("--min_swing_pct", type=float, default=0.002, help="笔的最小振幅占比(对近200根收盘均值)")
    args = ap.parse_args()

    print(f"Loading data from: {args.csv}")
    df = load_ohlc(args.csv)
    print(f"Loaded {len(df)} rows of data")
    # 可选择是否启用包含处理（默认启用平滑列，分型用原始高低）
    df_s = resolve_inclusion(df)

    # 用平滑高低来改善分型稳定性（可切到df["High"]/["Low"]）
    df_use = df.copy()
    if "High_smooth" in df_s.columns:
        df_use["High"] = df_s["High_smooth"]
        df_use["Low"]  = df_s["Low_smooth"]

    print("Finding fractals...")
    frs = find_fractals(df_use)
    print(f"Found {len(frs)} fractals")
    
    print("Building strokes...")
    strokes = build_strokes(df_use, frs, min_bars=args.min_bars, min_swing_pct=args.min_swing_pct)
    print(f"Found {len(strokes)} strokes")
    
    print("Building segments...")
    segs = build_segments(strokes, min_strokes_per_seg=3)
    print(f"Found {len(segs)} segments")
    
    print("Detecting zhongshu...")
    zses = detect_zhongshu(strokes)
    print(f"Found {len(zses)} zhongshu")
    
    print("Detecting divergences...")
    divs = detect_divergence(df_use, strokes)
    print(f"Found {len(divs)} divergences")

    print("Creating plot...")
    plot_candles_with_annotations(df, frs, strokes, segs, zses, divs, args.out)
    print(f"完成！输出：{args.out}")

if __name__ == "__main__":
    main()
