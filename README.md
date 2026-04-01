# Chan Theory Technical Analysis System

A technical analysis tool based on Chan Theory (缠论) for stock market fractal, stroke, segment, pivot zone, and divergence analysis.

## Project Structure

```
chan/
├── data/                    # Data files
│   ├── AAPL_1d_data.csv    # Daily data
│   ├── AAPL_1wk_data.csv   # Weekly data
│   ├── AAPL_1mo_data.csv   # Monthly data
│   ├── AAPL_company_info.json  # Company info
│   ├── AAPL_data_summary.json  # Data summary
│   ├── AAPL_news.csv       # News data
│   └── aapl_real_data.csv  # Raw data
├── scripts/                 # Scripts
│   ├── run.py              # Main analysis script
│   ├── download_yahoo_data.py  # Data download script
│   ├── chan_analysis_enhanced.py  # Enhanced analysis
│   └── chan_report_generator.py  # Report generator
├── reports/                 # Analysis reports
│   └── aapl_chan_analysis.png  # Analysis chart
├── docs/                    # Documentation
└── README.md
```

## Features

### 1. Data Retrieval
- Download multi-timeframe data from Yahoo Finance
- Supports 1-min, 5-min, 30-min, daily, weekly, and monthly data
- Automatic technical indicator calculation (MACD, RSI, Bollinger Bands, etc.)

### 2. Chan Theory Analysis
- **Fractal Detection**: Automatically identifies top and bottom fractals
- **Stroke Construction**: Connects adjacent fractals to form strokes
- **Segment Analysis**: Builds trend segments
- **Pivot Zone Detection**: Identifies price consolidation zones (中枢)
- **Divergence Analysis**: MACD-based divergence signal detection

### 3. Multi-Level Analysis
- Simultaneous analysis across multiple timeframes
- Level relationship analysis
- Multi-level resonance signals

### 4. Report Generation
- Automated analysis reports
- Visualization charts
- Trade signal alerts
- Risk warnings

## Usage

### 1. Download Data
```bash
python scripts/download_yahoo_data.py
```

### 2. Run Analysis
```bash
python scripts/run.py --csv data/AAPL_1d_data.csv --out reports/analysis.png
```

### 3. Generate Report
```bash
python scripts/chan_report_generator.py
```

## Configuration

### Fractal Parameters
- `fractal_window`: Fractal detection window (default: 3 bars)
- `min_fractal_gap`: Minimum gap between fractals

### Stroke Parameters
- `min_stroke_bars`: Minimum bars spanned by a stroke (default: 3)
- `min_stroke_pct`: Minimum amplitude percentage (default: 0.002)

### Segment Parameters
- `min_segment_strokes`: Minimum strokes per segment (default: 3)
- `segment_break_threshold`: Segment breakout threshold

### Pivot Zone Parameters
- `min_zhongshu_strokes`: Minimum strokes for a pivot zone (default: 3)
- `zhongshu_overlap_threshold`: Pivot zone overlap threshold

### MACD Parameters
- `macd_fast`: Fast line period (default: 12)
- `macd_slow`: Slow line period (default: 26)
- `macd_signal`: Signal line period (default: 9)

## Output

### Analysis Report Contents
1. **Market Overview**: Current price, change, volume
2. **Fractal Analysis**: Count, position, strength
3. **Stroke Analysis**: Count, direction, length, amplitude
4. **Segment Analysis**: Direction, range, strength
5. **Pivot Zone Analysis**: Position, status, breakout direction
6. **Divergence Analysis**: Signals, strength, position
7. **Multi-Level Analysis**: Results across different timeframes
8. **Trade Signals**: Buy/sell signals with rationale
9. **Risk Warnings**: Risk level and key risk factors
10. **Recommendations**: Suggested actions based on analysis

### Chart Output
- Candlestick chart showing price action
- Fractal annotations (top ▲, bottom ▼)
- Stroke lines showing trends
- Segment markers for major trends
- Pivot zone boxes for consolidation areas
- Divergence point labels

## Dependencies

```bash
pip install pandas numpy matplotlib yfinance
```

## Disclaimer

1. Data download requires internet access
2. Analysis results are for reference only and do not constitute investment advice
3. Recommended to use alongside other technical analysis methods
4. Practice proper risk management and set stop-loss levels

## Changelog

- v1.0: Basic Chan Theory analysis
- v1.1: Multi-timeframe analysis
- v1.2: Report generation
- v1.3: Improved file management and project structure
