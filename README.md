# 缠论技术分析系统

基于缠论理论的技术分析工具，用于股票市场的分型、笔、线段、中枢和背驰分析。

## 项目结构

```
缠论/
├── data/                    # 数据文件
│   ├── AAPL_1d_data.csv    # 日线数据
│   ├── AAPL_1wk_data.csv   # 周线数据
│   ├── AAPL_1mo_data.csv   # 月线数据
│   ├── AAPL_company_info.json  # 公司信息
│   ├── AAPL_data_summary.json  # 数据摘要
│   ├── AAPL_news.csv       # 新闻数据
│   └── aapl_real_data.csv  # 原始数据
├── scripts/                 # 脚本文件
│   ├── run.py              # 主分析脚本
│   ├── download_yahoo_data.py  # 数据下载脚本
│   ├── chan_analysis_enhanced.py  # 增强版分析
│   └── chan_report_generator.py  # 报告生成器
├── reports/                 # 分析报告
│   └── aapl_chan_analysis.png  # 分析图表
├── docs/                    # 文档
└── README.md               # 项目说明
```

## 功能特性

### 1. 数据获取
- 从Yahoo Finance下载多时间周期数据
- 支持1分钟、5分钟、30分钟、日线、周线、月线数据
- 自动计算技术指标（MACD、RSI、布林带等）

### 2. 缠论分析
- **分型识别**：自动识别顶分型和底分型
- **笔构建**：连接相邻分型形成笔
- **线段分析**：构建趋势线段
- **中枢检测**：识别价格震荡区间
- **背驰分析**：基于MACD的背驰信号检测

### 3. 多级别分析
- 支持多个时间周期的同时分析
- 级别关系分析
- 多级别共振信号

### 4. 报告生成
- 自动生成分析报告
- 可视化图表
- 交易信号提示
- 风险提示

## 使用方法

### 1. 下载数据
```bash
python scripts/download_yahoo_data.py
```

### 2. 运行分析
```bash
python scripts/run.py --csv data/AAPL_1d_data.csv --out reports/analysis.png
```

### 3. 生成报告
```bash
python scripts/chan_report_generator.py
```

## 参数配置

### 分型参数
- `fractal_window`: 分型识别窗口（默认3根K线）
- `min_fractal_gap`: 分型间最小间隔

### 笔参数
- `min_stroke_bars`: 笔的最小跨越K线数（默认3）
- `min_stroke_pct`: 笔的最小振幅占比（默认0.002）

### 线段参数
- `min_segment_strokes`: 线段最少笔数（默认3）
- `segment_break_threshold`: 线段突破阈值

### 中枢参数
- `min_zhongshu_strokes`: 中枢最少笔数（默认3）
- `zhongshu_overlap_threshold`: 中枢重叠阈值

### MACD参数
- `macd_fast`: 快线周期（默认12）
- `macd_slow`: 慢线周期（默认26）
- `macd_signal`: 信号线周期（默认9）

## 输出说明

### 分析报告包含
1. **市场概况**：当前价格、涨跌幅、成交量
2. **分型分析**：分型数量、位置、强度
3. **笔分析**：笔的数量、方向、长度、振幅
4. **线段分析**：线段方向、范围、强度
5. **中枢分析**：中枢位置、状态、突破方向
6. **背驰分析**：背驰信号、强度、位置
7. **多级别分析**：不同时间周期的分析结果
8. **交易信号**：买入/卖出信号及理由
9. **风险提示**：风险等级和主要风险点
10. **操作建议**：基于分析结果的操作建议

### 图表输出
- K线图显示价格走势
- 分型标注（顶分型▲，底分型▼）
- 笔连线显示趋势
- 线段标识主要趋势
- 中枢框显示震荡区间
- 背驰点文字标注

## 依赖库

- pandas
- numpy
- matplotlib
- yfinance

## 安装依赖

```bash
pip install pandas numpy matplotlib yfinance
```

## 注意事项

1. 数据下载需要网络连接
2. 分析结果仅供参考，不构成投资建议
3. 建议结合其他技术分析方法使用
4. 注意风险控制，设置止损位

## 更新日志

- v1.0: 基础缠论分析功能
- v1.1: 增加多时间周期分析
- v1.2: 增加报告生成功能
- v1.3: 优化文件管理和项目结构
