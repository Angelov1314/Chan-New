# 缠论技术分析系统架构文档

## 项目概述

本项目是一个基于缠论理论的技术分析系统，能够自动识别股票的分型、笔、线段、中枢和背驰信号，并提供可视化分析和准确性评估。

## 系统架构

```
缠论技术分析系统
├── 数据层 (Data Layer)
│   ├── 数据获取模块
│   ├── 数据预处理模块
│   └── 数据存储模块
├── 分析层 (Analysis Layer)
│   ├── 分型识别模块
│   ├── 笔构建模块
│   ├── 线段分析模块
│   ├── 中枢检测模块
│   └── 背驰分析模块
├── 评估层 (Evaluation Layer)
│   ├── 质量评估模块
│   ├── 准确性评估模块
│   └── 预测验证模块
├── 展示层 (Presentation Layer)
│   ├── 图表生成模块
│   ├── 报告生成模块
│   └── Web界面模块
└── 配置层 (Configuration Layer)
    ├── 参数配置
    └── 系统设置
```

## 目录结构

```
缠论/
├── README.md                           # 项目说明文档
├── architecture.md                     # 系统架构文档
├── config.json                         # 项目配置文件
├── data/                               # 数据文件目录
│   ├── AAPL_1d_data.csv               # 日线数据
│   ├── AAPL_1wk_data.csv              # 周线数据
│   ├── AAPL_1mo_data.csv              # 月线数据
│   ├── AAPL_2025_data.csv             # 2025年数据
│   ├── AAPL_company_info.json         # 公司信息
│   ├── AAPL_data_summary.json         # 数据摘要
│   ├── AAPL_news.csv                  # 新闻数据
│   └── aapl_real_data.csv             # 原始数据
├── scripts/                            # 脚本文件目录
│   ├── run_fixed.py                   # 主分析脚本
│   ├── download_yahoo_data.py         # 数据下载脚本
│   ├── download_2025_data.py          # 2025年数据下载
│   ├── generate_report.py             # 报告生成脚本
│   ├── evaluation_scoring.py          # 质量评估脚本
│   ├── accuracy_evaluation.py         # 准确性评估脚本
│   ├── chan_analysis_enhanced.py      # 增强版分析
│   └── chan_report_generator.py       # 报告生成器
├── reports/                            # 分析报告目录
│   ├── aapl_chan_analysis_demo.png    # 分析图表
│   └── chan_analysis_report_*.txt     # 分析报告
├── web/                                # Web应用目录
│   ├── app.py                         # Flask应用主文件
│   ├── templates/                     # HTML模板
│   │   ├── index.html                 # 主页面
│   │   ├── result.html                # 结果页面
│   │   └── base.html                  # 基础模板
│   ├── static/                        # 静态文件
│   │   ├── css/                       # 样式文件
│   │   ├── js/                        # JavaScript文件
│   │   └── images/                    # 图片文件
│   └── requirements.txt               # Web应用依赖
└── docs/                               # 文档目录
    └── (待添加更多文档)
```

## 核心模块详解

### 1. 数据层 (Data Layer)

#### 1.1 数据获取模块
- **文件**: `scripts/download_yahoo_data.py`
- **功能**: 从Yahoo Finance获取多时间周期股票数据
- **支持时间周期**: 1分钟、5分钟、30分钟、日线、周线、月线
- **技术指标**: MACD、RSI、布林带、移动平均线等

#### 1.2 数据预处理模块
- **文件**: `scripts/run_fixed.py` (load_ohlc函数)
- **功能**: 数据清洗、格式标准化、包含关系处理
- **输出**: 标准化的OHLCV数据

### 2. 分析层 (Analysis Layer)

#### 2.1 分型识别模块
- **文件**: `scripts/run_fixed.py` (find_fractals函数)
- **算法**: 3根K线窗口识别顶分型和底分型
- **参数**: 分型窗口、最小间隔、强度阈值

#### 2.2 笔构建模块
- **文件**: `scripts/run_fixed.py` (build_strokes函数)
- **算法**: 连接相邻分型形成笔
- **参数**: 最小跨越K线数、最小振幅占比

#### 2.3 线段分析模块
- **文件**: `scripts/run_fixed.py` (build_segments函数)
- **算法**: 合并同方向笔形成线段
- **参数**: 最小笔数、突破阈值

#### 2.4 中枢检测模块
- **文件**: `scripts/run_fixed.py` (detect_zhongshu函数)
- **算法**: 检测3笔以上重叠区间
- **参数**: 最小笔数、重叠阈值、合并规则

#### 2.5 背驰分析模块
- **文件**: `scripts/run_fixed.py` (detect_divergence函数)
- **算法**: 基于MACD的背驰检测
- **参数**: MACD参数、背驰阈值

### 3. 评估层 (Evaluation Layer)

#### 3.1 质量评估模块
- **文件**: `scripts/evaluation_scoring.py`
- **功能**: 评估分析质量，7个维度评分
- **评分维度**: 数据质量、分型识别、笔构建、线段分析、中枢检测、背驰信号、交易信号

#### 3.2 准确性评估模块
- **文件**: `scripts/accuracy_evaluation.py`
- **功能**: 使用未来数据验证分析准确性
- **评估维度**: 分型预测、笔预测、线段预测、中枢预测、背驰信号、价格预测、趋势预测

### 4. 展示层 (Presentation Layer)

#### 4.1 图表生成模块
- **文件**: `scripts/run_fixed.py` (plot_candles_with_annotations函数)
- **功能**: 生成带缠论标注的K线图
- **标注内容**: 分型、笔、线段、中枢、背驰信号

#### 4.2 报告生成模块
- **文件**: `scripts/generate_report.py`
- **功能**: 生成详细的文字分析报告
- **报告内容**: 市场概况、技术分析、交易信号、风险提示

#### 4.3 Web界面模块
- **文件**: `web/app.py`
- **功能**: 提供Web界面进行交互式分析
- **特性**: 股票代码输入、时间段选择、结果展示

## 技术栈

### 后端技术
- **Python 3.13**: 主要编程语言
- **pandas**: 数据处理和分析
- **numpy**: 数值计算
- **matplotlib**: 图表生成
- **yfinance**: 股票数据获取
- **Flask**: Web框架

### 前端技术
- **HTML5**: 页面结构
- **CSS3**: 样式设计
- **JavaScript**: 交互功能
- **Bootstrap**: UI框架

### 数据存储
- **CSV文件**: 股票数据存储
- **JSON文件**: 配置和元数据存储
- **PNG文件**: 图表输出

## 核心算法

### 1. 分型识别算法
```python
def find_fractals(df):
    for i in range(1, len(df)-1):
        # 顶分型: H[i] > H[i-1] and H[i] > H[i+1]
        if H[i] > H[i-1] and H[i] > H[i+1]:
            fractals.append(Fractal(idx=i, kind="top", price=H[i]))
        # 底分型: L[i] < L[i-1] and L[i] < L[i+1]
        if L[i] < L[i-1] and L[i] < L[i+1]:
            fractals.append(Fractal(idx=i, kind="bottom", price=L[i]))
```

### 2. 笔构建算法
```python
def build_strokes(fractals):
    for a, b in zip(ordered_fractals[:-1], ordered_fractals[1:]):
        if bars >= min_bars and swing_pct >= min_swing_pct:
            strokes.append(Stroke(start_idx=a.idx, end_idx=b.idx,
                                start_price=a.price, end_price=b.price,
                                direction=direction, swing=swing))
```

### 3. 中枢检测算法
```python
def detect_zhongshu(strokes):
    for i in range(len(strokes) - 2):
        a, b, c = strokes[i], strokes[i+1], strokes[i+2]
        upper = min(hi_a, hi_b, hi_c)  # 上界
        lower = max(lo_a, lo_b,lo_c)   # 下界
        if upper > lower:
            zhongshus.append(ZhongShu(upper=upper, lower=lower))
```

## 配置参数

### 分型参数
- `fractal_window`: 分型识别窗口 (默认: 3)
- `min_fractal_gap`: 分型间最小间隔 (默认: 2)

### 笔参数
- `min_stroke_bars`: 笔的最小跨越K线数 (默认: 3)
- `min_stroke_pct`: 笔的最小振幅占比 (默认: 0.002)

### 线段参数
- `min_segment_strokes`: 线段最少笔数 (默认: 3)
- `segment_break_threshold`: 线段突破阈值 (默认: 0.005)

### 中枢参数
- `min_zhongshu_strokes`: 中枢最少笔数 (默认: 3)
- `zhongshu_overlap_threshold`: 中枢重叠阈值 (默认: 0.001)

### MACD参数
- `macd_fast`: 快线周期 (默认: 12)
- `macd_slow`: 慢线周期 (默认: 26)
- `macd_signal`: 信号线周期 (默认: 9)

## 性能指标

### 分析质量评分
- **总分**: 100分
- **数据质量**: 20分
- **分型识别**: 20分
- **笔构建**: 15分
- **线段分析**: 15分
- **中枢检测**: 15分
- **背驰信号**: 10分
- **交易信号**: 5分

### 准确性评估
- **分型预测**: 100% (完美)
- **笔预测**: 100% (完美)
- **线段预测**: 100% (完美)
- **中枢预测**: 100% (完美)
- **背驰信号**: 100% (完美)
- **价格预测**: 40% (需改进)
- **趋势预测**: 32% (需改进)

## 部署说明

### 本地部署
1. 安装Python依赖: `pip install -r requirements.txt`
2. 运行数据下载: `python scripts/download_yahoo_data.py`
3. 运行分析: `python scripts/run_fixed.py --csv data/AAPL_1d_data.csv`
4. 启动Web应用: `python web/app.py`

### 生产部署
1. 使用Gunicorn作为WSGI服务器
2. 配置Nginx作为反向代理
3. 使用Redis进行缓存
4. 配置定时任务进行数据更新

## 扩展计划

### 短期目标
1. 完善Web界面功能
2. 增加更多技术指标
3. 优化预测算法
4. 添加实时数据支持

### 长期目标
1. 支持更多股票市场
2. 集成机器学习算法
3. 开发移动端应用
4. 提供API服务

## 维护说明

### 定期维护
1. 每日数据更新
2. 每周性能检查
3. 每月算法优化
4. 每季度系统升级

### 故障处理
1. 数据获取失败: 检查网络连接和API限制
2. 分析错误: 检查数据格式和参数设置
3. 图表生成失败: 检查matplotlib配置
4. Web服务异常: 检查Flask配置和端口占用

## 版本历史

- **v1.0**: 基础缠论分析功能
- **v1.1**: 增加多时间周期分析
- **v1.2**: 增加报告生成功能
- **v1.3**: 优化文件管理和项目结构
- **v1.4**: 增加质量评估和准确性评估
- **v1.5**: 开发Web界面 (当前版本)

## 联系方式

- **项目维护者**: 缠论分析团队
- **技术支持**: 通过GitHub Issues
- **文档更新**: 定期更新architecture.md
