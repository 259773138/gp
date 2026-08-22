"""
股票筛选系统 - 全局配置文件（自适应权重版）
包含：股票池、数据、指标、筛选、回测、自适应学习、调度、输出全部配置
"""

import os

# ============ 路径配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DAILY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "daily")
BACKTEST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "backtest")
DB_PATH = os.path.join(DATA_DIR, "screener.db")
DEMO_DB_PATH = os.path.join(DATA_DIR, "demo.db")
WEB_OUTPUT_DIR = os.path.join(REPO_ROOT, "docs")   # GitHub Pages 发布目录

# 自动创建目录
for d in [DATA_DIR, OUTPUT_DIR, DAILY_OUTPUT_DIR, BACKTEST_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ============ 股票池配置 ============
STOCK_POOL_CONFIG = {
    "hot_stock_count": 50,
    "index_stocks": {
        "000300": "沪深300",
        "000905": "中证500",
        "000852": "中证1000",
    },
    "min_pool_size": 120,
    "exclude_st": True,
    "exclude_kcb": False,
    "exclude_bse": True,
}

# ============ 数据参数 ============
DATA_CONFIG = {
    "kline_days": 750,        # 日线天数（加长，支持周线/年线与历史回放）
    "kline_period": "daily",
    "fetch_interval": 0.3,
    "max_retry": 3,           # 数据获取重试次数
    "retry_delay": 2,         # 重试间隔秒数
}

# ============ 技术指标参数 ============
INDICATOR_CONFIG = {
    "ma_periods": [5, 10, 20, 60, 120, 250],
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "rsi_period": 14,
    "boll_period": 20,
    "boll_std": 2,
    "atr_period": 14,
    "adx_period": 14,
    "vol_ma_periods": [5, 20, 60],
}

# ============ 周线指标参数 ============
WEEKLY_CONFIG = {
    "ma_periods": [5, 10, 20, 30, 60],
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "rsi_period": 14,
    # 周线信号权重（相对日线的加权倍数）——规则分类兜底用
    "signal_weight": 1.5,
}

# ============ 基本面参数 ============
FUNDAMENTAL_CONFIG = {
    "fetch_interval": 0.5,
    # 基本面评分权重（兜底默认；启用自适应后由学习模型接管打分）
    "weights": {
        "roe": 0.25,
        "revenue_growth": 0.20,
        "profit_growth": 0.20,
        "gross_margin": 0.15,
        "debt_ratio": 0.10,
        "pe_percentile": 0.10,
    },
    # 基本面筛选阈值
    "thresholds": {
        "roe_min": 8,              # 最低ROE(%)
        "revenue_growth_min": -10, # 最低营收增速(%)
        "debt_ratio_max": 70,      # 最高资产负债率(%)
        "gross_margin_min": 15,    # 最低毛利率(%)
    },
}

# ============ 筛选阈值（规则分类用） ============
SCREENER_CONFIG = {
    "reversal_up": {
        "rsi_oversold": 30,
        "volume_shrink_ratio": 0.3,
        "pe_percentile": 20,
        "macd_diff_threshold": 0,
        "min_drop_pct": -20,
    },
    "reversal_down": {
        "rsi_overbought": 70,
        "volume_surge_ratio": 2.5,
        "pe_percentile": 80,
        "min_rise_pct": 30,
    },
    "trend_continue": {
        "adx_threshold": 25,
        "ma_alignment_count": 4,
        "price_above_ma20": True,
    },
    "valuation": {
        "pe_max": 200,
        "pe_min": 0,
        "pb_max": 30,
    },
    # 兜底综合评分权重（学习模型未就绪时使用）。
    # 注意：启用自适应权重后，综合分改由回测数据训练出的模型生成，
    # 以下权重仅作为冷启动兜底，参见 weight_optimizer.py。
    "score_weights": {
        "technical": 0.40,
        "weekly": 0.25,
        "fundamental": 0.20,
        "valuation": 0.15,
    },
}

# ============ 回测配置 ============
BACKTEST_CONFIG = {
    # 回测检验周期（天）
    "check_periods": [5, 10, 20, 60],
    # 信号有效性阈值
    "success_thresholds": {
        "底部反转候选": {"min_return_20d": 5},     # 20天内涨5%算成功
        "顶部风险警示": {"max_return_20d": -5},    # 20天内跌5%算成功
        "强势上涨趋势": {"min_return_10d": 2},     # 10天内涨2%算成功
        "弱势下跌趋势": {"max_return_10d": -2},    # 10天内跌2%算成功
        "超跌反弹机会": {"min_return_5d": 3},      # 5天内涨3%算成功
    },
}

# ============ 自适应权重（学习模型）配置 ============
ADAPTIVE_CONFIG = {
    # 训练标签：check_period 天的前瞻收益 > label_return_threshold 记为 1（看多胜率）
    "label_check_period": 20,
    "label_return_threshold": 0.0,
    # 最少训练样本（joined 记录数）与最少不同日期数，不足则沿用默认权重
    "min_train_samples": 200,
    "min_train_dates": 8,
    # 时间序列切分：前 70% 日期训练，后 30% 日期做样本外验证
    "train_date_ratio": 0.7,
    # 逻辑回归超参（numpy 实现，标准化特征 + L2 正则）
    "l2_lambda": 1e-3,
    "learning_rate": 0.1,
    "max_iterations": 3000,
    # 权重文件
    "weights_file": os.path.join(DATA_DIR, "learned_weights.json"),
    # 展示胜率提升窗：每日取分数最高的 Top-N 计算胜率
    "top_n": 10,
}

# ============ 历史回放配置（冷启动：用历史K线“补跑”历史信号） ============
REPLAY_CONFIG = {
    "enabled": True,
    "max_stocks": 80,          # 参与回放的最大股票数（按股票池顺序取）
    "step_days": 5,            # 每隔多少个交易日回放一次
    "start_warmup": 260,       # 指标暖机：从第 260 根K线之后开始回放（约1年）
    "forward_margin": 61,      # 尾部预留天数，保证 60 日前瞻收益可计算
    "min_weekly_rows": 20,
}

# ============ 调度配置 ============
SCHEDULER_CONFIG = {
    # 每日执行时间（收盘后）
    "daily_run_time": "15:30",
    # 是否在周末执行（通常不需要）
    "run_on_weekends": False,
}

# ============ 输出配置 ============
OUTPUT_CONFIG = {
    "output_dir": OUTPUT_DIR,
    "console_max_rows": 20,
    # 图表配置
    "chart": {
        "dpi": 150,
        "figsize": (16, 10),
        "style": "seaborn-v0_8-whitegrid",
    },
}

# ============ 网站（GitHub Pages）配置 ============
WEB_CONFIG = {
    "site_title": "A股自适应量化筛选",
    "max_rows_per_category": 20,
    "archive_keep": 60,        # 站点保留的历史报告页数
}
