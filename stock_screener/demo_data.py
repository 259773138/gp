"""
演示数据生成器（离线用）
生成带不同市场形态（主升/主跌/筑底/见顶/震荡/超跌）的合成K线，
用于在没有行情网络的环境下端到端验证系统逻辑。
仅在 --mode demo 时使用，数据写入独立的 demo 数据库，不污染真实数据。
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

SCENARIOS = ["uptrend", "downtrend", "bottoming", "topping", "sideways", "oversold"]
NAMES = {
    "uptrend": "演示主升", "downtrend": "演示主跌", "bottoming": "演示筑底",
    "topping": "演示见顶", "sideways": "演示震荡", "oversold": "演示超跌",
}


def _gen_kline(seed: int, scenario: str, days: int = 750) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # 分段漂移：让最近一段呈现该形态特征
    drift = np.zeros(days)
    vol = np.full(days, 0.018)
    if scenario == "uptrend":
        drift[:] = 0.0015
        drift[-60:] = 0.003
    elif scenario == "downtrend":
        drift[:] = -0.0015
        drift[-60:] = -0.003
    elif scenario == "bottoming":
        drift[:-80] = -0.0025
        drift[-80:] = 0.0002
        drift[-5:] = 0.004
    elif scenario == "topping":
        drift[:-40] = 0.003
        drift[-40:] = -0.0005
        drift[-3:] = -0.008
    elif scenario == "oversold":
        drift[:-30] = 0.0
        drift[-30:-3] = -0.012
        drift[-3:] = 0.01
    else:  # sideways
        drift[:] = 0.0
        vol[:] = 0.012

    rets = drift + rng.normal(0, vol, days)
    close = 30 * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.004, days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, days)))
    base_v = rng.uniform(2e5, 2e6)
    volume = base_v * (1 + np.abs(rng.normal(0, 0.4, days))) * (1 + np.abs(rets) * 30)
    if scenario == "bottoming":
        volume[-10:] = base_v * 0.25   # 底部缩量
    if scenario == "topping":
        volume[-3:] = base_v * 3.0     # 顶部放量

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    df = pd.DataFrame({
        "date": dates, "open": open_, "close": close, "high": high, "low": low,
        "volume": volume.astype(int), "amount": (volume * close).astype(int),
    })
    df["pct_change"] = df["close"].pct_change() * 100
    df["turnover_rate"] = np.abs(rng.normal(3, 1.5, days)).round(2)
    return df


def make_demo_dataset(n_stocks: int = 36) -> Dict:
    """生成完整演示数据集：股票池、K线、周线、行情、基本面"""
    from data_fetcher import DataFetcher

    rows, klines, weekly, funds = [], {}, {}, {}
    for i in range(n_stocks):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        code = f"60{1000 + i}"
        name = f"{NAMES[scenario]}{i:03d}"
        rows.append({"code": code, "name": name, "source": f"演示-{scenario}"})
        df = _gen_kline(seed=1000 + i, scenario=scenario)
        klines[code] = df
        w = DataFetcher.build_weekly_from_daily(df)
        if w is not None:
            weekly[code] = w
        rng = np.random.default_rng(2000 + i)
        funds[code] = {
            "code": code,
            "roe": round(float(rng.uniform(-5, 30)), 1),
            "revenue_growth": round(float(rng.uniform(-30, 60)), 1),
            "profit_growth": round(float(rng.uniform(-40, 80)), 1),
            "gross_margin": round(float(rng.uniform(5, 70)), 1),
            "net_margin": None, "debt_ratio": round(float(rng.uniform(10, 80)), 1),
            "current_ratio": None, "eps": None, "bps": None,
            "fundamental_details": ["演示数据"],
        }

    # 用真实打分逻辑给演示基本面打分
    from fundamental import FundamentalAnalyzer
    fa = FundamentalAnalyzer()
    for code, f in funds.items():
        score, grade, details = fa._calculate_score(f)
        f["fundamental_score"] = score
        f["fundamental_grade"] = grade

    pool = pd.DataFrame(rows)

    quotes_rows = []
    for code, df in klines.items():
        last = df.iloc[-1]
        rng = np.random.default_rng(int(code))
        quotes_rows.append({
            "code": code, "name": pool.loc[pool["code"] == code, "name"].iloc[0],
            "price": round(float(last["close"]), 2),
            "pct_change": round(float(last["pct_change"]), 2),
            "volume": float(last["volume"]), "amount": float(last["amount"]),
            "turnover": 3.0,
            "pe": round(float(rng.uniform(5, 80)), 1),
            "pb": round(float(rng.uniform(0.5, 10)), 2),
            "total_mv": 1e10, "circ_mv": 8e9,
        })
    quotes = pd.DataFrame(quotes_rows)

    return {
        "pool": pool, "klines": klines, "weekly": weekly,
        "quotes": quotes, "fundamentals": funds,
    }
