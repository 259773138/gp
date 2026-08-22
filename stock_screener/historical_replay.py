"""
历史回放模块（冷启动训练数据生成器）
====================================
问题：自适应模型需要“信号 → 未来收益”的标注数据，纯实盘积累要等数月。
方案：用已有K线在过去每个回放日"假装运行一次筛选"：
  - 指标只使用该日及之前的数据（rolling/ewm 均为因果计算，天然无前视偏差）
  - 前瞻收益由后续K线精确计算
  - 同时写入 screening_records(source='replay') 与 backtest_results(source='replay')

这样项目第一天就能获得：数千条标注样本 → 自适应权重立即可训练 →
回测胜率矩阵立即可展示。基本面/估值特征在回放中取不到历史值，
统一记为缺失（模型按缺失特征处理，不参与该样本对应权重项的学习）。
"""

import time
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Optional
from config import REPLAY_CONFIG, BACKTEST_CONFIG
from db_manager import DBManager
from screener import StockScreener
from backtest import BacktestEngine


class HistoricalReplay:
    """历史信号回放器"""

    def __init__(self, db_path: str = None):
        self.db = DBManager(db_path) if db_path else DBManager()
        # 回放阶段不加载自适应模型（此时模型尚不存在/与回放数据可能同源）
        self.screener = StockScreener(model=None)
        self.cfg = REPLAY_CONFIG

    def already_done(self) -> bool:
        return self.db.get_meta("historical_replay_done") == "1"

    def run(self,
            stock_pool: pd.DataFrame,
            kline_data: Dict[str, pd.DataFrame],
            weekly_data: Dict[str, pd.DataFrame],
            max_stocks: int = None) -> int:
        """执行历史回放，返回生成的信号记录数"""
        if not self.cfg["enabled"]:
            print("  ⏸️ 历史回放已在配置中禁用")
            return 0
        if self.already_done():
            print("  ⏭️ 历史回放已完成过，跳过（如需重放请删除数据库后重跑）")
            return 0

        print("\n" + "=" * 60)
        print("⏪ 历史回放：为自适应权重生成训练数据...")
        print("=" * 60)

        max_stocks = max_stocks or self.cfg["max_stocks"]
        step = self.cfg["step_days"]
        warmup = self.cfg["start_warmup"]
        margin = self.cfg["forward_margin"]
        periods = BACKTEST_CONFIG["check_periods"]

        names = dict(zip(stock_pool["code"], stock_pool["name"]))
        records_by_date: Dict[str, List[Dict]] = defaultdict(list)
        outcomes_all: List[Dict] = []
        used = 0
        t0 = time.time()

        for code, daily in kline_data.items():
            if used >= max_stocks:
                break
            if daily is None or len(daily) < warmup + margin:
                continue

            name = names.get(code, "")
            weekly_full = weekly_data.get(code)
            closes = daily["close"].values
            dates = daily["date"].values

            idx = warmup
            while idx <= len(daily) - margin:
                dslice = daily.iloc[:idx + 1]
                t = daily.iloc[idx]["date"]
                scan_date = pd.Timestamp(t).strftime("%Y-%m-%d")

                # 周线切片（只含 ≤ 当日的周）
                wslice = None
                if weekly_full is not None:
                    wslice = weekly_full[weekly_full["date"] <= t]

                try:
                    res = self.screener.analyze_single_stock(code, name, dslice, wslice)
                except Exception:
                    res = None

                if res:
                    records_by_date[scan_date].append(res)
                    price0 = closes[idx]
                    # 前瞻收益与成功标注
                    for p in periods:
                        j = idx + p
                        if j < len(closes) and price0 > 0:
                            fwd = (closes[j] - price0) / price0 * 100
                            check_date = pd.Timestamp(dates[j]).strftime("%Y-%m-%d")
                            outcomes_all.append({
                                "scan_date": scan_date,
                                "check_date": check_date,
                                "check_period": p,
                                "code": code,
                                "name": name,
                                "category": res["category"],
                                "signal_score": res["signal_score"],
                                "price_at_signal": float(price0),
                                "price_at_check": float(closes[j]),
                                "actual_return": round(float(fwd), 2),
                                "is_success": BacktestEngine.judge_success(
                                    res["category"], fwd, p),
                                "source": "replay",
                            })
                idx += step

            used += 1
            if used % 10 == 0:
                print(f"  回放进度: {used} 支 | 已覆盖 {len(records_by_date)} 个交易日 | "
                      f"耗时 {time.time()-t0:.0f}s")

        # 批量落库
        print(f"\n  💾 批量写入数据库...")
        n_signals = 0
        for scan_date in sorted(records_by_date):
            self.db.save_screening_results(records_by_date[scan_date],
                                           scan_date=scan_date, source="replay", verbose=False)
            n_signals += len(records_by_date[scan_date])
        self.db.save_backtest_results(outcomes_all)

        self.db.set_meta("historical_replay_done", "1")
        self.db.set_meta("historical_replay_at", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"✅ 历史回放完成: {used} 支股票 → {n_signals} 条信号 / "
              f"{len(outcomes_all)} 条回测结果，耗时 {time.time()-t0:.0f}s")
        return n_signals
