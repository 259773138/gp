"""
回测验证模块
验证历史筛选信号的准确率，持续优化筛选系统。
改造（相对原版方案）：只处理 source='live' 的实盘记录；
历史回放（replay）记录在回放时已直接写入回测结果。
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional
from config import BACKTEST_CONFIG, BACKTEST_OUTPUT_DIR
from db_manager import DBManager
from data_fetcher import DataFetcher

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


class BacktestEngine:
    """回测验证引擎"""

    def __init__(self, db_path: str = None):
        self.db = DBManager(db_path) if db_path else DBManager()
        self.fetcher = DataFetcher()

    def run_backtest(self):
        """执行回测检验"""
        print("\n" + "=" * 60)
        print("🔬 开始回测验证（实盘记录）...")
        print("=" * 60)

        total_checked = 0

        for period in BACKTEST_CONFIG["check_periods"]:
            # 找到需要检验的日期
            pending_dates = self.db.get_pending_backtest_dates(period)

            if not pending_dates:
                print(f"  📅 {period}天回测: 无待检验数据")
                continue

            print(f"\n  📅 {period}天回测: {len(pending_dates)} 个日期待检验")

            for scan_date in pending_dates:
                checked = self._check_signals_for_date(scan_date, period)
                total_checked += checked

        print(f"\n✅ 回测完成: 共检验 {total_checked} 条信号")

        # 生成回测报告
        self._generate_backtest_report()

    def _check_signals_for_date(self, scan_date: str, check_period: int) -> int:
        """检验某天的筛选信号"""
        records = self.db.get_screening_by_date(scan_date, source="live")
        if not records:
            return 0

        # 计算检验日期
        scan_dt = datetime.strptime(scan_date, "%Y-%m-%d")
        check_dt = scan_dt + timedelta(days=check_period)
        check_date_str = check_dt.strftime("%Y-%m-%d")

        # 如果检验日期还没到，跳过
        if check_dt.date() > date.today():
            return 0

        checked = 0
        thresholds = BACKTEST_CONFIG["success_thresholds"]
        batch = []

        for record in records:
            code = record["code"]
            category = record["category"]
            price_at_signal = record["price_at_signal"]

            if price_at_signal is None or price_at_signal <= 0:
                continue

            # 获取检验日的价格
            price_at_check = self.fetcher.fetch_price_at_date(code, check_date_str)

            if price_at_check is None:
                continue

            # 计算实际收益率
            actual_return = (price_at_check - price_at_signal) / price_at_signal * 100

            # 判断信号是否成功
            is_success = self.judge_success(category, actual_return, check_period, thresholds)

            batch.append({
                "scan_date": scan_date,
                "check_date": check_date_str,
                "check_period": check_period,
                "code": code,
                "name": record.get("name", ""),
                "category": category,
                "signal_score": record.get("signal_score", 0),
                "price_at_signal": price_at_signal,
                "price_at_check": price_at_check,
                "actual_return": round(actual_return, 2),
                "is_success": is_success,
                "source": "live",
            })

            checked += 1
            time.sleep(0.2)

        if batch:
            self.db.save_backtest_results(batch)
        return checked

    @staticmethod
    def judge_success(category: str, actual_return: float,
                      check_period: int, thresholds: dict = None) -> bool:
        """判断信号是否成功"""
        thresholds = thresholds or BACKTEST_CONFIG["success_thresholds"]
        cat_thresholds = thresholds.get(category, {})

        if category == "底部反转候选":
            min_ret = cat_thresholds.get("min_return_20d", 5)
            return actual_return >= min_ret

        elif category == "顶部风险警示":
            max_ret = cat_thresholds.get("max_return_20d", -5)
            return actual_return <= max_ret

        elif category == "强势上涨趋势":
            min_ret = cat_thresholds.get("min_return_10d", 2)
            return actual_return >= min_ret

        elif category == "弱势下跌趋势":
            max_ret = cat_thresholds.get("max_return_10d", -2)
            return actual_return <= max_ret

        elif category == "超跌反弹机会":
            min_ret = cat_thresholds.get("min_return_5d", 3)
            return actual_return >= min_ret

        return False

    def _generate_backtest_report(self):
        """生成回测报告"""
        summary = self.db.get_backtest_summary()

        if summary.empty:
            print("\n  📊 暂无回测数据")
            return

        print("\n" + "─" * 70)
        print("📊 回测统计报告（含历史回放）")
        print("─" * 70)

        valid = set(BACKTEST_CONFIG["success_thresholds"].keys())
        for _, row in summary.iterrows():
            if row["category"] not in valid:
                continue
            success_icon = "✅" if row["success_rate"] >= 55 else "⚠️" if row["success_rate"] >= 45 else "❌"
            print(
                f"  {success_icon} {row['category']:10s} | "
                f"{row['check_period']:2d}天检验 | "
                f"信号{row['total_signals']:4d}条 | "
                f"成功率{row['success_rate']:5.1f}% | "
                f"平均收益{row['avg_return']:+6.2f}%"
            )

        # 生成图表
        if HAS_PLOT:
            self._plot_backtest_chart(summary)

    def _plot_backtest_chart(self, summary: pd.DataFrame):
        """生成回测图表"""
        try:
            plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # 1. 各类别成功率
            categories = summary["category"].unique()
            for cat in categories:
                cat_data = summary[summary["category"] == cat]
                axes[0].plot(
                    cat_data["check_period"],
                    cat_data["success_rate"],
                    marker="o", label=cat
                )
            axes[0].set_xlabel("Check Period (days)")
            axes[0].set_ylabel("Success Rate (%)")
            axes[0].set_title("Signal Success Rate by Category")
            axes[0].axhline(y=50, color="gray", linestyle="--", alpha=0.5)
            axes[0].legend(fontsize=8)
            axes[0].grid(True, alpha=0.3)

            # 2. 平均收益率
            for cat in categories:
                cat_data = summary[summary["category"] == cat]
                axes[1].bar(
                    cat_data["check_period"].astype(str) + "d\n" + cat[:4],
                    cat_data["avg_return"],
                    alpha=0.7, label=cat
                )
            axes[1].set_xlabel("Period & Category")
            axes[1].set_ylabel("Avg Return (%)")
            axes[1].set_title("Average Return by Signal Category")
            axes[1].axhline(y=0, color="black", linewidth=0.5)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            chart_path = f"{BACKTEST_OUTPUT_DIR}/backtest_report_{date.today().isoformat()}.png"
            plt.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\n  📈 回测图表已保存: {chart_path}")

        except Exception as e:
            print(f"\n  ⚠️ 图表生成失败: {e}")

    def get_signal_quality_analysis(self) -> Dict:
        """信号质量分析（用于优化阈值）"""
        summary = self.db.get_backtest_summary()
        if summary.empty:
            return {}

        analysis = {}
        for _, row in summary.iterrows():
            key = f"{row['category']}_{row['check_period']}d"
            analysis[key] = {
                "success_rate": row["success_rate"],
                "avg_return": row["avg_return"],
                "total_signals": row["total_signals"],
                "quality": "good" if row["success_rate"] >= 55 else "fair" if row["success_rate"] >= 45 else "poor"
            }

        return analysis
