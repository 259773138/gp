"""
报告生成模块（增强版）
含基本面、周线分析列、自适应 AI 分，以及回测统计
"""

import os
import pandas as pd
from datetime import datetime, date
from typing import Dict, List
from tabulate import tabulate
from config import OUTPUT_CONFIG, DAILY_OUTPUT_DIR
from screener import StockScreener
from db_manager import DBManager


class ReportGenerator:
    """报告生成器"""

    def __init__(self, results: Dict[str, List[Dict]], all_analyzed: List[Dict],
                 db_path: str = None):
        self.results = results
        self.all_analyzed = all_analyzed
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.db = DBManager(db_path) if db_path else DBManager()

    @staticmethod
    def _ai(s: Dict) -> str:
        v = s.get("adaptive_score")
        return f"{v:.0f}" if v is not None else "-"

    def print_console_report(self):
        """控制台报告"""
        print("\n")
        print("█" * 70)
        print(f"█  📊 股票市场自动筛选分析报告（自适应权重版）")
        print(f"█  📅 {self.timestamp}")
        print(f"█  📈 分析: {len(self.all_analyzed)} 支 | 日线+周线+基本面+AI自适应评分")
        print("█" * 70)

        for category, description in StockScreener.CATEGORIES.items():
            stocks = self.results.get(category, [])
            if not stocks:
                continue

            print(f"\n{'─' * 80}")
            print(f"  {description}")
            print(f"  共 {len(stocks)} 支 (显示前{OUTPUT_CONFIG['console_max_rows']}支)")
            print(f"{'─' * 80}")

            table_data = []
            for s in stocks[:OUTPUT_CONFIG["console_max_rows"]]:
                pe_str = f"{s['pe']:.1f}" if s.get('pe') and pd.notna(s.get('pe')) else "-"
                fund_str = s.get("fundamental_grade", "N/A")
                weekly_str = "✓" if s.get("weekly_confirm") else "-"

                table_data.append([
                    s["code"],
                    s["name"][:6],
                    f"{s['price']:.2f}",
                    f"{s['pct_change']:+.2f}%",
                    self._ai(s),
                    f"{s.get('composite_score', 0):.0f}",
                    f"{s['rsi']:.0f}",
                    s["ma_state"][:4],
                    s["vol_pattern"][:4],
                    f"{s['return_20d']:+.1f}%",
                    pe_str,
                    fund_str,
                    weekly_str,
                    s["signal_details"][:35],
                ])

            headers = [
                "代码", "名称", "现价", "涨跌", "AI分", "综合分", "RSI", "均线",
                "量能", "20日", "PE", "基本面", "周线", "信号详情"
            ]
            print(tabulate(table_data, headers=headers, tablefmt="simple",
                          stralign="left", numalign="right"))

        self._print_summary()
        self._print_backtest_summary()

    def _print_summary(self):
        """市场总结"""
        print(f"\n{'═' * 80}")
        print("📋 市场概况总结")
        print(f"{'═' * 80}")

        total = len(self.all_analyzed)
        if total == 0:
            return

        # 分布
        print("\n  📊 类别分布:")
        for cat in StockScreener.CATEGORIES:
            count = len(self.results.get(cat, []))
            pct = count / total * 100
            bar_len = int(pct / 2)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            print(f"    {cat:10s}: {count:4d}支 ({pct:5.1f}%) |{bar}|")

        # 温度计
        bull = len(self.results.get("强势上涨趋势", []))
        bear = len(self.results.get("弱势下跌趋势", []))
        bottom = len(self.results.get("底部反转候选", []))
        top = len(self.results.get("顶部风险警示", []))

        print("\n  🌡️ 市场温度:")
        bull_ratio = bull / total if total > 0 else 0
        bear_ratio = bear / total if total > 0 else 0

        if bull_ratio > 0.4:
            temp = "🔥 过热"
        elif bull_ratio > 0.25:
            temp = "☀️ 偏暖"
        elif bear_ratio > 0.4:
            temp = "❄️ 过冷"
        elif bear_ratio > 0.25:
            temp = "🌧️ 偏冷"
        elif bottom > top:
            temp = "🌱 筑底中"
        elif top > bottom:
            temp = "⚠️ 见顶风险"
        else:
            temp = "😐 震荡"
        print(f"    {temp}")

        # TOP信号
        print("\n  ⭐ 重点关注（AI 自适应分最高）:")
        for cat in ["底部反转候选", "强势上涨趋势", "超跌反弹机会"]:
            stocks = self.results.get(cat, [])
            if stocks:
                top3 = stocks[:3]
                print(f"    【{cat}】")
                for s in top3:
                    wc = "📅周确" if s.get("weekly_confirm") else ""
                    fg = f"基{s.get('fundamental_grade', '')}" if s.get("fundamental_grade", "N/A") != "N/A" else ""
                    print(f"      {s['code']} {s['name']} | AI分 {self._ai(s)} | {wc} {fg} | {s['signal_details'][:30]}")

        # 风险提示
        risk_stocks = self.results.get("顶部风险警示", [])
        if risk_stocks:
            print(f"\n  ⚠️ 风险警示 TOP3:")
            for s in risk_stocks[:3]:
                print(f"      {s['code']} {s['name']} | {s['signal_details'][:40]}")

        print(f"\n{'═' * 80}")
        print("⚠️ 免责声明: 本分析仅为技术面/统计参考，不构成投资建议。")
        print(f"{'═' * 80}")

    def _print_backtest_summary(self):
        """打印历史回测统计"""
        try:
            summary = self.db.get_backtest_summary()
            if summary.empty:
                print("\n  📊 暂无历史回测数据（持续运行后将自动积累）")
                return

            print(f"\n{'─' * 80}")
            print("📊 历史信号回测准确率")
            print(f"{'─' * 80}")

            from config import BACKTEST_CONFIG
            valid = set(BACKTEST_CONFIG["success_thresholds"].keys())
            for _, row in summary.iterrows():
                if row["category"] not in valid:
                    continue
                icon = "✅" if row["success_rate"] >= 55 else "⚠️" if row["success_rate"] >= 45 else "❌"
                print(
                    f"  {icon} {row['category']:10s} | "
                    f"{row['check_period']:2d}天 | "
                    f"成功率{row['success_rate']:5.1f}% | "
                    f"均收益{row['avg_return']:+.2f}% | "
                    f"共{row['total_signals']}条"
                )
        except Exception:
            pass

    def export_excel(self) -> str:
        """导出Excel"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screening_{timestamp_str}.xlsx"
        filepath = os.path.join(DAILY_OUTPUT_DIR, filename)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # 总览
            if self.all_analyzed:
                overview = self._build_overview_df()
                overview.to_excel(writer, sheet_name="总览", index=False)

            # 各分类
            for category in StockScreener.CATEGORIES:
                stocks = self.results.get(category, [])
                if not stocks:
                    continue
                sheet_name = category[:10]
                cat_df = self._build_category_df(stocks)
                cat_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # 统计页
            stats = self._build_stats_df()
            stats.to_excel(writer, sheet_name="统计", index=False)

            # 回测页
            try:
                bt_summary = self.db.get_backtest_summary()
                if not bt_summary.empty:
                    bt_summary.to_excel(writer, sheet_name="回测统计", index=False)
            except Exception:
                pass

        print(f"\n📁 Excel: {filepath}")
        return filepath

    def _build_overview_df(self) -> pd.DataFrame:
        rows = []
        for s in self.all_analyzed:
            rows.append({
                "代码": s["code"], "名称": s["name"], "现价": s["price"],
                "涨跌幅%": s["pct_change"], "分类": s["category"],
                "AI分": s.get("adaptive_score"),
                "综合分": s.get("composite_score", 0),
                "技术信号分": s["signal_score"],
                "RSI": s["rsi"], "ADX": s["adx"],
                "均线状态": s["ma_state"], "量能": s["vol_pattern"],
                "量比60": s["vol_ratio_60"], "量价关系": s["vol_price_relation"],
                "5日%": s["return_5d"], "20日%": s["return_20d"], "60日%": s["return_60d"],
                "20日偏离%": s["bias_20"],
                "底背离": "是" if s["macd_bottom_div"] else "",
                "顶背离": "是" if s["macd_top_div"] else "",
                "K线": s["candle_patterns"],
                "周线趋势": s.get("weekly_trend", ""),
                "周线确认": "✓" if s.get("weekly_confirm") else "",
                "周线RSI": s.get("weekly_rsi", ""),
                "基本面": s.get("fundamental_grade", ""),
                "基本面分": s.get("fundamental_score", ""),
                "ROE%": s.get("roe", ""),
                "营收增速%": s.get("revenue_growth", ""),
                "利润增速%": s.get("profit_growth", ""),
                "PE": s.get("pe", ""), "PB": s.get("pb", ""),
                "信号详情": s["signal_details"],
            })
        df = pd.DataFrame(rows)
        df = df.sort_values(["分类", "AI分"], ascending=[True, False], na_position="last")
        return df

    def _build_category_df(self, stocks: List[Dict]) -> pd.DataFrame:
        rows = []
        for rank, s in enumerate(stocks, 1):
            rows.append({
                "排名": rank, "代码": s["code"], "名称": s["name"],
                "现价": s["price"], "涨跌%": s["pct_change"],
                "AI分": s.get("adaptive_score"),
                "综合分": s.get("composite_score", 0),
                "信号分": s["signal_score"],
                "RSI": s["rsi"], "ADX": s["adx"],
                "均线": s["ma_state"], "量能": s["vol_pattern"],
                "量比": s["vol_ratio_60"],
                "5日%": s["return_5d"], "20日%": s["return_20d"],
                "60日%": s["return_60d"], "偏离20%": s["bias_20"],
                "底背离": "✓" if s["macd_bottom_div"] else "",
                "顶背离": "✓" if s["macd_top_div"] else "",
                "周线确认": "✓" if s.get("weekly_confirm") else "",
                "周线趋势": s.get("weekly_trend", ""),
                "基本面": s.get("fundamental_grade", ""),
                "ROE": s.get("roe", ""),
                "K线": s["candle_patterns"],
                "PE": s.get("pe", ""),
                "详情": s["signal_details"],
            })
        return pd.DataFrame(rows)

    def _build_stats_df(self) -> pd.DataFrame:
        rows = []
        for cat in StockScreener.CATEGORIES:
            stocks = self.results.get(cat, [])
            if not stocks:
                rows.append({"类别": cat, "数量": 0, "占比%": 0})
                continue
            count = len(stocks)
            total = len(self.all_analyzed)
            pct = count / total * 100 if total else 0
            avg_rsi = sum(s["rsi"] for s in stocks) / count
            avg_r20 = sum(s["return_20d"] for s in stocks) / count
            avg_composite = sum(s.get("composite_score", 0) for s in stocks) / count
            weekly_confirmed = sum(1 for s in stocks if s.get("weekly_confirm"))

            rows.append({
                "类别": cat, "数量": count, "占比%": round(pct, 1),
                "平均RSI": round(avg_rsi, 1),
                "平均20日%": round(avg_r20, 1),
                "平均综合分": round(avg_composite, 1),
                "周线确认数": weekly_confirmed,
            })
        return pd.DataFrame(rows)

    def save_to_db(self):
        """保存结果到数据库"""
        # 保存筛选记录
        self.db.save_screening_results(self.all_analyzed, source="live")

        # 保存每日统计
        total = len(self.all_analyzed)
        stats = {
            "total": total,
        }
        for cat in StockScreener.CATEGORIES:
            stats[cat] = len(self.results.get(cat, []))

        # 市场温度
        bull = stats.get("强势上涨趋势", 0)
        bear = stats.get("弱势下跌趋势", 0)
        if total > 0:
            if bull / total > 0.4:
                stats["market_temperature"] = "过热"
            elif bear / total > 0.4:
                stats["market_temperature"] = "过冷"
            else:
                stats["market_temperature"] = "正常"
        else:
            stats["market_temperature"] = "未知"

        self.db.save_daily_stats(stats)
