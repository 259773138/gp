"""
筛选引擎（自适应权重版）
整合日线信号 + 周线确认 + 基本面评分的多维度综合筛选。
改造（相对原版方案）：
  - 规则分类逻辑保留（用于类别归属）
  - 原来的"固定加权综合分"仅作冷启动兜底；当回测数据足够时，
    由 weight_optimizer 学出的自适应模型给出 adaptive_score(-AI分)，
    各分类内部以 AI 分排序（看多类降序、看空类升序）
  - 每只股票记录统一特征向量 features，供后续继续训练
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from config import SCREENER_CONFIG, INDICATOR_CONFIG, WEEKLY_CONFIG
from indicators import TechnicalIndicators
from weekly_indicators import WeeklyAnalyzer
from fundamental import FundamentalAnalyzer
from weight_optimizer import features_from_analysis, load_active_model, score_with_model


class StockScreener:
    """股票筛选引擎"""

    CATEGORIES = {
        "底部反转候选": "🟢 下跌可能结束，多信号共振+周线确认",
        "强势上涨趋势": "🔵 均线多头排列，趋势向上延续",
        "顶部风险警示": "🔴 上涨可能见顶，出现多个顶部信号",
        "弱势下跌趋势": "🟡 均线空头排列，趋势向下延续",
        "横盘震荡整理": "⚪ 趋势不明朗，等待方向选择",
        "超跌反弹机会": "🟣 短期超跌严重，可能出现技术性反弹",
    }
    BULL_CATEGORIES = {"底部反转候选", "强势上涨趋势", "超跌反弹机会"}
    BEAR_CATEGORIES = {"顶部风险警示", "弱势下跌趋势"}

    def __init__(self, model="auto", db_path: str = None):
        """
        model: "auto"=自动加载已训练权重; None=明确不使用模型; dict=直接使用
        """
        self.results: Dict[str, List[Dict]] = {cat: [] for cat in self.CATEGORIES}
        self.all_analyzed: List[Dict] = []
        self.fundamental_analyzer = FundamentalAnalyzer()
        if model == "auto":
            from config import DB_PATH as _DBP
            self.model = load_active_model(db_path or _DBP)
        else:
            self.model = model

    def analyze_single_stock(
        self, code: str, name: str,
        daily_df: pd.DataFrame,
        weekly_df: pd.DataFrame = None,
        realtime_info: dict = None,
        fundamental_info: dict = None
    ) -> Optional[Dict]:
        """分析单只股票"""
        # 日线指标
        df = TechnicalIndicators.calculate_all(daily_df)
        if df is None or len(df) < 60:
            return None

        last = df.iloc[-1]

        # 基本信息
        analysis = {
            "code": code,
            "name": name,
            "price": round(last["close"], 2),
            "pct_change": round(last.get("pct_change", 0), 2) if pd.notna(last.get("pct_change")) else 0,
        }

        # ---- 日线技术指标 ----
        analysis["rsi"] = round(last["rsi"], 2) if pd.notna(last.get("rsi")) else 50
        analysis["macd_dif"] = round(last["macd_dif"], 4) if pd.notna(last.get("macd_dif")) else 0
        analysis["macd_hist"] = round(last["macd_hist"], 4) if pd.notna(last.get("macd_hist")) else 0
        analysis["adx"] = round(last["adx"], 2) if pd.notna(last.get("adx")) else 0

        ma_info = TechnicalIndicators.detect_ma_alignment(df)
        analysis["ma_state"] = ma_info["state"]
        analysis["ma_bullish_count"] = ma_info["bullish_count"]
        analysis["ma_bearish_count"] = ma_info["bearish_count"]

        vol_info = TechnicalIndicators.detect_volume_pattern(df)
        analysis["vol_pattern"] = vol_info["pattern"]
        analysis["vol_ratio_60"] = vol_info["volume_ratio_60"]
        analysis["vol_price_relation"] = vol_info["vol_price_relation"]

        divergence = TechnicalIndicators.detect_macd_divergence(df)
        analysis["macd_bottom_div"] = divergence["bottom_divergence"]
        analysis["macd_top_div"] = divergence["top_divergence"]

        candle_patterns = TechnicalIndicators.detect_candlestick_patterns(df)
        analysis["candle_patterns"] = ", ".join(candle_patterns) if candle_patterns else "无"

        for col in ["return_5d", "return_10d", "return_20d", "return_60d", "return_120d"]:
            analysis[col] = round(last.get(col, 0), 2) if pd.notna(last.get(col)) else 0

        analysis["boll_pct_b"] = round(last.get("boll_pct_b", 0.5), 3) if pd.notna(last.get("boll_pct_b")) else 0.5
        analysis["price_pos_20d"] = round(last.get("price_position_20d", 0.5), 3) if pd.notna(last.get("price_position_20d")) else 0.5
        analysis["price_pos_60d"] = round(last.get("price_position_60d", 0.5), 3) if pd.notna(last.get("price_position_60d")) else 0.5
        analysis["bias_20"] = round(last.get("bias_20", 0), 2) if pd.notna(last.get("bias_20")) else 0
        analysis["bias_60"] = round(last.get("bias_60", 0), 2) if pd.notna(last.get("bias_60")) else 0
        analysis["atr_pct"] = round(last.get("atr_pct", 0), 2) if pd.notna(last.get("atr_pct")) else 0

        # ---- 实时估值 ----
        if realtime_info:
            analysis["pe"] = realtime_info.get("pe")
            analysis["pb"] = realtime_info.get("pb")
            analysis["total_mv"] = realtime_info.get("total_mv")
        else:
            analysis["pe"] = analysis["pb"] = analysis["total_mv"] = None

        # ---- 周线分析 ----
        weekly_result = None
        if weekly_df is not None:
            weekly_result = WeeklyAnalyzer.analyze(weekly_df)

        if weekly_result:
            analysis["weekly_trend"] = weekly_result["weekly_trend"]
            analysis["weekly_ma_state"] = weekly_result["weekly_ma_state"]
            analysis["weekly_rsi"] = weekly_result["weekly_rsi"]
            analysis["weekly_macd_cross"] = weekly_result["weekly_macd_cross"]
            analysis["weekly_confirm_bullish"] = weekly_result["weekly_confirm_bullish"]
            analysis["weekly_confirm_bearish"] = weekly_result["weekly_confirm_bearish"]
            analysis["weekly_bottom_div"] = weekly_result["weekly_bottom_divergence"]
            analysis["weekly_top_div"] = weekly_result["weekly_top_divergence"]
            analysis["weekly_score"] = weekly_result["weekly_score"]
            analysis["weekly_support"] = weekly_result["weekly_support_level"]
            analysis["weekly_resistance"] = weekly_result["weekly_resistance_level"]
        else:
            analysis["weekly_trend"] = "N/A"
            analysis["weekly_ma_state"] = "N/A"
            analysis["weekly_rsi"] = None
            analysis["weekly_macd_cross"] = "none"
            analysis["weekly_confirm_bullish"] = False
            analysis["weekly_confirm_bearish"] = False
            analysis["weekly_bottom_div"] = False
            analysis["weekly_top_div"] = False
            analysis["weekly_score"] = 0
            analysis["weekly_support"] = None
            analysis["weekly_resistance"] = None

        # ---- 基本面 ----
        if fundamental_info:
            analysis["fundamental_score"] = fundamental_info.get("fundamental_score", 0)
            analysis["fundamental_grade"] = fundamental_info.get("fundamental_grade", "N/A")
            analysis["roe"] = fundamental_info.get("roe")
            analysis["revenue_growth"] = fundamental_info.get("revenue_growth")
            analysis["profit_growth"] = fundamental_info.get("profit_growth")
            analysis["gross_margin"] = fundamental_info.get("gross_margin")
            analysis["debt_ratio"] = fundamental_info.get("debt_ratio")
            analysis["fundamental_details"] = " | ".join(fundamental_info.get("fundamental_details", []))
        else:
            analysis["fundamental_score"] = 0
            analysis["fundamental_grade"] = "N/A"
            analysis["roe"] = analysis["revenue_growth"] = None
            analysis["profit_growth"] = analysis["gross_margin"] = None
            analysis["debt_ratio"] = None
            analysis["fundamental_details"] = ""

        # ---- 规则分类 + 兜底综合分 ----
        signals = self._calculate_signals(analysis, df)
        analysis["signals"] = signals
        analysis["category"] = signals["category"]
        analysis["signal_score"] = signals["score"]
        analysis["signal_details"] = signals["details"]
        analysis["composite_score"] = signals["composite_score"]
        analysis["weekly_confirm"] = signals.get("weekly_confirmed", False)

        # ---- 自适应评分（AI 分）----
        analysis["features"] = features_from_analysis(analysis)
        ai_score = score_with_model(self.model, analysis["features"]) if self.model else None
        analysis["adaptive_score"] = ai_score

        return analysis

    @staticmethod
    def rank_key(stock: Dict) -> float:
        """排序键：优先 AI 分，缺失则用兜底综合分"""
        s = stock.get("adaptive_score")
        if s is not None:
            return s
        return stock.get("composite_score", 0)

    def _calculate_signals(self, analysis: Dict, df: pd.DataFrame) -> Dict:
        """规则分类与兜底综合分（冷启动固定权重）"""
        cfg_up = SCREENER_CONFIG["reversal_up"]
        cfg_down = SCREENER_CONFIG["reversal_down"]
        cfg_trend = SCREENER_CONFIG["trend_continue"]
        score_weights = SCREENER_CONFIG["score_weights"]

        # ====== 日线底部信号 ======
        bottom_score = 0
        bottom_details = []

        if analysis["rsi"] < cfg_up["rsi_oversold"]:
            bottom_score += 2
            bottom_details.append(f"RSI超卖({analysis['rsi']})")
        if analysis["macd_bottom_div"]:
            bottom_score += 3
            bottom_details.append("MACD底背离")
        if analysis["vol_ratio_60"] < cfg_up["volume_shrink_ratio"]:
            bottom_score += 2
            bottom_details.append(f"极度缩量({analysis['vol_ratio_60']})")
        if analysis["return_60d"] < cfg_up["min_drop_pct"]:
            bottom_score += 1
            bottom_details.append(f"60日跌{analysis['return_60d']}%")
        if analysis["boll_pct_b"] < 0.1:
            bottom_score += 2
            bottom_details.append("触及布林下轨")
        if "看涨吞没" in analysis["candle_patterns"]:
            bottom_score += 2
            bottom_details.append("看涨吞没")
        if "长下影线" in analysis["candle_patterns"] or "锤子" in analysis["candle_patterns"]:
            bottom_score += 1
            bottom_details.append("锤子线")
        if analysis["bias_20"] < -10:
            bottom_score += 2
            bottom_details.append(f"严重超跌({analysis['bias_20']}%)")

        # ====== 日线顶部信号 ======
        top_score = 0
        top_details = []

        if analysis["rsi"] > cfg_down["rsi_overbought"]:
            top_score += 2
            top_details.append(f"RSI超买({analysis['rsi']})")
        if analysis["macd_top_div"]:
            top_score += 3
            top_details.append("MACD顶背离")
        if analysis["vol_ratio_60"] > cfg_down["volume_surge_ratio"]:
            top_score += 2
            top_details.append(f"异常放量({analysis['vol_ratio_60']})")
        if analysis["return_60d"] > cfg_down["min_rise_pct"]:
            top_score += 1
            top_details.append(f"60日涨{analysis['return_60d']}%")
        if analysis["boll_pct_b"] > 0.95:
            top_score += 2
            top_details.append("触及布林上轨")
        if "看跌吞没" in analysis["candle_patterns"]:
            top_score += 2
            top_details.append("看跌吞没")
        if "长上影线" in analysis["candle_patterns"]:
            top_score += 1
            top_details.append("长上影线")
        if analysis["bias_20"] > 15:
            top_score += 2
            top_details.append(f"严重超涨({analysis['bias_20']}%)")

        # ====== 趋势信号 ======
        trend_bull = trend_bear = 0
        trend_details = []

        ma_state = analysis["ma_state"]
        if ma_state in ["完全多头", "偏多"]:
            trend_bull += 3
            trend_details.append(f"均线{ma_state}")
        elif ma_state in ["完全空头", "偏空"]:
            trend_bear += 3
            trend_details.append(f"均线{ma_state}")

        if analysis["bias_20"] > 0:
            trend_bull += 1
        else:
            trend_bear += 1

        if analysis["macd_hist"] > 0:
            trend_bull += 1
        else:
            trend_bear += 1

        if analysis["vol_price_relation"] == "量价齐升":
            trend_bull += 2
            trend_details.append("量价齐升")
        elif analysis["vol_price_relation"] == "放量下跌":
            trend_bear += 2
            trend_details.append("放量下跌")

        adx = analysis["adx"]
        strong_trend = adx > cfg_trend["adx_threshold"]
        if strong_trend:
            trend_details.append(f"ADX强趋势({adx})")

        # ====== 周线加权 ======
        weekly_weight = WEEKLY_CONFIG["signal_weight"]
        weekly_confirmed = False

        weekly_bull_bonus = 0
        weekly_bear_bonus = 0

        if analysis["weekly_confirm_bullish"]:
            weekly_bull_bonus = 3
            bottom_details.append("📅周线确认看多")
            trend_details.append("📅周线确认看多")
            weekly_confirmed = True
        if analysis["weekly_confirm_bearish"]:
            weekly_bear_bonus = 3
            top_details.append("📅周线确认看空")
            trend_details.append("📅周线确认看空")
            weekly_confirmed = True
        if analysis["weekly_bottom_div"]:
            weekly_bull_bonus += 2
            bottom_details.append("📅周线底背离")
        if analysis["weekly_top_div"]:
            weekly_bear_bonus += 2
            top_details.append("📅周线顶背离")
        if analysis["weekly_macd_cross"] == "golden":
            weekly_bull_bonus += 2
            bottom_details.append("📅周线MACD金叉")
        elif analysis["weekly_macd_cross"] == "death":
            weekly_bear_bonus += 2
            top_details.append("📅周线MACD死叉")

        bottom_score += int(weekly_bull_bonus * weekly_weight)
        top_score += int(weekly_bear_bonus * weekly_weight)
        trend_bull += int(weekly_bull_bonus * 0.5)
        trend_bear += int(weekly_bear_bonus * 0.5)

        # ====== 分类 ======
        category = self._decide_category(
            bottom_score, top_score, trend_bull, trend_bear, strong_trend, analysis
        )

        if category == "底部反转候选":
            tech_score = bottom_score
            details = bottom_details
        elif category == "顶部风险警示":
            tech_score = top_score
            details = top_details
        elif category in ["强势上涨趋势"]:
            tech_score = trend_bull
            details = trend_details
        elif category in ["弱势下跌趋势"]:
            tech_score = trend_bear
            details = trend_details
        elif category == "超跌反弹机会":
            tech_score = bottom_score
            details = bottom_details
        else:
            tech_score = 0
            details = ["趋势不明朗"]

        # ====== 兜底综合分（固定权重，模型未训练时使用）======
        tech_normalized = min(tech_score / 15 * 100, 100)
        weekly_score_val = analysis.get("weekly_score", 0)
        weekly_normalized = max(min((weekly_score_val + 10) / 20 * 100, 100), 0)
        fund_score = analysis.get("fundamental_score", 0)

        composite = (
            tech_normalized * score_weights["technical"] +
            weekly_normalized * score_weights["weekly"] +
            fund_score * (score_weights["fundamental"] + score_weights.get("valuation", 0))
        )

        # 基本面标注
        fund_grade = analysis.get("fundamental_grade", "N/A")
        if fund_grade in ["A", "B"]:
            details.append(f"基本面{fund_grade}级")
        elif fund_grade == "D":
            details.append(f"⚠️基本面D级")

        return {
            "category": category,
            "score": tech_score,
            "composite_score": round(composite, 1),
            "details": " | ".join(details) if details else "无明显信号",
            "bottom_score": bottom_score,
            "top_score": top_score,
            "trend_bull_score": trend_bull,
            "trend_bear_score": trend_bear,
            "weekly_confirmed": weekly_confirmed,
        }

    def _decide_category(self, bottom, top, trend_bull, trend_bear, strong_trend, analysis):
        if bottom >= 5 and trend_bear < 4:
            return "底部反转候选"
        if bottom >= 4 and analysis["return_20d"] < -15:
            return "超跌反弹机会"
        if top >= 5:
            return "顶部风险警示"
        if trend_bull >= 4 and strong_trend and trend_bear < 2:
            return "强势上涨趋势"
        if trend_bull >= 3 and trend_bear <= 1:
            return "强势上涨趋势"
        if trend_bear >= 4 and strong_trend and trend_bull < 2:
            return "弱势下跌趋势"
        if trend_bear >= 3 and trend_bull <= 1:
            return "弱势下跌趋势"
        return "横盘震荡整理"

    def screen_all(
        self, stock_pool: pd.DataFrame,
        kline_data: Dict[str, pd.DataFrame],
        weekly_data: Dict[str, pd.DataFrame] = None,
        realtime_quotes: pd.DataFrame = None,
        fundamental_data: Dict[str, Dict] = None
    ) -> Dict[str, List[Dict]]:
        """对所有股票进行筛选"""
        print("\n" + "=" * 60)
        print("🔍 开始综合筛选分析...")
        print("  🤖 自适应模型: " + ("✅ 已启用（回测学习权重）" if self.model else "⏸️ 未就绪（固定权重兜底）"))
        print("=" * 60)

        weekly_data = weekly_data or {}

        rt_dict = {}
        if realtime_quotes is not None and not realtime_quotes.empty:
            for _, row in realtime_quotes.iterrows():
                rt_dict[row["code"]] = row.to_dict()

        total = len(stock_pool)
        analyzed = 0

        for idx, row in stock_pool.iterrows():
            code = row["code"]
            name = row["name"]

            if code not in kline_data:
                continue

            daily_df = kline_data[code]
            weekly_df = weekly_data.get(code)
            rt_info = rt_dict.get(code)
            fund_info = fundamental_data.get(code) if fundamental_data else None

            try:
                result = self.analyze_single_stock(
                    code, name, daily_df, weekly_df, rt_info, fund_info
                )
                if result:
                    self.results[result["category"]].append(result)
                    self.all_analyzed.append(result)
                    analyzed += 1
            except Exception:
                continue

            if (idx + 1) % 20 == 0:
                print(f"  分析进度: {idx+1}/{total}")

        # 排序：看多类按 AI 分降序；看空类按 AI 分升序（越低越危险）
        for cat in self.results:
            reverse = cat not in self.BEAR_CATEGORIES
            self.results[cat] = sorted(
                self.results[cat],
                key=self.rank_key,
                reverse=reverse
            )

        print(f"\n✅ 分析完成: {analyzed} 支")
        for cat, stocks in self.results.items():
            emoji = self.CATEGORIES[cat].split(" ")[0]
            print(f"  {emoji} {cat}: {len(stocks)} 支")

        return self.results
