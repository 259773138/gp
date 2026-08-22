"""
周线级别分析模块
对日线信号进行高级别确认，提升信号可靠性
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from config import WEEKLY_CONFIG


class WeeklyAnalyzer:
    """周线分析器"""

    @staticmethod
    def analyze(weekly_df: pd.DataFrame) -> Optional[Dict]:
        """
        分析周线级别信号
        返回周线确认结果
        """
        if weekly_df is None or len(weekly_df) < 20:
            return None

        df = weekly_df.copy()

        # 计算周线指标
        df = WeeklyAnalyzer._calc_weekly_indicators(df)

        if len(df) < 20:
            return None

        last = df.iloc[-1]

        result = {
            "weekly_trend": "unknown",
            "weekly_ma_state": "unknown",
            "weekly_rsi": None,
            "weekly_macd_hist": None,
            "weekly_macd_cross": "none",
            "weekly_support_level": None,
            "weekly_resistance_level": None,
            "weekly_confirm_bullish": False,
            "weekly_confirm_bearish": False,
            "weekly_bottom_divergence": False,
            "weekly_top_divergence": False,
            "weekly_score": 0,
        }

        # 1. 周线均线状态
        ma_state = WeeklyAnalyzer._check_ma_state(df)
        result["weekly_ma_state"] = ma_state["state"]

        # 2. 周线RSI
        if pd.notna(last.get("w_rsi")):
            result["weekly_rsi"] = round(last["w_rsi"], 1)

        # 3. 周线MACD
        if pd.notna(last.get("w_macd_hist")):
            result["weekly_macd_hist"] = round(last["w_macd_hist"], 4)

        # MACD金叉/死叉判断
        if len(df) >= 3:
            prev_hist = df.iloc[-2].get("w_macd_hist", 0)
            curr_hist = last.get("w_macd_hist", 0)
            if pd.notna(prev_hist) and pd.notna(curr_hist):
                if prev_hist <= 0 < curr_hist:
                    result["weekly_macd_cross"] = "golden"
                elif prev_hist >= 0 > curr_hist:
                    result["weekly_macd_cross"] = "death"

        # 4. 周线背离检测
        divergence = WeeklyAnalyzer._detect_divergence(df)
        result["weekly_bottom_divergence"] = divergence["bottom"]
        result["weekly_top_divergence"] = divergence["top"]

        # 5. 支撑/阻力位
        result["weekly_support_level"] = round(df["low"].tail(10).min(), 2)
        result["weekly_resistance_level"] = round(df["high"].tail(10).max(), 2)

        # 6. 综合判断趋势
        result["weekly_trend"] = WeeklyAnalyzer._judge_trend(df, result)

        # 7. 确认信号
        bullish_score = 0
        bearish_score = 0

        # 均线多头
        if ma_state["state"] in ["完全多头", "偏多"]:
            bullish_score += 2
        elif ma_state["state"] in ["完全空头", "偏空"]:
            bearish_score += 2

        # RSI
        rsi = result["weekly_rsi"]
        if rsi is not None:
            if rsi < 30:
                bullish_score += 2  # 周线超卖是强信号
            elif rsi > 70:
                bearish_score += 2

        # MACD
        if result["weekly_macd_cross"] == "golden":
            bullish_score += 3  # 周线金叉权重大
        elif result["weekly_macd_cross"] == "death":
            bearish_score += 3

        # 背离
        if result["weekly_bottom_divergence"]:
            bullish_score += 3
        if result["weekly_top_divergence"]:
            bearish_score += 3

        # MACD柱状图方向
        hist = result["weekly_macd_hist"]
        if hist is not None:
            if hist > 0:
                bullish_score += 1
            else:
                bearish_score += 1

        result["weekly_confirm_bullish"] = bullish_score >= 3
        result["weekly_confirm_bearish"] = bearish_score >= 3
        result["weekly_score"] = bullish_score - bearish_score

        return result

    @staticmethod
    def _calc_weekly_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """计算周线级别的技术指标"""
        # 均线
        for period in WEEKLY_CONFIG["ma_periods"]:
            df[f"w_ma{period}"] = df["close"].rolling(window=period).mean()

        # MACD
        fast = WEEKLY_CONFIG["macd_fast"]
        slow = WEEKLY_CONFIG["macd_slow"]
        signal = WEEKLY_CONFIG["macd_signal"]
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        df["w_macd_dif"] = ema_fast - ema_slow
        df["w_macd_dea"] = df["w_macd_dif"].ewm(span=signal, adjust=False).mean()
        df["w_macd_hist"] = 2 * (df["w_macd_dif"] - df["w_macd_dea"])

        # RSI
        period = WEEKLY_CONFIG["rsi_period"]
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["w_rsi"] = 100 - (100 / (1 + rs))

        return df

    @staticmethod
    def _check_ma_state(df: pd.DataFrame) -> Dict:
        """检查周线均线排列"""
        last = df.iloc[-1]
        ma_values = []
        for period in WEEKLY_CONFIG["ma_periods"]:
            col = f"w_ma{period}"
            if col in df.columns and pd.notna(last[col]):
                ma_values.append((period, last[col]))

        if len(ma_values) < 3:
            return {"state": "unknown", "bullish": 0, "bearish": 0}

        bullish = bearish = 0
        for i in range(len(ma_values) - 1):
            if ma_values[i][1] > ma_values[i+1][1]:
                bullish += 1
            else:
                bearish += 1

        total = len(ma_values) - 1
        if bullish == total:
            state = "完全多头"
        elif bearish == total:
            state = "完全空头"
        elif bullish > bearish:
            state = "偏多"
        elif bearish > bullish:
            state = "偏空"
        else:
            state = "混乱"

        return {"state": state, "bullish": bullish, "bearish": bearish}

    @staticmethod
    def _detect_divergence(df: pd.DataFrame) -> Dict:
        """检测周线MACD背离"""
        result = {"bottom": False, "top": False}
        if len(df) < 30:
            return result

        recent = df.tail(30)
        prices = recent["close"].values
        dif = recent["w_macd_dif"].values
        if np.isnan(dif).all():
            return result

        try:
            half = len(prices) // 2
            # 底背离
            if np.nanmin(prices[half:]) < np.nanmin(prices[:half]):
                if np.nanmin(dif[half:]) > np.nanmin(dif[:half]):
                    result["bottom"] = True
            # 顶背离
            if np.nanmax(prices[half:]) > np.nanmax(prices[:half]):
                if np.nanmax(dif[half:]) < np.nanmax(dif[:half]):
                    result["top"] = True
        except Exception:
            pass

        return result

    @staticmethod
    def _judge_trend(df: pd.DataFrame, signals: Dict) -> str:
        """综合判断周线趋势"""
        last = df.iloc[-1]
        ma_state = signals["weekly_ma_state"]

        # 价格在周线20周均线之上/下
        w_ma20 = last.get("w_ma20")
        price = last["close"]

        if ma_state in ["完全多头", "偏多"] and pd.notna(w_ma20) and price > w_ma20:
            return "上涨趋势"
        elif ma_state in ["完全空头", "偏空"] and pd.notna(w_ma20) and price < w_ma20:
            return "下跌趋势"
        else:
            return "震荡"
