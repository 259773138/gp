"""
技术指标计算模块
"""

import numpy as np
import pandas as pd
from typing import Dict
from config import INDICATOR_CONFIG


class TechnicalIndicators:
    """技术指标计算器"""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) < 60:
            return df
        df = df.copy()
        df = TechnicalIndicators.calc_ma(df)
        df = TechnicalIndicators.calc_macd(df)
        df = TechnicalIndicators.calc_rsi(df)
        df = TechnicalIndicators.calc_bollinger(df)
        df = TechnicalIndicators.calc_atr(df)
        df = TechnicalIndicators.calc_adx(df)
        df = TechnicalIndicators.calc_volume_indicators(df)
        df = TechnicalIndicators.calc_auxiliary(df)
        return df

    @staticmethod
    def calc_ma(df: pd.DataFrame) -> pd.DataFrame:
        for period in INDICATOR_CONFIG["ma_periods"]:
            df[f"ma{period}"] = df["close"].rolling(window=period).mean()
        return df

    @staticmethod
    def calc_macd(df: pd.DataFrame) -> pd.DataFrame:
        fast = INDICATOR_CONFIG["macd_fast"]
        slow = INDICATOR_CONFIG["macd_slow"]
        signal = INDICATOR_CONFIG["macd_signal"]
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        df["macd_dif"] = ema_fast - ema_slow
        df["macd_dea"] = df["macd_dif"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = 2 * (df["macd_dif"] - df["macd_dea"])
        return df

    @staticmethod
    def calc_rsi(df: pd.DataFrame) -> pd.DataFrame:
        period = INDICATOR_CONFIG["rsi_period"]
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def calc_bollinger(df: pd.DataFrame) -> pd.DataFrame:
        period = INDICATOR_CONFIG["boll_period"]
        std_dev = INDICATOR_CONFIG["boll_std"]
        df["boll_mid"] = df["close"].rolling(window=period).mean()
        rolling_std = df["close"].rolling(window=period).std()
        df["boll_upper"] = df["boll_mid"] + std_dev * rolling_std
        df["boll_lower"] = df["boll_mid"] - std_dev * rolling_std
        df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / df["boll_mid"]
        df["boll_pct_b"] = (df["close"] - df["boll_lower"]) / \
                           (df["boll_upper"] - df["boll_lower"]).replace(0, np.nan)
        return df

    @staticmethod
    def calc_atr(df: pd.DataFrame) -> pd.DataFrame:
        period = INDICATOR_CONFIG["atr_period"]
        high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=period).mean()
        df["atr_pct"] = df["atr"] / df["close"] * 100
        return df

    @staticmethod
    def calc_adx(df: pd.DataFrame) -> pd.DataFrame:
        period = INDICATOR_CONFIG["adx_period"]
        high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr.replace(0, np.nan))
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        df["adx"] = dx.ewm(alpha=1/period, min_periods=period).mean()
        return df

    @staticmethod
    def calc_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        for period in INDICATOR_CONFIG["vol_ma_periods"]:
            df[f"vol_ma{period}"] = df["volume"].rolling(window=period).mean()
        df["volume_ratio"] = df["volume"] / df["vol_ma5"].replace(0, np.nan)
        df["volume_ratio_60"] = df["volume"] / df["vol_ma60"].replace(0, np.nan)
        return df

    @staticmethod
    def calc_auxiliary(df: pd.DataFrame) -> pd.DataFrame:
        for n in [5, 10, 20, 60, 120]:
            df[f"return_{n}d"] = df["close"].pct_change(n) * 100
        for period in INDICATOR_CONFIG["ma_periods"]:
            ma_col = f"ma{period}"
            if ma_col in df.columns:
                df[f"bias_{period}"] = (df["close"] - df[ma_col]) / df[ma_col].replace(0, np.nan) * 100
        df["high_20d"] = df["high"].rolling(20).max()
        df["low_20d"] = df["low"].rolling(20).min()
        df["high_60d"] = df["high"].rolling(60).max()
        df["low_60d"] = df["low"].rolling(60).min()
        range_20 = df["high_20d"] - df["low_20d"]
        df["price_position_20d"] = (df["close"] - df["low_20d"]) / range_20.replace(0, np.nan)
        range_60 = df["high_60d"] - df["low_60d"]
        df["price_position_60d"] = (df["close"] - df["low_60d"]) / range_60.replace(0, np.nan)
        return df

    @staticmethod
    def detect_macd_divergence(df: pd.DataFrame, lookback: int = 60) -> Dict:
        result = {"bottom_divergence": False, "top_divergence": False}
        if df is None or len(df) < lookback + 10:
            return result
        recent = df.tail(lookback).copy()
        prices = recent["close"].values
        dif = recent["macd_dif"].values
        if np.isnan(dif).all():
            return result
        try:
            half = len(prices) // 2
            if np.min(prices[half:]) < np.min(prices[:half]):
                if np.nanmin(dif[half:]) > np.nanmin(dif[:half]):
                    result["bottom_divergence"] = True
            if np.max(prices[half:]) > np.max(prices[:half]):
                if np.nanmax(dif[half:]) < np.nanmax(dif[:half]):
                    result["top_divergence"] = True
        except Exception:
            pass
        return result

    @staticmethod
    def detect_ma_alignment(df: pd.DataFrame) -> Dict:
        if df is None or len(df) < 2:
            return {"bullish_count": 0, "bearish_count": 0, "state": "unknown", "total_pairs": 0}
        last = df.iloc[-1]
        ma_values = []
        for period in INDICATOR_CONFIG["ma_periods"]:
            col = f"ma{period}"
            if col in df.columns and pd.notna(last[col]):
                ma_values.append((period, last[col]))
        if len(ma_values) < 3:
            return {"bullish_count": 0, "bearish_count": 0, "state": "unknown", "total_pairs": 0}
        bullish = bearish = 0
        for i in range(len(ma_values) - 1):
            if ma_values[i][1] > ma_values[i+1][1]:
                bullish += 1
            elif ma_values[i][1] < ma_values[i+1][1]:
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
        return {"bullish_count": bullish, "bearish_count": bearish, "state": state, "total_pairs": total}

    @staticmethod
    def detect_volume_pattern(df: pd.DataFrame) -> Dict:
        if df is None or len(df) < 5:
            return {"pattern": "unknown", "volume_ratio_60": 1.0, "vol_price_relation": "unknown"}
        last = df.iloc[-1]
        vol_ratio = last.get("volume_ratio_60", 1.0)
        if pd.isna(vol_ratio):
            vol_ratio = 1.0
        if vol_ratio > 3.0:
            pattern = "异常放量"
        elif vol_ratio > 2.0:
            pattern = "明显放量"
        elif vol_ratio > 1.5:
            pattern = "温和放量"
        elif vol_ratio > 0.8:
            pattern = "正常"
        elif vol_ratio > 0.4:
            pattern = "温和缩量"
        else:
            pattern = "极度缩量"
        recent = df.tail(5)
        price_up = recent["close"].iloc[-1] > recent["close"].iloc[0]
        vol_up = recent["volume"].iloc[-1] > recent["volume"].iloc[0]
        if price_up and vol_up:
            vp = "量价齐升"
        elif price_up and not vol_up:
            vp = "缩量上涨"
        elif not price_up and vol_up:
            vp = "放量下跌"
        else:
            vp = "缩量下跌"
        return {"pattern": pattern, "volume_ratio_60": round(vol_ratio, 2), "vol_price_relation": vp}

    @staticmethod
    def detect_candlestick_patterns(df: pd.DataFrame) -> list:
        patterns = []
        if df is None or len(df) < 3:
            return patterns
        last = df.iloc[-1]
        prev = df.iloc[-2]
        body = abs(last["close"] - last["open"])
        upper_shadow = last["high"] - max(last["close"], last["open"])
        lower_shadow = min(last["close"], last["open"]) - last["low"]
        total_range = last["high"] - last["low"]
        if total_range == 0:
            return patterns
        if upper_shadow > body * 2 and upper_shadow > total_range * 0.5:
            patterns.append("长上影线")
        if lower_shadow > body * 2 and lower_shadow > total_range * 0.5:
            patterns.append("长下影线/锤子")
        if body < total_range * 0.1:
            patterns.append("十字星")
        pct = last.get("pct_change", 0)
        if pd.notna(pct):
            if pct > 5:
                patterns.append("大阳线")
            elif pct < -5:
                patterns.append("大阴线")
        if (prev["close"] < prev["open"] and last["close"] > last["open"] and
            last["close"] > prev["open"] and last["open"] < prev["close"]):
            patterns.append("看涨吞没")
        if (prev["close"] > prev["open"] and last["close"] < last["open"] and
            last["close"] < prev["open"] and last["open"] > prev["close"]):
            patterns.append("看跌吞没")
        return patterns
