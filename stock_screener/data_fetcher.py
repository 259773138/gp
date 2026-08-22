"""
数据获取模块（增强版）
支持日线、周线数据获取，带重试机制。
修复（相对原版方案）：
  - K线显式限制起止日期，避免每次拉全历史
  - 周线 date 取该周最后一个实际交易日，保证“截至某日”的回放切片正确
"""

import time
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from config import STOCK_POOL_CONFIG, DATA_CONFIG


class DataFetcher:
    """数据获取器"""

    def __init__(self):
        self.stock_pool = pd.DataFrame()
        self.kline_cache: Dict[str, pd.DataFrame] = {}
        self.weekly_cache: Dict[str, pd.DataFrame] = {}
        self.basic_info_cache: Dict[str, dict] = {}

    # ========================================
    # 1. 构建股票池
    # ========================================
    def build_stock_pool(self) -> pd.DataFrame:
        """构建完整股票池"""
        print("=" * 60)
        print("📊 开始构建股票池...")
        print("=" * 60)

        all_stocks = []

        # 1) 热门股票
        all_stocks.append(self._get_hot_stocks())

        # 2) 指数成分股
        for index_code, index_name in STOCK_POOL_CONFIG["index_stocks"].items():
            all_stocks.append(self._get_index_stocks(index_code, index_name))
            time.sleep(0.5)

        # 3) 涨跌极端股票
        all_stocks.append(self._get_extreme_stocks())

        # 合并去重过滤
        pool = pd.concat(all_stocks, ignore_index=True)
        pool = pool.drop_duplicates(subset=["code"], keep="first")
        pool = self._filter_stocks(pool)

        # 补充
        if len(pool) < STOCK_POOL_CONFIG["min_pool_size"]:
            supplement = self._supplement_stocks(pool)
            if not supplement.empty:
                pool = pd.concat([pool, supplement], ignore_index=True)
                pool = pool.drop_duplicates(subset=["code"], keep="first")

        pool = pool.reset_index(drop=True)
        self.stock_pool = pool
        print(f"\n✅ 最终股票池: {len(pool)} 支")
        return pool

    def _get_hot_stocks(self) -> pd.DataFrame:
        try:
            print("  📌 获取热门股票...")
            df = ak.stock_hot_rank_em()
            if df is not None and not df.empty:
                result = pd.DataFrame({
                    "code": df["股票代码"].astype(str).str.zfill(6),
                    "name": df["股票名称"],
                    "source": "热门榜"
                }).head(STOCK_POOL_CONFIG["hot_stock_count"])
                print(f"    ✅ {len(result)} 支")
                return result
        except Exception as e:
            print(f"    ❌ 失败: {e}")
        return pd.DataFrame(columns=["code", "name", "source"])

    def _get_index_stocks(self, index_code: str, index_name: str) -> pd.DataFrame:
        try:
            print(f"  📌 获取{index_name}成分股...")
            df = ak.index_stock_cons(symbol=index_code)
            if df is not None and not df.empty:
                code_col = next((c for c in df.columns if "代码" in c or "code" in c.lower()), None)
                name_col = next((c for c in df.columns if "名称" in c or "name" in c.lower()), None)
                if code_col:
                    result = pd.DataFrame({
                        "code": df[code_col].astype(str).str.zfill(6),
                        "name": df[name_col] if name_col else "Unknown",
                        "source": index_name
                    })
                    print(f"    ✅ {len(result)} 支")
                    return result
        except Exception as e:
            print(f"    ❌ 失败: {e}")
        return pd.DataFrame(columns=["code", "name", "source"])

    def _get_extreme_stocks(self) -> pd.DataFrame:
        try:
            print("  📌 获取涨跌极端股票...")
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                df = df.dropna(subset=["涨跌幅"])
                extreme = pd.concat([
                    df.nlargest(20, "涨跌幅"),
                    df.nsmallest(20, "涨跌幅")
                ])
                result = pd.DataFrame({
                    "code": extreme["代码"].astype(str).str.zfill(6),
                    "name": extreme["名称"],
                    "source": "涨跌极端"
                })
                print(f"    ✅ {len(result)} 支")
                return result
        except Exception as e:
            print(f"    ❌ 失败: {e}")
        return pd.DataFrame(columns=["code", "name", "source"])

    def _supplement_stocks(self, existing_pool: pd.DataFrame) -> pd.DataFrame:
        try:
            print("  📌 补充活跃股票...")
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                existing_codes = set(existing_pool["code"].values)
                df = df[~df["代码"].astype(str).str.zfill(6).isin(existing_codes)]
                df = df.dropna(subset=["成交额"]).sort_values("成交额", ascending=False)
                need = STOCK_POOL_CONFIG["min_pool_size"] - len(existing_pool) + 20
                supplement = df.head(need)
                if supplement.empty:
                    return pd.DataFrame(columns=["code", "name", "source"])
                result = pd.DataFrame({
                    "code": supplement["代码"].astype(str).str.zfill(6),
                    "name": supplement["名称"],
                    "source": "活跃补充"
                })
                print(f"    ✅ 补充 {len(result)} 支")
                return result
        except Exception as e:
            print(f"    ❌ 补充失败: {e}")
        return pd.DataFrame(columns=["code", "name", "source"])

    def _filter_stocks(self, pool: pd.DataFrame) -> pd.DataFrame:
        original = len(pool)
        if STOCK_POOL_CONFIG["exclude_st"]:
            pool = pool[~pool["name"].str.contains("ST|退市", na=False)]
        if STOCK_POOL_CONFIG["exclude_bse"]:
            pool = pool[~pool["code"].str.startswith("8")]
        if STOCK_POOL_CONFIG["exclude_kcb"]:
            pool = pool[~pool["code"].str.startswith("688")]
        if len(pool) == 0:
            print("  ⚠️ 股票池为空")
            return pool
        filtered = original - len(pool)
        if filtered > 0:
            print(f"  🔍 过滤 {filtered} 支")
        return pool

    # ========================================
    # 2. K线数据（带重试）
    # ========================================
    def _fetch_with_retry(self, func, *args, **kwargs):
        """带重试的数据获取"""
        for attempt in range(DATA_CONFIG["max_retry"]):
            try:
                return func(*args, **kwargs)
            except Exception:
                if attempt < DATA_CONFIG["max_retry"] - 1:
                    time.sleep(DATA_CONFIG["retry_delay"])
                else:
                    return None

    def fetch_kline(self, code: str) -> Optional[pd.DataFrame]:
        """获取日线数据"""
        if code in self.kline_cache:
            return self.kline_cache[code]

        end_date = datetime.now().strftime("%Y%m%d")
        # 多取一些自然日，覆盖 kline_days 个交易日
        start_date = (datetime.now() - timedelta(days=int(DATA_CONFIG["kline_days"] * 1.8))).strftime("%Y%m%d")

        def _fetch():
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date=start_date, end_date=end_date, adjust="qfq")
            if df is None or df.empty or len(df) < 60:
                return None

            df = df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "振幅": "amplitude",
                "涨跌幅": "pct_change", "涨跌额": "price_change",
                "换手率": "turnover_rate"
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            df = df.tail(DATA_CONFIG["kline_days"]).reset_index(drop=True)
            return df

        df = self._fetch_with_retry(_fetch)
        if df is not None:
            self.kline_cache[code] = df
        return df

    @staticmethod
    def build_weekly_from_daily(daily: pd.DataFrame) -> Optional[pd.DataFrame]:
        """由日线合成周线；周线 date 取该周最后一个实际交易日"""
        if daily is None or len(daily) < 30:
            return None
        df = daily.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        weekly_last_dates = df["close"].resample("W").apply(lambda s: s.index.max())
        weekly = df.resample("W").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
        })
        weekly = weekly.dropna(subset=["close"])
        weekly["date"] = weekly_last_dates.reindex(weekly.index).values
        weekly = weekly.dropna(subset=["date"]).reset_index(drop=True)
        weekly["pct_change"] = weekly["close"].pct_change() * 100
        return weekly

    def fetch_weekly_kline(self, code: str) -> Optional[pd.DataFrame]:
        """获取周线数据（从日线合成）"""
        if code in self.weekly_cache:
            return self.weekly_cache[code]

        daily = self.fetch_kline(code)
        weekly = self.build_weekly_from_daily(daily)
        if weekly is None or len(weekly) < 20:
            return None

        self.weekly_cache[code] = weekly
        return weekly

    def fetch_all_klines(self) -> Dict[str, pd.DataFrame]:
        """批量获取所有K线"""
        print("\n" + "=" * 60)
        print("📈 获取K线数据...")
        print("=" * 60)

        total = len(self.stock_pool)
        success = fail = 0

        for idx, row in self.stock_pool.iterrows():
            code = row["code"]
            df = self.fetch_kline(code)
            if df is not None:
                # 同时生成周线
                self.fetch_weekly_kline(code)
                success += 1
            else:
                fail += 1

            if (idx + 1) % 10 == 0 or idx == total - 1:
                print(f"  进度: {idx+1}/{total} | ✅{success} ❌{fail}")

            time.sleep(DATA_CONFIG["fetch_interval"])

        print(f"\n✅ 完成: {success}/{total}")
        return self.kline_cache

    # ========================================
    # 3. 实时行情
    # ========================================
    def fetch_realtime_quotes(self) -> pd.DataFrame:
        try:
            print("\n📊 获取实时行情...")
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                quotes = pd.DataFrame({
                    "code": df["代码"].astype(str).str.zfill(6),
                    "name": df["名称"],
                    "price": pd.to_numeric(df["最新价"], errors="coerce"),
                    "pct_change": pd.to_numeric(df["涨跌幅"], errors="coerce"),
                    "volume": pd.to_numeric(df["成交量"], errors="coerce"),
                    "amount": pd.to_numeric(df["成交额"], errors="coerce"),
                    "turnover": pd.to_numeric(df["换手率"], errors="coerce"),
                    "pe": pd.to_numeric(df.get("市盈率-动态", pd.Series(dtype=float)), errors="coerce"),
                    "pb": pd.to_numeric(df.get("市净率", pd.Series(dtype=float)), errors="coerce"),
                    "total_mv": pd.to_numeric(df.get("总市值", pd.Series(dtype=float)), errors="coerce"),
                    "circ_mv": pd.to_numeric(df.get("流通市值", pd.Series(dtype=float)), errors="coerce"),
                })
                print(f"  ✅ {len(quotes)} 支")
                return quotes
        except Exception as e:
            print(f"  ❌ 失败: {e}")
        return pd.DataFrame()

    def fetch_single_price(self, code: str) -> Optional[float]:
        """获取单只股票当前价格（用于回测检验）"""
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
            if df is not None and not df.empty:
                return float(df.iloc[-1]["收盘"])
        except Exception:
            pass
        return None

    def fetch_price_at_date(self, code: str, target_date: str) -> Optional[float]:
        """获取指定日期的收盘价"""
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
            if df is not None and not df.empty:
                df["日期"] = pd.to_datetime(df["日期"])
                target = pd.to_datetime(target_date)
                # 找最接近的交易日
                df["diff"] = (df["日期"] - target).abs()
                closest = df.loc[df["diff"].idxmin()]
                # 确保在3个交易日内
                if closest["diff"].days <= 5:
                    return float(closest["收盘"])
        except Exception:
            pass
        return None
