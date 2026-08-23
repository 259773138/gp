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

    # K线数据源（按优先级）：em=东方财富 tx=腾讯 sina=新浪
    # 海外 Runner 访问东财/新浪可能被按 IP 段拒绝，运行期自动切换到可用源并"粘住"
    KLINE_SOURCES = ["em", "tx", "sina"]

    def __init__(self):
        self.stock_pool = pd.DataFrame()
        self.kline_cache: Dict[str, pd.DataFrame] = {}
        self.weekly_cache: Dict[str, pd.DataFrame] = {}
        self.basic_info_cache: Dict[str, dict] = {}
        self.kline_source_idx = 0        # 当前粘性数据源下标
        self.dead_sources = set()        # 本次运行中已确认不可用的源

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

    def _spot_snapshot(self) -> Optional[pd.DataFrame]:
        """全市场实时快照（东财 → 新浪 备用；统一列结构并缓存）"""
        if getattr(self, "_spot_cache", None) is not None:
            return self._spot_cache

        # 1) 东方财富（含 PE/PB/市值/换手率）
        try:
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                self._spot_cache = pd.DataFrame({
                    "code": df["代码"].astype(str).str.zfill(6),
                    "name": df["名称"],
                    "price": pd.to_numeric(df["最新价"], errors="coerce"),
                    "pct_change": pd.to_numeric(df["涨跌幅"], errors="coerce"),
                    "volume": pd.to_numeric(df["成交量"], errors="coerce"),
                    "amount": pd.to_numeric(df["成交额"], errors="coerce"),
                    "turnover": pd.to_numeric(df.get("换手率", pd.Series(dtype=float)), errors="coerce"),
                    "pe": pd.to_numeric(df.get("市盈率-动态", pd.Series(dtype=float)), errors="coerce"),
                    "pb": pd.to_numeric(df.get("市净率", pd.Series(dtype=float)), errors="coerce"),
                    "total_mv": pd.to_numeric(df.get("总市值", pd.Series(dtype=float)), errors="coerce"),
                    "circ_mv": pd.to_numeric(df.get("流通市值", pd.Series(dtype=float)), errors="coerce"),
                })
                return self._spot_cache
        except Exception as e:
            print(f"    ⚠️ 东财实时快照不可用: {str(e)[:60]}")

        # 2) 新浪备用（无 PE/PB/市值，其余字段可恢复；全市场约 60 秒）
        try:
            print("    🔀 尝试新浪全市场快照（约1分钟）...")
            df = ak.stock_zh_a_spot()
            if df is not None and not df.empty:
                self._spot_cache = pd.DataFrame({
                    "code": df["代码"].astype(str).str.zfill(6),
                    "name": df["名称"],
                    "price": pd.to_numeric(df["最新价"], errors="coerce"),
                    "pct_change": pd.to_numeric(df["涨跌幅"], errors="coerce"),
                    "volume": pd.to_numeric(df["成交量"], errors="coerce"),
                    "amount": pd.to_numeric(df["成交额"], errors="coerce"),
                    "turnover": np.nan,
                    "pe": np.nan,
                    "pb": np.nan,
                    "total_mv": np.nan,
                    "circ_mv": np.nan,
                })
                return self._spot_cache
        except Exception as e:
            print(f"    ⚠️ 新浪实时快照也不可用: {str(e)[:60]}")

        self._spot_cache = pd.DataFrame()
        return self._spot_cache

    def _get_extreme_stocks(self) -> pd.DataFrame:
        try:
            print("  📌 获取涨跌极端股票...")
            df = self._spot_snapshot()
            if df is not None and not df.empty:
                df = df.dropna(subset=["pct_change"])
                extreme = pd.concat([
                    df.nlargest(20, "pct_change"),
                    df.nsmallest(20, "pct_change")
                ])
                result = pd.DataFrame({
                    "code": extreme["code"].astype(str).str.zfill(6),
                    "name": extreme["name"],
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
            df = self._spot_snapshot()
            if df is not None and not df.empty:
                existing_codes = set(existing_pool["code"].values)
                df = df[~df["code"].astype(str).str.zfill(6).isin(existing_codes)]
                df = df.dropna(subset=["amount"]).sort_values("amount", ascending=False)
                need = STOCK_POOL_CONFIG["min_pool_size"] - len(existing_pool) + 20
                supplement = df.head(need)
                if supplement.empty:
                    return pd.DataFrame(columns=["code", "name", "source"])
                result = pd.DataFrame({
                    "code": supplement["code"].astype(str).str.zfill(6),
                    "name": supplement["name"],
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

    # ========================================
    # 2.1 K线多源获取（东财 → 腾讯 → 新浪，自动切换）
    # ========================================
    @staticmethod
    def _prefixed_code(code: str) -> str:
        """600519 -> sh600519；000001/300750 -> sz000001（腾讯/新浪需要带市场前缀）"""
        return f"sh{code}" if code.startswith(("6", "5", "9")) else f"sz{code}"

    def _kline_from_em(self, code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """东方财富日线（原始实现）"""
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
        return df

    def _kline_from_tx(self, code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """腾讯日线：date/open/close/high/low/volume/turnover(小数)/amount"""
        df = ak.stock_zh_a_hist_tx(symbol=self._prefixed_code(code),
                                   start_date=start_date, end_date=end_date, adjust="qfq")
        if df is None or df.empty or len(df) < 60:
            return None
        df["turnover_rate"] = pd.to_numeric(df.get("turnover"), errors="coerce") * 100
        return df

    def _kline_from_sina(self, code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """新浪日线：date/open/high/low/close/volume/amount/outstanding_share/turnover(小数)"""
        df = ak.stock_zh_a_daily(symbol=self._prefixed_code(code),
                                 start_date=start_date, end_date=end_date, adjust="qfq")
        if df is None or df.empty or len(df) < 60:
            return None
        df["turnover_rate"] = pd.to_numeric(df.get("turnover"), errors="coerce") * 100
        return df

    _KLINE_FETCHERS = {
        "em": _kline_from_em,
        "tx": _kline_from_tx,
        "sina": _kline_from_sina,
    }

    def _normalize_kline(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一列结构：补齐可由 OHLCV 推导的衍生列，裁剪到配置长度"""
        df["date"] = pd.to_datetime(df["date"])
        for col in ("open", "close", "high", "low", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

        prev_close = df["close"].shift(1)
        if "pct_change" not in df.columns or df["pct_change"].isna().all():
            df["pct_change"] = (df["close"] / prev_close - 1) * 100
        if "price_change" not in df.columns:
            df["price_change"] = df["close"] - prev_close
        if "amplitude" not in df.columns or df["amplitude"].isna().all():
            df["amplitude"] = (df["high"] - df["low"]) / prev_close * 100
        if "turnover_rate" not in df.columns:
            df["turnover_rate"] = np.nan
        if "amount" not in df.columns:
            df["amount"] = np.nan

        return df.tail(DATA_CONFIG["kline_days"]).reset_index(drop=True)

    def fetch_kline(self, code: str) -> Optional[pd.DataFrame]:
        """获取日线数据（多源：优先当前粘性源，失败自动切换并记住可用源）"""
        if code in self.kline_cache:
            return self.kline_cache[code]

        end_date = datetime.now().strftime("%Y%m%d")
        # 多取一些自然日，覆盖 kline_days 个交易日
        start_date = (datetime.now() - timedelta(days=int(DATA_CONFIG["kline_days"] * 1.8))).strftime("%Y%m%d")

        # 尝试顺序：粘性源优先，跳过已确认死掉的源
        order = [s for s in self.KLINE_SOURCES if s not in self.dead_sources]
        sticky = self.KLINE_SOURCES[self.kline_source_idx]
        if sticky in order:
            order.remove(sticky)
            order.insert(0, sticky)

        df = None
        for src in order:
            fetcher = self._KLINE_FETCHERS[src]
            for attempt in range(2):  # 每源最多试 2 次
                try:
                    raw = fetcher(self, code, start_date, end_date)
                    if raw is not None:
                        df = self._normalize_kline(raw)
                        break
                except Exception:
                    pass
                time.sleep(0.5)
            if df is not None:
                # 该源可用：粘住它
                if self.KLINE_SOURCES[self.kline_source_idx] != src:
                    print(f"  🔀 K线数据源切换: {self.KLINE_SOURCES[self.kline_source_idx]} → {src}")
                    self.kline_source_idx = self.KLINE_SOURCES.index(src)
                break
            # 该源连试 2 次都失败：标记为死源（若其余源还有希望）
            if len(self.dead_sources) < len(self.KLINE_SOURCES) - 1:
                if src not in self.dead_sources:
                    print(f"  ⚠️ K线数据源 {src} 不可用，切换备用源")
                    self.dead_sources.add(src)
            else:
                print(f"  ⚠️ K线数据源 {src} 不可用（已无其他可用源）")
                self.dead_sources.add(src)

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
                src = self.KLINE_SOURCES[self.kline_source_idx]
                print(f"  进度: {idx+1}/{total} | ✅{success} ❌{fail} | 源:{src}")

            # 快速失败：前 30 支全部失败说明所有数据源均被拒，继续跑只会浪费时间
            if idx + 1 == 30 and success == 0:
                raise RuntimeError(
                    f"前 30 支股票 K 线全部获取失败（数据源 {self.KLINE_SOURCES} 均不可用）。"
                    "可能是行情接口对当前 IP 限流/拒绝，请稍后 Re-run 或更换数据源。"
                )

            time.sleep(DATA_CONFIG["fetch_interval"])

        print(f"\n✅ 完成: {success}/{total}")
        return self.kline_cache

    # ========================================
    # 3. 实时行情
    # ========================================
    def fetch_realtime_quotes(self) -> pd.DataFrame:
        try:
            print("\n📊 获取实时行情...")
            quotes = self._spot_snapshot()
            if quotes is not None and not quotes.empty:
                print(f"  ✅ {len(quotes)} 支")
                return quotes
            print("  ⚠️ 所有实时行情源不可用，继续（报告将缺少 PE/PB 等估值字段）")
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
