"""
数据库管理模块
使用SQLite存储每日筛选结果、回测结果、自适应学习权重。
新增（相对原版方案）：
  - screening_records.features / composite_score / adaptive_score / source 列（含自动迁移）
  - model_weights 表（学习到的权重历史）
  - meta_kv 表（记录一次性任务标记，如历史回放）
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime, date
from typing import List, Dict, Optional
from config import DB_PATH


class DBManager:
    """SQLite数据库管理器"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _ensure_column(cursor, table: str, column: str, col_type: str):
        """若列不存在则添加（向后兼容旧库）"""
        cursor.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cursor.fetchall()]
        if column not in cols:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    def _init_db(self):
        """初始化数据库表结构"""
        conn = self._get_conn()
        cursor = conn.cursor()

        # 筛选记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS screening_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date TEXT NOT NULL,
                scan_time TEXT NOT NULL,
                code TEXT NOT NULL,
                name TEXT,
                category TEXT NOT NULL,
                signal_score REAL,
                price_at_signal REAL,
                rsi REAL,
                adx REAL,
                ma_state TEXT,
                vol_pattern TEXT,
                return_5d REAL,
                return_20d REAL,
                return_60d REAL,
                bias_20 REAL,
                macd_bottom_div INTEGER DEFAULT 0,
                macd_top_div INTEGER DEFAULT 0,
                signal_details TEXT,
                pe REAL,
                pb REAL,
                fundamental_score REAL,
                weekly_confirm INTEGER DEFAULT 0,
                extra_data TEXT,
                features TEXT,
                composite_score REAL,
                adaptive_score REAL,
                source TEXT DEFAULT 'live',
                UNIQUE(scan_date, code, source)
            )
        """)

        # 兼容旧库的增量列
        self._ensure_column(cursor, "screening_records", "features", "TEXT")
        self._ensure_column(cursor, "screening_records", "composite_score", "REAL")
        self._ensure_column(cursor, "screening_records", "adaptive_score", "REAL")
        self._ensure_column(cursor, "screening_records", "source", "TEXT DEFAULT 'live'")

        # 回测结果表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date TEXT NOT NULL,
                check_date TEXT NOT NULL,
                check_period INTEGER NOT NULL,
                code TEXT NOT NULL,
                name TEXT,
                category TEXT NOT NULL,
                signal_score REAL,
                price_at_signal REAL,
                price_at_check REAL,
                actual_return REAL,
                is_success INTEGER DEFAULT 0,
                source TEXT DEFAULT 'live',
                UNIQUE(scan_date, check_date, code, check_period)
            )
        """)
        self._ensure_column(cursor, "backtest_results", "source", "TEXT DEFAULT 'live'")

        # 每日统计表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date TEXT NOT NULL UNIQUE,
                total_analyzed INTEGER,
                bottom_reversal_count INTEGER,
                top_risk_count INTEGER,
                strong_uptrend_count INTEGER,
                weak_downtrend_count INTEGER,
                sideways_count INTEGER,
                oversold_bounce_count INTEGER,
                market_temperature TEXT,
                stats_json TEXT
            )
        """)

        # 学习到的权重历史
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trained_at TEXT NOT NULL,
                n_train INTEGER,
                n_test INTEGER,
                n_dates INTEGER,
                metrics_json TEXT,
                weights_json TEXT,
                is_active INTEGER DEFAULT 1
            )
        """)

        # 一次性任务/元信息
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meta_kv (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        conn.commit()
        conn.close()

    # ========================================
    # 元信息
    # ========================================
    def get_meta(self, key: str, default: str = None) -> Optional[str]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM meta_kv WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO meta_kv (key, value) VALUES (?, ?)", (key, value))
        conn.commit()
        conn.close()

    # ========================================
    # 写入操作
    # ========================================
    def save_screening_results(self, results: List[Dict], scan_date: str = None,
                               source: str = "live", verbose: bool = True):
        """保存筛选结果（批量；source: live/replay/demo）"""
        if scan_date is None:
            scan_date = date.today().isoformat()

        scan_time = datetime.now().strftime("%H:%M:%S")
        conn = self._get_conn()
        cursor = conn.cursor()

        saved = 0
        for s in results:
            try:
                extra = {
                    "vol_ratio_60": s.get("vol_ratio_60"),
                    "boll_pct_b": s.get("boll_pct_b"),
                    "atr_pct": s.get("atr_pct"),
                    "candle_patterns": s.get("candle_patterns"),
                    "vol_price_relation": s.get("vol_price_relation"),
                    "fundamental_grade": s.get("fundamental_grade"),
                }

                cursor.execute("""
                    INSERT OR REPLACE INTO screening_records
                    (scan_date, scan_time, code, name, category, signal_score,
                     price_at_signal, rsi, adx, ma_state, vol_pattern,
                     return_5d, return_20d, return_60d, bias_20,
                     macd_bottom_div, macd_top_div, signal_details,
                     pe, pb, fundamental_score, weekly_confirm, extra_data,
                     features, composite_score, adaptive_score, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    scan_date, scan_time,
                    s.get("code"), s.get("name"), s.get("category"), s.get("signal_score"),
                    s.get("price"), s.get("rsi"), s.get("adx"), s.get("ma_state"), s.get("vol_pattern"),
                    s.get("return_5d"), s.get("return_20d"), s.get("return_60d"), s.get("bias_20"),
                    1 if s.get("macd_bottom_div") else 0,
                    1 if s.get("macd_top_div") else 0,
                    s.get("signal_details", ""),
                    s.get("pe"), s.get("pb"),
                    s.get("fundamental_score", 0),
                    1 if s.get("weekly_confirm") else 0,
                    json.dumps(extra, ensure_ascii=False, default=str),
                    json.dumps(s.get("features") or {}, ensure_ascii=False, default=str),
                    s.get("composite_score"),
                    s.get("adaptive_score"),
                    source,
                ))
                saved += 1
            except Exception:
                continue

        conn.commit()
        conn.close()
        if verbose:
            print(f"  💾 已保存 {saved} 条筛选记录到数据库 (source={source})")

    def save_daily_stats(self, stats: Dict, scan_date: str = None):
        """保存每日统计"""
        if scan_date is None:
            scan_date = date.today().isoformat()

        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO daily_stats
                (scan_date, total_analyzed, bottom_reversal_count, top_risk_count,
                 strong_uptrend_count, weak_downtrend_count, sideways_count,
                 oversold_bounce_count, market_temperature, stats_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                scan_date,
                stats.get("total", 0),
                stats.get("底部反转候选", 0),
                stats.get("顶部风险警示", 0),
                stats.get("强势上涨趋势", 0),
                stats.get("弱势下跌趋势", 0),
                stats.get("横盘震荡整理", 0),
                stats.get("超跌反弹机会", 0),
                stats.get("market_temperature", ""),
                json.dumps(stats, ensure_ascii=False, default=str)
            ))
            conn.commit()
        except Exception as e:
            print(f"  ⚠️ 保存统计失败: {e}")
        finally:
            conn.close()

    def save_backtest_result(self, record: Dict):
        self.save_backtest_results([record])

    def save_backtest_results(self, records: List[Dict]):
        """批量保存回测结果"""
        conn = self._get_conn()
        cursor = conn.cursor()
        for record in records:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO backtest_results
                    (scan_date, check_date, check_period, code, name, category,
                     signal_score, price_at_signal, price_at_check, actual_return, is_success, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record["scan_date"], record["check_date"], record["check_period"],
                    record["code"], record.get("name", ""), record["category"],
                    record.get("signal_score", 0), record["price_at_signal"],
                    record["price_at_check"], record["actual_return"],
                    1 if record["is_success"] else 0,
                    record.get("source", "live"),
                ))
            except Exception:
                continue
        conn.commit()
        conn.close()

    def save_model_weights(self, weights_payload: Dict, metrics: Dict,
                           n_train: int, n_test: int, n_dates: int):
        """保存学习到的权重，旧记录标记为非激活"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("UPDATE model_weights SET is_active = 0")
        cursor.execute("""
            INSERT INTO model_weights (trained_at, n_train, n_test, n_dates,
                                       metrics_json, weights_json, is_active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            n_train, n_test, n_dates,
            json.dumps(metrics, ensure_ascii=False, default=str),
            json.dumps(weights_payload, ensure_ascii=False, default=str),
        ))
        conn.commit()
        conn.close()

    def get_active_model_weights(self) -> Optional[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT weights_json FROM model_weights WHERE is_active = 1 "
            "ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            try:
                return json.loads(row["weights_json"])
            except Exception:
                return None
        return None

    # ========================================
    # 读取操作
    # ========================================
    def get_screening_by_date(self, scan_date: str, source: str = None) -> List[Dict]:
        """获取指定日期的筛选记录"""
        conn = self._get_conn()
        cursor = conn.cursor()
        if source:
            cursor.execute(
                "SELECT * FROM screening_records WHERE scan_date = ? AND source = ? "
                "ORDER BY signal_score DESC",
                (scan_date, source)
            )
        else:
            cursor.execute(
                "SELECT * FROM screening_records WHERE scan_date = ? ORDER BY signal_score DESC",
                (scan_date,)
            )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_latest_scan_date(self, source: str = "live") -> Optional[str]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MAX(scan_date) AS d FROM screening_records WHERE source = ?", (source,)
        )
        row = cursor.fetchone()
        conn.close()
        return row["d"] if row and row["d"] else None

    def get_pending_backtest_dates(self, check_period: int) -> List[str]:
        """获取需要进行回测检验的日期（仅实盘记录，回放记录已自带结果）"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT sr.scan_date
            FROM screening_records sr
            WHERE sr.source = 'live'
            AND sr.scan_date <= date('now', ? || ' days')
            AND sr.scan_date NOT IN (
                SELECT DISTINCT scan_date FROM backtest_results
                WHERE check_period = ? AND source = 'live'
            )
            ORDER BY sr.scan_date
        """, (f"-{check_period}", check_period))

        rows = cursor.fetchall()
        conn.close()
        return [row["scan_date"] for row in rows]

    def get_training_dataset(self, check_period: int = 20) -> pd.DataFrame:
        """筛选记录 ⋈ 回测结果 → 自适应模型训练集"""
        conn = self._get_conn()
        query = """
            SELECT sr.scan_date, sr.code, sr.name, sr.category,
                   sr.features, sr.composite_score, sr.signal_score,
                   br.actual_return, br.is_success
            FROM screening_records sr
            JOIN backtest_results br
              ON sr.scan_date = br.scan_date AND sr.code = br.code
            WHERE br.check_period = ?
              AND sr.features IS NOT NULL
              AND br.actual_return IS NOT NULL
        """
        df = pd.read_sql(query, conn, params=(check_period,))
        conn.close()
        return df

    def get_backtest_summary(self, source: str = None) -> pd.DataFrame:
        """获取回测汇总统计"""
        conn = self._get_conn()
        where = "WHERE source = ?" if source else ""
        params = (source,) if source else ()
        query = f"""
            SELECT
                category,
                check_period,
                COUNT(*) as total_signals,
                SUM(is_success) as success_count,
                ROUND(AVG(actual_return), 2) as avg_return,
                ROUND(100.0 * SUM(is_success) / COUNT(*), 1) as success_rate,
                ROUND(AVG(signal_score), 1) as avg_signal_score
            FROM backtest_results
            {where}
            GROUP BY category, check_period
            ORDER BY category, check_period
        """
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df

    def get_daily_stats_history(self, days: int = 30) -> pd.DataFrame:
        """获取最近N天的每日统计历史"""
        conn = self._get_conn()
        query = f"""
            SELECT * FROM daily_stats
            ORDER BY scan_date DESC
            LIMIT {days}
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df

    def get_high_score_history(self, category: str, min_score: int = 5,
                                days: int = 30) -> pd.DataFrame:
        """获取历史高分信号"""
        conn = self._get_conn()
        query = """
            SELECT * FROM screening_records
            WHERE category = ? AND signal_score >= ?
            AND scan_date >= date('now', ? || ' days')
            ORDER BY scan_date DESC, signal_score DESC
        """
        df = pd.read_sql(query, conn, params=(category, min_score, f"-{days}"))
        conn.close()
        return df

    def count_rows(self, table: str, source: str = None) -> int:
        conn = self._get_conn()
        cursor = conn.cursor()
        if source:
            cursor.execute(f"SELECT COUNT(*) AS c FROM {table} WHERE source = ?", (source,))
        else:
            cursor.execute(f"SELECT COUNT(*) AS c FROM {table}")
        row = cursor.fetchone()
        conn.close()
        return row["c"] if row else 0
