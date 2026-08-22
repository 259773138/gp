"""
自适应权重学习模块（本项目核心改造）
====================================
原版方案中的综合评分使用固定权重（技术面40% + 周线25% + 基本面20%，
各技术信号固定打 1~3 分）。本模块把它替换为"由回测数据自适应学习"的权重：

  1. 每次扫描时，screener 为每只股票记录一个统一特征向量 features（JSON）。
  2. 回测模块随后给出该信号的未来收益（backtest_results）。
  3. 本模块把 features ⋈ 回测结果组成监督学习数据集，
     用带 L2 正则的逻辑回归（纯 numpy 实现）学习
         P(未来 N 日上涨) = sigmoid(b + Σ wᵢ · zᵢ)
     学习到的 w 即"回测胜率最优"的自适应权重。
  4. 为避免过拟合：按时间先后切分训练/验证/样本外测试集，
     在验证集上直接以"每日 Top-N 胜率"挑选正则强度，
     最终在样本外测试集上与"原固定权重综合分"对比胜率。
  5. 样本不足时自动回退到固定权重兜底（config.SCREENER_CONFIG['score_weights']）。

另外提供 HistoricalReplay（historical_replay.py）在项目首日即可生成
大量带标签的历史信号，使模型不必等待数月实盘积累即可启用。
"""

import json
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config import ADAPTIVE_CONFIG, DB_PATH


# ======================================================================
# 特征定义：名称 / 中文说明 / 原方案中的固定权重（仅为对照展示）
# ======================================================================
FEATURE_DEFS: List[Tuple[str, str, str]] = [
    # ---- 日线技术 ----
    ("rsi",                    "日线RSI",            "超卖/超买 2分"),
    ("macd_hist",              "MACD柱",             "趋势 1分"),
    ("macd_bottom_div",        "MACD底背离",         "3分"),
    ("macd_top_div",           "MACD顶背离",         "3分"),
    ("log_vol_ratio",          "量比60(对数)",       "缩量/放量 2分"),
    ("return_5d",              "5日涨跌幅",          "前期涨跌 1分"),
    ("return_20d",             "20日涨跌幅",         "-"),
    ("return_60d",             "60日涨跌幅",         "超跌/超涨 1分"),
    ("boll_pct_b",             "布林%B位置",         "触轨 2分"),
    ("bias_20",                "20日乖离率",         "超跌/超涨 2分"),
    ("bias_60",                "60日乖离率",         "-"),
    ("adx",                    "ADX趋势强度",        "阈值项"),
    ("ma_bullish_count",       "均线多头对数",       "多头排列 3分"),
    ("ma_bearish_count",       "均线空头对数",       "空头排列 3分"),
    ("atr_pct",                "ATR波动率%",         "-"),
    ("price_pos_60d",          "60日价格位置",       "-"),
    ("bull_engulf",            "看涨吞没",           "2分"),
    ("bear_engulf",            "看跌吞没",           "2分"),
    ("hammer",                 "锤子/长下影",         "1分"),
    ("long_upper",             "长上影线",           "1分"),
    ("doji",                   "十字星",             "-"),
    ("big_up",                 "大阳线",             "-"),
    ("big_down",               "大阴线",             "-"),
    # ---- 周线 ----
    ("weekly_score",           "周线综合分",         "×1.5加权"),
    ("weekly_confirm_bullish", "周线确认看多",       "3分×1.5"),
    ("weekly_confirm_bearish", "周线确认看空",       "3分×1.5"),
    ("weekly_rsi",             "周线RSI",            "2分×1.5"),
    ("weekly_bottom_div",      "周线底背离",         "2分×1.5"),
    ("weekly_top_div",         "周线顶背离",         "2分×1.5"),
    # ---- 基本面 / 估值 ----
    ("fundamental_score",      "基本面综合分",       "20%权重"),
    ("roe",                    "ROE%",               "25%子权重"),
    ("revenue_growth",         "营收增速%",          "20%子权重"),
    ("profit_growth",          "利润增速%",          "20%子权重"),
    ("gross_margin",           "毛利率%",            "15%子权重"),
    ("debt_ratio",             "资产负债率%",        "10%子权重"),
    ("pe_scaled",              "PE(0-200缩放)",      "估值15%权重"),
    ("pb_scaled",              "PB(0-30缩放)",       "估值15%权重"),
]

FEATURE_NAMES: List[str] = [f[0] for f in FEATURE_DEFS]
FEATURE_LABELS: Dict[str, str] = {f[0]: f[1] for f in FEATURE_DEFS}
FEATURE_LEGACY_W: Dict[str, str] = {f[0]: f[2] for f in FEATURE_DEFS}


def _num(v, default=None):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def features_from_analysis(a: Dict) -> Dict:
    """从 screener 的分析结果提取统一特征向量（缺失 → None，训练时按缺失处理）"""
    candles = a.get("candle_patterns", "") or ""
    vr = _num(a.get("vol_ratio_60"))
    pe = _num(a.get("pe"))
    pb = _num(a.get("pb"))
    feats = {
        # 日线
        "rsi": _num(a.get("rsi")),
        "macd_hist": _num(a.get("macd_hist")),
        "macd_bottom_div": 1.0 if a.get("macd_bottom_div") else 0.0,
        "macd_top_div": 1.0 if a.get("macd_top_div") else 0.0,
        "log_vol_ratio": math.log1p(min(max(vr, 0.0), 6.0)) if vr is not None else None,
        "return_5d": _num(a.get("return_5d")),
        "return_20d": _num(a.get("return_20d")),
        "return_60d": _num(a.get("return_60d")),
        "boll_pct_b": _num(a.get("boll_pct_b")),
        "bias_20": _num(a.get("bias_20")),
        "bias_60": _num(a.get("bias_60")),
        "adx": _num(a.get("adx")),
        "ma_bullish_count": _num(a.get("ma_bullish_count")),
        "ma_bearish_count": _num(a.get("ma_bearish_count")),
        "atr_pct": _num(a.get("atr_pct")),
        "price_pos_60d": _num(a.get("price_pos_60d")),
        "bull_engulf": 1.0 if "看涨吞没" in candles else 0.0,
        "bear_engulf": 1.0 if "看跌吞没" in candles else 0.0,
        "hammer": 1.0 if ("锤子" in candles or "长下影线" in candles) else 0.0,
        "long_upper": 1.0 if "长上影线" in candles else 0.0,
        "doji": 1.0 if "十字星" in candles else 0.0,
        "big_up": 1.0 if "大阳线" in candles else 0.0,
        "big_down": 1.0 if "大阴线" in candles else 0.0,
        # 周线
        "weekly_score": _num(a.get("weekly_score")),
        "weekly_confirm_bullish": 1.0 if a.get("weekly_confirm_bullish") else 0.0,
        "weekly_confirm_bearish": 1.0 if a.get("weekly_confirm_bearish") else 0.0,
        "weekly_rsi": _num(a.get("weekly_rsi")),
        "weekly_bottom_div": 1.0 if a.get("weekly_bottom_div") else 0.0,
        "weekly_top_div": 1.0 if a.get("weekly_top_div") else 0.0,
        # 基本面/估值
        "fundamental_score": _num(a.get("fundamental_score")),
        "roe": _num(a.get("roe")),
        "revenue_growth": _num(a.get("revenue_growth")),
        "profit_growth": _num(a.get("profit_growth")),
        "gross_margin": _num(a.get("gross_margin")),
        "debt_ratio": _num(a.get("debt_ratio")),
        "pe_scaled": (min(max(pe, 0.0), 200.0) / 200.0) if pe is not None else None,
        "pb_scaled": (min(max(pb, 0.0), 30.0) / 30.0) if pb is not None else None,
    }
    return feats


# ======================================================================
# 纯 numpy 逻辑回归
# ======================================================================
class LogisticModel:
    def __init__(self, n_features: int):
        self.w = np.zeros(n_features)
        self.b = 0.0

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float, l2: float, iters: int):
        n = X.shape[0]
        for i in range(iters):
            p = self._sigmoid(X @ self.w + self.b)
            grad_w = X.T @ (p - y) / n + l2 * self.w
            grad_b = float((p - y).mean())
            cur_lr = lr / (1.0 + i / (iters / 4.0))   # 简单退火
            self.w -= cur_lr * grad_w
            self.b -= cur_lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.w + self.b)


# ======================================================================
# 自适应权重优化器
# ======================================================================
class AdaptiveWeightOptimizer:
    def __init__(self, db_path: str = DB_PATH, weights_file: str = None):
        from db_manager import DBManager
        self.db = DBManager(db_path)
        self.cfg = ADAPTIVE_CONFIG
        self.weights_file = weights_file or ADAPTIVE_CONFIG["weights_file"]

    # ---------- 数据 ----------
    def build_dataset(self) -> Optional[pd.DataFrame]:
        df = self.db.get_training_dataset(self.cfg["label_check_period"])
        if df is None or df.empty:
            return None
        # 展开 features JSON
        rows = []
        for _, r in df.iterrows():
            try:
                feats = json.loads(r["features"]) if r["features"] else {}
            except Exception:
                feats = {}
            row = {
                "scan_date": r["scan_date"],
                "code": r["code"],
                "composite_score": r["composite_score"],
                "actual_return": r["actual_return"],
                "y": 1.0 if r["actual_return"] is not None and
                      float(r["actual_return"]) > self.cfg["label_return_threshold"] else 0.0,
            }
            for f in FEATURE_NAMES:
                row[f] = _num(feats.get(f))
            rows.append(row)
        out = pd.DataFrame(rows)
        out = out.dropna(subset=["actual_return"])
        return out if not out.empty else None

    @staticmethod
    def _matrix(df: pd.DataFrame, mean=None, std=None):
        X = df[FEATURE_NAMES].values.astype(float)  # NaN → 缺失
        if mean is None:
            mean = np.nanmean(X, axis=0)
            mean = np.where(np.isnan(mean), 0.0, mean)
        if std is None:
            std = np.nanstd(X, axis=0)
            std = np.where((std == 0) | np.isnan(std), 1.0, std)
        Z = (X - mean) / std
        Z = np.where(np.isnan(Z), 0.0, Z)   # 缺失特征 → 0（不贡献）
        return Z, mean, std

    @staticmethod
    def _top_n_win_rate(df: pd.DataFrame, scores: np.ndarray, top_n: int) -> float:
        """按日期分组，每日取分数最高的 Top-N，计算其中 y=1 的比例"""
        tmp = df[["scan_date", "y"]].copy()
        tmp["score"] = scores
        hits, cnt = 0, 0
        for _, g in tmp.groupby("scan_date"):
            g = g.sort_values("score", ascending=False).head(top_n)
            hits += float(g["y"].sum())
            cnt += len(g)
        return hits / cnt if cnt else 0.0

    # ---------- 训练 ----------
    def train(self) -> Optional[Dict]:
        df = self.build_dataset()
        min_samples = self.cfg["min_train_samples"]
        min_dates = self.cfg["min_train_dates"]

        if df is None or len(df) < min_samples or df["scan_date"].nunique() < min_dates:
            n = 0 if df is None else len(df)
            d = 0 if df is None else df["scan_date"].nunique()
            print(f"\n  🤖 自适应权重：样本不足（{n} 条 / {d} 天，"
                  f"需 ≥{min_samples} 条 / ≥{min_dates} 天），继续使用默认权重")
            return None

        if df["y"].nunique() < 2:
            print("\n  🤖 自适应权重：标签单一，跳过训练")
            return None

        # 按时间顺序切分
        dates = sorted(df["scan_date"].unique())
        cut = int(len(dates) * self.cfg["train_date_ratio"])
        cut = min(max(cut, 1), len(dates) - 1)
        train_dates, test_dates = set(dates[:cut]), set(dates[cut:])
        train_df = df[df["scan_date"].isin(train_dates)]
        test_df = df[df["scan_date"].isin(test_dates)]

        # 超参选择：在训练集内部再做时间切分，直接以验证集 Top-N 胜率为目标
        best_l2, best_val = self._select_l2(train_df)

        # 用最优 l2 在完整训练集上训练
        Ztr, mean, std = self._matrix(train_df)
        ytr = train_df["y"].values
        model = LogisticModel(len(FEATURE_NAMES)).fit(
            Ztr, ytr, lr=self.cfg["learning_rate"], l2=best_l2,
            iters=self.cfg["max_iterations"])

        # ---- 样本外评估 ----
        Zte, _, _ = self._matrix(test_df, mean, std)
        yte = test_df["y"].values
        probs = model.predict_proba(Zte)

        top_n = self.cfg["top_n"]
        test_base_rate = float(yte.mean())
        model_top_wr = self._top_n_win_rate(test_df, probs, top_n)
        legacy_scores = test_df["composite_score"].fillna(50).values.astype(float)
        legacy_top_wr = self._top_n_win_rate(test_df, legacy_scores, top_n)
        auc = self._auc(yte, probs)

        metrics = {
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_train": len(train_df), "n_test": len(test_df),
            "n_dates": len(dates),
            "label": f"{self.cfg['label_check_period']}日前瞻收益 > {self.cfg['label_return_threshold']}%",
            "best_l2": best_l2,
            "val_top_n_win_rate": round(best_val * 100, 1),
            "test_base_rate": round(test_base_rate * 100, 1),
            "test_accuracy": round(float(((probs > 0.5).astype(float) == yte).mean()) * 100, 1),
            "test_auc": round(auc, 3) if auc else None,
            "model_top_n_win_rate": round(model_top_wr * 100, 1),
            "legacy_top_n_win_rate": round(legacy_top_wr * 100, 1),
            "top_n": top_n,
            "win_rate_lift": round((model_top_wr - legacy_top_wr) * 100, 1),
        }

        payload = {
            "version": "logistic-v1",
            "feature_names": FEATURE_NAMES,
            "weights": {f: float(w) for f, w in zip(FEATURE_NAMES, model.w)},
            "intercept": float(model.b),
            "mean": {f: float(m) for f, m in zip(FEATURE_NAMES, mean)},
            "std": {f: float(s) for f, s in zip(FEATURE_NAMES, std)},
            "metrics": metrics,
            "trained_at": metrics["trained_at"],
        }

        # 持久化
        os.makedirs(os.path.dirname(self.weights_file), exist_ok=True)
        with open(self.weights_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.db.save_model_weights(payload, metrics, len(train_df), len(test_df), len(dates))

        self._print_report(payload)
        return payload

    def _select_l2(self, train_df: pd.DataFrame) -> Tuple[float, float]:
        """训练集内部时间切分，直接在验证集上挑 Top-N 胜率最高的 L2 强度"""
        dates = sorted(train_df["scan_date"].unique())
        if len(dates) < 4:
            return self.cfg["l2_lambda"], 0.0
        cut = max(1, int(len(dates) * 0.7))
        inner_tr = train_df[train_df["scan_date"].isin(set(dates[:cut]))]
        inner_va = train_df[train_df["scan_date"].isin(set(dates[cut:]))]
        if len(inner_tr) < 50 or len(inner_va) < 20 or inner_tr["y"].nunique() < 2:
            return self.cfg["l2_lambda"], 0.0

        best_l2, best_wr = self.cfg["l2_lambda"], -1.0
        for l2 in [1e-4, 1e-3, 1e-2]:
            Ztr, mean, std = self._matrix(inner_tr)
            m = LogisticModel(len(FEATURE_NAMES)).fit(
                Ztr, inner_tr["y"].values,
                lr=self.cfg["learning_rate"], l2=l2, iters=1500)
            Zva, _, _ = self._matrix(inner_va, mean, std)
            wr = self._top_n_win_rate(inner_va, m.predict_proba(Zva), self.cfg["top_n"])
            if wr > best_wr:
                best_wr, best_l2 = wr, l2
        return best_l2, max(best_wr, 0.0)

    @staticmethod
    def _auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
        try:
            pos = p[y == 1]
            neg = p[y == 0]
            if len(pos) == 0 or len(neg) == 0:
                return None
            # Mann-Whitney U
            greater = sum((pi > neg).sum() + 0.5 * (pi == neg).sum() for pi in pos)
            return float(greater) / (len(pos) * len(neg))
        except Exception:
            return None

    # ---------- 报告 ----------
    def _print_report(self, payload: Dict):
        m = payload["metrics"]
        print("\n" + "=" * 70)
        print("🤖 自适应权重训练完成（回测胜率导向）")
        print("=" * 70)
        print(f"  训练样本: {m['n_train']} 条 | 样本外测试: {m['n_test']} 条 "
              f"({m['n_dates']} 个交易日)")
        print(f"  标签定义: {m['label']} | 验证集选出的 L2 = {m['best_l2']}")
        print(f"  样本外基准上涨率: {m['test_base_rate']}% | "
              f"AUC: {m['test_auc']} | 准确率: {m['test_accuracy']}%")
        print(f"  ── Top-{m['top_n']} 日均胜率（样本外）──")
        print(f"    原固定权重综合分: {m['legacy_top_n_win_rate']}%")
        print(f"    自适应学习权重:   {m['model_top_n_win_rate']}% "
              f"(提升 {m['win_rate_lift']:+.1f}%)")
        print("-" * 70)
        print(f"  {'特征':<22}{'原固定权重':<14}{'学习权重':>10}   方向")
        print("-" * 70)
        ws = payload["weights"]
        for f in sorted(ws, key=lambda k: -abs(ws[k])):
            arrow = "看多 ↑" if ws[f] > 0.05 else ("看空 ↓" if ws[f] < -0.05 else "弱  ~")
            print(f"  {FEATURE_LABELS.get(f, f):<22}{FEATURE_LEGACY_W.get(f, '-'):<14}"
                  f"{ws[f]:>10.3f}   {arrow}")
        print("=" * 70)


# ======================================================================
# 模型加载与打分（供 screener / web 使用）
# ======================================================================
def load_active_model(db_path: str = DB_PATH) -> Optional[Dict]:
    """优先读 JSON 文件，其次读数据库"""
    wf = ADAPTIVE_CONFIG["weights_file"]
    if os.path.exists(wf):
        try:
            with open(wf, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("weights"):
                return payload
        except Exception:
            pass
    try:
        from db_manager import DBManager
        return DBManager(db_path).get_active_model_weights()
    except Exception:
        return None


def score_with_model(model: Dict, features: Dict) -> Optional[float]:
    """返回 0~100 的自适应分数（= 学习到的未来上涨概率 × 100）"""
    if not model or not features:
        return None
    names = model.get("feature_names", FEATURE_NAMES)
    w = model.get("weights", {})
    mean = model.get("mean", {})
    std = model.get("std", {})
    z = 0.0
    total = model.get("intercept", 0.0)
    for f in names:
        v = features.get(f)
        if v is None:
            continue
        s = std.get(f) or 1.0
        total += w.get(f, 0.0) * ((float(v) - mean.get(f, 0.0)) / s)
    prob = 1.0 / (1.0 + math.exp(-max(min(total, 30), -30)))
    return round(prob * 100, 1)
