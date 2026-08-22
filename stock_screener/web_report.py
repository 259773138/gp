"""
GitHub Pages 静态站点生成器
把筛选结果 / 回测胜率矩阵 / 自适应权重渲染为纯静态 HTML（无外部依赖），
输出到仓库 docs/ 目录，由 GitHub Pages 直接发布。
"""

import os
import json
import html
import shutil
from datetime import datetime
from typing import Dict, List, Optional

from config import WEB_OUTPUT_DIR, WEB_CONFIG, ADAPTIVE_CONFIG, BACKTEST_CONFIG
from db_manager import DBManager
from screener import StockScreener
from weight_optimizer import FEATURE_LABELS, FEATURE_LEGACY_W, load_active_model

CAT_ORDER = list(StockScreener.CATEGORIES.keys())
BEAR_CATS = StockScreener.BEAR_CATEGORIES

CSS = """
:root{--bg:#0f1420;--card:#1a2233;--border:#2a3550;--fg:#e8ecf4;--muted:#8b95ad;
--green:#22c55e;--red:#ef4444;--blue:#3b82f6;--yellow:#eab308;--purple:#a855f7;--gray:#64748b}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font-family:-apple-system,"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif;padding:24px;line-height:1.5}
h1{font-size:26px}h2{font-size:18px;margin-bottom:12px}
a{color:var(--blue);text-decoration:none}
.wrap{max-width:1200px;margin:0 auto}
.muted{color:var(--muted);font-size:13px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;justify-content:space-between;margin-bottom:16px}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;border:1px solid var(--border);margin-left:8px}
.badge.on{background:rgba(34,197,94,.12);color:var(--green);border-color:var(--green)}
.badge.off{background:rgba(234,179,8,.12);color:var(--yellow);border-color:var(--yellow)}
.grid{display:grid;gap:14px;margin:14px 0}
.grid.cols2{grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.grid.cols3{grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}
.stat{font-size:28px;font-weight:700}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:7px 9px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}
th:first-child,td:first-child,th.l,td.l{text-align:left}
th{color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--card)}
tr:hover td{background:rgba(59,130,246,.06)}
.bar{height:10px;border-radius:6px;background:#24304a;overflow:hidden;min-width:80px}
.bar>i{display:block;height:100%}
.rate{color:var(--muted)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.up{color:var(--red)}.down{color:var(--green)}
.wbar{display:inline-block;height:9px;border-radius:4px;vertical-align:middle}
.foot{margin:28px 0 12px;color:var(--muted);font-size:12px;border-top:1px solid var(--border);padding-top:14px}
.pill{font-size:11px;padding:1px 7px;border-radius:99px;border:1px solid var(--border);color:var(--muted)}
.cat-head{display:flex;align-items:center;gap:10px;margin:6px 0 10px}
.count{font-size:13px;color:var(--muted)}
details summary{cursor:pointer;color:var(--blue);font-size:13px;margin-bottom:8px}
"""

CAT_COLOR = {
    "底部反转候选": "var(--green)",
    "强势上涨趋势": "var(--blue)",
    "顶部风险警示": "var(--red)",
    "弱势下跌趋势": "var(--yellow)",
    "横盘震荡整理": "var(--gray)",
    "超跌反弹机会": "var(--purple)",
}


def esc(v) -> str:
    return html.escape("" if v is None else str(v))


def fmt_pct(v, digits=1, sign=True) -> str:
    if v is None or v == "":
        return "-"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    s = f"{v:+.{digits}f}%" if sign else f"{v:.{digits}f}%"
    cls = "up" if v > 0 else ("down" if v < 0 else "")
    return f'<span class="{cls}">{s}</span>'


def fmt_num(v, digits=2):
    if v is None or v == "":
        return "-"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


class WebReport:
    def __init__(self, out_dir: str = WEB_OUTPUT_DIR, db_path: str = None):
        self.out_dir = out_dir
        self.db = DBManager(db_path) if db_path else DBManager()
        os.makedirs(self.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def build(self, results: Dict[str, List[Dict]] = None,
              all_analyzed: List[Dict] = None,
              scan_date: str = None,
              model: Optional[Dict] = None) -> str:
        """生成网站，返回 index.html 路径"""
        # 数据来源：优先内存（刚跑完），否则从数据库读最近一次实盘扫描
        if results is None:
            scan_date = scan_date or self.db.get_latest_scan_date("live")
            results, all_analyzed = self._load_from_db(scan_date)

        if model is None:
            model = load_active_model(self.db.db_path)
        bt_summary = self.db.get_backtest_summary()
        daily_stats = self.db.get_daily_stats_history(30)

        html_text = self._render(results or {}, all_analyzed or [], scan_date,
                                 model, bt_summary, daily_stats)

        # index.html
        index_path = os.path.join(self.out_dir, "index.html")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(html_text)

        # .nojekyll（避免 Jekyll 处理）
        njk = os.path.join(self.out_dir, ".nojekyll")
        if not os.path.exists(njk):
            open(njk, "w").close()

        # 归档
        if scan_date:
            arch_dir = os.path.join(self.out_dir, "archive")
            os.makedirs(arch_dir, exist_ok=True)
            with open(os.path.join(arch_dir, f"{scan_date}.html"), "w", encoding="utf-8") as f:
                f.write(html_text)
            self._prune_archive(arch_dir)

        # 机器可读数据
        data_dir = os.path.join(self.out_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        counts = {c: len((results or {}).get(c, [])) for c in CAT_ORDER}
        with open(os.path.join(data_dir, "latest.json"), "w", encoding="utf-8") as f:
            json.dump({
                "scan_date": scan_date,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "total": len(all_analyzed or []),
                "counts": counts,
                "model": (model or {}).get("metrics"),
            }, f, ensure_ascii=False, indent=2)

        print(f"  🌐 网站已生成: {index_path}")
        return index_path

    def _prune_archive(self, arch_dir: str):
        keep = WEB_CONFIG.get("archive_keep", 60)
        files = sorted([f for f in os.listdir(arch_dir) if f.endswith(".html")])
        for f in files[:-keep]:
            try:
                os.remove(os.path.join(arch_dir, f))
            except OSError:
                pass

    def _load_from_db(self, scan_date: Optional[str]):
        if not scan_date:
            return {c: [] for c in CAT_ORDER}, []
        rows = self.db.get_screening_by_date(scan_date, source="live")
        all_analyzed, results = [], {c: [] for c in CAT_ORDER}
        for r in rows:
            extra = {}
            try:
                extra = json.loads(r.get("extra_data") or "{}")
            except Exception:
                pass
            s = {
                "code": r["code"], "name": r.get("name") or "",
                "price": r.get("price_at_signal"),
                "category": r["category"],
                "signal_score": r.get("signal_score"),
                "composite_score": r.get("composite_score"),
                "adaptive_score": r.get("adaptive_score"),
                "rsi": r.get("rsi"), "adx": r.get("adx"),
                "ma_state": r.get("ma_state"), "vol_pattern": r.get("vol_pattern"),
                "return_5d": r.get("return_5d"), "return_20d": r.get("return_20d"),
                "return_60d": r.get("return_60d"), "bias_20": r.get("bias_20"),
                "macd_bottom_div": bool(r.get("macd_bottom_div")),
                "macd_top_div": bool(r.get("macd_top_div")),
                "signal_details": r.get("signal_details") or "",
                "pe": r.get("pe"), "pb": r.get("pb"),
                "fundamental_score": r.get("fundamental_score"),
                "fundamental_grade": extra.get("fundamental_grade", "N/A"),
                "weekly_confirm": bool(r.get("weekly_confirm")),
                "pct_change": None,
            }
            all_analyzed.append(s)
            if s["category"] in results:
                results[s["category"]].append(s)
        # 与运行时相同的排序
        for cat in results:
            reverse = cat not in BEAR_CATS
            results[cat].sort(key=StockScreener.rank_key, reverse=reverse)
        return results, all_analyzed

    # ------------------------------------------------------------------
    def _render(self, results, all_analyzed, scan_date, model, bt_summary, daily_stats) -> str:
        title = WEB_CONFIG["site_title"]
        total = len(all_analyzed)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        archives = self._archive_links()

        model_badge = ('<span class="badge on">🤖 自适应权重已启用</span>' if model
                       else '<span class="badge off">🤖 样本积累中·固定权重兜底</span>')

        parts = [
            "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width,initial-scale=1'>",
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body><div class='wrap'>",
            f"<header><div><h1>📈 {esc(title)}{model_badge}</h1>",
            f"<div class='muted'>扫描日期: {esc(scan_date or '等待首次扫描')} · 页面生成: {now} · "
            f"分析股票 {total} 支</div></div>",
            f"<div class='muted'>{archives}</div>",
            "</header>",
            self._section_overview(results, total),
            self._section_model(model),
            self._section_backtest(bt_summary),
            self._section_stocks(results),
            f"<div class='foot'>⚠️ 免责声明：本页面由量化模型自动生成，仅为技术与统计参考，"
            f"不构成任何投资建议。股市有风险，投资需谨慎。<br>"
            f"自适应权重由历史信号回测结果训练得出（样本内/外验证），权重会随每日数据更新迭代。</div>",
            "</div></body></html>",
        ]
        return "\n".join(parts)

    def _archive_links(self) -> str:
        arch_dir = os.path.join(self.out_dir, "archive")
        links = []
        if os.path.isdir(arch_dir):
            files = sorted([f for f in os.listdir(arch_dir) if f.endswith(".html")], reverse=True)
            for f in files[:10]:
                links.append(f"<a href='archive/{f}'>{f[:-5]}</a>")
        if links:
            return "历史报告: " + " · ".join(links)
        return ""

    # ---------------- 各板块 ----------------
    def _section_overview(self, results, total) -> str:
        if total == 0:
            return ("<div class='card'><h2>📊 市场概况</h2>"
                    "<div class='muted'>暂无实盘扫描数据。历史回放与回测统计见下方。</div></div>")
        bull = len(results.get("强势上涨趋势", []))
        bear = len(results.get("弱势下跌趋势", []))
        bottom = len(results.get("底部反转候选", []))
        top = len(results.get("顶部风险警示", []))
        br, er = bull / total, bear / total
        if br > 0.4:
            temp = "🔥 过热"
        elif br > 0.25:
            temp = "☀️ 偏暖"
        elif er > 0.4:
            temp = "❄️ 过冷"
        elif er > 0.25:
            temp = "🌧️ 偏冷"
        elif bottom > top:
            temp = "🌱 筑底中"
        elif top > bottom:
            temp = "⚠️ 见顶风险"
        else:
            temp = "😐 震荡"

        rows = []
        for cat in CAT_ORDER:
            n = len(results.get(cat, []))
            pct = n / total * 100 if total else 0
            color = CAT_COLOR.get(cat, "var(--gray)")
            rows.append(
                f"<tr><td class='l'><span class='dot' style='background:{color}'></span>{esc(cat)}</td>"
                f"<td>{n}</td><td class='l'><div class='bar'><i style='width:{pct:.1f}%;background:{color}'></i></div></td>"
                f"<td class='muted'>{pct:.1f}%</td></tr>")

        return (
            "<div class='grid cols2'>"
            f"<div class='card'><h2>📊 市场概况</h2>"
            f"<div class='stat'>{temp}</div>"
            f"<div class='muted'>共分析 {total} 支 · 强势 {bull} / 弱势 {bear} / "
            f"底部信号 {bottom} / 顶部风险 {top}</div></div>"
            "<div class='card'><h2>类别分布</h2>"
            "<table><tr><th class='l'>类别</th><th>数量</th><th class='l' colspan='2'>占比</th></tr>"
            + "".join(rows) + "</table></div></div>")

    def _section_model(self, model) -> str:
        if not model or not model.get("metrics"):
            return ("<div class='card'><h2>🤖 自适应权重</h2>"
                    "<div class='muted'>训练样本不足，暂用原方案固定权重。首次历史回放下将自动训练。</div></div>")
        m = model["metrics"]
        lift = m.get("win_rate_lift", 0)
        lift_cls = "up" if lift >= 0 else "down"

        def cell(label, val, sub=""):
            return (f"<div class='card'><div class='muted'>{label}</div>"
                    f"<div class='stat'>{val}</div><div class='muted'>{sub}</div></div>")

        stats = (
            cell("样本外 Top-%s 胜率（自适应）" % m.get("top_n", 10),
                 f"{m.get('model_top_n_win_rate', '-')}%",
                 f"原固定权重: {m.get('legacy_top_n_win_rate', '-')}% · "
                 f"<span class='{lift_cls}'>提升 {lift:+.1f}%</span>"),
            cell("AUC / 准确率", f"{m.get('test_auc', '-')} / {m.get('test_accuracy', '-')}%",
                 f"基准上涨率 {m.get('test_base_rate', '-')}%"),
            cell("训练 / 测试样本", f"{m.get('n_train', 0)} / {m.get('n_test', 0)}",
                 f"{m.get('n_dates', 0)} 个交易日 · L2={m.get('best_l2')}"),
        )

        # 权重表
        ws = model.get("weights", {})
        rows = []
        if ws:
            wmax = max(abs(v) for v in ws.values()) or 1.0
            for f in sorted(ws, key=lambda k: -abs(ws[k])):
                w = ws[f]
                width = abs(w) / wmax * 120
                color = "var(--red)" if w > 0.03 else ("var(--green)" if w < -0.03 else "var(--gray)")
                direction = "看多" if w > 0.03 else ("看空" if w < -0.03 else "弱")
                rows.append(
                    f"<tr><td class='l'>{esc(FEATURE_LABELS.get(f, f))}</td>"
                    f"<td class='l muted'>{esc(FEATURE_LEGACY_W.get(f, '-'))}</td>"
                    f"<td class='l'><span class='wbar' style='width:{width:.0f}px;background:{color}'></span> "
                    f"{w:+.3f}</td><td>{direction}</td></tr>")
        weight_table = (
            "<details><summary>展开每个信号的自适应权重（按影响排序，红=看多 绿=看空）</summary>"
            "<table><tr><th class='l'>信号特征</th><th class='l'>原固定权重</th>"
            "<th class='l'>学习权重</th><th>方向</th></tr>" + "".join(rows) + "</table></details>")

        return (f"<div class='card'><h2>🤖 自适应权重（回测胜率导向 · 训练于 {esc(m.get('trained_at', ''))}）</h2>"
                f"<div class='grid cols3'>{''.join(stats)}</div>{weight_table}</div>")

    def _section_backtest(self, summary) -> str:
        if summary is None or summary.empty:
            return ("<div class='card'><h2>🧪 回测胜率矩阵</h2>"
                    "<div class='muted'>暂无回测数据。</div></div>")

        valid_cats = set(BACKTEST_CONFIG["success_thresholds"].keys())
        periods = sorted(summary["check_period"].unique())
        cats = [c for c in CAT_ORDER
                if c in valid_cats and c in set(summary["category"].unique())]
        head = "<tr><th class='l'>信号类别</th>" + "".join(
            f"<th>{p}天<br><span class='muted'>胜率/均收益</span></th>" for p in periods) + "</tr>"
        body = []
        for cat in cats:
            tds = [f"<td class='l'><span class='dot' style='background:{CAT_COLOR.get(cat)}'></span>{esc(cat)}</td>"]
            for p in periods:
                sub = summary[(summary["category"] == cat) & (summary["check_period"] == p)]
                if sub.empty:
                    tds.append("<td class='muted'>-</td>")
                    continue
                r = sub.iloc[0]
                rate, avg, n = r["success_rate"], r["avg_return"], r["total_signals"]
                bg = ("rgba(34,197,94,.15)" if rate >= 55 else
                      "rgba(234,179,8,.15)" if rate >= 45 else "rgba(239,68,68,.15)")
                tds.append(f"<td style='background:{bg}'><b>{rate:.1f}%</b> "
                           f"<span class='rate'>({avg:+.1f}%/{n}条)</span></td>")
            body.append("<tr>" + "".join(tds) + "</tr>")

        live_n = self.db.count_rows("backtest_results", "live")
        replay_n = self.db.count_rows("backtest_results", "replay")
        return ("<div class='card'><h2>🧪 回测胜率矩阵</h2>"
                f"<div class='muted' style='margin-bottom:8px'>"
                f"实盘信号 {live_n} 条 · 历史回放信号 {replay_n} 条（首日冷启动生成）</div>"
                f"<table>{head}{''.join(body)}</table></div>")

    def _section_stocks(self, results) -> str:
        max_rows = WEB_CONFIG.get("max_rows_per_category", 20)
        blocks = []
        for cat in CAT_ORDER:
            stocks = results.get(cat, [])
            if not stocks:
                continue
            color = CAT_COLOR.get(cat)
            rows = []
            for s in stocks[:max_rows]:
                ai = s.get("adaptive_score")
                ai_txt = f"{ai:.0f}" if ai is not None else "-"
                wk = "✓" if s.get("weekly_confirm") else "-"
                divs = []
                if s.get("macd_bottom_div"):
                    divs.append("底")
                if s.get("macd_top_div"):
                    divs.append("顶")
                fg = s.get("fundamental_grade", "N/A")
                rows.append(
                    "<tr>"
                    f"<td class='l'>{esc(s['code'])}</td>"
                    f"<td class='l'>{esc(s.get('name', ''))}</td>"
                    f"<td>{fmt_num(s.get('price'))}</td>"
                    f"<td><b>{ai_txt}</b></td>"
                    f"<td>{fmt_num(s.get('composite_score'), 0)}</td>"
                    f"<td>{fmt_num(s.get('signal_score'), 0)}</td>"
                    f"<td>{fmt_num(s.get('rsi'), 0)}</td>"
                    f"<td>{fmt_num(s.get('adx'), 0)}</td>"
                    f"<td class='l'>{esc(str(s.get('ma_state', '-'))[:4])}</td>"
                    f"<td class='l'>{esc(str(s.get('vol_pattern', '-'))[:4])}</td>"
                    f"<td>{fmt_pct(s.get('return_20d'))}</td>"
                    f"<td>{'/'.join(divs) if divs else '-'}</td>"
                    f"<td>{wk}</td>"
                    f"<td>{esc(fg) if fg and fg != 'N/A' else '-'}</td>"
                    f"<td class='l muted'>{esc(str(s.get('signal_details', ''))[:42])}</td>"
                    "</tr>")
            head = ("<tr><th class='l'>代码</th><th class='l'>名称</th><th>现价</th>"
                    "<th>AI分</th><th>综合</th><th>信号</th><th>RSI</th><th>ADX</th>"
                    "<th class='l'>均线</th><th class='l'>量能</th><th>20日</th>"
                    "<th>背离</th><th>周线</th><th>基本面</th><th class='l'>信号详情</th></tr>")
            blocks.append(
                f"<div class='card'><div class='cat-head'>"
                f"<h2><span class='dot' style='background:{color}'></span>{esc(cat)}</h2>"
                f"<span class='count'>{len(stocks)} 支（AI分{'降序' if cat not in BEAR_CATS else '升序'}，显示前{max_rows}）</span></div>"
                f"<div style='overflow-x:auto'><table>{head}{''.join(rows)}</table></div></div>")
        if not blocks:
            return ""
        return "<div class='grid'>" + "".join(blocks) + "</div>"
