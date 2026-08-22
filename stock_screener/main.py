"""
股票市场自动化筛选分析系统 v3.0（自适应权重版）
==============================================
相对原方案的增强:
  1. 日线 + 周线双级别信号确认
  2. 基本面分析 (ROE/营收/利润增速/毛利率/负债率)
  3. 固定加权综合分 → 回测胜率导向的自适应权重（逻辑回归，numpy 实现，
     时间序列切分验证，样本不足自动回退固定权重）
  4. 历史回放冷启动：首日即可训练自适应权重并展示回测胜率矩阵
  5. SQLite历史记录 + 自动回测验证
  6. 自动生成 GitHub Pages 静态站点（docs/）
  7. 定时调度：本地 schedule 模式 / GitHub Actions cron
==============================================
"""

import os
import sys
import time
import argparse
from datetime import datetime, date

from config import DB_PATH, DEMO_DB_PATH, DATA_DIR, WEB_OUTPUT_DIR
from db_manager import DBManager
from data_fetcher import DataFetcher
from screener import StockScreener
from fundamental import FundamentalAnalyzer
from backtest import BacktestEngine
from historical_replay import HistoricalReplay
from weight_optimizer import AdaptiveWeightOptimizer
from report import ReportGenerator
from web_report import WebReport
from scheduler import TaskScheduler


def run_analysis(limit: int = None, skip_replay: bool = False):
    """执行一次完整的分析流程（连接真实行情接口）"""
    db_path = DB_PATH
    start_time = time.time()

    print("╔" + "═" * 68 + "╗")
    print("║   📊 股票市场自动化筛选分析系统 v3.0 (自适应权重·回测驱动)      ║")
    print(f"║   🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " " * 43 + "║")
    print("╚" + "═" * 68 + "╝")

    # ========================================
    # Step 1: 数据获取
    # ========================================
    fetcher = DataFetcher()

    # 1.1 构建股票池
    stock_pool = fetcher.build_stock_pool()
    if stock_pool.empty:
        print("❌ 无法构建股票池")
        return False
    if limit:
        stock_pool = stock_pool.head(limit).reset_index(drop=True)
        fetcher.stock_pool = stock_pool
        print(f"  ⚡ 调试模式：仅取前 {limit} 支")

    # 1.2 实时行情
    realtime_quotes = fetcher.fetch_realtime_quotes()

    # 1.3 日线 + 周线数据
    kline_data = fetcher.fetch_all_klines()
    weekly_data = fetcher.weekly_cache

    if not kline_data:
        print("❌ 无法获取K线数据")
        return False

    # ========================================
    # Step 2: 历史回放（首次运行冷启动）→ 回测 → 训练自适应权重
    # ========================================
    if not skip_replay:
        try:
            HistoricalReplay(db_path).run(stock_pool, kline_data, weekly_data)
        except Exception as e:
            print(f"⚠️ 历史回放失败: {e}")

    try:
        BacktestEngine(db_path).run_backtest()
    except Exception as e:
        print(f"⚠️ 回测执行失败: {e}")

    model_payload = None
    try:
        model_payload = AdaptiveWeightOptimizer(db_path).train()
    except Exception as e:
        print(f"⚠️ 自适应权重训练失败: {e}")

    # ========================================
    # Step 3: 基本面分析
    # ========================================
    fund_analyzer = FundamentalAnalyzer()
    codes = list(kline_data.keys())
    names_dict = dict(zip(stock_pool["code"], stock_pool["name"]))
    fundamental_data = fund_analyzer.analyze_batch(codes, names_dict)

    # ========================================
    # Step 4: 综合筛选（已加载最新自适应权重）
    # ========================================
    screener = StockScreener(model=model_payload if model_payload else "auto",
                             db_path=db_path)
    results = screener.screen_all(
        stock_pool, kline_data, weekly_data,
        realtime_quotes, fundamental_data
    )

    # ========================================
    # Step 5: 报告生成
    # ========================================
    reporter = ReportGenerator(results, screener.all_analyzed, db_path=db_path)

    # 5.1 控制台
    reporter.print_console_report()

    # 5.2 Excel
    try:
        reporter.export_excel()
    except Exception as e:
        print(f"⚠️ Excel导出失败: {e}")

    # 5.3 保存到数据库
    try:
        reporter.save_to_db()
    except Exception as e:
        print(f"⚠️ 数据库保存失败: {e}")

    # ========================================
    # Step 6: 生成 GitHub Pages 静态站点
    # ========================================
    try:
        WebReport(db_path=db_path).build(results, screener.all_analyzed,
                                         scan_date=date.today().isoformat(),
                                         model=model_payload)
    except Exception as e:
        print(f"⚠️ 网站生成失败: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n⏱️ 总耗时: {elapsed:.1f} 秒")
    print("🎉 分析完成！\n")
    return True


def run_demo(n_stocks: int = 36, web_out: str = None):
    """离线演示模式：合成数据全链路验证（不访问行情接口）"""
    from demo_data import make_demo_dataset

    # 演示使用独立数据库与权重文件，不污染真实数据
    db_path = DEMO_DB_PATH
    demo_weights = os.path.join(DATA_DIR, "demo_weights.json")
    for p in [db_path, demo_weights]:
        if os.path.exists(p):
            os.remove(p)

    print("🧪 演示模式：使用合成数据（不访问网络行情接口）")
    ds = make_demo_dataset(n_stocks)

    # 历史回放 → 自适应权重
    HistoricalReplay(db_path).run(ds["pool"], ds["klines"], ds["weekly"],
                                  max_stocks=n_stocks)
    payload = AdaptiveWeightOptimizer(db_path, demo_weights).train()

    # “今日”扫描（应用刚学到的权重）
    screener = StockScreener(model=payload)
    results = screener.screen_all(ds["pool"], ds["klines"], ds["weekly"],
                                  ds["quotes"], ds["fundamentals"])

    reporter = ReportGenerator(results, screener.all_analyzed, db_path=db_path)
    reporter.print_console_report()
    try:
        reporter.save_to_db()
    except Exception as e:
        print(f"⚠️ 数据库保存失败: {e}")

    out = web_out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview_site")
    WebReport(out_dir=out, db_path=db_path).build(
        results, screener.all_analyzed,
        scan_date=date.today().isoformat(), model=payload)
    print(f"\n🌐 演示站点已生成于: {out}")
    return True


def main():
    parser = argparse.ArgumentParser(description="股票市场自动化筛选分析系统 v3.0")
    parser.add_argument(
        "--mode", type=str, default="once",
        choices=["once", "schedule", "backtest", "optimize", "replay", "web", "demo"],
        help="运行模式: once=执行一次, schedule=定时运行, backtest=仅回测,"
             "optimize=仅训练自适应权重, replay=仅历史回放, web=仅重建网站, demo=离线演示"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="仅处理股票池前 N 支（调试用）")
    parser.add_argument("--skip-replay", action="store_true",
                        help="跳过历史回放")
    parser.add_argument("--web-out", type=str, default=None,
                        help="网站输出目录（默认 docs/）")
    args = parser.parse_args()

    if args.mode == "once":
        ok = run_analysis(limit=args.limit, skip_replay=args.skip_replay)
        sys.exit(0 if ok else 1)

    elif args.mode == "schedule":
        print("📅 进入定时调度模式")
        print("   首次立即执行一次...")
        run_analysis()
        scheduler = TaskScheduler(run_analysis)
        scheduler.start()

    elif args.mode == "backtest":
        print("🔬 仅执行回测验证")
        BacktestEngine().run_backtest()

    elif args.mode == "optimize":
        print("🤖 仅训练自适应权重")
        payload = AdaptiveWeightOptimizer().train()
        sys.exit(0 if payload else 1)

    elif args.mode == "replay":
        print("⏪ 仅执行历史回放（使用实时K线数据）")
        fetcher = DataFetcher()
        pool = fetcher.build_stock_pool()
        if args.limit:
            pool = pool.head(args.limit).reset_index(drop=True)
            fetcher.stock_pool = pool
        klines = fetcher.fetch_all_klines()
        HistoricalReplay().run(pool, klines, fetcher.weekly_cache)

    elif args.mode == "web":
        print("🌐 仅从数据库重建网站")
        out = args.web_out or WEB_OUTPUT_DIR
        WebReport(out_dir=out).build()

    elif args.mode == "demo":
        ok = run_demo(n_stocks=args.limit or 36, web_out=args.web_out)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
