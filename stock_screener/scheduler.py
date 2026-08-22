"""
定时任务调度模块
每日收盘后自动执行分析（供本地/自有服务器使用；
GitHub Actions 定时部署请见 .github/workflows/daily_scan.yml）
"""

import time
import schedule
from datetime import datetime, date
from config import SCHEDULER_CONFIG


class TaskScheduler:
    """任务调度器"""

    def __init__(self, run_func):
        """
        Args:
            run_func: 要定时执行的函数（即main中的完整分析流程）
        """
        self.run_func = run_func

    def start(self):
        """启动定时任务"""
        run_time = SCHEDULER_CONFIG["daily_run_time"]

        print(f"\n⏰ 定时任务已启动")
        print(f"   每日执行时间: {run_time}")
        print(f"   周末执行: {'是' if SCHEDULER_CONFIG['run_on_weekends'] else '否'}")
        print(f"   等待中...\n")

        schedule.every().monday.at(run_time).do(self._safe_run)
        schedule.every().tuesday.at(run_time).do(self._safe_run)
        schedule.every().wednesday.at(run_time).do(self._safe_run)
        schedule.every().thursday.at(run_time).do(self._safe_run)
        schedule.every().friday.at(run_time).do(self._safe_run)

        if SCHEDULER_CONFIG["run_on_weekends"]:
            schedule.every().saturday.at(run_time).do(self._safe_run)
            schedule.every().sunday.at(run_time).do(self._safe_run)

        while True:
            schedule.run_pending()
            time.sleep(30)

    def _safe_run(self):
        """安全执行（捕获异常防止调度器崩溃）"""
        try:
            print(f"\n🚀 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始执行每日分析...")
            self.run_func()
            print(f"✅ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 每日分析完成")
        except Exception as e:
            print(f"❌ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 执行失败: {e}")
            import traceback
            traceback.print_exc()
