# 📈 A股自适应量化筛选系统

> 基于原“股票市场自动化分析系统 - 完整增强版”方案升级：
> **固定加权评分 → 回测胜率驱动的自适应权重**，并自动化部署到 GitHub Pages。

[![每日扫描](https://github.com/259773138/gp/actions/workflows/daily_scan.yml/badge.svg)](https://github.com/259773138/gp/actions/workflows/daily_scan.yml)

## 🚀 线上报告（GitHub Pages）

部署后访问：**https://259773138.github.io/gp/**

每个交易日北京时间 15:30（UTC 07:30）由 GitHub Actions 自动生成并更新。
等首次扫描完成后页面会展示：分类信号表、自适应 AI 分、回测胜率矩阵、学习到的信号权重。

## 🧠 自适应权重 —— 与原方案的差异

原方案综合评分使用**人工固定权重**：

```
综合评分 = 技术面(40%) + 周线(25%) + 基本面(20%) + 估值(15%)
各信号固定打分：MACD背离 3分、RSI 2分、…
```

本项目将其替换为**由回测数据学习得到的最优权重**：

1. 每次扫描为每只股票记录 37 维统一特征向量（日线技术 / 周线 / 基本面 / 估值）；
2. 回测模块给出每条信号 5/10/20/60 日后的真实收益；
3. `weight_optimizer.py` 用带 L2 正则的逻辑回归（纯 numpy 实现）学习
   `P(20日后上涨) = sigmoid(b + Σ wᵢ·zᵢ)`：
   - 按**时间顺序**切分训练/验证/样本外测试集，杜绝前视偏差；
   - 在验证集上**直接以“每日 Top-10 胜率”为优化目标**选择正则强度；
   - 样本外同时报告“原固定权重 vs 自适应权重”的 Top-N 胜率对比（胜率提升一目了然）；
4. 样本不足时自动回退原固定权重，不会出现空窗；
5. **历史回放冷启动**（`historical_replay.py`）：首次运行时，用历史K线回放过往约一年
   每周的信号与后续收益（rolling/ewm 均为因果计算，无前视），首日即可获得数千条带标注
   样本，自适应权重当天即可启用，无需等待数月实盘积累。

> ⚠️ 基本面/估值特征在历史回放中取不到历史点值，该部分权重主要随实盘数据逐步学习。

## 📁 项目结构

```
stock_screener/
├── config.py             # 全局配置（含自适应/回放参数）
├── data_fetcher.py       # 数据获取（akshare，重试机制）
├── indicators.py         # 日线技术指标
├── weekly_indicators.py  # 周线级别分析
├── fundamental.py        # 基本面分析（修复新浪列名模糊匹配）
├── screener.py           # 规则分类 + 自适应 AI 评分
├── weight_optimizer.py   # ★ 自适应权重学习（核心改造）
├── historical_replay.py  # ★ 历史回放冷启动（核心新增）
├── backtest.py           # 实盘信号回测验证
├── report.py             # 控制台 + Excel 报告
├── web_report.py         # ★ GitHub Pages 静态站点生成
├── db_manager.py         # SQLite（筛选/回测/权重/元信息，含自动迁移）
├── scheduler.py          # 本地定时调度
├── demo_data.py          # 离线演示数据生成
├── main.py               # 入口
└── requirements.txt
docs/                     # GitHub Pages 发布目录（自动生成）
.github/workflows/daily_scan.yml  # 每个交易日自动运行
```

## 🖥️ 本地使用

```bash
cd stock_screener
pip install -r requirements.txt

python main.py --mode once        # 完整分析一次（数据+筛选+回测+训练+建站）
python main.py --mode schedule    # 本机定时：每日 15:30 自动执行
python main.py --mode backtest    # 仅执行回测验证
python main.py --mode optimize    # 仅重新训练自适应权重
python main.py --mode replay      # 仅执行历史回放
python main.py --mode web         # 仅从数据库重建网站
python main.py --mode demo        # 离线演示（合成数据，无需行情网络）
python main.py --mode once --limit 10   # 调试：只跑前 10 支股票
```

## ⚙️ 自动化部署说明

- **数据源**：`akshare`（东方财富/新浪），GitHub Actions 海外 Runner 偶发限流时，
  在 Actions 页手动 `Re-run` 即可；
- **定时触发**：GitHub 的 `schedule` 只在**默认分支（main）**生效，合并 PR 后自动按
  cron 运行；当前分支可随时在 Actions 页手动触发（`workflow_dispatch`）；
- **数据持久化**：`stock_screener/data/screener.db`（筛选/回测历史）与
  `learned_weights.json`（最新自适应权重）由工作流自动提交回仓库，
  使每天的任务在昨天基础上持续进化；
- **Pages**：源为仓库 `docs/` 目录，已含 `.nojekyll`。

## ⚠️ 免责声明

本系统输出的所有信号与评分仅为技术面与统计参考，不构成任何投资建议。
历史回测胜率不代表未来表现。股市有风险，投资需谨慎。
