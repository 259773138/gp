"""
基本面分析模块
获取并分析财务指标：ROE、营收增速、毛利率、资产负债率等
修复（相对原版方案）：
  - 新浪财务接口实际列名带"(%)"、"(%)"等后缀，原版精确匹配会全部落空，
    这里统一改为“包含关键字”的模糊匹配。
"""

import time
import numpy as np
import pandas as pd
import akshare as ak
from typing import Dict, Optional
from config import FUNDAMENTAL_CONFIG, DATA_CONFIG


class FundamentalAnalyzer:
    """基本面分析器"""

    def __init__(self):
        self.cache: Dict[str, Dict] = {}

    # ---------- 工具 ----------
    @staticmethod
    def _find_col(df: pd.DataFrame, keywords: list) -> Optional[str]:
        """按关键字模糊匹配列名（优先短列名，即更精确的那个）"""
        candidates = []
        for col in df.columns:
            col_str = str(col)
            for kw in keywords:
                if kw in col_str:
                    candidates.append(col)
                    break
        if not candidates:
            return None
        candidates.sort(key=lambda c: len(str(c)))
        return candidates[0]

    @staticmethod
    def _to_float(val) -> Optional[float]:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            s = str(val).replace("%", "").replace(",", "").replace("--", "").strip()
            if s == "" or s.lower() == "nan":
                return None
            return float(s)
        except (ValueError, TypeError):
            return None

    def analyze_stock(self, code: str) -> Optional[Dict]:
        """分析单只股票的基本面"""
        if code in self.cache:
            return self.cache[code]

        result = {
            "code": code,
            "roe": None,
            "revenue_growth": None,
            "profit_growth": None,
            "gross_margin": None,
            "net_margin": None,
            "debt_ratio": None,
            "current_ratio": None,
            "eps": None,
            "bps": None,
            "fundamental_score": 0,
            "fundamental_grade": "N/A",
            "fundamental_details": [],
        }

        # 1. 获取财务指标数据（指标 + 增速来自同一接口，一次取齐）
        financial_data = self._fetch_financial_indicators(code)
        if financial_data is not None:
            result.update(financial_data)

        # 2. 计算基本面评分
        score, grade, details = self._calculate_score(result)
        result["fundamental_score"] = score
        result["fundamental_grade"] = grade
        result["fundamental_details"] = details

        self.cache[code] = result
        return result

    def analyze_batch(self, codes: list, names: dict = None) -> Dict[str, Dict]:
        """批量分析基本面"""
        print("\n" + "=" * 60)
        print("📊 获取基本面数据...")
        print("=" * 60)

        total = len(codes)
        success = 0
        results = {}

        for idx, code in enumerate(codes):
            try:
                data = self.analyze_stock(code)
                if data:
                    results[code] = data
                    success += 1
            except Exception:
                continue

            if (idx + 1) % 10 == 0 or idx == total - 1:
                print(f"  进度: {idx+1}/{total} | ✅{success}")

            time.sleep(FUNDAMENTAL_CONFIG["fetch_interval"])

        print(f"\n✅ 基本面分析完成: {success}/{total}")
        return results

    def _fetch_financial_indicators(self, code: str) -> Optional[Dict]:
        """获取主要财务指标（含增速）"""
        try:
            df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2022")
            if df is None or df.empty:
                return None

            # 取最新一期数据
            latest = df.iloc[0]

            result = {}

            # 关键字模糊匹配（列名通常带单位后缀，如 "净资产收益率(%)"）
            column_mapping = {
                "roe": ["加权净资产收益率", "净资产收益率", "摊薄净资产收益率"],
                "gross_margin": ["销售毛利率", "毛利率"],
                "net_margin": ["销售净利率", "净利率"],
                "debt_ratio": ["资产负债率"],
                "current_ratio": ["流动比率"],
                "eps": ["基本每股收益", "每股收益"],
                "bps": ["每股净资产"],
                "revenue_growth": ["主营业务收入增长率", "营业收入增长率", "营业总收入同比增长率"],
                "profit_growth": ["净利润增长率", "归属母公司股东的净利润同比增长率"],
            }

            for key, keywords in column_mapping.items():
                col = self._find_col(df, keywords)
                if col is not None:
                    val = self._to_float(latest[col])
                    if val is not None:
                        result[key] = val

            return result if result else None

        except Exception:
            return None

    def _calculate_score(self, data: Dict) -> tuple:
        """
        计算基本面综合评分
        返回: (score: float 0-100, grade: str, details: list)
        """
        weights = FUNDAMENTAL_CONFIG["weights"]
        thresholds = FUNDAMENTAL_CONFIG["thresholds"]

        scores = {}
        details = []
        valid_weight_sum = 0

        # 1. ROE评分
        roe = data.get("roe")
        if roe is not None:
            if roe >= 20:
                scores["roe"] = 100
                details.append(f"ROE优秀({roe:.1f}%)")
            elif roe >= 15:
                scores["roe"] = 80
                details.append(f"ROE良好({roe:.1f}%)")
            elif roe >= 10:
                scores["roe"] = 60
                details.append(f"ROE一般({roe:.1f}%)")
            elif roe >= thresholds["roe_min"]:
                scores["roe"] = 40
                details.append(f"ROE偏低({roe:.1f}%)")
            else:
                scores["roe"] = 20
                details.append(f"ROE较差({roe:.1f}%)")
            valid_weight_sum += weights["roe"]

        # 2. 营收增速评分
        rev_growth = data.get("revenue_growth")
        if rev_growth is not None:
            if rev_growth >= 30:
                scores["revenue_growth"] = 100
                details.append(f"营收高增({rev_growth:.1f}%)")
            elif rev_growth >= 15:
                scores["revenue_growth"] = 80
                details.append(f"营收稳增({rev_growth:.1f}%)")
            elif rev_growth >= 0:
                scores["revenue_growth"] = 60
                details.append(f"营收微增({rev_growth:.1f}%)")
            elif rev_growth >= thresholds["revenue_growth_min"]:
                scores["revenue_growth"] = 40
                details.append(f"营收微降({rev_growth:.1f}%)")
            else:
                scores["revenue_growth"] = 20
                details.append(f"营收大降({rev_growth:.1f}%)")
            valid_weight_sum += weights["revenue_growth"]

        # 3. 利润增速评分
        profit_growth = data.get("profit_growth")
        if profit_growth is not None:
            if profit_growth >= 30:
                scores["profit_growth"] = 100
            elif profit_growth >= 15:
                scores["profit_growth"] = 80
            elif profit_growth >= 0:
                scores["profit_growth"] = 60
            elif profit_growth >= -20:
                scores["profit_growth"] = 40
            else:
                scores["profit_growth"] = 20
            valid_weight_sum += weights["profit_growth"]
            details.append(f"利润增速{profit_growth:.1f}%")

        # 4. 毛利率评分
        gross = data.get("gross_margin")
        if gross is not None:
            if gross >= 50:
                scores["gross_margin"] = 100
                details.append(f"毛利率高({gross:.1f}%)")
            elif gross >= 30:
                scores["gross_margin"] = 80
            elif gross >= thresholds["gross_margin_min"]:
                scores["gross_margin"] = 60
            else:
                scores["gross_margin"] = 30
                details.append(f"毛利率低({gross:.1f}%)")
            valid_weight_sum += weights["gross_margin"]

        # 5. 资产负债率评分
        debt = data.get("debt_ratio")
        if debt is not None:
            if debt <= 30:
                scores["debt_ratio"] = 100
            elif debt <= 50:
                scores["debt_ratio"] = 80
            elif debt <= thresholds["debt_ratio_max"]:
                scores["debt_ratio"] = 60
            else:
                scores["debt_ratio"] = 30
                details.append(f"负债率高({debt:.1f}%)")
            valid_weight_sum += weights["debt_ratio"]

        # 加权平均
        if valid_weight_sum > 0 and scores:
            total_score = 0
            for key, score in scores.items():
                w = weights.get(key, 0)
                total_score += score * w
            total_score = total_score / valid_weight_sum
        else:
            total_score = 0

        # 等级
        if total_score >= 80:
            grade = "A"
        elif total_score >= 65:
            grade = "B"
        elif total_score >= 50:
            grade = "C"
        elif total_score > 0:
            grade = "D"
        else:
            grade = "N/A"

        return round(total_score, 1), grade, details
