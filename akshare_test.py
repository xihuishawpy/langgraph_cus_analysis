import time
import warnings
from typing import Iterable, List, Dict

import akshare as ak
import pandas as pd
from requests.exceptions import JSONDecodeError

warnings.filterwarnings('ignore')

RATE_LIMIT_SECONDS = 0.5
OUTPUT_PATH = "chip_industry_results.csv"


def fetch_chip_industry_codes(stock_df: pd.DataFrame) -> List[Dict[str, str]]:
    """遍历证券列表，筛选行业含“芯片”的标的。"""
    results = []
    for _, row in stock_df.iterrows():
        symbol = row["证券代码"]
        try:
            stock_industry_change_cninfo_df = ak.stock_industry_change_cninfo(
                symbol=symbol,
                start_date="20091227",
                end_date="20220708",
            )
        except (KeyError, JSONDecodeError, TypeError):
            # 个别证券缺少 records 字段，或接口返回异常结构，跳过
            continue

        # 控制请求节奏，避免频繁访问
        time.sleep(RATE_LIMIT_SECONDS)

        target_series = stock_industry_change_cninfo_df.loc[
            stock_industry_change_cninfo_df["分类标准"] == "申银万国行业分类标准",
            "行业大类",
        ].dropna()

        if target_series.empty:
            continue

        industry_value = target_series.iloc[0]
        if isinstance(industry_value, str) and "芯片" in industry_value:
            results.append({"代码": symbol, "行业大类": industry_value})

    return results


def main():
    # 上证、深证列表
    stock_info_sh_name_code_df = ak.stock_info_sh_name_code(symbol="主板A股")
    stock_info_sz_name_code_df = ak.stock_info_sz_name_code(symbol="A股列表")

    all_results: List[Dict[str, str]] = []
    for stock_df in (stock_info_sh_name_code_df, stock_info_sz_name_code_df):
        all_results.extend(fetch_chip_industry_codes(stock_df))

    result_df = pd.DataFrame(all_results).drop_duplicates()
    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"共收录 {len(result_df)} 条记录，结果已保存至 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
