
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

import akshare as ak


REQUIRED_INFO_COLUMNS = {"行业代码", "行业名称"}


def _validate_info_columns(columns: Iterable[str]) -> None:
    """Ensure the info DataFrame includes the fields we need."""
    missing = REQUIRED_INFO_COLUMNS.difference(columns)
    if missing:
        raise ValueError(
            f"缺少必要的列: {', '.join(sorted(missing))}，请确认 akshare 返回的数据结构未发生变化。"
        )


def fetch_all_sw_third_constituents(
    output_path: str | Path, *, limit: int | None = None
) -> Tuple[Path, List[str]]:
    """
    返回所有申万三级行业成份股，并保存到 Excel 文件中。
    """

    info_df = ak.sw_index_third_info()
    _validate_info_columns(info_df.columns)

    frames: List[pd.DataFrame] = []
    failures: List[str] = []

    for idx, (_, row) in enumerate(info_df.iterrows(), start=1):
        if limit is not None and idx > limit:
            break
        code = row["行业代码"]
        name = row["行业名称"]
        parent = row.get("上级行业", "")

        try:
            cons_df = ak.sw_index_third_cons(symbol=code)
        except Exception as exc:  # pragma: no cover - network dependent
            failures.append(code)
            print(f"[WARN] 无法获取 {code}-{name} 的成份股: {exc}")
            continue

        if cons_df.empty:
            print(f"[INFO] {code}-{name} 未返回成份数据，已跳过。")
            continue

        cons_df = cons_df.copy()
        cons_df.insert(0, "所属三级行业名称", name)
        cons_df.insert(0, "所属三级行业代码", code)
        if parent:
            cons_df.insert(0, "所属二级行业", parent)

        frames.append(cons_df)
        print(f"[OK] 已获取 {code}-{name} 成份股 {len(cons_df)} 条。")

    if not frames:
        raise RuntimeError("未能获取任何成份股数据，请稍后重试。")

    combined_df = pd.concat(frames, ignore_index=True)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_excel(output, index=False)

    print(f"已导出 {len(combined_df)} 条记录至 {output}")
    if failures:
        print(f"以下行业获取失败：{', '.join(failures)}")

    return output, failures


if __name__ == "__main__":
    fetch_all_sw_third_constituents("sw_third_industry_constituents.xlsx")