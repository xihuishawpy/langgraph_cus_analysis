"""
Export all Eastmoney concept-board constituents (股票-概念映射) via Akshare.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

import akshare as ak

CONCEPT_NAME_CANDIDATES: Sequence[str] = ("概念名称", "板块名称", "名称")
CONCEPT_CODE_CANDIDATES: Sequence[str] = ("概念代码", "板块代码", "代码")
STOCK_CODE_CANDIDATES: Sequence[str] = ("股票代码", "证券代码", "代码")
STOCK_NAME_CANDIDATES: Sequence[str] = ("股票简称", "股票名称", "名称", "证券简称")


def _normalize_columns(columns: Iterable[object]) -> List[str]:
    return [str(col).strip() for col in columns]


def _locate_column(
    columns: Sequence[str],
    candidates: Sequence[str],
    *,
    fallback_contains: str | None = None,
    exclude_contains: Sequence[str] = (),
    label: str = "",
) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate

    if fallback_contains:
        lowered_excludes = tuple(word.lower() for word in exclude_contains)
        for col in columns:
            lowered_col = col.lower()
            if any(excl in lowered_col for excl in lowered_excludes):
                continue
            if fallback_contains.lower() in lowered_col:
                return col

    raise ValueError(f"无法在列 {columns} 中找到“{label or fallback_contains or candidates}”")


def _fetch_concept_metadata() -> tuple[pd.DataFrame, str, str]:
    """Download Eastmoney concept table and detect key columns."""
    df = ak.stock_board_concept_name_em()
    if df.empty:
        raise ValueError("未获取到任何概念板块数据。")

    df.columns = _normalize_columns(df.columns)
    name_col = _locate_column(df.columns, CONCEPT_NAME_CANDIDATES, fallback_contains="名称", label="概念名称")
    code_col = _locate_column(df.columns, CONCEPT_CODE_CANDIDATES, fallback_contains="代码", label="概念代码")
    return df, name_col, code_col


def _fetch_single_concept(
    identifiers: Sequence[str],
    *,
    retries: int,
    pause: float,
) -> pd.DataFrame | None:
    """Try downloading one concept by trying multiple identifiers (name, code)."""
    last_error: Exception | None = None
    for symbol in identifiers:
        if not symbol:
            continue
        for attempt in range(1, retries + 1):
            try:
                df = ak.stock_board_concept_cons_em(symbol=symbol)
                df.columns = _normalize_columns(df.columns)
                return df
            except Exception as exc:
                last_error = exc
                sleep_time = pause * attempt
                print(f"[WARN] 拉取概念 {symbol} 失败({attempt}/{retries}): {exc}. {sleep_time:.1f}s 后重试。")
                time.sleep(sleep_time)
    if last_error:
        print(f"[ERROR] 全部标识符 {identifiers} 均获取失败: {last_error}")
    return None


def _standardize_constituents(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure constituent dataframe has 股票代码/股票简称列。"""
    if df.empty:
        return df

    columns = list(df.columns)
    code_col = _locate_column(
        columns,
        STOCK_CODE_CANDIDATES,
        fallback_contains="代码",
        exclude_contains=("板块", "概念"),
        label="股票代码",
    )
    name_col = _locate_column(
        columns,
        STOCK_NAME_CANDIDATES,
        fallback_contains="名",
        exclude_contains=("板块", "概念"),
        label="股票名称",
    )

    rename_map = {}
    if code_col != "股票代码":
        rename_map[code_col] = "股票代码"
    if name_col != "股票简称":
        rename_map[name_col] = "股票简称"

    return df.rename(columns=rename_map)


def export_eastmoney_concepts(
    output_path: str | Path,
    *,
    limit: int | None = None,
    retries: int = 3,
    pause: float = 1.0,
) -> tuple[Path, list[str]]:
    """Download all Eastmoney concept constituents and export to Excel."""
    concept_df, name_col, code_col = _fetch_concept_metadata()
    concept_df = concept_df.dropna(subset=[name_col]).drop_duplicates(subset=[name_col])

    frames: list[pd.DataFrame] = []
    failures: list[str] = []

    iterable = concept_df[[name_col, code_col]].itertuples(index=False, name=None)
    for idx, row in enumerate(iterable, start=1):
        if limit is not None and idx > limit:
            break
        concept_name = str(row[0]).strip()
        raw_code = row[1] if len(row) > 1 else ""
        concept_code = "" if raw_code is None else str(raw_code).strip()

        if not concept_name:
            continue

        identifiers = (concept_name, concept_code)
        cons_df = _fetch_single_concept(identifiers, retries=retries, pause=pause)
        if cons_df is None or cons_df.empty:
            failures.append(concept_name)
            continue

        cons_df = _standardize_constituents(cons_df)
        cons_df.insert(0, "概念名称", concept_name)
        cons_df.insert(1, "概念代码", concept_code)
        frames.append(cons_df)
        print(f"[{idx}] {concept_name} -> {len(cons_df)} 条成份股。")

    if not frames:
        raise RuntimeError("未能成功获取任何概念成份股，请稍后重试。")

    combined = pd.concat(frames, ignore_index=True)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_excel(output, index=False)

    print(f"已导出 {len(combined)} 条股票-概念映射到 {output}")
    if failures:
        print(f"以下概念获取失败或无数据: {', '.join(failures)}")

    return output, failures


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出东方财富概念 -> 股票映射 Excel。")
    parser.add_argument(
        "-o",
        "--output",
        default="eastmoney_concept_constituents.xlsx",
        help="输出 Excel 路径（默认: eastmoney_concept_constituents.xlsx）",
    )
    parser.add_argument("--limit", type=int, default=None, help="仅抓取前 N 个概念，调试用。")
    parser.add_argument("--retries", type=int, default=3, help="单个概念请求重试次数。")
    parser.add_argument("--pause", type=float, default=1.0, help="重试间隔的基准秒数。")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    export_eastmoney_concepts(
        args.output,
        limit=args.limit,
        retries=args.retries,
        pause=args.pause,
    )


if __name__ == "__main__":
    main()
