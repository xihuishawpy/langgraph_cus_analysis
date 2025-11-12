from __future__ import annotations

import argparse
import time
from io import StringIO
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
import requests

import akshare as ak

CONCEPT_NAME_CANDIDATES: Sequence[str] = ("概念名称", "板块名称", "名称")
CONCEPT_CODE_CANDIDATES: Sequence[str] = ("概念代码", "板块代码", "代码")
STOCK_CODE_CANDIDATES: Sequence[str] = ("股票代码", "证券代码", "代码")
STOCK_NAME_CANDIDATES: Sequence[str] = ("股票简称", "股票名称", "名称", "证券简称")

THS_DETAIL_URL = "http://q.10jqka.com.cn/gn/detail/code/{code}/"
THS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "http://q.10jqka.com.cn/gn/",
}
THS_TIMEOUT = 15


def _normalize_columns(columns: Iterable[object]) -> List[str]:
    """标准化列名为去空格的字符串列表。"""
    return [str(col).strip() for col in columns]


def _locate_column(
    columns: Sequence[str],
    candidates: Sequence[str],
    *,
    fallback_contains: str | None = None,
    exclude_contains: Sequence[str] = (),
    label: str = "",
) -> str:
    """根据候选名或包含规则定位目标列名。"""
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


def _fetch_concept_metadata(retries: int, pause: float) -> tuple[pd.DataFrame, str, str]:
    """抓取同花顺概念列表并识别名称/代码列。"""
    last_error: Exception | None = None
    df: pd.DataFrame | None = None
    for attempt in range(1, retries + 1):
        try:
            df = ak.stock_board_concept_name_ths()
            break
        except Exception as exc:
            last_error = exc
            sleep_time = pause * attempt
            print(
                f"[WARN] 获取同花顺概念列表失败({attempt}/{retries}): {exc}. "
                f"{sleep_time:.1f}s 后重试。"
            )
            time.sleep(sleep_time)

    if df is None:
        raise RuntimeError(f"同花顺概念列表多次获取失败: {last_error}") from last_error
    if df.empty:
        raise ValueError("未获取到任何同花顺概念板块数据。")

    df = df.rename(columns={"name": "概念名称", "code": "概念代码"})
    df["概念名称"] = df["概念名称"].astype(str).str.strip()
    df["概念代码"] = df["概念代码"].astype(str).str.strip()
    print(f"[INFO] 已获取同花顺概念 {len(df)} 个。")
    return df, "概念名称", "概念代码"


def _fetch_single_concept(
    identifiers: Sequence[str],
    *,
    retries: int,
    pause: float,
) -> pd.DataFrame | None:
    """请求并解析单个概念的成份股表。"""
    concept_name = identifiers[0] if identifiers else ""
    concept_code = identifiers[1] if len(identifiers) > 1 else ""
    if not concept_code:
        print(f"[WARN] 同花顺概念 {concept_name} 缺少代码，已跳过。")
        return None

    url = THS_DETAIL_URL.format(code=concept_code)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=THS_HEADERS, timeout=THS_TIMEOUT)
            resp.raise_for_status()
            tables = pd.read_html(StringIO(resp.text))
            if not tables:
                raise ValueError("未解析到成份表")

            selected: pd.DataFrame | None = None
            for table in tables:
                normalized_cols = _normalize_columns(table.columns)
                has_code = any(col in normalized_cols for col in STOCK_CODE_CANDIDATES)
                has_name = any(col in normalized_cols for col in STOCK_NAME_CANDIDATES)
                if has_code and has_name:
                    selected = table.copy()
                    selected.columns = normalized_cols
                    break

            if selected is None:
                raise ValueError("未找到包含代码/名称列的成份表")

            if "代码" in selected.columns:
                selected["代码"] = selected["代码"].astype(str).str.zfill(6)
            return selected
        except Exception as exc:
            last_error = exc
            sleep_time = pause * attempt
            print(
                f"[WARN] 拉取同花顺概念 {concept_name}({concept_code}) "
                f"失败({attempt}/{retries}): {exc}. {sleep_time:.1f}s 后重试。"
            )
            time.sleep(sleep_time)
    if last_error:
        print(
            f"[ERROR] 同花顺概念 {concept_name}({concept_code}) 获取失败: {last_error}"
        )
    return None


def _standardize_constituents(df: pd.DataFrame) -> pd.DataFrame:
    """规范成份股表列名为"股票代码/股票简称"。"""
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
    """抓取全部概念成份股并导出为Excel。"""
    concept_df, name_col, code_col = _fetch_concept_metadata(
        retries=retries, pause=pause
    )
    concept_df = concept_df.dropna(subset=[name_col]).drop_duplicates(subset=[name_col])

    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    print("[INFO] 当前数据源: 同花顺")

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
