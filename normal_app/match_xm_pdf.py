#!/usr/bin/env python3
"""匹配 xm.pdf 中的服务项目与财报中的检测项目

Usage (from repo root):

    python -m normal_app.match_xm_pdf

或：

    python normal_app/match_xm_pdf.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import fitz  # PyMuPDF
from difflib import SequenceMatcher

from normal_app.run import (
    SectionAnalysis,
    _extract_json,
    _init_client,
    _message_content_to_text,
)


LLM_MATCH_SYSTEM_PROMPT = """
你是一名熟悉半导体/电子行业检测与失效分析的 TIC 专家。

现在有两类信息：
1）“检测项目”：来自财报分析提炼的关键检测项目名称，如“金线拉力测试（Wire Pull）”、“ESD 静电放电测试”等；
2）“服务项目”：来自第三方实验室报价清单，每条包含项目代码（如 ESD001、FA041、RA068）和中文/英文说明。

你的任务：根据技术含义，判断每个“检测项目”在报价清单中是否存在对应或高度相关的“服务项目”，并给出项目代码和简短理由。

- 只在技术内容高度匹配或明显属于同一类测试时才认为“匹配”。
- 名称相似但技术内容完全不同时，必须标记为不匹配。
- 如果在候选列表中没有合适项目，可以返回空数组。

输出必须是 JSON，格式如下：
{
  "test_item": "原始检测项目名称",
  "matches": [
    {
      "code": "ESD001",
      "reason": "与静电放电敏感度测试同属 ESD 测试，描述中包含 HBM/MM 等典型模式"
    }
  ]
}
严禁输出 JSON 以外的内容。
"""


# 额外的“实验室项目描述”列表，用于与财报检测项目做额外匹配
EXTRA_PROJECT_DESCRIPTIONS: List[str] = [
    "TEM/EDX",
    "定点X-S TEM样品制备(N-Si)",
    "SIMS-元素分析",
    "二次离子质谱仪 7f",
    "FIB-FA(高)",
    "非定点X-S TEM样品制备(N-Si)",
    "Engineering Charge 工程费用",
    "SIMS定点样品制备",
    "定点SEM样品制备",
    "SEM/EDX",
    "热影像分析仪",
    "定点展阻分析仪",
    "拉曼光谱分析-委外",
    "高解析度3D X-Ray显微镜",
    "XPS/ESCA",
    "X光绕射分析仪",
    "非定点展阻分析仪",
    "PCB layer 研磨",
    "离子束截面研磨 Normally",
    "超高解析3D光学显微镜",
    "InGaAs 微光显微镜",
    "Others-特殊样品制备",
    "开盖後补线",
    "I-V curve 量测",
    "特殊取晶粒(PCB、CIS…)",
    "取晶粒 一般件",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sections_json_path() -> Path:
    # 约定的缓存路径：data/三安光电_sections.json
    return _project_root() / "data" / "三安光电_sections.json"


def load_analyses_from_json(path: Path) -> List[SectionAnalysis]:
    """Load SectionAnalysis list from a JSON cache file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    # 允许两种结构：
    # 1) 直接是 SectionAnalysis 列表
    # 2) { "analyses": [ ... ] }
    if isinstance(data, dict) and "analyses" in data:
        items = data["analyses"]
    else:
        items = data
    return [SectionAnalysis.model_validate(item) for item in items]


@dataclass
class TestItemContext:
    """单条检测项目及其上下文信息，用于更稳健的 LLM 匹配。"""

    test_item: str
    industry: str
    target: str
    features: str
    services: List[str]
    section_title: str


def match_test_items_to_extra(
    test_items: List[TestItemContext],
    extra_descriptions: List[str],
    model: str,
) -> List[Tuple[str, str, str]]:
    """使用 LLM 将财报检测项目与实验室项目描述做语义匹配。

    返回 (financial_test_item, lab_project_code, lab_project_description) 列表。
    这里复用 _llm_match_one 的思路，将实验室项目列表包装成候选“服务项目”。
    """
    client = _init_client()

    # 将实验室项目列表包装成 ServiceRecord，code 就直接使用描述文本本身
    extra_records: List[ServiceRecord] = []
    for desc in extra_descriptions:
        text = (desc or "").strip()
        if not text:
            continue
        extra_records.append(ServiceRecord(code=text, lines=[text]))

    all_matches: List[Tuple[str, str, str]] = []
    for item_ctx in test_items:
        # 先用简单相似度筛一轮候选，再交给 LLM 精筛，避免一次塞太多内容
        candidates = _rank_candidates_for_item(item_ctx, extra_records, top_k=8)
        if not candidates:
            continue

        try:
            _, codes = _llm_match_one(client, model, item_ctx, candidates)
        except Exception as exc:  # noqa: BLE001
            print(f"LLM 匹配财报项目 '{item_ctx.test_item}' 到实验室项目失败，跳过：{exc}")
            continue

        if not codes:
            continue

        code_set = set(codes)
        for rec in candidates:
            if rec.code in code_set:
                all_matches.append((item_ctx.test_item, rec.code, rec.text))

    return all_matches


def collect_test_items(analyses: Iterable[SectionAnalysis]) -> List[TestItemContext]:
    """Collect unique test_items from all opportunities, with context."""
    seen: set[tuple[str, str, str]] = set()
    items: List[TestItemContext] = []
    for ana in analyses:
        for opp in ana.opportunities:
            for raw in getattr(opp, "test_items", []) or []:
                item = raw.strip()
                if not item:
                    continue
                sig = (item, opp.industry, opp.target)
                if sig in seen:
                    continue
                seen.add(sig)
                items.append(
                    TestItemContext(
                        test_item=item,
                        industry=opp.industry,
                        target=opp.target,
                        features=opp.features,
                        services=list(opp.services or []),
                        section_title=ana.title,
                    )
                )
    return items


def load_xm_lines(pdf_path: Path) -> List[str]:
    """Extract all non-empty lines from xm.pdf."""
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"项目清单 PDF 不存在：{pdf_path}")

    doc = fitz.open(pdf_path)
    lines: List[str] = []
    try:
        for page in doc:
            text = page.get_text("text") or ""
            for line in text.splitlines():
                line = line.strip()
                if line:
                    lines.append(line)
    finally:
        doc.close()
    return lines


class ServiceRecord:
    """Structured representation of a service row parsed from xm.pdf."""

    def __init__(self, code: str, lines: List[str]) -> None:
        self.code = code
        # 合并该项目相关的所有行，供匹配与 LLM 参考
        self.lines = lines

    @property
    def text(self) -> str:
        return " ".join(self.lines)


def build_service_records(xm_lines: List[str]) -> List[ServiceRecord]:
    """Group xm.pdf lines into ServiceRecord objects by project code."""
    import re

    records: List[ServiceRecord] = []
    i = 0
    n = len(xm_lines)

    while i < n:
        line = xm_lines[i].strip()
        if re.fullmatch(r"[A-Z]{2,3}\d{3}", line):
            code = line
            group = [line]
            j = i + 1
            # 聚合直到下一个代码或文件结束
            while j < n and not re.fullmatch(r"[A-Z]{2,3}\d{3}", xm_lines[j].strip()):
                group.append(xm_lines[j].strip())
                j += 1
            records.append(ServiceRecord(code, group))
            i = j
        else:
            i += 1

    return records


def _find_code_nearby(lines: List[str], idx: int) -> str:
    """Look up a project code (e.g. ESD001 / FA041 / RA068) a few lines above idx."""
    import re

    for j in range(max(0, idx - 4), idx):
        s = lines[j].strip()
        if re.fullmatch(r"[A-Z]{2,3}\d{3}", s):
            return s
    return ""


def find_matches(
    test_items: List[str],
    xm_lines: List[str],
    threshold: float = 0.45,
) -> List[Tuple[str, str, str, float]]:
    """Return list of (test_item, code, line, score) for matched entries."""
    matches: List[Tuple[str, str, str, float]] = []

    for item in test_items:
        norm_item = item.replace(" ", "").lower()
        if not norm_item:
            continue

        best_score = 0.0
        best_idx = -1

        for idx, line in enumerate(xm_lines):
            norm_line = line.replace(" ", "").lower()

            # quick path: substring match
            if norm_item in norm_line or norm_line in norm_item:
                score = 1.0
            else:
                score = SequenceMatcher(None, norm_item, norm_line).ratio()

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0 and best_score >= threshold:
            code = _find_code_nearby(xm_lines, best_idx)
            matches.append((item, code, xm_lines[best_idx], best_score))

    # sort by code then by score desc for stable, readable output
    matches.sort(key=lambda x: (x[1] or "ZZZ", -x[3], x[0]))
    return matches


def _rank_candidates_for_item(
    item: TestItemContext,
    records: List[ServiceRecord],
    top_k: int = 8,
) -> List[ServiceRecord]:
    """Use simple string similarity to pick top-k candidate service records."""
    norm_item = item.test_item.replace(" ", "").lower()
    scored: List[Tuple[float, ServiceRecord]] = []

    for rec in records:
        norm_text = rec.text.replace(" ", "").lower()
        if not norm_item or not norm_text:
            continue

        if norm_item in norm_text or norm_text in norm_item:
            score = 1.0
        else:
            score = SequenceMatcher(None, norm_item, norm_text).ratio()

        scored.append((score, rec))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [rec for score, rec in scored[:top_k] if score > 0.1]


def _llm_match_one(
    client,
    model: str,
    item_ctx: TestItemContext,
    candidates: List[ServiceRecord],
) -> Tuple[str, List[str]]:
    """Ask LLM to judge which candidate service records match the test item.

    Returns (test_item, [matched_codes]).
    """
    if not candidates:
        return item_ctx.test_item, []

    # 构造用户提示，将检测项目及其上下文 + 候选服务项目列表都给到模型
    lines: List[str] = [
        f"检测项目：{item_ctx.test_item}",
        f"所属行业/类别：{item_ctx.industry}",
        f"具体产品/对象：{item_ctx.target}",
        f"章节标题：{item_ctx.section_title}",
        f"原文关键特征/应用场景：{item_ctx.features}",
        "推荐 TIC 服务方案：" + ("；".join(item_ctx.services) if item_ctx.services else "（无）"),
        "",
        "候选服务项目列表：",
    ]
    for idx, rec in enumerate(candidates, start=1):
        lines.append(f"{idx}. [{rec.code}] {rec.text}")
    user_prompt = "\n".join(lines)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LLM_MATCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        extra_body={"enable_thinking": True},
    )

    raw = _message_content_to_text(resp.choices[0].message.content)
    payload = json.loads(_extract_json(raw.strip()))
    matches = payload.get("matches", []) or []

    codes: List[str] = []
    for m in matches:
        code = (m or {}).get("code")
        if isinstance(code, str) and code.strip():
            codes.append(code.strip())

    return item_ctx.test_item, codes


def llm_match(
    test_items: List[TestItemContext],
    records: List[ServiceRecord],
    model: str,
) -> List[Tuple[str, str, str]]:
    """Use LLM to match test_items to xm service records.

    Returns list of (test_item, code, text).
    """
    client = _init_client()

    # 先为每个检测项目挑选若干候选，再用 LLM 精筛
    all_matches: List[Tuple[str, str, str]] = []
    for item_ctx in test_items:
        candidates = _rank_candidates_for_item(item_ctx, records, top_k=8)
        try:
            _, codes = _llm_match_one(client, model, item_ctx, candidates)
        except Exception as exc:  # noqa: BLE001
            print(f"LLM 匹配 '{item_ctx.test_item}' 失败，跳过：{exc}")
            continue

        if not codes:
            continue

        # 仅保留 LLM 选中的代码对应的记录
        code_set = set(codes)
        for rec in candidates:
            if rec.code in code_set:
                all_matches.append((item_ctx.test_item, rec.code, rec.text))

    return all_matches


def main() -> None:
    root = _project_root()
    sections_json = _sections_json_path()

    if not sections_json.exists():
        raise FileNotFoundError(
            f"找不到章节分析缓存文件：{sections_json}\n"
            "请先将 LLM 的章节分析结果保存为 JSON（SectionAnalysis 列表），"
            "或调整 _sections_json_path() 指向你的 JSON 文件。"
        )

    analyses = load_analyses_from_json(sections_json)
    test_items = collect_test_items(analyses)
    print(f"从章节分析 JSON 中收集到 {len(test_items)} 个检测项目候选。\n")

    xm_pdf = root / "data" / "xm.pdf"
    xm_lines = load_xm_lines(xm_pdf)
    print(f"从 xm.pdf 中提取到 {len(xm_lines)} 行文本。\n")

    # 解析报价清单为结构化记录，供 LLM 参考
    records = build_service_records(xm_lines)
    print(f"解析得到 {len(records)} 条服务项目记录。\n")

    # 使用与 run.py 相同的模型配置
    import os

    model = os.getenv("QWEN_MODEL_NAME", "qwen3-max")

    print("开始使用 LLM 进行语义匹配（按检测项目逐个匹配）...\n")
    llm_matches = llm_match(test_items, records, model=model)

    print("===== 使用 LLM 与 xm 报价清单匹配到的检测项目 =====")
    for item, code, text in llm_matches:
        code_str = f"[{code}]" if code else "[未知代码]"
        print(f"财报检测项目：{item} ---> {code_str} {text} ")

    print(f"\n共匹配到 {len(llm_matches)} 个项目。")

    # 将匹配结果结构化保存到 JSON 文件，便于后续分析或导入其他系统
    result_by_item: dict[str, dict] = {}
    for item, code, text in llm_matches:
        rec = result_by_item.setdefault(
            item,
            {
                "financial_test_item": item,  # 财报推断的检测项目
                "matches": [],  # 对应 xm.pdf 中的服务/检测项目（可能有多个）
            },
        )
        rec["matches"].append(
            {
                "code": code,
                "service_text": text,
            }
        )

    output_path = root / "data" / "xm_match_results.json"
    output_path.write_text(
        json.dumps(list(result_by_item.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n匹配结果已保存到：{output_path}")

    # 额外导出为 Excel / CSV，方便在表格中查看与筛选
    # 行粒度为「一条财报检测项目与 xm.pdf 中的一个服务项目」。
    try:
        import pandas as pd

        rows: List[dict[str, str]] = []
        for item, code, text in llm_matches:
            rows.append(
                {
                    "financial_test_item": item,
                    "xm_service_code": code,
                    "xm_service_text": text,
                }
            )

        if rows:
            df = pd.DataFrame(rows)
            excel_path = root / "data" / "xm_match_results.xlsx"
            try:
                # 默认使用 openpyxl / xlsxwriter 引擎，如未安装则降级为 CSV
                df.to_excel(excel_path, index=False)
                print(f"匹配结果已导出为 Excel：{excel_path}")
            except Exception as exc:  # noqa: BLE001
                csv_path = root / "data" / "xm_match_results.csv"
                # 使用 utf-8-sig 方便在 Excel 中直接打开
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                print(
                    "写入 Excel 文件失败，已降级导出为 CSV："
                    f"{csv_path}（错误信息：{exc}）"
                )
        else:
            print("没有匹配结果，不生成 Excel/CSV 文件。")
    except Exception as exc:  # noqa: BLE001
        print(f"导出 Excel/CSV 时出现异常：{exc}")

    # =========================
    # 额外：财报检测项目 ↔ 实验室项目描述（LLM 语义匹配）
    # =========================
    print("\n开始对财报检测项目与自定义实验室项目列表进行 LLM 语义匹配...\n")
    extra_llm_matches = match_test_items_to_extra(
        test_items, EXTRA_PROJECT_DESCRIPTIONS, model=model
    )

    print("===== 财报检测项目 与 自定义实验室项目 的匹配结果（LLM） =====")
    for fin_item, lab_code, lab_text in extra_llm_matches:
        code_str = f"[{lab_code}]" if lab_code else "[未知代码]"
        print(f"财报检测项目：{fin_item}  ---> 实验室项目：{code_str} {lab_text}")

    print(f"\n财报检测项目与自定义列表共产生 {len(extra_llm_matches)} 条 LLM 匹配记录。")

    # 保存为 JSON（按财报检测项目聚合）
    extra_result_by_item: dict[str, dict] = {}
    for fin_item, lab_code, lab_text in extra_llm_matches:
        rec = extra_result_by_item.setdefault(
            fin_item,
            {
                "financial_test_item": fin_item,
                "matches": [],
            },
        )
        rec["matches"].append(
            {
                "lab_project_code": lab_code,
                "lab_project_description": lab_text,
            }
        )

    extra_json_path = root / "data" / "xm_match_financial_to_extra.json"
    extra_json_path.write_text(
        json.dumps(list(extra_result_by_item.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n财报检测项目与自定义实验室项目的 LLM 匹配结果已保存到：{extra_json_path}")

    # 保存为 Excel / CSV（逐行展开）
    try:
        import pandas as pd

        if extra_llm_matches:
            extra_rows: List[dict[str, object]] = []
            for fin_item, lab_code, lab_text in extra_llm_matches:
                extra_rows.append(
                    {
                        "financial_test_item": fin_item,
                        "lab_project_code": lab_code,
                        "lab_project_description": lab_text,
                    }
                )

            extra_df = pd.DataFrame(extra_rows)
            extra_excel_path = root / "data" / "xm_match_financial_to_extra.xlsx"
            try:
                extra_df.to_excel(extra_excel_path, index=False)
                print(
                    f"财报检测项目与自定义实验室项目的 LLM 匹配结果已导出为 Excel：{extra_excel_path}"
                )
            except Exception as exc:  # noqa: BLE001
                extra_csv_path = root / "data" / "xm_match_financial_to_extra.csv"
                extra_df.to_csv(extra_csv_path, index=False, encoding="utf-8-sig")
                print(
                    "写入财报↔自定义实验室项目 Excel 文件失败，已降级导出为 CSV："
                    f"{extra_csv_path}（错误信息：{exc}）"
                )
        else:
            print("财报检测项目与自定义列表无有效 LLM 匹配，不生成额外的 Excel/CSV 文件。")
    except Exception as exc:  # noqa: BLE001
        print(f"导出财报↔自定义实验室项目 Excel/CSV 时出现异常：{exc}")


if __name__ == "__main__":
    main()
