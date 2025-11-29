"""按章节拆分 Markdown，逐段调用 LLM 分析并输出汇总。"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

dotenv.load_dotenv()


TIC_SYSTEM_PROMPT = """
# Role
你是一位拥有 20 年经验的全球 TIC（检测、检验、认证）行业资深顾问。你精通 ISO、IEC、ASTM、GB、FDA、CE、UL 等跨行业标准，熟悉从原材料、零部件到成品全生命周期的合规与质量控制要求。

# Task
阅读我提供的章节内容，从中精准提取所有存在潜在 TIC 业务机会的产品、服务或设施。

# Extraction Strategy
按照以下四个维度扫描：
1. **合规与市场准入**：出口、车规、医规、安规、环保等。
2. **安全性与可靠性**：耐高低温、阻燃、寿命、环境适应等。
3. **性能与效能**：能效、传输速度、精度、成分、纯度等指标验证。
4. **供应链管控**：原料采购、工艺、验货、良率、供应商管理等。

# Output Format
请先按章节整理结果，之后我会汇总为统一表格。每条机会需要包含所属行业/类别、具体产品/对象、关键特征/应用场景、推荐 TIC 服务方案、对应的关键检测项目。

# Constraints
仅基于文本中的明确信息或强烈暗示，给出专业、具体的检测/认证建议。
"""

SECTION_ANALYSIS_REQUIREMENT = """
请阅读章节内容，仅输出 JSON：
{
  "title": "章节标题",
  "opportunities": [
    {
      "industry": "所属行业/类别",
      "target": "具体产品/对象",
      "features": "来自原文的关键特征或应用描述",
      "services": ["推荐 TIC 服务方案（使用具体标准或测试名称）"],
      "test_items": ["服务对应的具体检测项目（例如：绝缘耐压测试、IP 防护等级测试、盐雾腐蚀测试等，尽可能详细列出）"]
    }
  ]
}
如无可提取内容，opportunities 传回空数组。严禁输出 JSON 以外内容。
"""


@dataclass
class Section:
    title: str
    content: str


class TicOpportunity(BaseModel):
    industry: str
    target: str
    features: str
    services: List[str] = Field(default_factory=list)
    # 具体的检测/试验项目名称列表（与服务方案相对应）
    test_items: List[str] = Field(default_factory=list)


class SectionAnalysis(BaseModel):
    title: str
    opportunities: List[TicOpportunity] = Field(default_factory=list)


def _default_md_path() -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    return base_dir / "data" / "三安光电.md"


def load_sections(md_path: Path) -> List[Section]:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    sections: List[Section] = []
    current_title: str | None = None
    buffer: List[str] = []
    heading_pattern = re.compile(r"^##\s+(.*)$")

    for raw_line in lines:
        match = heading_pattern.match(raw_line.strip())
        if match:
            if current_title and buffer:
                sections.append(Section(current_title, "\n".join(buffer).strip()))
                buffer = []
            current_title = match.group(1).strip()
        else:
            if current_title:
                buffer.append(raw_line)

    if current_title and buffer:
        sections.append(Section(current_title, "\n".join(buffer).strip()))

    ignored_titles = {"目录", "备查文件目录"}
    filtered = [sec for sec in sections if sec.title not in ignored_titles and sec.content]
    return filtered


def _init_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未配置，无法调用通义千问 API。")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            text_value = getattr(part, "text", None)
            if text_value is None and isinstance(part, dict):
                text_value = part.get("text")
            if text_value:
                chunks.append(text_value)
        return "".join(chunks)
    if content is None:
        return ""
    return str(content)


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError("LLM 返回内容不包含 JSON。")
    return text[start : end + 1]


def _normalize_section_payload(payload: Any) -> Any:
    """递归清洗 LLM 返回的 JSON，主要是去掉 key 两侧的空格，避免校验失败。"""

    if isinstance(payload, dict):
        cleaned: dict[Any, Any] = {}
        for k, v in payload.items():
            if isinstance(k, str):
                k = k.strip()
            cleaned[k] = _normalize_section_payload(v)
        # 如果 opportunities 不是列表，直接置空，避免报错
        if "opportunities" in cleaned and not isinstance(cleaned["opportunities"], list):
            cleaned["opportunities"] = []
        return cleaned
    if isinstance(payload, list):
        return [_normalize_section_payload(item) for item in payload]
    return payload


def analyze_section(client: OpenAI, model: str, section: Section) -> SectionAnalysis:
    user_prompt = (
        f"章节标题：《{section.title}》\n"
        f"请仅基于以下文本提取所有潜在 TIC 业务机会，谨慎遵循系统指令的四大维度与表格要求。"
        f"\n章节内容：\n{section.content}\n\n{SECTION_ANALYSIS_REQUIREMENT}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # 启用 Qwen3 的 thinking 模式
        extra_body={"enable_thinking": True},
    )
    raw = _message_content_to_text(resp.choices[0].message.content)
    payload = json.loads(_extract_json(raw.strip()))
    payload = _normalize_section_payload(payload)
    return SectionAnalysis.model_validate(payload)


def render_final_table(analyses: List[SectionAnalysis]) -> str:
    rows: List[tuple[str, str, str, str, str]] = []
    for section in analyses:
        for opp in section.opportunities:
            services = "、".join(opp.services)
            test_items = "、".join(opp.test_items)
            rows.append((opp.industry, opp.target, opp.features, services, test_items))

    if not rows:
        return "未在文本中发现可提取的 TIC 商机。"

    header = "| 所属行业/类别 | 具体产品/对象 | 关键特征/应用场景 | 推荐 TIC 服务方案 | 关键检测项目 |"
    separator = "| :--- | :--- | :--- | :--- | :--- |"
    lines = [header, separator]
    for industry, target, features, services, test_items in rows:
        lines.append(f"| {industry} | {target} | {features} | {services} | {test_items} |")

    lines.append(f"\n共提取 {len(rows)} 条潜在 TIC 机会。")
    return "\n".join(lines)


def print_section_analysis(analysis: SectionAnalysis) -> None:
    print(f"## {analysis.title}")
    if not analysis.opportunities:
        print("（本章节未发现可提取的 TIC 机会）\n")
        return

    header = "| 所属行业/类别 | 具体产品/对象 | 关键特征/应用场景 | 推荐 TIC 服务方案 | 关键检测项目 |"
    separator = "| :--- | :--- | :--- | :--- | :--- |"
    print(header)
    print(separator)
    for opp in analysis.opportunities:
        services = "、".join(opp.services)
        test_items = "、".join(opp.test_items)
        print(f"| {opp.industry} | {opp.target} | {opp.features} | {services} | {test_items} |")
    print("")


def main() -> None:
    md_path = Path(os.getenv("MD_SOURCE_PATH", _default_md_path()))
    if not md_path.exists():
        raise FileNotFoundError(f"找不到 Markdown 文件：{md_path}")

    sections = load_sections(md_path)
    print(f"共检测到 {len(sections)} 个章节，开始逐段调用 LLM...\n")

    client = _init_client()
    # 默认使用 Qwen3-Max 推理模型，可通过环境变量 QWEN_MODEL_NAME 覆盖
    model = os.getenv("QWEN_MODEL_NAME", "qwen3-max")
    analyses: List[SectionAnalysis] = []
    for section in sections:
        try:
            analysis = analyze_section(client, model, section)
        except Exception as exc:
            print(f"章节《{section.title}》分析失败：{exc}")
            continue
        analyses.append(analysis)
        print_section_analysis(analysis)

    if not analyses:
        raise RuntimeError("未获取任何章节分析结果，无法汇总。")

    print("===== 汇总表格 =====")
    table = render_final_table(analyses)
    print(table)


if __name__ == "__main__":
    main()
