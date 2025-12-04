"""Generate TIC solution proposals based on a topic, with web-searched case studies.

Usage examples (from repo root):

    python plan_app/run.py "连接器的可靠性与信号完整性——第三方全方位验证方案"

或：

    python -m plan_app.run "新能源汽车电池包第三方可靠性验证方案"
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import dotenv
from openai import OpenAI
from tavily import TavilyClient

dotenv.load_dotenv()


# =========================
# 基础组件：LLM & Tavily
# =========================


def _init_qwen_client() -> OpenAI:
    """Initialize DashScope-compatible OpenAI client for Qwen models."""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未配置，无法调用通义千问 API。")
    base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return OpenAI(api_key=api_key, base_url=base_url)


def _message_content_to_text(content: Any) -> str:
    """Normalize OpenAI-style message content to plain text."""
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
    """Extract the JSON object from a free-form LLM output string."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError("LLM 返回内容不包含 JSON。")
    return text[start : end + 1]


@dataclass
class WebContext:
    """Structured representation of web search context."""

    summary: str
    sources: List[Dict[str, str]]


def _build_tavily_client() -> TavilyClient | None:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        # 允许在没有 Tavily Key 的情况下退化为“仅凭经验生成方案”
        return None
    return TavilyClient(api_key=api_key)


def search_cases_with_tavily(topic: str, max_results: int = 6) -> WebContext:
    """Use Tavily to fetch public case/material around the topic.

    The result will be fed into the LLM as background for the 'case_studies' block.
    """
    client = _build_tavily_client()
    if client is None:
        return WebContext(
            summary="（未配置 TAVILY_API_KEY，无法获取网络检索资料。）",
            sources=[],
        )

    # 构造偏向“第三方检测案例 / 项目”的查询
    query = f"{topic} 第三方检测 可靠性 认证 案例 项目 客户"
    response: Dict[str, Any] = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=True,
        include_raw_content=False,
    )

    results = response.get("results") or []
    answer = (response.get("answer") or "").strip()

    sources: List[Dict[str, str]] = []
    lines: List[str] = []
    if answer:
        lines.append("【Tavily 综合回答摘要】")
        lines.append(answer)
        lines.append("")

    if results:
        lines.append("【主要网页资料（供你抽取或改写为案例）】")
    for idx, item in enumerate(results, start=1):
        title = (item.get("title") or item.get("url") or "").strip()
        url = (item.get("url") or "").strip()
        snippet = (item.get("content") or "").strip()
        src_id = f"S{idx}"
        sources.append({"id": src_id, "title": title, "url": url})

        block_lines = [f"[{src_id}] {title}"]
        if url:
            block_lines.append(f"URL: {url}")
        if snippet:
            block_lines.append(f"摘要: {snippet}")
        lines.append("\n".join(block_lines))
        lines.append("")

    summary_text = "\n".join(lines) if lines else "（未从网络检索到可用资料。）"
    return WebContext(summary=summary_text, sources=sources)


# =========================
# 方案生成 Prompt
# =========================

SOLUTION_SYSTEM_PROMPT = """
你是一名在第三方检测（TIC：Testing, Inspection, Certification）行业深耕 15 年的技术与市场方案顾问，
熟悉电子、电气、汽车、新能源等领域的可靠性、安规与性能测试。

你的任务：根据用户给定的“方案题目”和提供的“网络检索资料”，
生成一份可直接用于 PPT 的第三方检测方案，框架必须包含以下 5 个一级模块：

1. 核心价值（相对通用）
2. 我们的核心测试能力（针对性强）
3. 案例分享（针对性强，PPT 的亮点，优先基于网络检索资料改写或抽象）
4. 测试标准与我们的资质（相对通用）
5. 服务流程与优势总结（相对通用）

【输出要求】
- 只输出 JSON，不要有任何解释性文字。
- JSON 结构必须为：

{
  "topic": "题目原文",
  "target_customer": "建议面向的客户画像（行业 + 角色，如“汽车电子连接器厂商的研发/质量团队”）",
  "application_scenario": "一句话概括业务/应用场景",

  "core_value": {
    "title": "核心价值",
    "points": ["...", "..."]
  },

  "core_testing_capability": {
    "title": "我们的核心测试能力",
    "positioning": "一句话定位（例如：围绕 XXX 的一站式验证平台）",
    "capability_blocks": [
      {
        "name": "能力模块名称",
        "typical_items": ["典型测试项目 1", "典型测试项目 2"],
        "typical_standards": ["相关标准或协议列表"],
        "applicable_products": "适用产品/场景"
      }
    ]
  },

  "case_studies": [
    {
      "name": "案例名称（可根据公开信息进行适度改写或泛化）",
      "customer_type": "客户类型",
      "challenge": "客户面临的关键问题",
      "solution": "我们提供的整体方案（包含测试+技术支持）",
      "highlights": ["方案亮点 1", "方案亮点 2"],
      "value": "给客户带来的价值（如缩短周期、降低失败风险等）",
      "source_ids": ["S1", "S2"]
    }
  ],

  "standards_and_qualifications": {
    "key_standards": ["关键测试标准/法规列表"],
    "accreditations": ["实验室资质/认可"],
    "lab_capabilities": ["实验室平台能力/重要设备"]
  },

  "service_flow_and_advantages": {
    "service_flow": ["步骤 1", "步骤 2", "步骤 3", "步骤 4", "步骤 5"],
    "advantages": ["我们的 3-6 个差异化优势"]
  }
}

- 所有内容全部使用简体中文。
- `case_studies` 模块中，请尽量基于提供的网络资料进行抽象和再组织，可以适度泛化公司名称或敏感信息；
  如果网络资料不足，则可以给出符合行业常识的典型虚构案例，但不要捏造具体企业的机密信息。
- 如果题目过于宽泛，请先在 mind 中假设一个最典型、最有利于成交的应用场景，再据此生成方案。
"""


def generate_solution(topic: str, model: str | None = None) -> Dict[str, Any]:
    """Generate a structured TIC solution proposal JSON."""
    client = _init_qwen_client()
    model_name = model or os.getenv("QWEN_MODEL_NAME", "qwen3-max")

    web_ctx = search_cases_with_tavily(topic, max_results=6)

    user_prompt = (
        f"方案题目：{topic}\n\n"
        "以下是围绕该题目的网络检索资料，请重点用于“案例分享”模块，"
        "也可辅助完善核心价值与测试能力描述：\n\n"
        f"{web_ctx.summary}\n\n"
        "请严格按照系统提示中的 JSON 结构，生成一份完整的第三方检测解决方案。"
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SOLUTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # 对于 Qwen3 推理模型可以开启思维扩展；若是普通对话模型会自动忽略
        extra_body={"enable_thinking": True},
    )

    raw = _message_content_to_text(resp.choices[0].message.content)
    payload = json.loads(_extract_json(raw.strip()))

    # 附带上用于生成的来源列表，方便后续引用或排查
    if isinstance(payload, dict):
        payload.setdefault("_web_sources", web_ctx.sources)

    return payload


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="根据题目生成第三方检测（TIC）方案（结构化 JSON）。"
    )
    parser.add_argument(
        "topic",
        help="方案题目，例如：连接器的可靠性与信号完整性——第三方全方位验证方案",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("QWEN_MODEL_NAME", "qwen3-max"),
        help="通义千问模型名称（默认从环境变量 QWEN_MODEL_NAME 读取，缺省为 qwen3-max）",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="是否以缩进格式输出 JSON，便于阅读。",
    )
    args = parser.parse_args(argv)

    data = generate_solution(args.topic, model=args.model)
    text = json.dumps(data, ensure_ascii=False, indent=2 if args.pretty else None)
    print(text)


if __name__ == "__main__":
    main()

