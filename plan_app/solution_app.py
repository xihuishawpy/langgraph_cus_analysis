"""Streamlit front-end for the TIC solution generation agent.

Run from repo root:

    streamlit run plan_app/solution_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import dotenv
import streamlit as st

# Ensure project root is on sys.path so `plan_app` package can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plan_app.run import (  # type: ignore[import]
    generate_solution,
    _init_qwen_client,
    _message_content_to_text,
)

dotenv.load_dotenv()


def _solution_to_markdown(data: Dict[str, Any]) -> str:
    """Render solution JSON into human-readable Markdown."""

    def add_list(lines: List[str], items: List[str] | None, prefix: str = "- ") -> None:
        if not items:
            return
        for item in items:
            item = (item or "").strip()
            if item:
                lines.append(f"{prefix}{item}")

    topic = (data.get("topic") or "第三方检测解决方案").strip()
    target_customer = (data.get("target_customer") or "").strip()
    application_scenario = (data.get("application_scenario") or "").strip()

    lines: List[str] = [f"# {topic}", ""]

    if target_customer:
        lines.append(f"> 面向客户：{target_customer}")
    if application_scenario:
        lines.append(f"> 应用场景：{application_scenario}")
    if target_customer or application_scenario:
        lines.append("")

    # 一、核心价值
    core_value = data.get("core_value") or {}
    cv_title = (core_value.get("title") or "核心价值").strip()
    lines.append(f"## 一、{cv_title}")
    add_list(lines, core_value.get("points") or [])
    lines.append("")

    # 二、核心测试能力
    ctc = data.get("core_testing_capability") or {}
    ctc_title = (ctc.get("title") or "我们的核心测试能力").strip()
    lines.append(f"## 二、{ctc_title}")
    positioning = (ctc.get("positioning") or "").strip()
    if positioning:
        lines.append(f"**定位：**{positioning}")
        lines.append("")
    blocks = ctc.get("capability_blocks") or []
    for idx, block in enumerate(blocks, start=1):
        name = (block.get("name") or f"能力模块 {idx}").strip()
        lines.append(f"### {idx}. {name}")
        typical_items = block.get("typical_items") or []
        typical_standards = block.get("typical_standards") or []
        applicable_products = (block.get("applicable_products") or "").strip()
        if typical_items:
            lines.append("**典型测试项目：**")
            add_list(lines, typical_items)
        if typical_standards:
            lines.append("")
            lines.append("**相关标准/规范：**")
            add_list(lines, typical_standards)
        if applicable_products:
            lines.append("")
            lines.append(f"**典型适用产品/场景：**{applicable_products}")
        lines.append("")

    # 三、案例分享
    lines.append("## 三、案例分享")
    case_studies = data.get("case_studies") or []
    if not case_studies:
        lines.append("- （可根据实际项目补充典型案例）")
    else:
        for idx, case in enumerate(case_studies, start=1):
            name = (case.get("name") or f"案例 {idx}").strip()
            customer_type = (case.get("customer_type") or "").strip()
            challenge = (case.get("challenge") or "").strip()
            solution = (case.get("solution") or "").strip()
            highlights = case.get("highlights") or []
            value = (case.get("value") or "").strip()

            lines.append(f"### 案例 {idx}：{name}")
            if customer_type:
                lines.append(f"- **客户类型：**{customer_type}")
            if challenge:
                lines.append(f"- **客户挑战：**{challenge}")
            if solution:
                lines.append(f"- **我们的方案：**{solution}")
            if highlights:
                lines.append("- **方案亮点：**")
                add_list(lines, highlights, prefix="  - ")
            if value:
                lines.append(f"- **客户价值：**{value}")
            lines.append("")

    # 四、测试标准与资质
    saq = data.get("standards_and_qualifications") or {}
    lines.append("## 四、测试标准与我们的资质")
    key_standards = saq.get("key_standards") or []
    accs = saq.get("accreditations") or []
    lab_caps = saq.get("lab_capabilities") or []
    if key_standards:
        lines.append("**关键测试标准/法规：**")
        add_list(lines, key_standards)
        lines.append("")
    if accs:
        lines.append("**实验室资质/认可：**")
        add_list(lines, accs)
        lines.append("")
    if lab_caps:
        lines.append("**实验室平台与能力：**")
        add_list(lines, lab_caps)
        lines.append("")

    # 五、服务流程与优势
    sfa = data.get("service_flow_and_advantages") or {}
    lines.append("## 五、服务流程与优势总结")
    flow = sfa.get("service_flow") or []
    advs = sfa.get("advantages") or []
    if flow:
        lines.append("**标准服务流程：**")
        add_list(lines, flow)
        lines.append("")
    if advs:
        lines.append("**我们的优势：**")
        add_list(lines, advs)
        lines.append("")

    # 参考来源
    sources = data.get("_web_sources") or []
    if sources:
        lines.append("## 参考来源")
        for src in sources:
            title = (src.get("title") or "").strip() or src.get("url") or ""
            url = (src.get("url") or "").strip()
            if url:
                lines.append(f"- [{title}]({url})")
            elif title:
                lines.append(f"- {title}")

    return "\n".join(lines).strip() + "\n"


POLISH_SYSTEM_PROMPT = """
你是一名资深技术营销文案顾问，擅长为检测认证（TIC）行业撰写解决方案和 PPT 文案。

现在给你一份已经按结构整理好的 Markdown 方案，请在遵守以下约束的前提下进行润色：
- 保持各级标题结构（#、##、### 等）不变，不要新增或删除章节；
- 可以调整段落和要点的表述，使其更加专业、流畅、有说服力；
- 可以适度补充过渡语或增强价值表述，但不要虚构具体企业机密或不合理的夸大承诺；
- 输出必须仍然是 Markdown 文本。
"""


def polish_markdown(markdown: str, model: str | None = None) -> str:
    """Let LLM polish the generated Markdown while keeping structure."""
    client = _init_qwen_client()
    model_name = model or os.getenv("QWEN_MODEL_NAME", "qwen3-max")

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": POLISH_SYSTEM_PROMPT},
            {"role": "user", "content": markdown},
        ],
        extra_body={"enable_thinking": True},
    )
    raw = _message_content_to_text(resp.choices[0].message.content)
    return (raw or "").strip()


def main() -> None:
    st.set_page_config(page_title="TIC 方案生成 Agent", layout="wide")
    st.title("📄 TIC 方案生成 Agent")
    st.caption("输入一个方案题目，自动生成结构化 JSON → Markdown → LLM 润色后的 PPT 方案。")

    with st.sidebar:
        st.markdown("### 运行配置")
        default_model = os.getenv("QWEN_MODEL_NAME", "qwen3-max")
        model_name = st.text_input("Qwen 模型名称", value=default_model)
        show_json = st.checkbox("显示原始 JSON 结构", value=True)

    topic = st.text_input(
        "方案题目",
        value="连接器的可靠性与信号完整性——第三方全方位验证方案",
        help="可以替换为任意你需要的检测方案标题，例如“动力电池包可靠性第三方验证方案”。",
    )
    run_button = st.button("生成并润色方案", type="primary", disabled=not topic.strip())

    if run_button:
        try:
            with st.spinner("正在生成结构化方案（含联网案例检索）..."):
                solution = generate_solution(topic.strip(), model=model_name.strip() or None)
        except Exception as exc:  # noqa: BLE001
            st.error(f"生成方案时出错：{exc}")
            return

        if show_json:
            with st.expander("原始 JSON 方案结构", expanded=False):
                st.json(solution)

        # 转为 Markdown 初稿
        markdown_raw = _solution_to_markdown(solution)
        st.subheader("Markdown 方案（初稿）")
        st.code(markdown_raw, language="markdown")

        # 使用 LLM 润色 Markdown
        try:
            with st.spinner("LLM 正在润色方案文案..."):
                markdown_polished = polish_markdown(markdown_raw, model=model_name.strip() or None)
        except Exception as exc:  # noqa: BLE001
            st.error(f"润色方案时出错：{exc}")
            return

        st.subheader("润色后方案（可直接用于 PPT / 文档）")
        st.markdown(markdown_polished)


if __name__ == "__main__":
    main()
