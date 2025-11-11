"""Simple Streamlit front-end for the Pro Search agent."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from agent.graph import graph  # noqa: E402

st.set_page_config(page_title="Pro Search Agent 调试台", layout="wide")
st.title("🔍 Pro Search Agent 调试台")

REQUIRED_KEYS = ["DASHSCOPE_API_KEY", "TAVILY_API_KEY"]
missing_keys = [name for name in REQUIRED_KEYS if not os.getenv(name)]
if missing_keys:
    st.error(
        "缺少以下环境变量，请在终端或 .env 中配置后再运行：\n" + ", ".join(missing_keys)
    )
    st.stop()

with st.sidebar:
    st.header("运行配置")
    initial_queries = st.number_input("初始搜索查询数量", min_value=1, max_value=5, value=3)
    max_loops = st.slider("最大研究循环", min_value=1, max_value=5, value=2)
    use_kb_search = st.checkbox("启用内部知识库检索", value=False)
    kb_top_k = st.slider(
        "知识库返回条数",
        min_value=1,
        max_value=10,
        value=3,
        disabled=not use_kb_search,
    )
    query_model = st.text_input("查询生成模型", value="qwen-plus")
    reflection_model = st.text_input("反思模型", value="qwen-plus")
    answer_model = st.text_input("回答模型", value="qwen-plus")
    reasoning_model = st.text_input("推理模型 (可选)", value="")
    embedding_model = st.text_input(
        "知识库向量模型", value="text-embedding-v3", disabled=not use_kb_search
    )
    llm_backend = st.selectbox("LLM 后端", options=["dashscope", "local"], index=0)
    enable_tongyi_search_summary = st.checkbox(
        "使用通义千问生成搜索摘要", value=False, help="默认关闭，可在需要更详细综述时开启"
    )

if "runs" not in st.session_state:
    st.session_state["runs"] = []

st.write("输入调研问题，点击“开始调研”即可查看完整链路输出。")
user_query = st.text_area("研究问题", height=120, placeholder="例如：PCB 增长较好的企业分析")
run_button = st.button("开始调研", type="primary", disabled=not user_query.strip())

configurable_overrides: Dict[str, Any] = {
    "number_of_initial_queries": int(initial_queries),
    "max_research_loops": int(max_loops),
    "knowledge_base_top_k": int(kb_top_k),
    "query_generator_model": query_model.strip() or "qwen-plus",
    "reflection_model": reflection_model.strip() or "qwen-plus",
    "answer_model": answer_model.strip() or "qwen-plus",
    "knowledge_base_paths": os.getenv(
        "KNOWLEDGE_BASE_PATHS",
        "eastmoney_concept_constituents.xlsx,sw_third_industry_constituents.xlsx",
    ),
    "knowledge_base_embedding_model": embedding_model.strip() or "text-embedding-v3",
    "llm_backend": llm_backend,
    "enable_knowledge_base_search": bool(use_kb_search),
    "enable_tongyi_search_summary": bool(enable_tongyi_search_summary),
}

if run_button:
    st.session_state["runs"].insert(0, {"query": user_query.strip(), "status": "running"})
    try:
        with st.spinner("智能体正在执行..."):
            state: Dict[str, Any] = {
                "messages": [HumanMessage(content=user_query.strip())],
            }
            if reasoning_model.strip():
                state["reasoning_model"] = reasoning_model.strip()
            result = graph.invoke(state, config={"configurable": configurable_overrides})
    except Exception as exc:  # noqa: BLE001
        st.session_state["runs"][0] = {
            "query": user_query.strip(),
            "status": "error",
            "error": str(exc),
        }
    else:
        st.session_state["runs"][0] = {
            "query": user_query.strip(),
            "status": "success",
            "result": result,
        }

for idx, run in enumerate(st.session_state["runs"], start=1):
    with st.expander(f"运行 {idx}: {run['query'][:40]}" + ("..." if len(run["query"]) > 40 else ""), expanded=(idx == 1)):
        if run["status"] == "error":
            st.error(run.get("error", "未知错误"))
            continue
        if run["status"] == "running":
            st.info("任务执行中...")
            continue
        result = run["result"]
        messages: List[Any] = result.get("messages", [])
        answer = None
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                answer = message.content
                break
        if answer:
            st.subheader("最终回答")
            st.markdown(answer)
        summaries = result.get("web_research_result", [])
        if summaries:
            st.subheader("阶段性摘要")
            for i, summary in enumerate(summaries, start=1):
                st.markdown(f"**摘要 {i}:**\n{summary}")
        sources = result.get("sources_gathered", [])
        if sources:
            st.subheader("引用来源")
            for source in sources:
                label = source.get("label") or source.get("short_url")
                url = source.get("value") or source.get("short_url")
                st.markdown(f"- [{label}]({url})")
        st.caption("原始状态: " + repr({k: v for k, v in result.items() if k != "messages"}))
