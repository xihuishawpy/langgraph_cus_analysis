"""Streamlit front-end with LangGraph and CrewAI entry points."""
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
from crewai_app.configuration import Configuration  # noqa: E402
from crewai_app.crew_builder import ProSearchCrewBuilder  # noqa: E402

st.set_page_config(page_title="Pro Search Agent", layout="wide")
st.title("🔍 Pro Search Agent")

REQUIRED_KEYS = ["DASHSCOPE_API_KEY", "TAVILY_API_KEY"]
missing_keys = [name for name in REQUIRED_KEYS if not os.getenv(name)]
if missing_keys:
    st.error(
        "缺少以下环境变量，请在终端或 .env 中配置后再运行：\n" + ", ".join(missing_keys)
    )
    st.stop()

if "langgraph_runs" not in st.session_state:
    st.session_state["langgraph_runs"] = []
if "crewai_runs" not in st.session_state:
    st.session_state["crewai_runs"] = []


def render_langgraph_results(runs: List[Dict[str, Any]]) -> None:
    for idx, run in enumerate(runs, start=1):
        header = f"运行 {idx}: {run['query'][:40]}" + ("..." if len(run["query"]) > 40 else "")
        with st.expander(header, expanded=(idx == 1)):
            status = run["status"]
            if status == "error":
                st.error(run.get("error", "未知错误"))
                continue
            if status == "running":
                st.info("任务执行中...")
                continue
            result = run["result"]
            messages: List[Any] = result.get("messages", [])
            answer = next((msg.content for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
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
            st.caption("原始状态 " + repr({k: v for k, v in result.items() if k != "messages"}))


def render_crewai_results(runs: List[Dict[str, Any]]) -> None:
    for idx, run in enumerate(runs, start=1):
        header = f"运行 {idx}: {run['query'][:40]}" + ("..." if len(run["query"]) > 40 else "")
        with st.expander(header, expanded=(idx == 1)):
            status = run["status"]
            if status == "error":
                st.error(run.get("error", "未知错误"))
                continue
            if status == "running":
                st.info("任务执行中...")
                continue
            st.subheader("CrewAI 输出")
            st.markdown(run["result_markdown"])


def run_langgraph_agent(query: str, overrides: Dict[str, Any], reasoning_model: str) -> Dict[str, Any]:
    state: Dict[str, Any] = {"messages": [HumanMessage(content=query)]}
    if reasoning_model.strip():
        state["reasoning_model"] = reasoning_model.strip()
    return graph.invoke(state, config={"configurable": overrides})


def run_crewai_agent(query: str, overrides: Dict[str, Any]) -> str:
    configuration = Configuration.from_runnable_config({"configurable": overrides})
    builder = ProSearchCrewBuilder(configuration, verbose=False)
    crew = builder.build(query)
    result = crew.kickoff(inputs={"topic": query})
    return getattr(result, "raw", str(result))


tab_lang, tab_crewai = st.tabs(["LangGraph 工作流", "CrewAI 工作流"])

with tab_lang:
    st.subheader("LangGraph 多节点智能体")
    cfg_col, run_col = st.columns([1, 2])
    with cfg_col:
        st.markdown("#### 运行配置")
        initial_queries = st.number_input("初始搜索查询数量", min_value=1, max_value=6, value=3, key="lg_initial_queries")
        max_loops = st.slider("最大研究循环次数", min_value=1, max_value=5, value=1, key="lg_max_loops")
        use_kb_search = st.checkbox("启用内部知识库检索", value=True, key="lg_use_kb")
        kb_top_k = st.slider(
            "知识库返回条数",
            min_value=1,
            max_value=30,
            value=10,
            disabled=not use_kb_search,
            key="lg_kb_topk",
        )
        query_model = st.text_input("查询生成模型", value="qwen-plus", key="lg_query_model")
        reflection_model = st.text_input("反思模型", value="qwen-plus", key="lg_reflection_model")
        answer_model = st.text_input("回答模型", value="qwen-plus", key="lg_answer_model")
        reasoning_model = st.text_input("推理模型 (可选)", value="", key="lg_reasoning_model")
        embedding_model = st.text_input(
            "知识库向量模型", value="text-embedding-v3", disabled=not use_kb_search, key="lg_embedding_model"
        )
        llm_backend = st.selectbox("LLM 后端", options=["dashscope", "local"], index=0, key="lg_llm_backend")
        enable_tongyi_search_summary = st.checkbox(
            "使用通义千问生成搜索摘要", value=False, help="默认关闭，可在需要更详细综述时开启", key="lg_tongyi_summary"
        )
    with run_col:
        st.markdown("#### 调研输入")
        user_query = st.text_area("研究问题", height=120, placeholder="例如：PCB 增长较好的企业分布？", key="lg_query")
        run_button = st.button("开始 LangGraph 调研", type="primary", key="lg_run_button", disabled=not user_query.strip())
    overrides: Dict[str, Any] = {
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
        st.session_state["langgraph_runs"].insert(0, {"query": user_query.strip(), "status": "running"})
        try:
            with st.spinner("LangGraph 智能体执行中..."):
                result = run_langgraph_agent(user_query.strip(), overrides, reasoning_model)
        except Exception as exc:  # noqa: BLE001
            st.session_state["langgraph_runs"][0] = {
                "query": user_query.strip(),
                "status": "error",
                "error": str(exc),
            }
        else:
            st.session_state["langgraph_runs"][0] = {
                "query": user_query.strip(),
                "status": "success",
                "result": result,
            }
    st.markdown("---")
    render_langgraph_results(st.session_state["langgraph_runs"])

with tab_crewai:
    st.subheader("CrewAI 多 Agent 工作流")
    cfg_col, run_col = st.columns([1, 2])
    with cfg_col:
        st.markdown("#### 运行配置")
        crew_initial_queries = st.number_input("初始搜索查询数量", min_value=1, max_value=8, value=1, key="crew_initial_queries")
        crew_max_loops = st.slider("最大迭代轮数", min_value=1, max_value=4, value=1, key="crew_max_loops")
        crew_enable_kb = st.checkbox("启用 Excel 知识库", value=True, key="crew_enable_kb")
        crew_kb_top_k = st.slider(
            "知识库返回条数",
            min_value=1,
            max_value=20,
            value=5,
            key="crew_kb_topk",
            disabled=not crew_enable_kb,
        )
        crew_disable_industry = st.checkbox("关闭行业报告模板", value=False, key="crew_disable_industry")
        crew_kb_paths = st.text_input(
            "知识库路径 (逗号分隔)",
            value=os.getenv(
                "KNOWLEDGE_BASE_PATHS",
                "eastmoney_concept_constituents.xlsx,sw_third_industry_constituents.xlsx",
            ),
            key="crew_kb_paths",
        )
        crew_query_model = st.text_input("查询生成模型", value="qwen-plus", key="crew_query_model")
        crew_reflection_model = st.text_input("反思模型", value="qwen-plus", key="crew_reflection_model")
        crew_answer_model = st.text_input("回答模型", value="qwen-plus", key="crew_answer_model")
    with run_col:
        st.markdown("#### 调研输入")
        crew_query = st.text_area("研究问题", height=120, placeholder="例如：半导体设备国产替代有哪些机会？", key="crew_query")
        crew_button = st.button("运行 CrewAI 工作流", type="primary", key="crew_run_button", disabled=not crew_query.strip())
    crew_overrides = {
        "number_of_initial_queries": int(crew_initial_queries),
        "max_research_loops": int(crew_max_loops),
        "enable_knowledge_base_search": bool(crew_enable_kb),
        "knowledge_base_top_k": int(crew_kb_top_k),
        "knowledge_base_paths": crew_kb_paths.strip(),
        "enable_industry_report_mode": not crew_disable_industry,
        "query_generator_model": crew_query_model.strip() or "qwen-plus",
        "reflection_model": crew_reflection_model.strip() or "qwen-plus",
        "answer_model": crew_answer_model.strip() or "qwen-plus",
    }
    if crew_button:
        st.session_state["crewai_runs"].insert(0, {"query": crew_query.strip(), "status": "running"})
        try:
            with st.spinner("CrewAI 智能体执行中..."):
                output = run_crewai_agent(crew_query.strip(), crew_overrides)
        except Exception as exc:  # noqa: BLE001
            st.session_state["crewai_runs"][0] = {
                "query": crew_query.strip(),
                "status": "error",
                "error": str(exc),
            }
        else:
            st.session_state["crewai_runs"][0] = {
                "query": crew_query.strip(),
                "status": "success",
                "result_markdown": output,
            }
    st.markdown("---")
    render_crewai_results(st.session_state["crewai_runs"])
