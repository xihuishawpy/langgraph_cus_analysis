import os
from pathlib import Path
from typing import Dict, List

import dotenv
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from tavily import TavilyClient

from agent.configuration import Configuration
from agent.prompts import (
    answer_instructions,
    get_current_date,
    kb_searcher_instructions,
    query_writer_instructions,
    reflection_instructions,
    web_searcher_instructions,
)
from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.tools_and_schemas import Reflection, SearchQueryList
from agent.knowledge_base import ExcelKnowledgeBase
from agent.utils import (
    get_research_topic,
    replace_citation_tokens,
    resolve_urls,
)
from langchain_community.chat_models import ChatTongyi

dotenv.load_dotenv()

if os.getenv("DASHSCOPE_API_KEY") is None:
    raise ValueError("DASHSCOPE_API_KEY is not set")

if os.getenv("TAVILY_API_KEY") is None:
    raise ValueError("TAVILY_API_KEY is not set")

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
PROJECT_ROOT = Path(__file__).resolve().parents[3]
KB_CACHE: Dict[str, ExcelKnowledgeBase] = {}


# LangGraph 节点定义

def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """根据用户问题生成搜索查询的 LangGraph 节点。

    通过通义千问模型基于用户问题生成适合网络调研的优化查询列表。

    参数:
        state: 当前图状态，包含用户的问题
        config: 可运行配置，含大模型供应商等设置

    返回:
        含状态更新的数据字典，search_query 键保存生成的查询列表
    """
    configurable = Configuration.from_runnable_config(config)

    # 检查是否提供了自定义的初始查询数量
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # 若禁用外部 LLM，则直接用用户问题作为单条查询
    if (configurable.llm_backend or "dashscope").lower() != "dashscope":
        return {"search_query": [get_research_topic(state["messages"]) ]}

    # 初始化通义千问模型
    llm = ChatTongyi(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # 格式化提示词
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # 触发生成搜索查询
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """将搜索查询派发至网络调研节点的 LangGraph 节点。

    根据查询条数启动对应数量的网络调研节点，每条查询生成一个分支。
    """
    return [
        Send(
            "web_research",
            {"search_query": search_query, "id": str(idx), "query_text": search_query},
        )
        for idx, search_query in enumerate(state["search_query"])
    ]


def _format_tavily_sources(results: List[dict], run_id: str) -> tuple[str, List[dict]]:
    """将 Tavily 检索结果格式化为提示文本与引用信息。"""
    urls = [item["url"] for item in results]
    resolved_urls = resolve_urls(urls, run_id)

    formatted_blocks: List[str] = []
    sources: List[dict] = []
    for idx, item in enumerate(results, start=1):
        source_id = f"S{idx}"
        title = item.get("title") or item["url"]
        url = item["url"]
        short_url = resolved_urls[url]
        snippet = item.get("content") or ""
        published = item.get("published_date")
        block_lines = [
            f"[{source_id}] {title}",
            f"URL: {short_url}",
        ]
        if published:
            block_lines.append(f"时间: {published}")
        if snippet:
            block_lines.append(f"摘要: {snippet}")
        formatted_blocks.append("\n".join(block_lines))
        sources.append(
            {
                "id": source_id,
                "title": title,
                "short_url": short_url,
                "url": url,
            }
        )
    return "\n\n".join(formatted_blocks), sources


def _resolve_kb_paths(path_string: str) -> List[Path]:
    paths: List[Path] = []
    for raw in path_string.split(","):
        candidate = raw.strip()
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = PROJECT_ROOT / candidate
        paths.append(path)
    return paths


def _get_excel_kb(configurable: Configuration) -> ExcelKnowledgeBase:
    key = (
        f"{configurable.knowledge_base_paths}|"
        f"{configurable.knowledge_base_embedding_model}|"
        f"{configurable.knowledge_base_embedding_backend.lower()}"
    )
    if key not in KB_CACHE:
        paths = _resolve_kb_paths(configurable.knowledge_base_paths)
        KB_CACHE[key] = ExcelKnowledgeBase(
            paths,
            embedding_model=configurable.knowledge_base_embedding_model,
            embedding_backend=configurable.knowledge_base_embedding_backend,
            embedding_batch_size=int(
                getattr(configurable, "knowledge_base_embedding_batch_size", 10)
            ),
        )
    return KB_CACHE[key]


def _format_kb_sources(records) -> tuple[str, List[dict]]:
    formatted_blocks: List[str] = []
    sources: List[dict] = []
    for idx, record in enumerate(records, start=1):
        source_id = f"K{idx}"
        location = f"{record.source} 行 {record.row_index}"
        block_lines = [
            f"[{source_id}] 数据源: {record.source}",
            f"行号: {record.row_index}",
            f"内容: {record.text}",
        ]
        formatted_blocks.append("\n".join(block_lines))
        short_url = f"kb://{record.source}#R{record.row_index}"
        sources.append(
            {
                "id": source_id,
                "title": location,
                "short_url": short_url,
                "url": short_url,
            }
        )
    return "\n\n".join(formatted_blocks), sources


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """使用 Tavily API 执行网络调研的 LangGraph 节点。

    结合 Tavily 检索结果与通义千问模型生成带引用的结构化综述。

    参数:
        state: 当前图状态，包含搜索查询与研究循环计数
        config: 可运行配置，包含搜索 API 相关设置

    返回:
        状态更新字典，包含 sources_gathered、research_loop_count 与 web_research_result
    """
    configurable = Configuration.from_runnable_config(config)
    is_dashscope_backend = (configurable.llm_backend or "dashscope").lower() == "dashscope"
    use_tongyi_summary = (
        is_dashscope_backend and configurable.enable_tongyi_search_summary
    )
    query_text = state.get("query_text") or state["search_query"]
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=query_text,
    )

    search_response = tavily_client.search(
        query=query_text,
        search_depth="advanced",
        max_results=8,
        include_answer=False,
        include_raw_content=False,
    )
    results = search_response.get("results", [])

    if not results:
        return {
            "sources_gathered": [],
            "search_query": [query_text],
            "web_research_result": [
                "未能从可靠来源检索到与该查询相关的公开信息。"
            ],
            "query_texts": [query_text],
        }

    source_section, sources = _format_tavily_sources(results, str(state["id"]))
    prompt = (
        f"{formatted_prompt}\n\n"
        "以下是 Tavily 搜索得到的候选资料。撰写综述时请仅引用这些资料，"
        "并在使用某条信息时保留对应的引用标记（例如 [S1]）：\n\n"
        f"{source_section}\n\n"
        "输出直接为综述正文。"
    )

    # 当禁用 LLM 或未开启通义千问搜索摘要时，直接整理要点
    if not use_tongyi_summary:
        bullet_lines = []
        for s in sources:
            bullet_lines.append(f"- {s['title']} ({s['short_url']})")
        raw_text = "\n".join(bullet_lines) or "未能从搜索结果中提取到信息。"
        modified_text, sources_gathered = replace_citation_tokens(raw_text, sources)
    else:
        summarization_model = state.get("reasoning_model", configurable.answer_model)
        llm = ChatTongyi(
            model=summarization_model,
            temperature=0.2,
            max_retries=2,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        response = llm.invoke(prompt)
        if isinstance(response, AIMessage):
            raw_text = response.content
        else:
            raw_text = getattr(response, "content", str(response))

        modified_text, sources_gathered = replace_citation_tokens(raw_text, sources)

    return {
        "sources_gathered": sources_gathered,
        "search_query": [query_text],
        "web_research_result": [modified_text],
        "query_texts": [query_text],
    }


def knowledge_base_research(state: OverallState, config: RunnableConfig) -> OverallState:
    """查询本地 Excel 知识库并生成带引用的摘要。"""

    configurable = Configuration.from_runnable_config(config)
    if not configurable.enable_knowledge_base_search:
        return {}

    texts = state.get("query_texts") or []
    query_text = texts[-1] if texts else None
    kb = _get_excel_kb(configurable)

    if not query_text:
        return {
            "web_research_result": ["【内部知识库】未接收到有效的查询上下文，已跳过。"],
            # 不变更 query_texts
        }

    if kb.is_empty:
        return {
            "web_research_result": ["【内部知识库】知识库文件未找到或为空，无法检索。"],
            # 不变更 query_texts
        }

    records = kb.search(query_text, top_k=configurable.knowledge_base_top_k)
    if not records:
        return {
            "web_research_result": ["【内部知识库】未发现与当前查询相关的记录。"],
            # 不变更 query_texts
        }

    record_section, sources = _format_kb_sources(records)
    # 无 LLM 后端：直接把知识库记录转换为要点
    if (configurable.llm_backend or "dashscope").lower() != "dashscope":
        bullet = []
        for s in sources:
            bullet.append(f"- {s['title']}: {s['short_url']}")
        raw_text = "\n".join(bullet)
        modified_text, kb_sources = replace_citation_tokens(raw_text, sources)
    else:
        prompt = kb_searcher_instructions.format(
            current_date=get_current_date(),
            research_topic=query_text,
            table_rows=record_section,
        )

        summarization_model = state.get("reasoning_model") or configurable.answer_model
        llm = ChatTongyi(
            model=summarization_model,
            temperature=0.3,
            max_retries=2,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        response = llm.invoke(prompt)
        if isinstance(response, AIMessage):
            raw_text = response.content
        else:
            raw_text = getattr(response, "content", str(response))

        modified_text, kb_sources = replace_citation_tokens(raw_text, sources)
    summary = f"【内部知识库】{modified_text}"

    return {
        "web_research_result": [summary],
        "sources_gathered": kb_sources,
        # 不变更 query_texts
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """识别知识空白并生成后续查询的 LangGraph 节点。

    分析当前摘要以定位需要继续调研的领域，并生成可能的后续查询；通过结构化输出来提取 JSON 格式结果。

    参数:
        state: 当前图状态，包含摘要进度与研究主题
        config: 可运行配置，包含大模型供应商等设置

    返回:
        状态更新字典，search_query 键存储生成的后续查询
    """
    configurable = Configuration.from_runnable_config(config)
    # 增加研究循环计数并选取推理模型
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # 格式化提示词
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # 无 LLM 后端：直接认为信息充足，结束循环
    if (configurable.llm_backend or "dashscope").lower() != "dashscope":
        return {
            "is_sufficient": True,
            "knowledge_gap": "",
            "follow_up_queries": [],
            "research_loop_count": state["research_loop_count"],
            "number_of_ran_queries": len(state["search_query"]),
        }
    # 初始化推理模型
    llm = ChatTongyi(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """控制研究流程走向的 LangGraph 路由函数。

    根据配置的研究循环上限判断是继续收集信息还是进入最终总结环节。

    参数:
        state: 当前图状态，包含研究循环计数
        config: 可运行配置，含 max_research_loops 设置

    返回:
        下一步要访问的节点名称（"web_research" 或 "finalize_answer"）
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": str(state["number_of_ran_queries"] + int(idx)),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """生成最终研究结果的 LangGraph 节点。

    对来源去重并格式化后，与已有摘要整合，生成带有正确引用的结构化研究报告。

    参数:
        state: 当前图状态，包含摘要内容与收集到的来源

    返回:
        状态更新字典，其中 running_summary 键包含格式化后的最终摘要及引用
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # 格式化提示词
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # 无 LLM 后端：直接将阶段性摘要合并为最终回答
    if (configurable.llm_backend or "dashscope").lower() != "dashscope":
        class Dummy:
            content = formatted_prompt
        result = Dummy()
    else:
        # 初始化推理模型，默认为通义千问
        llm = ChatTongyi(
            model=reasoning_model,
            temperature=0,
            max_retries=2,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        result = llm.invoke(formatted_prompt)

    # 将短链接替换为原始链接，并记录最终引用的来源
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


################################################### 构建图#########################################
builder = StateGraph(OverallState, config_schema=Configuration)

# 定义核心节点
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("knowledge_base_research", knowledge_base_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# 将入口节点设为 `generate_query`（最先执行）
builder.add_edge(START, "generate_query")
# 添加条件边以便并行处理多条搜索查询
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# 在完成网络检索后进入反思节点
# 先运行内部知识库检索，再进入反思环节
builder.add_edge("web_research", "knowledge_base_research")
builder.add_edge("knowledge_base_research", "reflection")
# 评估是否继续研究
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# 生成最终答复
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
