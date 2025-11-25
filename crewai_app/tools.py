from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, List

from crewai.tools import BaseTool
from tavily import TavilyClient

from crewai_app import REPO_ROOT
from crewai_app.configuration import Configuration
from crewai_app.knowledge_base import ExcelKnowledgeBase
from crewai_app.utils import replace_citation_tokens, resolve_urls


class TavilyResearchTool(BaseTool):
    name: str = "tavily_research_tool"
    description: str = (
        "使用 Tavily 搜索 API 获取与查询相关的最新网页情报，输出包含引用编号的结构化笔记。"
    )
    def __init__(self, *, max_results: int = 8):
        super().__init__()
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY is not set, cannot use web research tool.")
        self._client = TavilyClient(api_key=api_key)
        self._max_results = max_results

    def _run(self, query: str) -> str:
        response = self._client.search(
            query=query,
            search_depth="advanced",
            max_results=self._max_results,
            include_answer=False,
            include_raw_content=False,
        )
        results = response.get("results", [])
        if not results:
            return f"【Tavily】未能检索到与“{query}”相关的公开网页，请尝试调整关键词。"

        formatted, sources = _format_tavily_sources(results, run_id=uuid.uuid4().hex[:8])
        markdown, _ = replace_citation_tokens(formatted, sources)
        catalog_lines = [f"- {source['id']}: [{source['title']}]({source['short_url']})" for source in sources]
        catalog = "\n".join(catalog_lines)

        return (
            f"## Tavily: {query}\n\n{markdown}\n\n### 来源清单\n{catalog}"
        )


class KnowledgeBaseSearchTool(BaseTool):
    name: str = "excel_knowledge_base_search"
    description: str = (
        "查询 Excel 构建的内部知识库，返回命中的行及引用编号（K#）。"
    )
    def __init__(self, configuration: Configuration):
        super().__init__()
        self._configuration = configuration
        self._kb: ExcelKnowledgeBase | None = None

    def _ensure_kb(self) -> ExcelKnowledgeBase:
        if self._kb is None:
            paths = _resolve_kb_paths(self._configuration.knowledge_base_paths)
            self._kb = ExcelKnowledgeBase(
                paths,
                embedding_model=self._configuration.knowledge_base_embedding_model,
                embedding_backend=self._configuration.knowledge_base_embedding_backend,
                embedding_batch_size=int(self._configuration.knowledge_base_embedding_batch_size),
            )
        return self._kb

    def _run(self, query: str) -> str:
        if not self._configuration.enable_knowledge_base_search:
            return "【内部知识库】当前配置禁用了 Excel 语料查询。"

        kb = self._ensure_kb()
        if kb.is_empty:
            return "【内部知识库】尚未构建索引，请先运行 knowledge/init_kb.py。"

        records = kb.search(query, top_k=self._configuration.knowledge_base_top_k)
        if not records:
            return f"【内部知识库】未找到与“{query}”匹配的行。"
        formatted, _ = _format_kb_sources(records)
        return formatted


def _format_tavily_sources(results: List[dict], run_id: str) -> tuple[str, List[dict]]:
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
            path = REPO_ROOT / candidate
        paths.append(path)
    return paths


def _format_kb_sources(records):
    formatted_blocks: List[str] = []
    sources: List[dict] = []
    for idx, record in enumerate(records, start=1):
        source_id = f"K{idx}"
        location = f"{record.source} 第 {record.row_index} 行"
        block_lines = [
            f"[{source_id}] 来源: {record.source}",
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
