from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from claude_agent_sdk import create_sdk_mcp_server, tool

from crewai_app.configuration import Configuration
from crewai_app.tools import KnowledgeBaseSearchTool, TavilyResearchTool


@dataclass
class ToolingConfig:
    """Bundle of MCP tooling configuration for the Claude Agent SDK."""

    server_name: str
    mcp_servers: Dict[str, Any]
    allowed_tools: List[str]
    tool_ids: Dict[str, str]


def create_research_tooling(configuration: Configuration) -> ToolingConfig:
    """Create MCP tooling (Tavily + optional Excel KB) for the Claude agent."""

    server_name = "tic_research"
    tavily_tool = TavilyResearchTool()
    tavily_description = getattr(
        tavily_tool,
        "description",
        "使用 Tavily 搜索 API 获取网页情报。",
    )
    kb_tool: Optional[KnowledgeBaseSearchTool] = None
    if configuration.enable_knowledge_base_search:
        kb_tool = KnowledgeBaseSearchTool(configuration)
    kb_description = (
        getattr(
            kb_tool,
            "description",
            "查询 Excel 构建的内部知识库。",
        )
        if kb_tool
        else ""
    )

    decorated_tools = []
    allowed_tools: List[str] = []
    tool_ids: Dict[str, str] = {}

    @tool("tavily_research_tool", tavily_description, {"query": str})
    async def tavily_executor(args: Dict[str, Any]) -> Dict[str, Any]:
        query_text = (args.get("query") or "").strip()
        if not query_text:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "tavily_research_tool: query 参数不能为空。",
                    }
                ],
                "is_error": True,
            }

        def _run() -> str:
            return tavily_tool._run(query_text)  # type: ignore[attr-defined]

        try:
            result = await asyncio.to_thread(_run)
        except Exception as exc:  # pragma: no cover - network errors
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"tavily_research_tool 执行失败: {exc}",
                    }
                ],
                "is_error": True,
            }

        return {"content": [{"type": "text", "text": result}]}

    decorated_tools.append(tavily_executor)
    tavily_id = _tool_id(server_name, "tavily_research_tool")
    allowed_tools.append(tavily_id)
    tool_ids["tavily"] = tavily_id

    if kb_tool is not None:

        @tool(
            "excel_knowledge_base_search",
            kb_description,
            {"query": str},
        )
        async def excel_kb_executor(args: Dict[str, Any]) -> Dict[str, Any]:
            query_text = (args.get("query") or "").strip()
            if not query_text:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "excel_knowledge_base_search: query 参数不能为空。",
                        }
                    ],
                    "is_error": True,
                }

            def _run() -> str:
                return kb_tool._run(query_text)  # type: ignore[attr-defined]

            try:
                result = await asyncio.to_thread(_run)
            except Exception as exc:  # pragma: no cover - kb exceptions
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"excel_knowledge_base_search 执行失败: {exc}",
                        }
                    ],
                    "is_error": True,
                }

            return {"content": [{"type": "text", "text": result}]}

        decorated_tools.append(excel_kb_executor)
        kb_id = _tool_id(server_name, "excel_knowledge_base_search")
        allowed_tools.append(kb_id)
        tool_ids["knowledge_base"] = kb_id

    server_config = create_sdk_mcp_server(
        name=server_name,
        version="0.1.0",
        tools=decorated_tools,
    )

    return ToolingConfig(
        server_name=server_name,
        mcp_servers={server_name: server_config},
        allowed_tools=allowed_tools,
        tool_ids=tool_ids,
    )


def _tool_id(server_name: str, tool_name: str) -> str:
    return f"mcp__{server_name}__{tool_name}"
