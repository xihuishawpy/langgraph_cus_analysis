from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

import yaml
from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    query,
)

from crewai_app import PACKAGE_ROOT as CREW_PACKAGE_ROOT
from crewai_app.configuration import Configuration
from crewai_app.prompts import (
    answer_instructions,
    get_current_date,
    industry_report_instructions,
    kb_searcher_instructions,
    query_writer_instructions,
    reflection_instructions,
    web_searcher_instructions,
)
from crewai_app.query_patterns import is_broad_topic

from claude_sdk.tools import ToolingConfig, create_research_tooling


AGENT_TEMPLATE_PATH = CREW_PACKAGE_ROOT / "config" / "agents.yaml"


@dataclass
class ResearchContext:
    topic: str
    current_date: str
    number_of_queries: int
    use_industry_template: bool
    max_loops: int
    has_knowledge_base: bool


class ProSearchClaudeWorkflow:
    """Claude Agent SDK orchestration that mirrors the CrewAI research flow."""

    def __init__(self, configuration: Configuration, *, model: Optional[str] = None) -> None:
        self._configuration = configuration
        self._model = (model or os.getenv("CLAUDE_AGENT_MODEL") or "sonnet").strip()
        self._agent_templates = _load_agent_templates()
        self._tooling: ToolingConfig | None = None

    async def run(self, topic: str, *, verbose: bool = False) -> str:
        context = self._build_context(topic)
        tooling = self._ensure_tooling()
        agent_definitions = self._build_agent_definitions(context, tooling.tool_ids)
        options = self._build_agent_options(context, tooling, agent_definitions)
        user_prompt = self._build_user_prompt(context)
        return await self._collect_response(user_prompt, options, verbose=verbose)

    def _build_context(self, topic: str) -> ResearchContext:
        return ResearchContext(
            topic=topic,
            current_date=get_current_date(),
            number_of_queries=int(self._configuration.number_of_initial_queries),
            use_industry_template=self._should_use_industry_template(topic),
            max_loops=int(self._configuration.max_research_loops),
            has_knowledge_base=self._configuration.enable_knowledge_base_search,
        )

    def _should_use_industry_template(self, topic: str) -> bool:
        if not self._configuration.enable_industry_report_mode:
            return False
        if not self._configuration.industry_report_auto_detect:
            return True
        return is_broad_topic(topic)

    def _ensure_tooling(self) -> ToolingConfig:
        if self._tooling is None:
            self._tooling = create_research_tooling(self._configuration)
        return self._tooling

    def _build_agent_definitions(
        self,
        context: ResearchContext,
        tool_ids: Dict[str, str],
    ) -> Dict[str, AgentDefinition]:
        definitions: Dict[str, AgentDefinition] = {}
        sequence = [
            "query_strategist",
            "web_researcher",
            "knowledge_analyst",
            "reflection_partner",
            "reporting_analyst",
        ]
        for name in sequence:
            if name == "knowledge_analyst" and (
                not context.has_knowledge_base or "knowledge_base" not in tool_ids
            ):
                continue
            template = self._agent_templates.get(name)
            if not template:
                continue
            instructions = self._agent_instruction(name, context)
            if not instructions:
                continue
            prompt = self._format_agent_prompt(template, instructions, context.topic)
            description = template["goal"].format(topic=context.topic).strip()
            tools = self._agent_tools(name, tool_ids)
            definitions[name] = AgentDefinition(
                description=description,
                prompt=prompt,
                tools=tools if tools else None,
                model=None,
            )
        return definitions

    def _agent_instruction(self, name: str, context: ResearchContext) -> str:
        if name == "query_strategist":
            return query_writer_instructions.format(
                current_date=context.current_date,
                research_topic=context.topic,
                number_queries=context.number_of_queries,
            )
        if name == "web_researcher":
            return (
                web_searcher_instructions.format(
                    current_date=context.current_date,
                    research_topic=context.topic,
                )
                + "\n务必针对查询调用 tavily_research_tool 并引用 [S#]。"
            )
        if name == "knowledge_analyst":
            return kb_searcher_instructions.format(
                current_date=context.current_date,
                research_topic=context.topic,
                table_rows="（请调用 excel_knowledge_base_search 获取原始记录）",
            )
        if name == "reflection_partner":
            return reflection_instructions.format(
                current_date=context.current_date,
                research_topic=context.topic,
                summaries="（请审阅先前阶段输出）",
            )
        if name == "reporting_analyst":
            template = (
                industry_report_instructions
                if context.use_industry_template
                else answer_instructions
            )
            return template.format(
                current_date=context.current_date,
                research_topic=context.topic,
                summaries="（综合全部调研摘要）",
            )
        return ""

    def _agent_tools(self, name: str, tool_ids: Dict[str, str]) -> List[str]:
        if name == "web_researcher":
            return [tool_ids["tavily"]] if "tavily" in tool_ids else []
        if name == "knowledge_analyst":
            kb_id = tool_ids.get("knowledge_base")
            return [kb_id] if kb_id else []
        return []

    def _format_agent_prompt(self, template: Dict[str, Any], instructions: str, topic: str) -> str:
        role = template["role"].format(topic=topic).strip()
        goal = template["goal"].format(topic=topic).strip()
        backstory = template["backstory"].format(topic=topic).strip()
        return (
            f"角色: {role}\n\n"
            f"目标:\n{goal}\n\n"
            f"背景信息:\n{backstory}\n\n"
            f"操作指南:\n{instructions.strip()}"
        )

    def _build_agent_options(
        self,
        context: ResearchContext,
        tooling: ToolingConfig,
        agents: Dict[str, AgentDefinition],
    ) -> ClaudeAgentOptions:
        system_prompt = self._build_system_prompt(context, "knowledge_base" in tooling.tool_ids)
        # Simplified configuration without agents/MCP servers to avoid CLI transport issues
        return ClaudeAgentOptions(
            system_prompt=system_prompt,
            # agents=agents,  # Commented out to avoid CLI transport
            # mcp_servers=tooling.mcp_servers,  # Commented out to avoid CLI transport
            # allowed_tools=tooling.allowed_tools,  # Commented out to avoid CLI transport
            model=self._model or None,
            max_turns=self._max_turns(context),
        )

    def _build_system_prompt(self, context: ResearchContext, has_kb: bool) -> str:
        steps = [
            "你是 TIC 行业研究的首席协调，已配置多个专业子智能体与外部工具。",
            f"当前日期: {context.current_date}",
            f"研究主题: {context.topic}",
            f"首轮至少生成 1 条、至多 {context.number_of_queries} 条互补的搜索查询。",
            "必须引用来自工具输出的证据，引用格式使用 [S#] (网页) 或 [K#] (Excel)。",
        ]
        steps.append(
            "流程:\n"
            "1) 启动 query_strategist 拆解搜索意图，输出 JSON（字段 rationale, query[]）。\n"
            "2) 将 query 列表交给 web_researcher，针对每条调用 tavily_research_tool 并整合客户/需求/标准。\n"
            + (
                "3) 调用 knowledge_analyst 使用 excel_knowledge_base_search，补充内部 [K#] 线索。\n"
                if has_kb
                else ""
            )
            + "3/4) 让 reflection_partner 评估是否充分，若提供 follow_up_queries，可在剩余回合中继续搜索。\n"
            "最后由 reporting_analyst 根据行业模板整合成结构化 Markdown，附引用及下一步建议。"
        )
        steps.append(
            "无论过程如何，最终答复必须由 reporting_analyst 完成，并强调可执行建议。"
        )
        steps.append(
            f"允许的总循环 ≤ {context.max_loops}，请自我管理对话回合并避免冗余输出。"
        )
        return "\n\n".join(step for step in steps if step)

    def _build_user_prompt(self, context: ResearchContext) -> str:
        return (
            f"请围绕“{context.topic}”运行完整的研究流程，并返回结构化 Markdown 报告。"
        )

    def _max_turns(self, context: ResearchContext) -> int:
        base = max(6, context.number_of_queries + 4)
        return base + max(0, context.max_loops - 1) * 2

    async def _collect_response(
        self,
        prompt: str,
        options: ClaudeAgentOptions,
        *,
        verbose: bool,
    ) -> str:
        chunks: List[str] = []
        final_result: Optional[str] = None
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
                        if verbose:
                            print(block.text, end="", flush=True)
                    elif isinstance(block, ToolUseBlock) and verbose:
                        print(f"\n[tool] {block.name}: {block.input}")
                    elif isinstance(block, ToolResultBlock) and verbose:
                        print(f"\n[result] {block.content}")
            elif isinstance(message, ResultMessage):
                final_result = message.result or final_result
        if verbose:
            print()
        if final_result:
            return final_result.strip()
        return "".join(chunks).strip()


@lru_cache(maxsize=1)
def _load_agent_templates() -> Dict[str, Dict[str, Any]]:
    with AGENT_TEMPLATE_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
