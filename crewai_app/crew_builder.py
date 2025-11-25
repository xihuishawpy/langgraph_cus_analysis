from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml
from crewai import Agent, Crew, Process, Task
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput
from langchain_core.messages import HumanMessage

from crewai_app import PACKAGE_ROOT
from crewai_app.configuration import Configuration
from crewai_app.llm import create_tongyi_llm
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
from crewai_app.schemas import Reflection, SearchQueryList
from crewai_app.tools import KnowledgeBaseSearchTool, TavilyResearchTool


@dataclass
class CrewContext:
    topic: str
    messages: List[HumanMessage]
    current_date: str
    number_of_queries: int
    use_industry_template: bool


class ProSearchCrewBuilder:
    """将 backend 中的节点逻辑映射为 CrewAI Agents + Tasks。"""

    def __init__(self, configuration: Configuration, *, verbose: bool = True):
        self._configuration = configuration
        self._verbose = verbose
        with (PACKAGE_ROOT / "config" / "agents.yaml").open("r", encoding="utf-8") as f:
            self._agent_templates: Dict[str, dict] = yaml.safe_load(f)
        self._tavily_tool = TavilyResearchTool()
        self._kb_tool = (
            KnowledgeBaseSearchTool(configuration)
            if configuration.enable_knowledge_base_search
            else None
        )

    def build(self, topic: str, *, history: Optional[List[HumanMessage]] = None) -> Crew:
        messages = history or [HumanMessage(content=topic)]
        context = CrewContext(
            topic=topic,
            messages=messages,
            current_date=get_current_date(),
            number_of_queries=int(self._configuration.number_of_initial_queries),
            use_industry_template=self._should_use_industry_template(topic),
        )
        agents = self._create_agents(context)
        tasks = self._create_tasks(context, agents)
        crew = Crew(
            agents=list({agent.role: agent for agent in agents.values()}.values()),
            tasks=tasks,
            process=Process.sequential,
            verbose=self._verbose,
        )
        return crew

    def _create_agents(self, context: CrewContext) -> Dict[str, Agent]:
        substitutions = {"topic": context.topic}
        agents: Dict[str, Agent] = {}
        agents["query_strategist"] = Agent(
            **self._format_agent_config("query_strategist", substitutions),
            llm=create_tongyi_llm(self._configuration.query_generator_model, temperature=1.0),
            allow_delegation=False,
        )
        web_llm_name = self._configuration.answer_model
        agents["web_researcher"] = Agent(
            **self._format_agent_config("web_researcher", substitutions),
            llm=create_tongyi_llm(web_llm_name, temperature=0.3),
            tools=[self._tavily_tool],
            allow_delegation=False,
        )
        if self._kb_tool:
            agents["knowledge_analyst"] = Agent(
                **self._format_agent_config("knowledge_analyst", substitutions),
                llm=create_tongyi_llm(web_llm_name, temperature=0.35),
                tools=[self._kb_tool],
                allow_delegation=False,
            )
        agents["reflection_partner"] = Agent(
            **self._format_agent_config("reflection_partner", substitutions),
            llm=create_tongyi_llm(self._configuration.reflection_model, temperature=0.8),
            allow_delegation=False,
        )
        agents["reporting_analyst"] = Agent(
            **self._format_agent_config("reporting_analyst", substitutions),
            llm=create_tongyi_llm(self._configuration.answer_model, temperature=0.0),
            allow_delegation=False,
        )
        return agents

    def _create_tasks(self, context: CrewContext, agents: Dict[str, Agent]) -> List[Task]:
        query_prompt = query_writer_instructions.format(
            current_date=context.current_date,
            research_topic=context.topic,
            number_queries=context.number_of_queries,
        )
        query_task = Task(
            description=query_prompt,
            expected_output=(
                "JSON 对象，字段包含 rationale（文本）与 query（字符串数组，数量不超过 "
                f"{context.number_of_queries} 条）。"
            ),
            agent=agents["query_strategist"],
            output_pydantic=SearchQueryList,
        )

        web_prompt = web_searcher_instructions.format(
            current_date=context.current_date,
            research_topic=context.topic,
        )
        web_task = Task(
            description=web_prompt
            + "\n请针对 query_task 的每条查询至少调用一次 tavily_research_tool，并在摘要中引用 [S#]。",
            expected_output="结构化 Markdown，包含客户、需求信号、标准映射及来源清单。",
            agent=agents["web_researcher"],
            context=[query_task],
        )

        tasks: List[Task] = [query_task, web_task]

        if self._kb_tool:
            kb_prompt = kb_searcher_instructions.format(
                current_date=context.current_date,
                research_topic=context.topic,
                table_rows="（请调用 excel_knowledge_base_search 工具获取原始记录）",
            )
            kb_task = Task(
                description=kb_prompt
                + "\n务必调用 excel_knowledge_base_search 工具，并以 [K#] 标注文内引用。",
                expected_output="一段带 [K#] 引用的摘要，突出企业/材料/产能/认证等线索。",
                agent=agents["knowledge_analyst"],
                context=[query_task],
            )
            tasks.append(kb_task)
        else:
            kb_task = None

        reflection_prompt = reflection_instructions.format(
            current_date=context.current_date,
            research_topic=context.topic,
            summaries="(内容会自动来自前置 Task Context)",
        )
        reflection_task = Task(
            description=reflection_prompt,
            expected_output="JSON，包含 is_sufficient、knowledge_gap、follow_up_queries。",
            agent=agents["reflection_partner"],
            context=[web_task] + ([kb_task] if kb_task else []),
            output_pydantic=Reflection,
        )
        tasks.append(reflection_task)

        followup_task = None
        if int(self._configuration.max_research_loops) > 1:
            followup_task = ConditionalTask(
                description=(
                    "若 reflection_task 指出知识缺口且提供 follow_up_queries，"
                    "请对每条查询调用 tavily_research_tool 并补充新的洞察。否则跳过本任务。"
                ),
                expected_output="若被执行，则输出补充的网页洞察与引用列表；若跳过无需输出。",
                condition=_needs_follow_up,
                agent=agents["web_researcher"],
                context=[reflection_task],
            )
            tasks.append(followup_task)

        final_template = industry_report_instructions if context.use_industry_template else answer_instructions
        final_prompt = final_template.format(
            current_date=context.current_date,
            research_topic=context.topic,
            summaries="(内容来自全部调研 Task)",
        )
        final_context = [web_task, reflection_task]
        if kb_task:
            final_context.insert(1, kb_task)
        if followup_task:
            final_context.append(followup_task)
        final_task = Task(
            description=final_prompt
            + "\n所有事实必须引用 Markdown 链接，并给出下一步建议。",
            expected_output="结构化 Markdown，含洞察、引用、下一步建议。",
            agent=agents["reporting_analyst"],
            context=final_context,
        )
        tasks.append(final_task)

        return tasks

    def _format_agent_config(self, key: str, substitutions: Dict[str, str]) -> dict:
        template = self._agent_templates[key]
        return {
            "role": template["role"].format(**substitutions),
            "goal": template["goal"].format(**substitutions),
            "backstory": template["backstory"].format(**substitutions),
            "verbose": True,
        }

    def _should_use_industry_template(self, topic: str) -> bool:
        if not self._configuration.enable_industry_report_mode:
            return False
        if not self._configuration.industry_report_auto_detect:
            return True
        return is_broad_topic(topic)


def _needs_follow_up(output: TaskOutput) -> bool:
    data = getattr(output, "pydantic", None)
    if data is None:
        try:
            data = Reflection.model_validate(json.loads(output.raw))
        except Exception:
            return False
    if data.is_sufficient:
        return False
    return bool(data.follow_up_queries)
