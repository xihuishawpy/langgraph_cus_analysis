from __future__ import annotations

import logging
import os
from typing import Any, List

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


class TongyiLLM(BaseLLM):
    """CrewAI BaseLLM 适配层，内部调用 LangChain ChatTongyi。"""

    def __init__(self, model: str, *, temperature: float = 0.2) -> None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set, cannot instantiate Tongyi models.")
        super().__init__(model=model, temperature=temperature, api_key=api_key, provider="dashscope")
        self._client = ChatTongyi(
            model=model,
            temperature=temperature,
            max_retries=2,
            api_key=api_key,
        )

    def call(  # type: ignore[override]
        self,
        messages: str | List[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ) -> str | Any:
        formatted_messages = self._format_messages(messages)
        self._emit_call_started_event(
            formatted_messages,
            tools=tools,
            callbacks=callbacks,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
        )

        if tools:
            logging.warning("TongyiLLM currently ignores function-calling tools; proceeding without tool execution.")

        langchain_messages = _convert_to_langchain_messages(formatted_messages)
        try:
            response = self._client.invoke(langchain_messages)
        except Exception as exc:
            self._emit_call_failed_event(str(exc), from_task=from_task, from_agent=from_agent)
            raise

        if isinstance(response, AIMessage):
            content = response.content
        elif isinstance(response, BaseMessage):
            content = response.content
        else:
            content = getattr(response, "content", str(response))

        content = self._apply_stop_words(str(content))
        parsed = self._validate_structured_output(content, response_model)
        self._emit_call_completed_event(
            response=content,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=formatted_messages,
        )
        return parsed


def _convert_to_langchain_messages(messages: list[dict[str, Any]]) -> list[BaseMessage]:
    result: list[BaseMessage] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
        else:
            result.append(HumanMessage(content=content))
    return result


def create_tongyi_llm(model_name: str, *, temperature: float = 0.2) -> TongyiLLM:
    return TongyiLLM(model=model_name, temperature=temperature)
