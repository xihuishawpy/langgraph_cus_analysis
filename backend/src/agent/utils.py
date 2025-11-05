from typing import Dict, List, Tuple

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


def get_research_topic(messages: List[AnyMessage]) -> str:
    """从消息列表中提取研究主题。"""
    # 判断是否只有首轮消息，或需要拼接对话历史
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def resolve_urls(urls_to_resolve: List[str], run_id: str, prefix: str = "https://tav.link/") -> Dict[str, str]:
    """为提供的原始链接生成唯一且更短的映射，便于在提示和引用中使用。

    参数:
        urls_to_resolve: 需要生成短链接的原始 URL 列表。
        run_id: 当前搜索/节点的唯一标识，用于保证短链接唯一性。
        prefix: 生成短链接时使用的前缀。

    返回:
        dict: 原始 URL 到短链接的映射。
    """

    resolved_map: Dict[str, str] = {}
    for idx, url in enumerate(urls_to_resolve):
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{run_id}-{idx}"
    return resolved_map


def replace_citation_tokens(text: str, sources: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
    """将文本中的 [S#] 标记替换为 Markdown 链接，并返回已引用的来源列表。

    参数:
        text: 含有引用占位符的原始文本。
        sources: 来源信息列表，每项需要包含 id/title/short_url/url 键。

    返回:
        tuple: (替换后的文本, 被引用的来源信息列表)。
    """

    cited_sources: List[Dict[str, str]] = []
    updated_text = text
    for source in sources:
        token = f"[{source['id']}]"
        if token in updated_text:
            short_url = source.get("short_url") or source.get("url", "")
            markdown_link = f"[{source['title']}]({short_url})"
            updated_text = updated_text.replace(token, markdown_link)
            cited_sources.append(
                {
                    "label": source["title"],
                    "short_url": short_url,
                    "value": source["url"],
                }
            )
    return updated_text, cited_sources
