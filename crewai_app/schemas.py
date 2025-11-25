from typing import List
from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="用于网络调研的搜索查询列表。"
    )
    rationale: str = Field(
        description="简要说明这些查询与研究主题的关联性。"
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="给定摘要是否足以回答用户问题。"
    )
    knowledge_gap: str = Field(
        description="尚缺失或需要澄清的信息描述。"
    )
    follow_up_queries: List[str] = Field(
        description="用于弥补知识缺口的后续查询列表。"
    )
