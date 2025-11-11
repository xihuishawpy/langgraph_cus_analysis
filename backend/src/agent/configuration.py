import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """智能体的配置选项。"""

    query_generator_model: str = Field(
        default="qwen-plus",
        metadata={
            "description": "用于智能体生成搜索查询的通义千问模型名称。",
        },
    )

    reflection_model: str = Field(
        default="qwen-plus",
        metadata={
            "description": "用于智能体反思环节的通义千问模型名称。",
        },
    )

    answer_model: str = Field(
        default="qwen-plus",
        metadata={
            "description": "用于生成最终答复的通义千问模型名称。",
        },
    )

    llm_backend: str = Field(
        default="dashscope",
        metadata={
            "description": "LLM 后端：dashscope 或 local（禁用外部 LLM，走本地规则汇总）",
        },
    )

    enable_knowledge_base_search: bool = Field(
        default=True,
        metadata={
            "description": "是否启用本地知识库检索，默认关闭仅使用 Tavily 搜索",
        },
    )

    enable_tongyi_search_summary: bool = Field(
        default=False,
        metadata={
            "description": "是否在网络调研阶段调用通义千问生成摘要，默认关闭",
        },
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "需要生成的初始搜索查询数量。"},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "允许运行的研究循环最大次数。"},
    )

    knowledge_base_paths: str = Field(
        default="eastmoney_concept_constituents.xlsx,sw_third_industry_constituents.xlsx",
        metadata={
            "description": "逗号分隔的 Excel 路径列表，用于构建内部知识库。",
        },
    )

    knowledge_base_top_k: int = Field(
        default=3,
        metadata={
            "description": "每次查询内部知识库时返回的最大行数。",
        },
    )

    knowledge_base_embedding_model: str = Field(
        default="text-embedding-v3",
        metadata={
            "description": "用于构建 FAISS 知识库索引的 Qwen/DashScope 向量模型名称。",
        },
    )

    knowledge_base_embedding_backend: str = Field(
        default="dashscope",
        metadata={
            "description": "知识库向量后端，可选 dashscope 或 local（SentenceTransformer）。",
        },
    )

    knowledge_base_embedding_batch_size: int = Field(
        default=10,
        metadata={
            "description": "向量编码批大小；DashScope 限制最大 10，超出将被自动截断。",
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """从 RunnableConfig 创建 Configuration 实例。"""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # 优先从环境变量或传入配置中获取原始值
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # 过滤掉值为 None 的键
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
