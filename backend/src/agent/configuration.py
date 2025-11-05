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

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "需要生成的初始搜索查询数量。"},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "允许运行的研究循环最大次数。"},
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
