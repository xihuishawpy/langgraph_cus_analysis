from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from google.adk.planners import PlanReActPlanner
import os
from pydantic import BaseModel, Field

class TimeOutput(BaseModel):
    city: str = Field(description="返回指定城市中的当前时间。")


def get_current_time(city: str) -> dict:
    """返回指定城市中的当前时间。"""
    return {"status": "success", "city": city, "time": "10:30 AM"}

root_agent = LlmAgent(
    model=LiteLlm(model="openai/qwen-plus",
                    api_key=os.environ['DASHSCOPE_API_KEY'],
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                    ), 
    name='root_agent',
    planner=PlanReActPlanner(),
    description="Tells the current time in a specified city.给定一个城市名称，只能以 JSON 对象形式回复当前时间。格式：{'city': 'time'}",
    instruction="You are a helpful assistant that tells the current time in cities. Use the 'get_current_time' tool for this purpose.",
    tools=[get_current_time],
    output_schema=TimeOutput,

)