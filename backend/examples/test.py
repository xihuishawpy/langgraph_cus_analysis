
from agent.graph import graph
from agent.state import OverallState
from langchain_core.messages import HumanMessage

state = OverallState(
    messages=[HumanMessage(content="华测检测25年3季度财报")],
    search_query=[],
    web_research_result=[],
    sources_gathered=[],
    initial_search_query_count=3,
    max_research_loops=2,
    research_loop_count=0,
    reasoning_model="qwen-plus",
)

config = {"configurable": {}}

result = graph.invoke(state, config=config)
print(result)
