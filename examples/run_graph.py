import os, sys
from pathlib import Path
sys.path.insert(0, str(Path('backend/src').resolve()))
from agent.graph import graph
from agent.state import OverallState
from langchain_core.messages import HumanMessage
state = OverallState(  # type: ignore
    messages=[HumanMessage(content='水冷板')],
    search_query=[],
    web_research_result=[],
    sources_gathered=[],
    initial_search_query_count=3,
    max_research_loops=1,
    research_loop_count=0,
    reasoning_model=''
)
config = {
    'configurable': {
        'llm_backend': 'local',
        'enable_knowledge_base_search': False,
        'number_of_initial_queries': 3,
        'max_research_loops': 1,
        'knowledge_base_top_k': 3,
    }
}
result = graph.invoke(state, config=config)
print(result)
