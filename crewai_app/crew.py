from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from openai import OpenAI
import os, dotenv
dotenv.load_dotenv()


from crewai import LLM

llm = LLM(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.2,
)


@CrewBase
class TicOpportunityCrew:
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def tic_consultant(self) -> Agent:
        return Agent(
            config=self.agents_config['tic_consultant'],
            verbose=True,
            llm=llm,
        )

    @task
    def analyze_markdown(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_markdown'],
            agent=self.tic_consultant(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
