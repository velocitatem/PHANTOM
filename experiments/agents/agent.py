from os import environ
from base import Agent as BaseAgent
from browser_use import Browser, Agent, ChatOpenAI
from enum import Enum

class AgentTypes(str, Enum):
    GENERIC_BROWSER_USE_AGENT = "generic_browser_use_agent"

def _build_prompt(goal : str, environment_url : str) -> str:
    #TODO: Improve prompt engineering here and experiment with
    return f"""You are an autonomous agent tasked with achieving the following goal: {goal}
You have access to a web browser to interact with the environment at {environment_url}.
Use the browser to navigate, gather information, and perform actions necessary to accomplish your goal.
Be thorough and ensure you complete the task fully."""

class GenericBrowserUseAgent(BaseAgent):
    def __init__(self,
                 goal: str,
                 url: str = "http://localhost:3000",
                 timeout: int = 300,
                 llm_model: str = "gpt-5-mini",
                 headless: bool = True):
        super().__init__(goal, url, timeout)
        self.llm_model = ChatOpenAI(model=llm_model)
        self.browser = Browser(headless=headless)
        self.agent = Agent(task=_build_prompt(goal, url),
                           llm=self.llm_model,
                           browser=self.browser)
    async def act(self) -> str:
        self.result = await self.agent.run()
        return self.result.final_result()

def get_agent(agent_type: AgentTypes,  **kwargs) -> Agent:
    if agent_type == AgentTypes.GENERIC_BROWSER_USE_AGENT:
        return GenericBrowserUseAgent(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

if __name__ == "__main__":
    import asyncio
    JTBD= "Name of the company of this website"
    agent = get_agent(AgentTypes.GENERIC_BROWSER_USE_AGENT, goal=JTBD, url="https://ie.edu", timeout=300)
    R=asyncio.run(agent.act())
    print(R)
