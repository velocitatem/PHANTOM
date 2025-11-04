import asyncio

class Agent:
    def __init__(self,
                 goal : str = "Get Information on All Prices",
                 environment_url : str = "https://www.example.com",
                 timeout : int = 60 * 5):
        self.goal = goal
        self.environment_url = environment_url
        self.timeout = timeout
        self.result = None
        # TODO: implement agent initialization
        pass
    async def act(self):
        # set the self.result to whatever text result the agents returns
        pass # return await _async_method()

    def final_result(self) -> str|None:
        return self.result


# asyncio.run(Agent(...).act())
if __name__ == "__main__":
    print("Testing Agent...")
    agent = Agent(goal="Find the best price for a laptop", environment_url="https://www.example.com")
    asyncio.run(agent.act())
    print(f"Agent Result: {agent.final_result()}")
