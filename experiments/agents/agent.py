from base import Agent


class GenericComputerUseAgent(Agent):
    # TODO: implement code using computer-use library
    """
    async def do_job(jtbd):
    browser = Browser(headless=True)
    llm = ChatOpenAI(model="gpt-5-mini")
    task = "You are an agent for doing anything on http://localhost:3000 (wait for the website to load) for the user. The user wants you to do the following job: "+jtbd + ". Use the browser to do it. Be careful to follow all instructions and complete the job fully. You have the power to purchase or to do anything."
    agent = Agent(task=task, llm=llm, browser=browser)
    return await agent.run()

if __name__ == "__main__":
    JTBD= "Book the best room"
    R=asyncio.run(do_job(JTBD))
    print(R.final_result())
    """
    pass




def get_agent():
    # construct
    pass
