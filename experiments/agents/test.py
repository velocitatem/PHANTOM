import pytest
import asyncio
from experiments.agents.agent import get_agent, AgentTypes
import os


def test_agent_init():
    agent = get_agent(AgentTypes.GENERIC_BROWSER_USE_AGENT, goal="test", url="http://example.com", timeout=100)
    assert agent.goal == "test"
    assert agent.url == "http://example.com"
    assert agent.timeout == 100


def test_invalid_agent():
    with pytest.raises(ValueError):
        get_agent("invalid", goal="test")


@pytest.mark.asyncio
@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="OPENAI_API_KEY not set")
async def test_agent_execution():
    agent = get_agent(AgentTypes.GENERIC_BROWSER_USE_AGENT, goal="get page title", url="https://example.com", timeout=60)

    result = await agent.act()
    assert result
    assert agent.final_result()
    assert agent.final_result().history[-1].result[-1].is_done == True
    assert isinstance(result, str)
    assert "example" in result.lower()
    assert len(result) > 0
