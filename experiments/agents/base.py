from abc import ABC, abstractmethod
from typing import Optional

class Agent(ABC):
    """Base interface for browser automation agents"""

    def __init__(self, goal: str, url: str = "http://localhost:3000", timeout: int = 300):
        self.goal = goal
        self.url = url
        self.timeout = timeout
        self.result: Optional[str] = None

    @abstractmethod
    async def act(self) -> str:
        """Execute goal and return result text"""
        pass

    def final_result(self) -> Optional[str]:
        return self.result
