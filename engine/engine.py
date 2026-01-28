from sys import platform
import numpy as np
from .lib.demand import generate_demand, estimate_demand
from .lib.behavior import sample_behavior
from logging import INFO, getLogger
logger = getLogger(__name__)
logger.setLevel(INFO)



class MarketEngine():
    def __init__(self,
                 alpha = 0.5,
                 N = 100,
                 demand_distribution = (50, 10),
                 demand_sampling_function = np.random.normal):
        self.Nagents = int(N*alpha)
        self.Nhumans = int(N*(1-alpha))
        self.demand = (demand_sampling_function, demand_distribution)

    def act(self, prices):
        demand = generate_demand(prices, *self.demand)
        sample_n = lambda n, human: [sample_behavior(demand, human=human) for _ in range(n)]
        human_t, agent_t = sample_n(100, True), sample_n(100, False)
        trajectories = human_t + agent_t
        demand_estimate = estimate_demand(trajectories)
        return demand_estimate

    def measure(self):
        pass

class PricingEngine():
    def __init__(self,
                 ) -> None:
        pass

    def act(self, demand):
        return np.random.uniform(low=25, high=100, size=10)



class Limbo():
    def __init__(self,
                 platform,
                 market
                 ) -> None:
        self.platform_turn = True
        self.platform = platform
        self.market = market
        self.output = None

    def step(self):
        # we could code golf this a little bit
        if self.platform_turn:
            self.output = self.platform.act(self.output)
        else:
            self.output = self.market.act(self.output)
        print(self.output)
        self.platform_turn = not self.platform_turn

if __name__ == "__main__":
    platform = PricingEngine()
    market = MarketEngine()
    limbo = Limbo(platform, market)
    for _ in range(10):
        limbo.step()
