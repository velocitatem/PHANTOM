from sys import platform
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .lib.demand import generate_demand_for_actor, estimate_demand
from .lib.behavior import get_adjusted_transitions, sample_behavior_from_transitions
from logging import INFO, getLogger

logger = getLogger(__name__)
logger.setLevel(INFO)

# shared pool; reused across act() calls to avoid per-call thread-spawn overhead
_pool = ThreadPoolExecutor(max_workers=4)


class MarketEngine:
    """implements separate demand distributions for humans and agents per Section 3.1.1"""

    def __init__(
        self,
        alpha: float,
        N: int,
        human_params: tuple,
        agent_params: tuple,
        demand_distribution=np.random.normal,
        noise_std: float = 1.0,
        action_weights: dict | None = None,
    ):
        # no defaults for D_H, D_A - force explicit experiment design
        self.alpha = alpha
        self.N = int(N)
        self.Nagents = int(N * alpha)
        self.Nhumans = int(N * (1 - alpha))
        self.human_params = human_params
        self.agent_params = agent_params
        self.noise_std = noise_std
        self.demand_dist = demand_distribution
        self.action_weights = action_weights

    def act(self, prices):
        # generate separate demands d() per actor type
        demand_h = generate_demand_for_actor(
            prices,
            self.human_params,
            self.noise_std,
            distribution_method=self.demand_dist,
        )
        demand_a = generate_demand_for_actor(
            prices,
            self.agent_params,
            self.noise_std,
            distribution_method=self.demand_dist,
        )
        human_transitions = get_adjusted_transitions(demand_h, human=True)
        agent_transitions = get_adjusted_transitions(demand_a, human=False)
        # sample N trajectories in parallel; each chain is independent so threads
        # do not share state and numpy's per-call RNG is thread-safe
        h_futs = [
            _pool.submit(sample_behavior_from_transitions, human_transitions)
            for _ in range(self.Nhumans)
        ]
        a_futs = [
            _pool.submit(sample_behavior_from_transitions, agent_transitions)
            for _ in range(self.Nagents)
        ]
        human_t = [f.result() for f in h_futs]
        agent_t = [f.result() for f in a_futs]
        # store trajectories for agent probability calculation
        self.last_trajectories = human_t + agent_t
        return estimate_demand(self.last_trajectories, self.action_weights)

    def measure(self):
        pass


class PricingEngine:
    def __init__(
        self,
    ) -> None:
        pass

    def act(self, demand):
        return np.random.uniform(low=25, high=100, size=10)


class Limbo:
    def __init__(self, platform, market) -> None:
        self.platform_turn = True
        self.platform = platform
        self.market = market
        self.output = None

    def step(self):
        if self.platform_turn:
            self.output = self.platform.act(self.output)
        else:
            self.output = self.market.act(self.output)
        self.platform_turn = not self.platform_turn
        return self.output

    def reset(self):
        self.platform_turn = True
        self.output = None


if __name__ == "__main__":
    platform = PricingEngine()
    market = MarketEngine(
        alpha=0.3, N=100, human_params=(50, 10), agent_params=(45, 15)
    )
    limbo = Limbo(platform, market)
    for _ in range(10):
        limbo.step()
