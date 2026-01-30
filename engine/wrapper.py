import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .engine import Limbo, MarketEngine, PricingEngine
from .lib.render import DashboardRenderer


class PHANTOM(gym.Env):
    """Gymnasium wrapper for the Limbo pricing-market simulation. Platform sets prices, market responds with demand."""
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self,
                 n_products: int = 10,
                 alpha: float = 0.3,
                 N: int = 100,
                 price_bounds: tuple = (10.0, 150.0),
                 lambda_coi: float = 0.1,
                 render_mode: str = None):
        super().__init__()
        self.n_products = n_products
        self.price_bounds = price_bounds
        self.lambda_coi = lambda_coi
        self.render_mode = render_mode
        self.alpha = alpha
        self.N = N

        self.market = MarketEngine(alpha=alpha, N=N)
        self._platform_stub = PricingEngine()
        self._limbo = Limbo(self._platform_stub, self.market)

        self.action_space = spaces.Box(
            low=price_bounds[0], high=price_bounds[1],
            shape=(n_products,), dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            "demand": spaces.Box(low=0.0, high=100.0, shape=(n_products,), dtype=np.float32),
            "prices": spaces.Box(low=price_bounds[0], high=price_bounds[1], shape=(n_products,), dtype=np.float32),
        })

        self._prices = None
        self._demand = None
        self._step_count = 0
        self._demand_history = []
        self._price_history = []
        self._revenue_history = []
        self._renderer = None

    def _get_obs(self) -> dict:
        demand_arr = np.array([self._demand.get(i, 0.0) for i in range(self.n_products)], dtype=np.float32)
        return {"demand": demand_arr, "prices": self._prices.astype(np.float32)}

    def _compute_reward(self, prices: np.ndarray, demand: dict) -> float:
        revenue = np.sum(prices * np.array([demand.get(i, 0.0) for i in range(self.n_products)]))
        # TODO: implement supra-competitive price punishment
        return float(revenue)

    def _record_history(self):
        demand_arr = np.array([self._demand.get(i, 0.0) for i in range(self.n_products)])
        self._demand_history.append(demand_arr)
        self._price_history.append(self._prices.copy())
        self._revenue_history.append(np.sum(self._prices * demand_arr))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._prices = np.random.uniform(*self.price_bounds, size=self.n_products)
        self._demand = self.market.act(self._prices)
        self._step_count = 0
        self._demand_history, self._price_history, self._revenue_history = [], [], []
        self._record_history()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._prices = np.clip(action, *self.price_bounds)
        self._demand = self.market.act(self._prices)
        self._step_count += 1
        self._record_history()

        reward = self._compute_reward(self._prices, self._demand)
        terminated = self._step_count >= 100

        return self._get_obs(), reward, terminated, False, {"step": self._step_count}

    def _compute_elasticity(self) -> np.ndarray:
        """point elasticity: e = (dQ/dP) * (P/Q) via finite differences, clipped to [-5, 5]"""
        if len(self._price_history) < 2:
            return np.zeros(self.n_products)
        p, q = np.array(self._price_history), np.array(self._demand_history)
        dp, dq = np.diff(p, axis=0), np.diff(q, axis=0)
        valid = np.abs(dp) > 0.5
        with np.errstate(divide='ignore', invalid='ignore'):
            elasticity = np.where(valid, (dq / dp) * (p[:-1] / np.maximum(q[:-1], 1.0)), 0.0)
            elasticity = np.nan_to_num(np.clip(elasticity, -5.0, 5.0), nan=0.0)
        return np.mean(elasticity, axis=0) if len(elasticity) > 0 else np.zeros(self.n_products)

    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                self._renderer = DashboardRenderer()
            self._renderer.render(self)
        elif self.render_mode == "ansi":
            return f"step={self._step_count}, prices={self._prices}, demand={self._demand}"
        return None

    def close(self):
        if self._renderer:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    env = PHANTOM(n_products=15, alpha=0.3, N=100, render_mode="human")
    obs, _ = env.reset()
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        env.render()
        if term: break
    env.close()
