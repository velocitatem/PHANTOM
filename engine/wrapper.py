import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .engine import Limbo, MarketEngine, PricingEngine
from .lib.render import DashboardRenderer
from .lib.coi import compute_coi_proxy
from .lib.wrappers import EconomicMetricsWrapper


class PHANTOM(gym.Env):
    """Gymnasium wrapper for the Limbo pricing-market simulation. Platform sets prices, market responds with demand."""
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self,
                 n_products: int = 10,
                 alpha: float = 0.3,
                 N: int = 100,
                 human_params: tuple = (50.0, 10.0),
                 agent_params: tuple = (45.0, 15.0),
                 noise_std: float = 1.0,
                 price_bounds: tuple = (10.0, 150.0),
                 lambda_coi: float = 0.1,
                 coi_window: int = 10,
                 render_mode: str = None):
        super().__init__()
        self.n_products = n_products
        self.price_bounds = price_bounds
        self.lambda_coi = lambda_coi
        self.coi_window = coi_window  # K steps for rolling COI calculation
        self.render_mode = render_mode
        self.alpha = alpha
        self.N = N
        self.human_params = human_params
        self.agent_params = agent_params

        self.market = MarketEngine(
            alpha=alpha, N=N,
            human_params=human_params, agent_params=agent_params, noise_std=noise_std
        )
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
        self._initial_episode_prices = None  # prices at episode start for COI calc

    def _get_obs(self) -> dict:
        demand_arr = np.array([self._demand.get(i, 0.0) for i in range(self.n_products)], dtype=np.float32)
        return {"demand": demand_arr, "prices": self._prices.astype(np.float32)}

    def _compute_coi_proxy(self):
        return compute_coi_proxy(
            self._price_history, self._demand_history, self._initial_episode_prices,
            self._prices, self.price_bounds, self.alpha, self.coi_window
        )

    def _compute_reward(self, prices: np.ndarray, demand: dict) -> float:
        revenue = np.sum(prices * np.array([demand.get(i, 0.0) for i in range(self.n_products)]))
        coi_penalty = self.lambda_coi * self._compute_coi_proxy()
        return float(revenue - coi_penalty)

    def _record_history(self):
        demand_arr = np.array([self._demand.get(i, 0.0) for i in range(self.n_products)])
        self._demand_history.append(demand_arr)
        self._price_history.append(self._prices.copy())
        self._revenue_history.append(np.sum(self._prices * demand_arr))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._prices = np.random.uniform(*self.price_bounds, size=self.n_products)
        self._initial_episode_prices = self._prices.copy()  # snapshot for COI calculation
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

        coi_proxy = self._compute_coi_proxy()
        reward = self._compute_reward(self._prices, self._demand)
        terminated = self._step_count >= 100

        info = {
            "step": self._step_count,
            "coi_proxy": coi_proxy,
            "coi_penalty": self.lambda_coi * coi_proxy,
            "raw_revenue": np.sum(self._prices * np.array([self._demand.get(i, 0.0) for i in range(self.n_products)])),
        }
        return self._get_obs(), reward, terminated, False, info

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
    import wandb
    from .lib import MetricsCallback

    class RandomPolicy:
        """Minimal SB3-compatible random policy for baseline testing."""
        def __init__(self, env):
            self.env = env
            self.num_timesteps = 0

        def learn(self, total_timesteps, callback=None):
            callback.model = self
            callback.num_timesteps = 0
            callback.locals = {}
            callback.on_training_start({}, {})

            obs, _ = self.env.reset()
            for step in range(total_timesteps):
                action = self.env.action_space.sample()
                obs, reward, term, trunc, info = self.env.step(action)
                self.num_timesteps = step + 1
                callback.num_timesteps = self.num_timesteps
                callback.locals = {"infos": [info]}
                callback.on_step()
                if term or trunc:
                    callback.on_rollout_end()
                    obs, _ = self.env.reset()
            return self

        def predict(self, obs, **kwargs):
            return self.env.action_space.sample(), None

    wandb.init(project="phantom-pricing", config={"policy": "random", "alpha": 0.3})
    env = EconomicMetricsWrapper(PHANTOM(n_products=15, alpha=0.3, render_mode=None))

    model = RandomPolicy(env)
    model.learn(total_timesteps=1000, callback=MetricsCallback())

    print(f"Episode revenue: {env.episode_revenue:.1f}")
    wandb.finish()
    env.close()
