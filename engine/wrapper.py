import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .engine import Limbo, MarketEngine, PricingEngine
from .lib.render import DashboardRenderer
from .lib.coi import (
    compute_coi_leakage,
    compute_erosion_metrics,
    compute_agent_probability,
)
from .lib.behavior import get_transition_models, trajectory_to_events
from .lib.wrappers import EconomicMetricsWrapper


class PHANTOM(gym.Env):
    """Gymnasium wrapper for Limbo pricing-market simulation implementing thesis COI framework

    reward = R(p,d) - λ·COI_leak(p,τ') per thesis Section on DR-RL
    COI_leak uses behavioral divergence to estimate agent probability f(τ')
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        n_products: int = 10,
        alpha: float = 0.3,
        N: int = 100,
        human_params: tuple = (50.0, 10.0),
        agent_params: tuple = (45.0, 15.0),
        noise_std: float = 1.0,
        price_bounds: tuple = (10.0, 150.0),
        lambda_coi: float = 0.1,
        coi_window: int = 10,
        render_mode: str = None,
    ):
        super().__init__()
        self.n_products = n_products
        self.price_bounds = price_bounds
        self.lambda_coi = lambda_coi
        self.coi_window = coi_window
        self.render_mode = render_mode
        self.alpha = alpha
        self.N = N
        self.human_params = human_params
        self.agent_params = agent_params

        self.market = MarketEngine(
            alpha=alpha,
            N=N,
            human_params=human_params,
            agent_params=agent_params,
            noise_std=noise_std,
        )
        self._platform_stub = PricingEngine()
        self._limbo = Limbo(self._platform_stub, self.market)

        self.action_space = spaces.Box(
            low=price_bounds[0],
            high=price_bounds[1],
            shape=(n_products,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "demand": spaces.Box(
                    low=0.0, high=100.0, shape=(n_products,), dtype=np.float32
                ),
                "prices": spaces.Box(
                    low=price_bounds[0],
                    high=price_bounds[1],
                    shape=(n_products,),
                    dtype=np.float32,
                ),
            }
        )

        self._prices = None
        self._demand = None
        self._step_count = 0
        self._demand_history = []
        self._price_history = []
        self._revenue_history = []
        self._renderer = None
        self._initial_episode_prices = None
        self._trajectories = []  # session trajectories for agent prob calculation

        # load behavioral models for agent probability estimation
        try:
            self._human_trans, self._agent_trans = get_transition_models()
        except Exception:
            # fallback if behavioral data unavailable
            self._human_trans, self._agent_trans = None, None

    def _get_obs(self) -> dict:
        demand_arr = np.array(
            [self._demand.get(i, 0.0) for i in range(self.n_products)], dtype=np.float32
        )
        return {"demand": demand_arr, "prices": self._prices.astype(np.float32)}

    def _compute_agent_prob(self) -> float:
        """estimate agent probability from accumulated trajectories using KL divergence"""
        if (
            not self._trajectories
            or self._human_trans is None
            or self._agent_trans is None
        ):
            return self.alpha  # fallback to contamination level

        # aggregate all trajectories from this episode
        all_events = []
        for traj in self._trajectories:
            all_events.extend(trajectory_to_events(traj))

        if len(all_events) < 2:
            return self.alpha

        return compute_agent_probability(
            all_events, self._human_trans, self._agent_trans
        )

    def _compute_reward(self, prices: np.ndarray, demand: dict) -> float:
        revenue = np.sum(
            prices * np.array([demand.get(i, 0.0) for i in range(self.n_products)])
        )

        # compute agent probability from behavioral trajectories
        agent_prob = self._compute_agent_prob()

        # COI leakage: minimal implementation per thesis
        coi_leakage = compute_coi_leakage(prices, agent_prob)
        coi_penalty = self.lambda_coi * coi_leakage

        return float(revenue - coi_penalty)

    def _record_history(self):
        demand_arr = np.array(
            [self._demand.get(i, 0.0) for i in range(self.n_products)]
        )
        self._demand_history.append(demand_arr)
        self._price_history.append(self._prices.copy())
        self._revenue_history.append(np.sum(self._prices * demand_arr))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._prices = np.random.uniform(*self.price_bounds, size=self.n_products)
        self._initial_episode_prices = self._prices.copy()
        self._demand = self.market.act(self._prices)
        self._step_count = 0
        self._demand_history, self._price_history, self._revenue_history = [], [], []
        self._trajectories = []
        self._record_history()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._prices = np.clip(action, *self.price_bounds)
        self._demand = self.market.act(self._prices)
        self._step_count += 1
        self._record_history()

        # capture trajectories generated by market for agent prob estimation
        if hasattr(self.market, "last_trajectories"):
            self._trajectories.extend(self.market.last_trajectories)

        agent_prob = self._compute_agent_prob()
        coi_leakage = compute_coi_leakage(self._prices, agent_prob)
        reward = self._compute_reward(self._prices, self._demand)
        terminated = self._step_count >= 100

        # legacy erosion metrics for comparison
        erosion = compute_erosion_metrics(
            self._price_history,
            self._demand_history,
            self._initial_episode_prices,
            self._prices,
            self.price_bounds,
            self.alpha,
            self.coi_window,
        )

        info = {
            "step": self._step_count,
            "agent_prob": agent_prob,
            "coi_leakage": coi_leakage,
            "coi_penalty": self.lambda_coi * coi_leakage,
            "erosion_metrics": erosion,
            "raw_revenue": np.sum(
                self._prices
                * np.array([self._demand.get(i, 0.0) for i in range(self.n_products)])
            ),
        }
        return self._get_obs(), reward, terminated, False, info

    def _compute_elasticity(self) -> np.ndarray:
        """point elasticity: e = (dQ/dP) * (P/Q) via finite differences, clipped to [-5, 5]"""
        if len(self._price_history) < 2:
            return np.zeros(self.n_products)
        p, q = np.array(self._price_history), np.array(self._demand_history)
        dp, dq = np.diff(p, axis=0), np.diff(q, axis=0)
        valid = np.abs(dp) > 0.5
        with np.errstate(divide="ignore", invalid="ignore"):
            elasticity = np.where(
                valid, (dq / dp) * (p[:-1] / np.maximum(q[:-1], 1.0)), 0.0
            )
            elasticity = np.nan_to_num(np.clip(elasticity, -5.0, 5.0), nan=0.0)
        return (
            np.mean(elasticity, axis=0)
            if len(elasticity) > 0
            else np.zeros(self.n_products)
        )

    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                self._renderer = DashboardRenderer()
            self._renderer.render(self)
        elif self.render_mode == "ansi":
            return (
                f"step={self._step_count}, prices={self._prices}, demand={self._demand}"
            )
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
