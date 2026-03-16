import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .engine import Limbo, MarketEngine, PricingEngine
from .lib.render import DashboardRenderer
from .lib.coi import (
    compute_uplift_coi,
    extract_purchases,
    compute_agent_probability,
)
from .lib.behavior import get_transition_models, trajectory_to_events
from .lib.wrappers import EconomicMetricsWrapper
from .jax.robust import select_adversarial_alpha_jax, _JAX_OK


class _ActionPricingEngine(PricingEngine):
    def __init__(self, n_products: int, price_bounds: tuple):
        self._prices = np.full(n_products, price_bounds[0], dtype=float)

    def set_prices(self, prices: np.ndarray):
        self._prices = np.asarray(prices, dtype=float)

    def act(self, _):
        return self._prices


class PHANTOM(gym.Env):
    """Gymnasium wrapper for Limbo pricing-market simulation implementing thesis COI framework

    reward = R(p,d) - λ·COI_leak(p,τ') per thesis Section on DR-RL
    COI_leak uses behavioral divergence to estimate agent probability f(τ')
    robust inner step: min over alpha in Wasserstein interval around nominal alpha
    actions are discrete global price-scale moves
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
        robust_radius: float = 0.0,
        robust_points: int = 5,
        robust_rollouts: int = 1,
        info_value: float = 1.0,
        eta_ux: float = 0.5,
        reward_profit_weight: float = 1.0,
        action_levels: int = 9,
        action_scale_low: float = 0.9,
        action_scale_high: float = 1.1,
        max_steps: int = 100,
        margin_floor: float = 0.05,
        margin_floor_patience: int = 5,
        render_mode: str = None,
    ):
        super().__init__()
        self.n_products = n_products
        self.price_bounds = price_bounds
        self.lambda_coi = lambda_coi
        self.coi_window = coi_window
        self.max_steps = max(1, int(max_steps))
        self.margin_floor = float(
            margin_floor
        )  # terminate if avg margin stays below this for patience steps
        self.margin_floor_patience = max(1, int(margin_floor_patience))
        self.render_mode = render_mode
        self.alpha = float(alpha)
        self.nominal_alpha = float(alpha)
        self.N = N
        self.human_params = human_params
        self.agent_params = agent_params
        self.robust_radius = max(0.0, float(robust_radius))
        self.robust_points = max(1, int(robust_points))
        self.robust_rollouts = max(1, int(robust_rollouts))
        self.info_value = float(info_value)
        self.eta_ux = float(eta_ux)
        self.reward_profit_weight = float(reward_profit_weight)
        self.action_levels = max(2, int(action_levels))
        self._action_scales = np.linspace(
            float(action_scale_low), float(action_scale_high), self.action_levels
        )

        self.market = MarketEngine(
            alpha=alpha,
            N=N,
            human_params=human_params,
            agent_params=agent_params,
            noise_std=noise_std,
        )
        self._platform_stub = _ActionPricingEngine(n_products, price_bounds)
        self._limbo = Limbo(self._platform_stub, self.market)
        self._set_market_mix(self.nominal_alpha)

        self.action_space = spaces.Discrete(self.action_levels)
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
                "signals": spaces.Box(
                    low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                    shape=(4,),
                    dtype=np.float32,
                ),
            }
        )

        self._prices = None
        self._demand = None
        self._step_count = 0
        self._global_step = 0  # monotonic; used as JAX RNG seed across resets
        self._demand_history = []
        self._price_history = []
        self._revenue_history = []
        self._renderer = None
        self._initial_episode_prices = None
        self._trajectories = []  # session trajectories for agent prob calculation
        self.baseline_prices = np.full(self.n_products, self.price_bounds[0])
        self._low_margin_streak = 0  # consecutive steps below margin_floor
        self._last_agent_prob = float(self.alpha)
        self._last_alpha_adv = float(self.alpha)

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
        signals = np.array(
            [
                float(np.clip(self._last_agent_prob, 0.0, 1.0)),
                float(np.clip(self._last_alpha_adv, 0.0, 1.0)),
                float(np.clip(self.nominal_alpha, 0.0, 1.0)),
                float(np.clip(self.robust_radius, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return {
            "demand": demand_arr,
            "prices": self._prices.astype(np.float32),
            "signals": signals,
        }

    def _set_market_mix(self, alpha: float):
        alpha = float(np.clip(alpha, 0.0, 1.0))
        n_agents = int(self.N * alpha)
        self.alpha = alpha
        self.market.alpha = alpha
        self.market.Nagents = n_agents
        self.market.Nhumans = self.N - n_agents

    def _decode_action(self, action) -> np.ndarray:
        base = (
            self._prices
            if self._prices is not None
            else np.full(self.n_products, self.price_bounds[0], dtype=float)
        )
        if np.isscalar(action):
            idx = int(np.clip(int(action), 0, self.action_levels - 1))
            return np.clip(base * self._action_scales[idx], *self.price_bounds)
        a = np.asarray(action)
        if a.size == 1:
            idx = int(np.clip(int(a.reshape(-1)[0]), 0, self.action_levels - 1))
            return np.clip(base * self._action_scales[idx], *self.price_bounds)
        return np.clip(a.astype(float), *self.price_bounds)

    def _compute_agent_prob(self, trajectories=None) -> float:
        trajectories = (
            self.market.last_trajectories if trajectories is None else trajectories
        )
        if not trajectories or self._human_trans is None or self._agent_trans is None:
            return float(self.market.alpha)
        probs = []
        for traj in trajectories:
            events = trajectory_to_events(traj)
            if len(events) < 2:
                continue
            probs.append(
                compute_agent_probability(events, self._human_trans, self._agent_trans)
            )
        return float(np.mean(probs)) if probs else float(self.market.alpha)

    def _compute_reward(
        self, prices: np.ndarray, demand: dict, agent_prob: float, trajectories: list
    ) -> tuple[float, dict]:
        demand_arr = np.array(
            [demand.get(i, 0.0) for i in range(self.n_products)], dtype=float
        )
        revenue = float(np.dot(prices, demand_arr))
        floor_cost = float(np.dot(self.baseline_prices, demand_arr))
        profit = revenue - floor_cost
        purchases = extract_purchases(trajectories)
        coi_mix = compute_uplift_coi(prices, purchases, self.baseline_prices)

        coi_leakage = float(agent_prob * self.info_value)
        info_budget = max(floor_cost, 1.0)
        coi_penalty = self.lambda_coi * coi_leakage * info_budget

        if len(self._price_history) > 0:
            prev_prices = np.asarray(self._price_history[-1], dtype=float)
            rel_change = (prices - prev_prices) / np.maximum(prev_prices, 1.0)
            volatility = float(np.mean(np.abs(rel_change)))
            upward_volatility = float(np.mean(np.clip(rel_change, 0.0, None)))
        else:
            volatility = 0.0
            upward_volatility = 0.0
        ux_penalty = self.eta_ux * info_budget * (volatility + 0.5 * upward_volatility)

        competitive_anchor = float(
            np.clip(float(self.human_params[0]) * 1.2, *self.price_bounds)
        )
        price_ratio = prices / max(competitive_anchor, 1.0)
        supra_excess = np.clip(price_ratio - 1.0, 0.0, None)
        supra_penalty = (
            0.5 * self.eta_ux * info_budget * float(np.mean(np.square(supra_excess)))
        )
        supra_share = float(np.mean(supra_excess > 0.0))

        reward_revenue = self.reward_profit_weight * profit
        reward = reward_revenue - coi_penalty - ux_penalty - supra_penalty

        return reward, {
            "revenue": revenue,
            "cost_floor": floor_cost,
            "profit": profit,
            "coi_mix": float(coi_mix),
            "coi_base": 0.0,
            "coi_leakage": coi_leakage,
            "coi_penalty": coi_penalty,
            "coi_info_budget": info_budget,
            "ux_penalty": ux_penalty,
            "volatility": volatility,
            "upward_volatility": upward_volatility,
            "supra_penalty": supra_penalty,
            "supra_share": supra_share,
            "competitive_anchor": competitive_anchor,
            "reward_revenue": reward_revenue,
            "reward_total": reward,
        }

    def _alpha_candidates(self) -> np.ndarray:
        if self.robust_radius <= 0.0 or self.robust_points == 1:
            return np.array([self.nominal_alpha], dtype=float)
        lo = max(0.0, self.nominal_alpha - self.robust_radius)
        hi = min(1.0, self.nominal_alpha + self.robust_radius)
        return np.linspace(lo, hi, self.robust_points)

    def _evaluate_candidate(self, alpha: float, prices: np.ndarray) -> float:
        self._set_market_mix(alpha)
        rewards = []
        for _ in range(self.robust_rollouts):
            demand = self.market.act(prices)
            trajectories = list(self.market.last_trajectories)
            agent_prob = self._compute_agent_prob(trajectories)
            reward, _ = self._compute_reward(prices, demand, agent_prob, trajectories)
            rewards.append(float(reward))
        return float(np.mean(rewards)) if rewards else 0.0

    def _select_adversarial_alpha(self, prices: np.ndarray) -> float:
        """inner robust step: pick worst-case alpha from the ambiguity interval.

        when JAX is available and robust_rollouts==1 we use a vmapped pass over
        all K candidates in a single call (no Python loop, no market.act overhead).
        the JAX path approximates demand as the mixed closed-form d(p;theta) signal
        rather than running full trajectory sampling, which is accurate for the
        alpha-selection decision while being dramatically cheaper.

        when robust_rollouts>1 or JAX is unavailable we fall back to the sequential
        market.act() loop so behavior is identical to the original implementation.
        """
        candidates = self._alpha_candidates()
        if len(candidates) == 1:
            return float(candidates[0])

        if _JAX_OK and self.robust_rollouts == 1:
            best_alpha, _ = select_adversarial_alpha_jax(
                candidates=candidates,
                prices=prices,
                human_params=self.market.human_params,
                agent_params=self.market.agent_params,
                noise_std=self.market.noise_std,
                baseline_prices=self.baseline_prices,
                lambda_coi=self.lambda_coi,
                info_value=self.info_value,
                reward_profit_weight=self.reward_profit_weight,
                rng_seed=self._global_step,
            )
            return best_alpha

        # fallback: full trajectory-based sequential evaluation
        evaluations = [
            (float(alpha), self._evaluate_candidate(float(alpha), prices))
            for alpha in candidates
        ]
        best_alpha, _ = min(evaluations, key=lambda x: x[1])
        return best_alpha

    def _record_history(self):
        demand_arr = np.array(
            [self._demand.get(i, 0.0) for i in range(self.n_products)]
        )
        self._demand_history.append(demand_arr)
        self._price_history.append(self._prices.copy())
        self._revenue_history.append(np.sum(self._prices * demand_arr))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._set_market_mix(self.nominal_alpha)
        self._limbo.reset()
        self._prices = np.random.uniform(*self.price_bounds, size=self.n_products)
        self._platform_stub.set_prices(self._prices)
        self._limbo.step()
        self._demand = self._limbo.step()
        self._initial_episode_prices = self._prices.copy()
        self._step_count = 0
        self._low_margin_streak = 0
        self._demand_history, self._price_history, self._revenue_history = [], [], []
        self._trajectories = list(getattr(self.market, "last_trajectories", []))
        self._last_agent_prob = float(self.nominal_alpha)
        self._last_alpha_adv = float(self.nominal_alpha)
        self._record_history()
        return self._get_obs(), {}

    def step(self, action):
        self._prices = self._decode_action(action)
        alpha_adv = self._select_adversarial_alpha(self._prices)
        self._global_step += 1  # always increment; JAX path may have already done so
        self._set_market_mix(alpha_adv)
        self._platform_stub.set_prices(self._prices)
        self._step_count += 1

        self._demand = self.market.act(self._prices)
        trajectories = list(self.market.last_trajectories)
        agent_prob = self._compute_agent_prob(trajectories)
        self._trajectories.extend(trajectories)
        self._last_agent_prob = float(agent_prob)
        self._last_alpha_adv = float(alpha_adv)

        reward, metrics = self._compute_reward(
            self._prices, self._demand, agent_prob, trajectories
        )
        self._record_history()

        # soft early termination when margin collapses for too long
        avg_margin = float(np.mean(self._prices) - self.price_bounds[0]) / max(
            float(np.mean(self._prices)), 1e-6
        )
        if avg_margin < self.margin_floor:
            self._low_margin_streak += 1
        else:
            self._low_margin_streak = 0
        margin_collapsed = self._low_margin_streak >= self.margin_floor_patience
        terminated = self._step_count >= self.max_steps or margin_collapsed

        info = {
            "step": self._step_count,
            "agent_prob": agent_prob,
            "alpha_adv": float(alpha_adv),
            "alpha_nominal": float(self.nominal_alpha),
            "wasserstein_radius": float(self.robust_radius),
            "robust_rollouts": int(self.robust_rollouts),
            **metrics,
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

    wandb.init(project="capstone", config={"policy": "random", "alpha": 0.3})
    env = EconomicMetricsWrapper(PHANTOM(n_products=15, alpha=0.3, render_mode=None))

    model = RandomPolicy(env)
    model.learn(total_timesteps=1000, callback=MetricsCallback())

    print(f"Episode revenue: {env.episode_revenue:.1f}")
    wandb.finish()
    env.close()
