"""Gymnasium-compatible RL environment for thesis pricing system.

Wraps simplified.System with standard Gym interface for training pricing policies.
Supports multiple reward modes and contamination scenarios.

Action: price multipliers [0.5, 1.5] applied to reference prices
Observation: [prices, demand_agg, alpha_est, margins, position_proxy]
Reward: configurable objective (revenue, profit, robust, coi-aware)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

from .simplified import System, Session, Event, Limbo, put_prices_to_market, compute_demand, estimate_alpha
from .coi import COIWindow, compute_coi_window, coi_erosion


@dataclass
class EnvConfig:
    n_products: int = 5
    max_steps: int = 200
    sessions_per_step: int = 30
    alpha_true: float = 0.2
    alpha_drift: float = 0.0
    alpha_bounds: Tuple[float, float] = (0.0, 0.6)
    lambda_coi: float = 0.5
    lambda_vol: float = 0.1
    reward_mode: str = "robust"  # revenue | profit | robust | coi_aware
    normalize_reward: bool = True
    seed: int | None = 42


def aggregate_purchases(sessions: list[Session], n_products: int, costs: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Aggregate purchases from sessions, returns (counts, revenue, cost)."""
    purchases = np.zeros(n_products, dtype=float)
    revenue, cost = 0.0, 0.0
    for sess in sessions:
        for e in sess.events:
            if e.action == "purchase" and 0 <= e.product_idx < n_products:
                purchases[e.product_idx] += 1.0
                revenue += float(e.price_seen)
                cost += float(costs[e.product_idx])
    return purchases, revenue, cost


class PricingEnv(gym.Env if HAS_GYM else object):
    """RL environment for dynamic pricing under agent contamination.

    Platform sets prices p_t, market responds with mixture demand Q(p) = (1-alpha)*D_H + alpha*D_A.
    Agent estimates contamination alpha_hat from behavioral signals.
    Reward balances profit vs COI leakage.
    """
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, cfg: EnvConfig | None = None):
        if not HAS_GYM:
            raise ImportError("gymnasium required")
        self.cfg = cfg or EnvConfig()
        self.n = self.cfg.n_products
        self._sys: System | None = None
        self._t = 0
        self._alpha = self.cfg.alpha_true
        self._last_prices: np.ndarray | None = None
        self._last_demand: Dict[str, float] | None = None
        self._episode_rewards: list[float] = []
        self._demand_agg = np.zeros(self.n)

        self.action_space = spaces.Box(low=0.5, high=1.5, shape=(self.n,), dtype=np.float32)
        obs_dim = self.n + self.n + 1 + 1 + self.n + 1  # prices + demand + alpha_hat + alpha + margins + t
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _build_obs(self) -> np.ndarray:
        if self._sys is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        prices = self._last_prices if self._last_prices is not None else self._sys.refs
        return np.concatenate([
            prices / (self._sys.refs + 1e-6),
            self._demand_agg / (np.sum(self._demand_agg) + 1e-6),
            [self._sys.alpha, self._alpha],
            (prices - self._sys.costs) / (self._sys.costs + 1e-6),
            [self._t / self.cfg.max_steps],
        ]).astype(np.float32)

    def _compute_reward(self, prices: np.ndarray, demand: Dict[str, float]) -> float:
        cfg, sys = self.cfg, self._sys
        if sys is None:
            return 0.0

        # aggregate demand per product
        agg = np.zeros(self.n)
        for sid, q in demand.items():
            sess = next((s for s in sys._sessions if s.sid == sid), None)
            if sess and sess.events:
                agg[sess.events[0].product_idx] += q
        self._demand_agg = agg

        _, revenue, cost = aggregate_purchases(sys._last_sessions, self.n, sys.costs)
        profit = revenue - cost

        vol_penalty = 0.0
        if self._last_prices is not None:
            vol_penalty = cfg.lambda_vol * float(np.mean(np.abs(prices - self._last_prices) / (sys.refs + 1e-6)))

        coi = compute_coi_window(sys._last_sessions, sys.costs, demand_mapping=demand)
        leak = float(coi.leak)

        reward_fns = {
            "revenue": lambda: revenue,
            "profit": lambda: profit,
            "robust": lambda: profit - cfg.lambda_coi * leak - vol_penalty,
            "coi_aware": lambda: profit - cfg.lambda_coi * (1 + 2 * sys.alpha) * leak - vol_penalty,
        }
        r = reward_fns.get(cfg.reward_mode, lambda: profit)()
        return float(r / (float(np.sum(sys.refs)) + 1e-6)) if cfg.normalize_reward else float(r)

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
        seed = seed if seed is not None else self.cfg.seed
        self._sys = System(n_products=self.n, lambda_coi=self.cfg.lambda_coi, seed=seed)
        self._t, self._alpha = 0, self.cfg.alpha_true
        self._last_prices, self._last_demand = None, None
        self._episode_rewards, self._demand_agg = [], np.zeros(self.n)
        return self._build_obs(), {"alpha_true": self._alpha, "alpha_est": self._sys.alpha,
                                   "costs": self._sys.costs.copy(), "refs": self._sys.refs.copy()}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._sys is None:
            raise RuntimeError("call reset() first")

        action = np.clip(action, 0.5, 1.5)
        prices = np.clip(self._sys.refs * action.astype(np.float64), self._sys.costs * 1.01, self._sys.refs * 2.0)
        demand = self._sys.observe_demand(prices, alpha_true=self._alpha, n_sessions=self.cfg.sessions_per_step)
        self._sys.limbo.add_update("prices", prices)
        self._sys._alpha_est = self._sys._estimate_alpha_from_sessions()

        reward = self._compute_reward(prices, demand)
        self._episode_rewards.append(reward)
        self._last_prices, self._last_demand = prices.copy(), demand
        self._t += 1

        # compute info metrics using shared helper
        purchases, revenue, cost = aggregate_purchases(self._sys._last_sessions, self.n, self._sys.costs)
        n_agents = int(self._alpha * self.cfg.sessions_per_step)
        coi = compute_coi_window(self._sys._last_sessions, self._sys.costs, demand_mapping=demand)

        info = {
            "alpha_true": self._alpha, "alpha_est": self._sys.alpha,
            "alpha_error": abs(self._alpha - self._sys.alpha),
            "revenue": float(revenue), "profit": float(revenue - cost), "cost": float(cost),
            "n_purchases": int(np.sum(purchases)),
            "avg_margin": float(np.mean((prices - self._sys.costs) / self._sys.costs)),
            "n_sessions": len(demand), "n_agents": n_agents, "price_std": float(np.std(prices)),
            "coi_erosion": coi_erosion(coi.policy, coi.agent),
            "coi_policy": float(coi.policy), "coi_agent": float(coi.agent),
            "coi_leakage": float(coi.leak), "coi_survival": float(coi.survival_ratio),
            "cumulative_reward": sum(self._episode_rewards), "step": self._t,
        }
        return self._build_obs(), reward, self._t >= self.cfg.max_steps, False, info

    def render(self, mode: str = "human") -> str | None:
        if self._sys is None or self._last_prices is None:
            return None
        out = f"t={self._t}/{self.cfg.max_steps} | alpha_true={self._alpha:.3f} alpha_hat={self._sys.alpha:.3f} | " \
              f"prices: {self._last_prices.round(1)} | demand: {self._demand_agg.round(2)} | " \
              f"reward: {self._episode_rewards[-1] if self._episode_rewards else 0:.3f}"
        if mode == "human":
            print(out)
        return out

    def close(self) -> None:
        pass


class ContaminationSweepEnv(PricingEnv):
    """Environment that sweeps through contamination levels during training."""

    def __init__(self, cfg: EnvConfig | None = None, alpha_schedule: list[float] | None = None):
        super().__init__(cfg)
        self._schedule = alpha_schedule or [0.1, 0.2, 0.3, 0.4, 0.5]
        self._schedule_idx = 0

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
        if options and options.get("advance_schedule", False):
            self._schedule_idx = (self._schedule_idx + 1) % len(self._schedule)
        self.cfg.alpha_true = self._schedule[self._schedule_idx]
        return super().reset(seed, options)


class AdversarialEnv(PricingEnv):
    """Environment with adversarial contamination dynamics.

    Contamination increases when prices are predictable (agents exploit).
    """

    def __init__(self, cfg: EnvConfig | None = None, exploitation_rate: float = 0.02):
        super().__init__(cfg)
        self._exploit_rate = exploitation_rate
        self._price_history: list[np.ndarray] = []

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, term, trunc, info = super().step(action)
        if self._last_prices is not None:
            self._price_history.append(self._last_prices.copy())
        predictability = 0.0
        if len(self._price_history) > 10:
            predictability = 1.0 / (float(np.std(self._price_history[-10:])) + 0.1)
            self._alpha = np.clip(self._alpha + self._exploit_rate * predictability * self._sys.rng.random(), *self.cfg.alpha_bounds)
        info["predictability"] = predictability
        return obs, reward, term, trunc, info

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
        self._price_history = []
        return super().reset(seed, options)


def make_env(cfg: EnvConfig | None = None, env_type: str = "standard") -> PricingEnv:
    return {"sweep": ContaminationSweepEnv, "adversarial": AdversarialEnv}.get(env_type, PricingEnv)(cfg)


# baseline policies
fixed_price_policy = lambda refs, margin=0.0: np.ones(len(refs), dtype=np.float32) * (1.0 + margin)
random_policy = lambda n, rng=None: (rng or np.random.default_rng()).uniform(0.7, 1.3, n).astype(np.float32)
adaptive_policy = lambda obs, n, base=0.1: np.ones(n, dtype=np.float32) * (1.0 + base * (1.0 - 0.4 * obs[2 * n]))


if __name__ == "__main__":
    cfg = EnvConfig(n_products=100, max_steps=100, alpha_true=0.25, reward_mode="robust")
    env = make_env(cfg)
    obs, info = env.reset()
    print(f"initial: alpha={info['alpha_true']:.2f}")

    total_reward = 0.0
    for t in range(cfg.max_steps):
        action = adaptive_policy(obs, cfg.n_products)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        if t % 10 == 0:
            env.render()
        if done:
            break

    print(f"\ntotal reward: {total_reward:.2f}, final alpha_hat: {info['alpha_est']:.3f}")
