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

from .simplified import (System, Session, Event, Limbo, put_prices_to_market,
                         compute_demand, estimate_alpha, coi_erosion, TRANS_H, TRANS_A)


@dataclass
class EnvConfig:
    """Configuration for pricing environment."""
    n_products: int = 5
    max_steps: int = 200
    sessions_per_step: int = 30
    alpha_true: float = 0.2           # true contamination level
    alpha_drift: float = 0.0          # per-step drift in α
    alpha_bounds: Tuple[float, float] = (0.0, 0.6)
    lambda_coi: float = 0.5           # COI penalty weight
    lambda_vol: float = 0.1           # volatility penalty weight
    reward_mode: str = "robust"       # revenue | profit | robust | coi_aware
    normalize_reward: bool = True
    seed: int | None = 42


class PricingEnv:
    """RL environment for dynamic pricing under agent contamination.

    Implements the thesis formulation where:
    - Platform sets prices p_t
    - Market responds with mixture demand Q(p) = (1-α)D_H + αD_A
    - Agent estimates contamination α̂ from behavioral signals
    - Reward balances profit vs COI leakage

    Observation space (normalized):
        [0:n]     - current prices / ref_prices
        [n:2n]    - aggregated demand per product
        [2n]      - estimated contamination α̂
        [2n+1]    - true contamination α (if observable, else 0)
        [2n+2:3n+2] - current margins (prices - costs) / costs
        [3n+2]    - step / max_steps

    Action space:
        price multipliers in [0.5, 1.5] applied to reference prices
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

        # gymnasium spaces
        self.action_space = spaces.Box(low=0.5, high=1.5, shape=(self.n,), dtype=np.float32)
        obs_dim = self.n + self.n + 1 + 1 + self.n + 1  # prices + demand + α̂ + α + margins + t
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _build_obs(self) -> np.ndarray:
        """Construct observation vector."""
        if self._sys is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        prices = self._last_prices if self._last_prices is not None else self._sys.refs
        price_ratio = prices / (self._sys.refs + 1e-6)
        demand_norm = self._demand_agg / (np.sum(self._demand_agg) + 1e-6)
        margins = (prices - self._sys.costs) / (self._sys.costs + 1e-6)
        t_norm = self._t / self.cfg.max_steps

        obs = np.concatenate([
            price_ratio,                          # [0:n]
            demand_norm,                          # [n:2n]
            [self._sys.alpha],                    # [2n] estimated α̂
            [self._alpha],                        # [2n+1] true α
            margins,                              # [2n+2:3n+2]
            [t_norm],                             # [3n+2]
        ])
        return obs.astype(np.float32)

    def _compute_reward(self, prices: np.ndarray, demand: Dict[str, float]) -> float:
        """Compute reward based on configured mode."""
        cfg, sys = self.cfg, self._sys
        if sys is None:
            return 0.0

        # aggregate demand per product
        agg = np.zeros(self.n)
        for sid, q in demand.items():
            sess = next((s for s in sys._sessions if s.sid == sid), None)
            if sess and sess.events:
                pidx = sess.events[0].product_idx
                agg[pidx] += q
        self._demand_agg = agg

        revenue = float(np.dot(prices, agg))
        cost = float(np.dot(sys.costs, np.clip(agg, 0, 1)))  # simplified cost model
        profit = revenue - cost

        # volatility penalty (price changes)
        vol_penalty = 0.0
        if self._last_prices is not None:
            price_change = np.abs(prices - self._last_prices) / (sys.refs + 1e-6)
            vol_penalty = cfg.lambda_vol * float(np.mean(price_change))

        # COI leakage penalty
        avg_margin = float(np.mean(prices - sys.costs))
        coi_leak = sys.alpha * avg_margin

        if cfg.reward_mode == "revenue":
            r = revenue
        elif cfg.reward_mode == "profit":
            r = profit
        elif cfg.reward_mode == "robust":
            # robust objective: profit - λ_coi * COI_leak - λ_vol * volatility
            r = profit - cfg.lambda_coi * coi_leak - vol_penalty
        elif cfg.reward_mode == "coi_aware":
            # adaptive: heavier penalty at high contamination
            adaptive_lambda = cfg.lambda_coi * (1 + 2 * sys.alpha)
            r = profit - adaptive_lambda * coi_leak - vol_penalty
        else:
            r = profit

        if cfg.normalize_reward:
            r = r / (float(np.sum(sys.refs)) + 1e-6)  # normalize by potential revenue

        return float(r)

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        seed = seed if seed is not None else self.cfg.seed
        self._sys = System(n_products=self.n, lambda_coi=self.cfg.lambda_coi, seed=seed)
        self._t = 0
        self._alpha = self.cfg.alpha_true
        self._last_prices = None
        self._last_demand = None
        self._episode_rewards = []
        self._demand_agg = np.zeros(self.n)

        info = {"alpha_true": self._alpha, "alpha_est": self._sys.alpha,
                "costs": self._sys.costs.copy(), "refs": self._sys.refs.copy()}
        return self._build_obs(), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step.

        Args:
            action: price multipliers in [0.5, 1.5]

        Returns:
            obs, reward, terminated, truncated, info
        """
        if self._sys is None:
            raise RuntimeError("call reset() first")

        # convert action to prices
        action = np.clip(action, 0.5, 1.5)
        prices = self._sys.refs * action.astype(np.float64)
        prices = np.clip(prices, self._sys.costs * 1.01, self._sys.refs * 2.0)

        # drift contamination
        if self.cfg.alpha_drift != 0:
            self._alpha = np.clip(
                self._alpha + self.cfg.alpha_drift * self._sys.rng.normal(),
                *self.cfg.alpha_bounds)

        # observe demand
        demand = self._sys.observe_demand(prices, alpha_true=self._alpha, n_sessions=self.cfg.sessions_per_step)
        self._sys.limbo.add_update("prices", prices)

        # update α estimate
        self._sys._alpha_est = self._sys._estimate_alpha_from_sessions()

        reward = self._compute_reward(prices, demand)
        self._episode_rewards.append(reward)

        self._last_prices = prices.copy()
        self._last_demand = demand
        self._t += 1

        terminated = self._t >= self.cfg.max_steps
        truncated = False

        info = {
            "alpha_true": self._alpha,
            "alpha_est": self._sys.alpha,
            "revenue": float(np.dot(prices, self._demand_agg)),
            "avg_margin": float(np.mean((prices - self._sys.costs) / self._sys.costs)),
            "n_sessions": len(demand),
            "coi_erosion": coi_erosion(int(self._alpha * self.cfg.sessions_per_step), float(np.std(prices))),
        }

        return self._build_obs(), reward, terminated, truncated, info

    def render(self, mode: str = "human") -> str | None:
        """Render environment state."""
        if self._sys is None or self._last_prices is None:
            return None

        lines = [
            f"t={self._t}/{self.cfg.max_steps}",
            f"α_true={self._alpha:.3f} α̂={self._sys.alpha:.3f}",
            f"prices: {self._last_prices.round(1)}",
            f"demand: {self._demand_agg.round(2)}",
            f"reward: {self._episode_rewards[-1] if self._episode_rewards else 0:.3f}",
        ]
        out = " | ".join(lines)
        if mode == "human":
            print(out)
        return out

    def close(self) -> None:
        pass


class ContaminationSweepEnv(PricingEnv):
    """Environment that sweeps through contamination levels during training.

    Useful for curriculum learning: start with low α, gradually increase.
    """

    def __init__(self, cfg: EnvConfig | None = None, alpha_schedule: list[float] | None = None):
        super().__init__(cfg)
        self._schedule = alpha_schedule or [0.1, 0.2, 0.3, 0.4, 0.5]
        self._schedule_idx = 0

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
        # advance schedule on reset
        if options and options.get("advance_schedule", False):
            self._schedule_idx = (self._schedule_idx + 1) % len(self._schedule)
        self.cfg.alpha_true = self._schedule[self._schedule_idx]
        return super().reset(seed, options)


class AdversarialEnv(PricingEnv):
    """Environment with adversarial contamination dynamics.

    The contamination level responds to pricing policy: if prices are too predictable,
    agents learn to exploit and α increases.
    """

    def __init__(self, cfg: EnvConfig | None = None, exploitation_rate: float = 0.02):
        super().__init__(cfg)
        self._exploit_rate = exploitation_rate
        self._price_history: list[np.ndarray] = []

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, term, trunc, info = super().step(action)

        # track price history for predictability
        if self._last_prices is not None:
            self._price_history.append(self._last_prices.copy())

        # increase α if prices are predictable (low variance over recent history)
        if len(self._price_history) > 10:
            recent = np.array(self._price_history[-10:])
            predictability = 1.0 / (float(np.std(recent)) + 0.1)
            self._alpha = np.clip(
                self._alpha + self._exploit_rate * predictability * self._sys.rng.random(),
                *self.cfg.alpha_bounds)

        info["predictability"] = predictability if len(self._price_history) > 10 else 0.0
        return obs, reward, term, trunc, info

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
        self._price_history = []
        return super().reset(seed, options)


def make_env(cfg: EnvConfig | None = None, env_type: str = "standard") -> PricingEnv:
    """Factory for creating pricing environments."""
    if env_type == "sweep":
        return ContaminationSweepEnv(cfg)
    elif env_type == "adversarial":
        return AdversarialEnv(cfg)
    return PricingEnv(cfg)


# simple baseline policies for benchmarking
def fixed_price_policy(refs: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Fixed markup policy: always return ref * (1 + margin)."""
    return np.ones(len(refs), dtype=np.float32) * (1.0 + margin)


def random_policy(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random policy for exploration baseline."""
    rng = rng or np.random.default_rng()
    return rng.uniform(0.7, 1.3, n).astype(np.float32)


def adaptive_policy(obs: np.ndarray, n: int, base_margin: float = 0.1) -> np.ndarray:
    """Simple adaptive policy: reduce margins when α̂ is high."""
    alpha_est = obs[2 * n]  # α̂ is at position 2n in observation
    margin_scale = 1.0 - 0.4 * alpha_est  # defensive when α̂ high
    return np.ones(n, dtype=np.float32) * (1.0 + base_margin * margin_scale)


if __name__ == "__main__":
    # demo run
    cfg = EnvConfig(n_products=100, max_steps=100, alpha_true=0.25, reward_mode="robust")
    env = make_env(cfg)
    obs, info = env.reset()
    print(f"initial: α={info['alpha_true']:.2f}")

    total_reward = 0.0
    for t in range(cfg.max_steps):
        action = adaptive_policy(obs, cfg.n_products)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        if t % 10 == 0:
            env.render()
        if done:
            break

    print(f"\ntotal reward: {total_reward:.2f}, final α̂: {info['alpha_est']:.3f}")
