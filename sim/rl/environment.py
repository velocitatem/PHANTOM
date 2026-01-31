from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError("sim.rl.environment requires gymnasium") from e

from sim.case.thesis_simplified.coi import COIWindow, coi_erosion, compute_coi_window
from sim.case.thesis_simplified.separability import estimate_alpha as estimate_session_alpha
from sim.case.thesis_simplified.simplified import Limbo, Session, put_prices_to_market
from sim.rl.thesis_core import aggregate_demand_by_product, aggregate_purchases, constrain_prices


@dataclass(frozen=True)
class BusinessLogicConstraints:
    product_catalogue_size: int = 100
    max_steps: int = 2000
    sessions_per_step: int = 250

    system_max_price: float = 500.0
    system_min_price: float = 1.0
    max_price_adjustment: float = 0.30
    min_margin_pct: float = 0.05

    agent_share: float = 0.2
    alpha_drift: float = 0.0
    alpha_bounds: tuple[float, float] = (0.0, 0.8)

    coi_strength: float = 0.25
    w_volatility: float = 5.0
    w_estimation_error: float = 0.25

    seed: int = 7


def make_env(constraints: Optional[BusinessLogicConstraints] = None) -> "PHANTOMEnv":
    return PHANTOMEnv(constraints=constraints or BusinessLogicConstraints())


class PHANTOMEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, constraints: Optional[BusinessLogicConstraints] = None):
        super().__init__()
        self.c = constraints or BusinessLogicConstraints()
        self.n = int(self.c.product_catalogue_size)

        self._rng = np.random.default_rng(self.c.seed)
        self._t = 0
        self._alpha_true = float(self.c.agent_share)
        self._alpha_hat = float(self.c.agent_share)
        self._costs = np.zeros(self.n, dtype=np.float32)
        self._refs = np.zeros(self.n, dtype=np.float32)
        self._prices: Optional[np.ndarray] = None
        self._last_sessions: list[Session] = []
        self._last_coi: COIWindow | None = None
        self._limbo = Limbo()

        self.action_space = spaces.Box(
            low=np.full((self.n,), self.c.system_min_price, dtype=np.float32),
            high=np.full((self.n,), self.c.system_max_price, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "elasticity": spaces.Dict(
                    {
                        "price": spaces.Box(
                            low=np.full((self.n,), self.c.system_min_price, dtype=np.float32),
                            high=np.full((self.n,), self.c.system_max_price, dtype=np.float32),
                            dtype=np.float32,
                        ),
                        "demand": spaces.Box(
                            low=np.zeros((self.n,), dtype=np.float32),
                            high=np.full((self.n,), 1e9, dtype=np.float32),
                            dtype=np.float32,
                        ),
                    }
                ),
                "market": spaces.Dict(
                    {
                        "alpha_hat": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                        "revenue_rate": spaces.Box(low=0.0, high=1e12, shape=(1,), dtype=np.float32),
                        "conversion_rate": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                        "price_volatility": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    }
                ),
                "cost": spaces.Box(
                    low=np.zeros((self.n,), dtype=np.float32),
                    high=np.full((self.n,), self.c.system_max_price, dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )

    def _reset_catalogue(self) -> None:
        self._costs = self._rng.uniform(15.0, 60.0, size=self.n).astype(np.float32)
        margins = self._rng.uniform(0.2, 0.6, size=self.n).astype(np.float32)
        self._refs = (self._costs * (1.0 + margins)).astype(np.float32)
        self._prices = self._refs.copy()

    def _observe_market(
        self, prices: np.ndarray
    ) -> tuple[list[Session], Dict[str, float], np.ndarray, np.ndarray, float, float, int]:
        sessions, demand_map = put_prices_to_market(
            prices,
            costs=self._costs,
            alpha=self._alpha_true,
            n_sessions=int(self.c.sessions_per_step),
            seed=int(self._rng.integers(0, 2**31 - 1)),
        )
        demand_by_product = aggregate_demand_by_product(sessions, demand_map, self.n)
        purchases, revenue, cost, n_agents = aggregate_purchases(sessions, self._costs, self.n)
        conversion = float(np.sum(purchases) / max(len(sessions), 1))
        return sessions, demand_map, demand_by_product, purchases, revenue, cost, n_agents

    def _update_alpha_hat(self, sessions: list[Session]) -> float:
        scores = [estimate_session_alpha(s) for s in sessions if s.events]
        if not scores:
            return self._alpha_hat
        alpha_step = float(np.mean(scores))
        self._alpha_hat = 0.8 * self._alpha_hat + 0.2 * alpha_step
        self._alpha_hat = float(np.clip(self._alpha_hat, 0.0, 1.0))
        return self._alpha_hat

    def _reward(self, prices: np.ndarray, revenue: float, cost: float, volatility: float) -> float:
        profit = float(revenue - cost)
        coi_leak = float(self._last_coi.leak) if self._last_coi else 0.0
        alpha_err = abs(self._alpha_hat - self._alpha_true)
        return profit - self.c.coi_strength * coi_leak - self.c.w_volatility * volatility - self.c.w_estimation_error * alpha_err

    def _build_obs(
        self,
        prices: np.ndarray,
        demand_by_product: np.ndarray,
        revenue: float,
        conversion: float,
        volatility: float,
    ) -> Dict[str, Any]:
        return {
            "elasticity": {"price": prices.astype(np.float32), "demand": demand_by_product.astype(np.float32)},
            "market": {
                "alpha_hat": np.array([self._alpha_hat], dtype=np.float32),
                "revenue_rate": np.array([revenue], dtype=np.float32),
                "conversion_rate": np.array([conversion], dtype=np.float32),
                "price_volatility": np.array([volatility], dtype=np.float32),
            },
            "cost": self._costs.astype(np.float32),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._alpha_true = float(np.clip(self.c.agent_share, *self.c.alpha_bounds))
        self._alpha_hat = float(self.c.agent_share)
        self._reset_catalogue()
        self._limbo = Limbo()
        self._last_sessions = []
        self._last_coi = None

        prices = self._prices if self._prices is not None else np.zeros(self.n, dtype=np.float32)
        obs = self._build_obs(prices, np.zeros(self.n, dtype=np.float32), 0.0, 0.0, 0.0)
        return obs, {"alpha_true": self._alpha_true}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self._prices is None:
            raise RuntimeError("reset() must be called before step()")

        prev = self._prices
        prices = constrain_prices(
            prev,
            np.asarray(action, dtype=np.float32),
            costs=self._costs,
            min_price=float(self.c.system_min_price),
            max_price=float(self.c.system_max_price),
            max_adjustment=float(self.c.max_price_adjustment),
            min_margin_pct=float(self.c.min_margin_pct),
        )
        self._prices = prices
        self._limbo.add_update("prices", prices)

        sessions, demand_map, demand_by_product, purchases, revenue, cost, n_agents = self._observe_market(prices)
        self._last_sessions = sessions
        self._limbo.add_update("demand", demand_map)

        self._update_alpha_hat(self._last_sessions)
        self._last_coi = compute_coi_window(self._last_sessions, self._costs, demand_mapping=demand_map)

        self._alpha_true = float(np.clip(self._alpha_true + self.c.alpha_drift, *self.c.alpha_bounds))
        volatility = float(np.std((prices - prev) / (prev + 1e-6)))
        reward = float(self._reward(prices, revenue, cost, volatility))
        conversion = float(np.sum(purchases) / max(len(self._last_sessions), 1))

        self._t += 1
        terminated = self._t >= int(self.c.max_steps)

        obs = self._build_obs(prices, demand_by_product, revenue, conversion, min(volatility, 1.0))
        info = {
            "step": self._t,
            "reward": reward,
            "revenue": float(revenue),
            "profit": float(revenue - cost),
            "n_sessions": int(self.c.sessions_per_step),
            "n_agents": int(n_agents),
            "alpha_true": float(self._alpha_true),
            "alpha_hat": float(self._alpha_hat),
            "alpha_error": float(abs(self._alpha_hat - self._alpha_true)),
            "price_std": float(np.std(prices)),
            "price_volatility": float(volatility),
        }
        if self._last_coi is not None:
            info.update(
                {
                    "coi_policy": float(self._last_coi.policy),
                    "coi_agent": float(self._last_coi.agent),
                    "coi_leakage": float(self._last_coi.leak),
                    "coi_survival": float(self._last_coi.survival_ratio),
                    "coi_erosion": float(coi_erosion(self._last_coi.policy, self._last_coi.agent)),
                }
            )
        return obs, reward, terminated, False, info

    def render(self, mode: str = "human") -> str | None:
        if self._prices is None:
            return None
        out = (
            f"t={self._t}/{self.c.max_steps} "
            f"alpha_true={self._alpha_true:.3f} alpha_hat={self._alpha_hat:.3f} "
            f"price_std={float(np.std(self._prices)):.2f}"
        )
        if mode == "human":
            print(out)
        return out

    def close(self) -> None:
        return
