from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class PolicyLike(Protocol):
    def predict(self, obs: np.ndarray, deterministic: bool = True): ...


class StaticPolicy:
    def __init__(self, n_actions: int):
        self._action = int(max(0, n_actions // 2))

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self._action, None


class SurgePolicy:
    def __init__(
        self,
        n_actions: int,
        n_products: int,
        high_threshold: float = 60.0,
        low_threshold: float = 30.0,
    ):
        self.n_actions = int(n_actions)
        self.n_products = int(n_products)
        self.mid = self.n_actions // 2
        self.high_t = float(high_threshold)
        self.low_t = float(low_threshold)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        obs_arr = np.asarray(obs, dtype=np.float32)
        demand = obs_arr[: self.n_products]
        demand_mean = float(np.mean(demand)) if demand.size > 0 else 0.0
        if demand_mean >= self.high_t:
            return min(self.mid + 2, self.n_actions - 1), None
        if demand_mean <= self.low_t:
            return max(self.mid - 2, 0), None
        return self.mid, None


@dataclass
class LinearElasticityPolicy:
    n_actions: int
    n_products: int
    price_low: float
    price_high: float

    def __post_init__(self):
        self.n_actions = int(self.n_actions)
        self.n_products = int(self.n_products)
        self.price_low = float(self.price_low)
        self.price_high = float(self.price_high)
        self._target_price = 0.5 * (self.price_low + self.price_high)
        self._action_scales = np.linspace(0.8, 1.2, self.n_actions)

    def fit(self, env, warmup_steps: int = 800, seed: int = 42):
        rng = np.random.default_rng(int(seed))
        obs, _ = env.reset(seed=int(seed))
        prices: list[float] = []
        demands: list[float] = []

        for _ in range(int(max(10, warmup_steps))):
            action = int(rng.integers(0, self.n_actions))
            obs, _, term, trunc, info = env.step(action)
            done = bool(term or trunc)

            p = np.asarray(info.get("prices", []), dtype=np.float32)
            d = np.asarray(info.get("demand", []), dtype=np.float32)
            if p.size > 0 and d.size > 0:
                prices.append(float(np.mean(p)))
                demands.append(float(np.mean(d)))

            if done:
                obs, _ = env.reset()

        if len(prices) < 8:
            self._target_price = 0.5 * (self.price_low + self.price_high)
            return self

        slope, intercept = np.polyfit(np.asarray(prices), np.asarray(demands), 1)
        if slope < -1e-6:
            p_star = -intercept / (2.0 * slope)
            self._target_price = float(np.clip(p_star, self.price_low, self.price_high))
        else:
            self._target_price = 0.5 * (self.price_low + self.price_high)
        return self

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        obs_arr = np.asarray(obs, dtype=np.float32)
        cur_prices = obs_arr[self.n_products : 2 * self.n_products]
        cur_mean = (
            float(np.mean(cur_prices)) if cur_prices.size > 0 else self._target_price
        )
        scale = self._target_price / max(cur_mean, 1e-6)
        action = int(np.argmin(np.abs(self._action_scales - scale)))
        return int(np.clip(action, 0, self.n_actions - 1)), None
