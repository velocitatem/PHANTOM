"""Economic metrics wrapper - calculates thesis-aligned KPIs and injects into info dict."""

import gymnasium as gym
import numpy as np


class EconomicMetricsWrapper(gym.Wrapper):
    """Calculates thesis-aligned economic metrics per step, injects into info.

    Metrics follow thesis definitions:
    - COI level: E[P] - p_min (Definition 1)
    - Margin: (avg_price - p_min) / avg_price
    - Regret: 1 - (revenue / baseline_revenue)
    """

    def __init__(
        self, env: gym.Env, p_min: float = 10.0, baseline_revenue: float | None = None
    ):
        super().__init__(env)
        self.p_min = p_min
        self.baseline_revenue = baseline_revenue
        self._price_history: list[np.ndarray] = []
        self._revenue_history: list[float] = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._price_history = []
        self._revenue_history = []
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # extract from unwrapped env
        prices = self.env.unwrapped._prices
        demand_dict = self.env.unwrapped._demand
        demand = np.array([demand_dict.get(i, 0.0) for i in range(len(prices))])
        alpha = self.env.unwrapped.alpha

        # core calculations
        revenue = float(np.sum(prices * demand))
        avg_price = float(np.mean(prices))
        margin = (avg_price - self.p_min) / max(avg_price, 1e-6)
        coi_level = avg_price - self.p_min  # E[P] - p_min per thesis Def 1

        self._price_history.append(prices.copy())
        self._revenue_history.append(revenue)

        # regret vs baseline (golden path)
        regret = 0.0
        if self.baseline_revenue and self.baseline_revenue > 0:
            regret = 1.0 - (revenue / self.baseline_revenue)

        # inject structured metrics into info
        info["economics"] = {
            "revenue": revenue,
            "margin": margin,
            "coi_level": coi_level,
            "regret": regret,
        }
        for key in ("coi_mix", "coi_base", "coi_leakage", "coi_penalty"):
            if key in info:
                info["economics"][key] = info[key]
        info["prices"] = prices.copy()
        info["demand"] = demand.copy()

        return obs, reward, terminated, truncated, info

    @property
    def episode_revenue(self) -> float:
        return sum(self._revenue_history)

    @property
    def episode_mean_price(self) -> float:
        if not self._price_history:
            return 0.0
        return float(np.mean([np.mean(p) for p in self._price_history]))
