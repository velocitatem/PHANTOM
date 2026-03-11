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
        quoted_prices = np.asarray(self.env.unwrapped._prices, dtype=float)
        effective_prices = np.asarray(
            info.get("effective_prices", quoted_prices), dtype=float
        )
        if effective_prices.shape != quoted_prices.shape:
            effective_prices = quoted_prices
        demand_dict = self.env.unwrapped._demand
        demand = np.array([demand_dict.get(i, 0.0) for i in range(len(quoted_prices))])

        # core calculations
        revenue = float(info.get("revenue", np.sum(effective_prices * demand)))
        quoted_revenue = float(np.sum(quoted_prices * demand))
        avg_price = float(np.mean(effective_prices))
        margin = (avg_price - self.p_min) / max(avg_price, 1e-6)
        coi_level = avg_price - self.p_min  # E[P] - p_min per thesis Def 1

        self._price_history.append(effective_prices.copy())
        self._revenue_history.append(revenue)

        # regret vs baseline (golden path)
        regret = 0.0
        if self.baseline_revenue and self.baseline_revenue > 0:
            regret = 1.0 - (revenue / self.baseline_revenue)

        # inject structured metrics into info
        info["economics"] = {
            "revenue": revenue,
            "quoted_revenue": quoted_revenue,
            "margin": margin,
            "coi_level": coi_level,
            "regret": regret,
        }
        for key in (
            "coi_mix",
            "coi_base",
            "coi_leakage",
            "coi_penalty",
            "ux_penalty",
            "volatility",
            "profit",
            "cost_floor",
            "reward_revenue",
            "reward_total",
            "agent_prob",
            "alpha_adv",
            "alpha_nominal",
            "erosion_share",
            "effective_price_mean",
        ):
            if key in info:
                info["economics"][key] = info[key]
        info["prices"] = quoted_prices.copy()
        info["effective_prices"] = effective_prices.copy()
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
