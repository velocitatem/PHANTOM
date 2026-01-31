from os import kill
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any
from sim.rl.environment import BusinessLogicConstraints

"""
An angine by default should have its own demand estimation mechanism from the observed observations whihc are the computer feature.
From these features we then follow the researc hstructure of q -> p with a testable and must be updatable mechanism.
"""

class BasePricingEngine(ABC):
    """base interface for all pricing engines"""
    def __init__(self, constraints: BusinessLogicConstraints, seed: int = 0):
        self.c = constraints
        self.rng = np.random.default_rng(seed)
        self.step_count = 0


    @abstractmethod
    def compute_prices(self, current_prices: np.ndarray, observation: Dict[str, Any]) -> np.ndarray:
        """compute new prices given current state and observation from environment

        args:
            current_prices: current price vector [N]
            observation: dict containing 'price', 'demand', and possibly interaction data

        returns:
            new_prices: updated price vector [N]
        """
        pass

    def update(self, observation: Dict[str, Any], reward: float, done: bool, info: Dict[str, Any]) -> None:
        """Default no-op update. Engines can override as needed."""
        self.last_observation = observation
        self.last_reward = reward
        self.last_info = info




    def reset(self):
        """reset engine state for new episode"""
        self.step_count = 0


class WildPricingEngine(BasePricingEngine):
    """production-like pricing using online elasticity estimation via EWMA regression"""
    def __init__(self, constraints: BusinessLogicConstraints, seed: int = 0):
        super().__init__(constraints, seed)
        # per-product unit costs (unknown to customers; known to platform)
        self.unit_cost = self.rng.uniform(8.0, 40.0, size=self.c.product_catalogue_size).astype(np.float32)
        # online elasticity estimate (start moderately elastic)
        self.e_hat = np.full((self.c.product_catalogue_size,), -1.3, dtype=np.float32)
        # EWMA state for log-log regression
        self.mu_logp = np.zeros(self.c.product_catalogue_size, dtype=np.float32)
        self.mu_logq = np.zeros(self.c.product_catalogue_size, dtype=np.float32)
        self.cov_pq  = np.zeros(self.c.product_catalogue_size, dtype=np.float32)
        self.var_p   = np.ones(self.c.product_catalogue_size, dtype=np.float32)
        # knobs typical in production
        self.lr = 0.08
        self.ewma = 0.05
        self.eps_explore = 0.03
        self.explore_scale = 0.03

    def _safe_elasticity(self, e: np.ndarray) -> np.ndarray:
        return np.clip(e, -5.0, -1.05)

    def reset(self):
        super().reset()
        self.e_hat = np.full((self.c.product_catelogue_size,), -1.3, dtype=np.float32)
        self.mu_logp = np.zeros(self.c.product_catelogue_size, dtype=np.float32)
        self.mu_logq = np.zeros(self.c.product_catelogue_size, dtype=np.float32)
        self.cov_pq = np.zeros(self.c.product_catelogue_size, dtype=np.float32)
        self.var_p = np.ones(self.c.product_catelogue_size, dtype=np.float32)

    def compute_prices(self, current_prices: np.ndarray, observation: Dict[str, Any]) -> np.ndarray:
        self.step_count += 1
        # extract demand signal (from env observation) as proxy for sales
        demand = observation.get('demand', np.zeros(self.c.product_catelogue_size, dtype=np.float32))
        return self._update_from_demand(current_prices, demand)

    def _update_from_demand(self, prices: np.ndarray, sold: np.ndarray) -> np.ndarray:
        # log transforms (add 1 to handle zeros)
        logp = np.log(np.clip(prices, 1e-3, None)).astype(np.float32)
        logq = np.log(sold + 1.0).astype(np.float32)
        # EWMA moments for per-product regression: logq ≈ a + e*logp
        a = self.ewma
        dp = logp - self.mu_logp
        dq = logq - self.mu_logq
        self.mu_logp = (1 - a) * self.mu_logp + a * logp
        self.mu_logq = (1 - a) * self.mu_logq + a * logq
        self.cov_pq = (1 - a) * self.cov_pq + a * (dp * dq)
        self.var_p = (1 - a) * self.var_p + a * (dp * dp + 1e-6)
        e_new = self.cov_pq / (self.var_p + 1e-6)
        self.e_hat = self._safe_elasticity(0.9 * self.e_hat + 0.1 * e_new)
        # profit-optimal price for isoelastic demand (if e < -1)
        e = self.e_hat
        p_star = self.unit_cost * (e / (e + 1.0))
        # smooth toward p_star
        new_prices = (1 - self.lr) * prices + self.lr * p_star
        # exploration (small random perturbations)
        if self.rng.random() < self.eps_explore:
            noise = self.rng.normal(0.0, self.explore_scale, size=new_prices.shape).astype(np.float32)
            new_prices = new_prices * (1.0 + noise)
        # apply business guardrails (max change + bounds)
        max_adj = self.c.max_price_adjustment
        ratio = np.clip(new_prices / (prices + 1e-6), 1 - max_adj, 1 + max_adj)
        new_prices = prices * ratio
        new_prices = np.clip(new_prices, self.c.system_min_price, self.c.system_max_price).astype(np.float32)
        return new_prices


class StaticPricingEngine(BasePricingEngine):
    """baseline: fixed prices throughout episode"""
    def __init__(self, constraints: BusinessLogicConstraints, seed: int = 0):
        super().__init__(constraints, seed)
        self.fixed_prices = None

    def reset(self):
        super().reset()
        self.fixed_prices = None

    def compute_prices(self, current_prices: np.ndarray, observation: Dict[str, Any]) -> np.ndarray:
        self.step_count += 1
        if self.fixed_prices is None:
            self.fixed_prices = current_prices.copy()
        return self.fixed_prices.copy()


class SimpleDemandEngine(BasePricingEngine):
    """demand-driven pricing: increase price when demand rises, decrease when it falls"""
    def __init__(self, constraints: BusinessLogicConstraints, seed: int = 0):
        super().__init__(constraints, seed)
        self.prev_demand = None
        self.lr = 0.05

    def reset(self):
        super().reset()
        self.prev_demand = None

    def compute_prices(self, current_prices: np.ndarray, observation: Dict[str, Any]) -> np.ndarray:
        self.step_count += 1
        demand = _extract_demand(observation, self.c.product_catalogue_size)
        if self.prev_demand is None:
            self.prev_demand = demand.copy()
            return current_prices.copy()
        # simple rule: if demand increases, raise price; if decreases, lower price
        delta_d = demand - self.prev_demand
        price_adj = self.lr * np.sign(delta_d) * np.abs(delta_d) / (np.abs(self.prev_demand) + 1.0)
        new_prices = current_prices * (1.0 + price_adj)
        self.prev_demand = demand.copy()
        # apply constraints
        max_adj = self.c.max_price_adjustment
        ratio = np.clip(new_prices / (current_prices + 1e-6), 1 - max_adj, 1 + max_adj)
        new_prices = current_prices * ratio
        return np.clip(new_prices, self.c.system_min_price, self.c.system_max_price).astype(np.float32)


class RandomWalkEngine(BasePricingEngine):
    """random walk pricing with mean reversion"""
    def __init__(self, constraints: BusinessLogicConstraints, seed: int = 0):
        super().__init__(constraints, seed)
        self.target_price = None
        self.volatility = 0.02

    def reset(self):
        super().reset()
        self.target_price = None

    def compute_prices(self, current_prices: np.ndarray, observation: Dict[str, Any]) -> np.ndarray:
        self.step_count += 1
        if self.target_price is None:
            self.target_price = current_prices.copy()
        # random walk with mean reversion toward target
        noise = self.rng.normal(0.0, self.volatility, size=current_prices.shape).astype(np.float32)
        reversion = 0.01 * (self.target_price - current_prices)
        new_prices = current_prices * (1.0 + noise) + reversion
        # apply constraints
        max_adj = self.c.max_price_adjustment
        ratio = np.clip(new_prices / (current_prices + 1e-6), 1 - max_adj, 1 + max_adj)
        new_prices = current_prices * ratio
        return np.clip(new_prices, self.c.system_min_price, self.c.system_max_price).astype(np.float32)


class ThompsonSamplingEngine(BasePricingEngine):
    """bayesian bandit approach per product treating price as discrete action"""
    def __init__(self, constraints: BusinessLogicConstraints, seed: int = 0):
        super().__init__(constraints, seed)
        self.n_price_levels = 5
        self.alpha = np.ones((self.c.product_catalogue_size, self.n_price_levels), dtype=np.float32)
        self.beta = np.ones((self.c.product_catalogue_size, self.n_price_levels), dtype=np.float32)
        self.price_grid = None
        self.last_actions = None

    def reset(self):
        super().reset()
        self.alpha = np.ones((self.c.product_catalogue_size, self.n_price_levels), dtype=np.float32)
        self.beta = np.ones((self.c.product_catalogue_size, self.n_price_levels), dtype=np.float32)
        self.price_grid = None
        self.last_actions = None

    def compute_prices(self, current_prices: np.ndarray, observation: Dict[str, Any]) -> np.ndarray:
        self.step_count += 1
        if self.price_grid is None:
            # define price grid per product
            lo = current_prices * 0.7
            hi = current_prices * 1.3
            self.price_grid = np.linspace(lo, hi, self.n_price_levels).T
        demand = _extract_demand(observation, self.c.product_catalogue_size)
        # update beliefs based on last action
        if self.last_actions is not None:
            for i in range(self.c.product_catalogue_size):
                a = self.last_actions[i]
                reward = demand[i]
                if reward > 0.5:
                    self.alpha[i, a] += reward
                else:
                    self.beta[i, a] += 1.0
        # thompson sampling: sample from posterior, pick best
        new_prices = np.zeros(self.c.product_catalogue_size, dtype=np.float32)
        actions = np.zeros(self.c.product_catalogue_size, dtype=int)
        for i in range(self.c.product_catalogue_size):
            theta = self.rng.beta(self.alpha[i], self.beta[i]).astype(np.float32)
            actions[i] = int(np.argmax(theta))
            new_prices[i] = self.price_grid[i, actions[i]]
        self.last_actions = actions
        return np.clip(new_prices, self.c.system_min_price, self.c.system_max_price).astype(np.float32)


def _extract_demand(observation: Dict[str, Any], n: int) -> np.ndarray:
    if "elasticity" in observation and isinstance(observation["elasticity"], dict):
        d = observation["elasticity"].get("demand")
        if d is not None:
            return np.asarray(d, dtype=np.float32)
    d = observation.get("demand")
    if d is not None:
        return np.asarray(d, dtype=np.float32)
    return np.zeros(n, dtype=np.float32)
