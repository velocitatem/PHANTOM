import numpy as np
import pandas as pd
from procesing.pricers.base import PricingFunction


class StaticPricer(PricingFunction):
    """Static pricing: always return fixed base prices"""

    def __init__(self, base_prices: np.ndarray = None):
        self.base_prices = base_prices

    def fit(self, historical_data: pd.DataFrame):
        """Extract base prices from historical data"""
        if 'base_price' in historical_data.columns:
            self.base_prices = historical_data['base_price'].values
        elif 'price' in historical_data.columns:
            self.base_prices = historical_data['price'].values
        else:
            raise ValueError("historical_data must contain 'base_price' or 'price' column")
        return self

    def predict(self, state_space) -> np.ndarray:
        """Return static base prices regardless of state"""
        if self.base_prices is None:
            raise ValueError("Must call fit() or provide base_prices in constructor")
        return self.base_prices.copy()


class RandomPricer(PricingFunction):
    """Random pricing within bounds (for baseline comparison)"""

    def __init__(self, price_min: float = 50.0, price_max: float = 500.0, seed: int = None):
        self.price_min = price_min
        self.price_max = price_max
        self.seed = seed
        self.n_products = None
        self.rng = np.random.default_rng(seed)

    def fit(self, historical_data: pd.DataFrame):
        """Learn number of products"""
        self.n_products = len(historical_data)
        return self

    def predict(self, state_space) -> np.ndarray:
        """Generate random prices"""
        if self.n_products is None:
            self.n_products = len(state_space.demand)
        return self.rng.uniform(self.price_min, self.price_max, size=self.n_products)


class SimpleSurgePricer(PricingFunction):
    """
    Rule-based surge pricer adjusting prices via demand thresholds.
    Logic: if demand > high_threshold -> surge, if demand < low_threshold -> discount.
    Simpler and more controllable than curve fitting approaches.
    """

    def __init__(self,
                 base_prices: np.ndarray = None,
                 high_threshold: int = 10,
                 low_threshold: int = 2,
                 surge_multiplier: float = 1.2,
                 discount_multiplier: float = 0.9):
        self.base_prices = base_prices
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.surge_multiplier = surge_multiplier
        self.discount_multiplier = discount_multiplier

    def fit(self, historical_data: pd.DataFrame):
        """Extract base prices from product catalog or historical averages"""
        if 'base_price' in historical_data.columns:
            self.base_prices = historical_data['base_price'].values
        elif 'price' in historical_data.columns:
            self.base_prices = historical_data.groupby('productId')['price'].mean().values
        else:
            raise ValueError("historical_data must contain 'base_price' or 'price'")
        return self

    def predict(self, state_space) -> np.ndarray:
        """
        Adjust prices based on current demand using surge rules.
        state_space.demand: demand counts per product
        state_space.prices: current prices (fallback if base_prices not set)
        """
        current_prices = self.base_prices if self.base_prices is not None else state_space.prices
        demand = state_space.demand
        new_prices = current_prices.copy()

        high_mask = demand >= self.high_threshold
        new_prices[high_mask] *= self.surge_multiplier

        low_mask = demand <= self.low_threshold
        new_prices[low_mask] *= self.discount_multiplier

        return new_prices
