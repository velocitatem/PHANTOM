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
