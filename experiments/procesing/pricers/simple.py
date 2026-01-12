import numpy as np
import pandas as pd
from procesing.pricers.base import PricingFunction


def session_features_to_demand(session_features: pd.DataFrame) -> float:
    """
    Map session behavioral features to demand proxy.
    THIS is the critical θ̂ → D transformation for rule-based pricing.

    Logic:
      - High velocity → agent behavior → price up (revenue recovery)
      - High cart ratio → purchase intent → price up
      - Low activity → discount to convert

    Returns: demand proxy score (0-20 range, higher = more demand)
    """
    if session_features.empty:
        return 1.0

    feat = session_features.iloc[0] if len(session_features) > 0 else {}

    velocity = feat.get('interaction_velocity', 0)
    cart_ratio = feat.get('cart_to_view_ratio', 0)
    item_views = feat.get('item_views', 0)
    cart_adds = feat.get('cart_adds', 0)

    # baseline demand
    demand = 1.0

    # agent detection: high velocity → treat as high "demand" to price up
    if velocity > 2.0:
        demand += 10.0  # strong agent signal

    # conversion intent: cart interaction → price up
    if cart_ratio > 0.1 or cart_adds > 0:
        demand += 5.0

    # browsing depth: many views → interest signal
    if item_views > 3:
        demand += min(item_views, 5.0)

    return min(demand, 20.0)  # cap at 20


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

    def fit(self, market_data: pd.DataFrame):
        """Extract base prices from product catalog or historical averages"""
        self.base_prices = market_data['base_price'].to_numpy() if 'base_price' in market_data.columns else market_data['price'].values
        return self

    def predict(self, state_space) -> np.ndarray:
        """
        Adjust prices based on current demand using surge rules.
        state_space.demand: demand proxy per product (from session features)
        state_space.prices: base prices
        """
        demand = np.asarray(state_space.demand) if state_space and hasattr(state_space, 'demand') else np.array([0])
        base = np.asarray(state_space.prices) if state_space and hasattr(state_space, 'prices') else self.base_prices

        if base is None:
            base = np.ones(len(demand)) * 99.99

        new_prices = base.copy()
        high_mask = demand >= self.high_threshold
        new_prices[high_mask] *= self.surge_multiplier

        low_mask = demand <= self.low_threshold
        new_prices[low_mask] *= self.discount_multiplier

        return new_prices
