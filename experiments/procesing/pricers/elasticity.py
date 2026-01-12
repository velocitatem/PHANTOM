import numpy as np
import pandas as pd
from procesing.pricers.base import PricingFunction


class ElasticityBasedPricer(PricingFunction):
    """
    Pricing based on demand elasticity estimates.
    f(Q, S) = base_price * (1 + alpha * elasticity * demand_deviation)
    """

    def __init__(self, alpha: float = 0.1, price_floor: float = 0.0, price_ceil: float = np.inf):
        self.alpha = alpha
        self.price_floor = price_floor
        self.price_ceil = price_ceil
        self.elasticity = None
        self.base_prices = None
        self.mean_demand = None

    def fit(self, historical_data: pd.DataFrame):
        """
        Calibrate from historical elasticity estimates.
        Expects: [productId, elasticity, base_price, mean_demand]
        """
        if 'elasticity' not in historical_data.columns:
            raise ValueError("historical_data must contain 'elasticity' column")

        self.elasticity = historical_data['elasticity'].values
        self.base_prices = (historical_data['base_price'].values
                           if 'base_price' in historical_data.columns
                           else np.ones(len(historical_data)) * 100)
        self.mean_demand = (historical_data['mean_demand'].values
                           if 'mean_demand' in historical_data.columns
                           else np.ones(len(historical_data)) * 10)
        return self

    def predict(self, state_space) -> np.ndarray:
        """
        Adjust prices based on demand deviation and elasticity.
        Higher demand -> increase price (but less for elastic goods)
        """
        if self.elasticity is None:
            raise ValueError("Must call fit() before predict()")

        demand = np.asarray(state_space.demand)
        if len(demand) != len(self.elasticity):
            raise ValueError(f"Demand vector size {len(demand)} != elasticity size {len(self.elasticity)}")

        # compute demand deviation from mean
        demand_dev = (demand - self.mean_demand) / (self.mean_demand + 1e-6)

        # adjust price: if demand high and elastic, don't increase much
        # if demand high and inelastic, increase more
        price_multiplier = 1 + self.alpha * np.abs(self.elasticity) * demand_dev
        prices = self.base_prices * price_multiplier

        # enforce bounds
        prices = np.clip(prices, self.price_floor, self.price_ceil)
        return prices

    def _get_features(self, state_space=None) -> np.ndarray:
        """Extract elasticity, demand, and demand deviation for each product"""
        if state_space is None or self.elasticity is None:
            n = len(self.elasticity) if self.elasticity is not None else 0
            return np.zeros((n, 3))

        demand = np.asarray(state_space.demand)
        demand_dev = (demand - self.mean_demand) / (self.mean_demand + 1e-6)
        return np.column_stack([self.elasticity, demand, demand_dev])
