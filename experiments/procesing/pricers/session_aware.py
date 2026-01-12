"""
Session-aware pricing functions that leverage behavioral features S_t.
These pricers aim to minimize L_agent = R_oracle - R_observed.
"""
import numpy as np
import pandas as pd
from procesing.pricers.base import PricingFunction
from procesing.pricers.elasticity import ElasticityBasedPricer


class SessionAwarePricer(PricingFunction):
    """
    Extends elasticity-based pricing with session behavioral signals.

    f(Q, P, S) = base_price * elasticity_factor * session_factor

    Where session_factor adjusts for:
        - interaction_velocity (agent detection proxy)
        - product_view_depth (interest signal)
        - cart_to_view_ratio (conversion intent)

    Strategy: charge higher prices to suspected agents (high velocity)
    to recover oracle revenue from reconnaissance sessions.
    """

    def __init__(self,
                 alpha: float = 0.1,
                 beta_velocity: float = 0.05,
                 beta_attention: float = 0.03,
                 agent_velocity_threshold: float = 5.0,
                 agent_markup: float = 1.2,
                 price_floor: float = 0.0,
                 price_ceil: float = np.inf):
        """
        Args:
            alpha: elasticity sensitivity
            beta_velocity: interaction velocity weight
            beta_attention: product attention weight
            agent_velocity_threshold: velocity above which to apply agent markup
            agent_markup: price multiplier for suspected agent sessions
            price_floor, price_ceil: price bounds
        """
        self.alpha = alpha
        self.beta_velocity = beta_velocity
        self.beta_attention = beta_attention
        self.agent_velocity_threshold = agent_velocity_threshold
        self.agent_markup = agent_markup
        self.price_floor = price_floor
        self.price_ceil = price_ceil

        # fitted parameters
        self.elasticity = None
        self.base_prices = None
        self.mean_demand = None

    def fit(self, historical_data: pd.DataFrame, **kwargs):
        """Calibrate from historical elasticity data."""
        if 'elasticity' not in historical_data.columns:
            raise ValueError("historical_data must contain 'elasticity'")

        self.elasticity = historical_data['elasticity'].values
        self.base_prices = (historical_data['base_price'].values
                           if 'base_price' in historical_data.columns
                           else np.ones(len(historical_data)) * 100)
        self.mean_demand = (historical_data['mean_demand'].values
                           if 'mean_demand' in historical_data.columns
                           else np.ones(len(historical_data)) * 10)
        return self

    def predict(self, state_space) -> np.ndarray:
        """Generate prices with session awareness."""
        if self.elasticity is None:
            raise ValueError("Must call fit() before predict()")

        demand = np.asarray(state_space.demand)
        n_products = len(demand)

        # base elasticity-driven pricing
        demand_dev = (demand - self.mean_demand) / (self.mean_demand + 1e-6)
        elasticity_factor = 1 + self.alpha * np.abs(self.elasticity) * demand_dev

        # session-aware adjustments
        session_factor = np.ones(n_products)

        if not state_space.session_features.empty:
            sf = state_space.session_features.iloc[0]  # single session features

            # agent detection via velocity
            velocity = sf.get('interaction_velocity', 0.0)
            if velocity > self.agent_velocity_threshold:
                # suspected agent: apply markup to recover oracle revenue
                session_factor *= self.agent_markup

            # attention signal: higher view depth -> user interested -> can charge more
            view_depth = sf.get('product_view_depth', 0)
            if view_depth > 0:
                attention_boost = 1 + self.beta_attention * np.log1p(view_depth)
                session_factor *= attention_boost

            # cart presence: if user has items in cart, slightly increase prices
            cart_to_view = sf.get('cart_to_view_ratio', 0.0)
            if cart_to_view > 0.1:
                session_factor *= (1 + 0.02)  # small boost for conversion intent

        prices = self.base_prices * elasticity_factor * session_factor
        prices = np.clip(prices, self.price_floor, self.price_ceil)

        return prices

    def _get_features(self, state_space=None) -> np.ndarray:
        """Extract elasticity, demand, and session features"""
        if state_space is None or self.elasticity is None:
            n = len(self.elasticity) if self.elasticity is not None else 0
            return np.zeros((n, 5))

        demand = np.asarray(state_space.demand)
        n_products = len(demand)

        # extract session features
        velocity = 0.0
        view_depth = 0.0
        cart_to_view = 0.0

        if not state_space.session_features.empty:
            sf = state_space.session_features.iloc[0]
            velocity = sf.get('interaction_velocity', 0.0)
            view_depth = sf.get('product_view_depth', 0.0)
            cart_to_view = sf.get('cart_to_view_ratio', 0.0)

        # broadcast session features to all products
        features = np.column_stack([
            self.elasticity,
            demand,
            np.full(n_products, velocity),
            np.full(n_products, view_depth),
            np.full(n_products, cart_to_view)
        ])
        return features


class ProductSpecificSessionPricer(PricingFunction):
    """
    Session-aware pricer with product-specific demand signals.

    Uses S_t to extract per-product interaction counts and adjusts pricing
    for products the user has already viewed/hovered.

    Strategy: products viewed multiple times = high interest -> price up
    """

    def __init__(self,
                 alpha: float = 0.1,
                 view_boost: float = 0.02,
                 max_view_boost: float = 0.15,
                 price_floor: float = 0.0,
                 price_ceil: float = np.inf):
        self.alpha = alpha
        self.view_boost = view_boost
        self.max_view_boost = max_view_boost
        self.price_floor = price_floor
        self.price_ceil = price_ceil

        self.elasticity = None
        self.base_prices = None
        self.mean_demand = None
        self.product_ids = None

    def fit(self, historical_data: pd.DataFrame, **kwargs):
        if 'elasticity' not in historical_data.columns or 'productId' not in historical_data.columns:
            raise ValueError("historical_data must contain 'elasticity' and 'productId'")

        self.elasticity = historical_data['elasticity'].values
        self.base_prices = (historical_data['base_price'].values
                           if 'base_price' in historical_data.columns
                           else np.ones(len(historical_data)) * 100)
        self.mean_demand = (historical_data['mean_demand'].values
                           if 'mean_demand' in historical_data.columns
                           else np.ones(len(historical_data)) * 10)
        self.product_ids = historical_data['productId'].values
        return self

    def predict(self, state_space) -> np.ndarray:
        if self.elasticity is None:
            raise ValueError("Must call fit() before predict()")

        demand = np.asarray(state_space.demand)
        n_products = len(demand)

        # base pricing
        demand_dev = (demand - self.mean_demand) / (self.mean_demand + 1e-6)
        base_prices = self.base_prices * (1 + self.alpha * np.abs(self.elasticity) * demand_dev)

        # product-specific session adjustments
        if not state_space.session_features.empty and state_space.product_ids is not None:
            # extract product interaction counts from session metadata
            # (this would require session features to include per-product signals)
            # for now, use uniform boost as placeholder
            # TODO: extend session feature extraction to include product-specific counts
            pass

        prices = np.clip(base_prices, self.price_floor, self.price_ceil)
        return prices

    def _get_features(self, state_space=None) -> np.ndarray:
        """Extract elasticity and demand features for product-specific pricing"""
        if state_space is None or self.elasticity is None:
            n = len(self.elasticity) if self.elasticity is not None else 0
            return np.zeros((n, 2))

        demand = np.asarray(state_space.demand)
        return np.column_stack([self.elasticity, demand])
