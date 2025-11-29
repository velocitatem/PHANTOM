import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from experiments.procesing.pricers.simple import StaticPricer
from procesing.steps.base import BaseContextStep
from procesing.pricers import ElasticityBasedPricer

@dataclass
class StateSpace:
    """
    State representation for pricing functions.

    Components:
        Q_t: demand ∈ R^n (current demand signal per product)
        P_t: prices ∈ R^n (current/base prices)
        S_t: session_features (behavioral signals, interaction data)
        H_t: history = {Q_{t-k}, P_{t-k}, S_{t-k}} for k in [1, history_length]

    Additionally stores:
        - product_ids: product identifiers (n,)
        - elasticity: price elasticity per product (n,)
        - metadata: arbitrary context (experiment_id, timestamp, etc.)
    """
    demand: np.ndarray  # Q_t ∈ R^n
    prices: np.ndarray  # P_t ∈ R^n
    session_features: pd.DataFrame = field(default_factory=pd.DataFrame)  # S_t

    # augmented state components
    product_ids: Optional[np.ndarray] = None
    elasticity: Optional[np.ndarray] = None

    # historical trajectory H_t = {(Q_{t-k}, P_{t-k}, S_{t-k})}
    history: List[Dict[str, Any]] = field(default_factory=list)

    # metadata for context
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate dimensions."""
        n = len(self.demand)
        assert len(self.prices) == n, "demand and prices must have same dimension"
        if self.elasticity is not None:
            assert len(self.elasticity) == n, "elasticity must match dimension"
        if self.product_ids is not None:
            assert len(self.product_ids) == n, "product_ids must match dimension"

    @property
    def n_products(self) -> int:
        """Number of products in state space."""
        return len(self.demand)

    def add_history(self, q: np.ndarray, p: np.ndarray, s: pd.DataFrame, max_length: int = 10):
        """Append historical state to trajectory H_t."""
        self.history.append({'demand': q, 'prices': p, 'session_features': s})
        if len(self.history) > max_length:
            self.history.pop(0)

    def get_history_window(self, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve last k historical states."""
        return self.history[-k:] if len(self.history) >= k else self.history


class BuildStateSpaceStep(BaseContextStep):
    """
    Build state space from elasticity, demand, and price data.

    Input: elasticity_df [productId, elasticity, ...], optional demand_df
    Output: StateSpace instance with Q_t, P_t, elasticity, product_ids
    """

    def transform(self, elasticity_df: pd.DataFrame, demand_df: Optional[pd.DataFrame] = None):
        products = self.context.products

        # extract base prices from product metadata
        products_with_prices = products.copy()
        if 'metadata' in products_with_prices.columns:
            products_with_prices['base_price'] = products_with_prices['metadata'].apply(
                lambda m: m.get('base_price', 0) if isinstance(m, dict) else 0
            )
        else:
            products_with_prices['base_price'] = 0

        # merge with elasticity
        merged = products_with_prices[['id', 'base_price']].rename(
            columns={'id': 'productId'}
        ).merge(
            elasticity_df[['productId', 'elasticity']],
            on='productId',
            how='left'
        ).fillna({'elasticity': 0.0, 'base_price': 0.0})

        # merge with demand if provided, else use default
        if demand_df is not None and 'demand' in demand_df.columns:
            merged = merged.merge(
                demand_df[['productId', 'demand']],
                on='productId',
                how='left'
            ).fillna({'demand': 0.0})
            demand_vector = merged['demand'].values
        else:
            # default: uniform demand or use elasticity as proxy
            demand_vector = np.ones(len(merged)) * 10.0

        return StateSpace(
            demand=demand_vector,
            prices=merged['base_price'].values,
            session_features=pd.DataFrame(),
            product_ids=merged['productId'].values,
            elasticity=merged['elasticity'].values,
            metadata={'timestamp': pd.Timestamp.now().isoformat()}
        )


class FitPricingFunctionStep(BaseContextStep):
    """
    Fit pricing function using data.
    Input: pricing_data
    Output: fitted pricing function instance
    """

    def transform(self, pricing_data: pd.DataFrame):
        pricing_class = self.context.config.get('pricing_function_class', StaticPricer)
        pricing_params = self.context.config.get('pricing_function_params', {})

        pricer = pricing_class(**pricing_params)
        pricer.fit(pricing_data)

        return pricer


class PredictPricesStep(BaseContextStep):
    """
    Predict optimal prices using fitted pricing function.
    Input: (pricer, state_space)
    Output: prices_df [productId, predicted_price]
    """

    def transform(self, data: tuple):
        pricer, state_space = data

        products = self.context.products
        product_ids = products['id'].values

        predicted_prices = pricer.predict(state_space)

        return pd.DataFrame({
            'productId': product_ids,
            'predicted_price': predicted_prices
        })
