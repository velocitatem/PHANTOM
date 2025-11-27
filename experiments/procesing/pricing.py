r"""
Our state space comes as:
$Q_t in R^n$ - our demand at a time t
$P_t in R^n$ - prices at time t
$S_t$ some form of interaction session features

This is a single sate which we map under

$f: (Q, S, H) \to P_{t+1}$

With:

$H_t = \{Q_{t-k}, P_{t-k}, S_{t-k}\}$


We can have f be literally anything, analytical or learned or rule based or an RL policy.

Our goal is to mazimize the expected revenue:

$E[R_T] = E[\sum_{t=1}^T P_t^T \dot Q_t]$

subject to Q_t = g(P_t, S_t) : demand response to price (estimated via elasticity) and P_t ≥ C : prices above cost floor and additionally minimizing the following:

$L_{agent} = R_{oracle} - R_{observed}

where: R_oracle = revenue if we knew agent intentions (from recon session) and R_observed = revenue under current pricing policy f

I would start be defning a pricing function interface and standardizing how to train that based on historical data and define how to make it behave for online training (if we do that)

We also need to develop a solid benchmark with mapping revenue and full KPIs from session interactions to measure differences between different price learning methods
"""

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import os
from supabase import create_client, Client
from pipeline import interaction_pipeline, price_data_pipeline, elasticity_pipeline

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def expected_revenue(prices: np.ndarray, demand: np.ndarray) -> float:
    """Returns: expected revenue R_t = P_t^T * Q_t"""
    return float(np.dot(prices, demand))

class StateSpace:
    def __init__(self,
                 demand : np.ndarray, # at time t, only values (assuming aligned by productId order)
                 prices : np.ndarray, # at time t, only values (assuming aligned by productId order)
                 session_features : pd.DataFrame):
        self.demand = demand  # Q_t
        self.prices = prices  # P_t
        self.session_features = session_features  # S_t
        self.history = []  # H_t

class PricingFunction(BaseEstimator, TransformerMixin, ABC):
    def __init__(self):
        pass

    def fit(self, historical_data):
        """
        Train the pricing function based on historical data.
        historical_data: list of StateSpace instances with known outcomes
        """
        raise NotImplementedError("Train method must be implemented by subclass.")

    def transform(self, state_space) -> np.ndarray:
        """
        Predict the next prices given the current state space.
        state_space: StateSpace instance
        Returns: predicted prices P_{t+1}
        """
        raise NotImplementedError("Predict method must be implemented by subclass.")


class SimpleLinearPricingFunction(PricingFunction):
    def __init__(self, price_sensitivity: float = -0.1):
        super().__init__()
        self.price_sensitivity = price_sensitivity  # simple coefficient

    def fit(self, historical_data):
        return self

    def transform(self, state_space: StateSpace) -> np.ndarray:
        # Simple linear adjustment: P_{t+1} = P_t + sensitivity * Q_t
        new_prices = state_space.prices + self.price_sensitivity * state_space.demand # this is not great
        return np.maximum(new_prices, 0)

# Example usage:
if __name__ == "__main__":
    store_mode = 'hotel'
    interaction_data = interaction_pipeline.fit_transform(None)
    price_data = price_data_pipeline.fit_transform(None)

    elasticity_df = elasticity_pipeline(interaction_data, price_data, window_size="30s", store_mode=store_mode)

    # fetch all products with base prices from database
    products_resp = supabase.table(f'{store_mode}_products').select("id, metadata").execute()
    products_df = pd.DataFrame(products_resp.data)

    # extract base_price from metadata
    products_df['base_price'] = products_df['metadata'].apply(lambda m: m.get('base_price', 0) if isinstance(m, dict) else 0)
    products_df = products_df.rename(columns={'id': 'productId'})[['productId', 'base_price']]

    # override with logged prices where available
    if not price_data.empty:
        if 'ts' in price_data.columns and not pd.api.types.is_datetime64_any_dtype(price_data['ts']):
            price_data['ts'] = pd.to_datetime(price_data['ts'])

        # get latest logged price per product
        price_logs_agg = price_data.sort_values('ts').groupby('productId', as_index=False).last()

        # merge: start with all products (base prices), override with logged prices
        products_df = products_df.merge(
            price_logs_agg[['productId', 'price']],
            on='productId',
            how='left'
        )
        products_df['final_price'] = products_df['price'].fillna(products_df['base_price'])
    else:
        products_df['final_price'] = products_df['base_price']

    # merge with elasticity
    if elasticity_df is not None and not elasticity_df.empty:
        price_data_merged = products_df[['productId', 'final_price']].merge(
            elasticity_df[['productId', 'elasticity']],
            on='productId',
            how='left'
        ).fillna({'elasticity': 0.0})

        prices = price_data_merged['final_price'].values
        elasticities = price_data_merged['elasticity'].values
    else:
        prices = np.array([])
        elasticities = np.array([])

    print(elasticities)
    print(prices)

    state_space = StateSpace(
        demand=elasticities,
        prices=prices,
        session_features=interaction_data
    )

    pricing_function = SimpleLinearPricingFunction(price_sensitivity=-0.05)
    pricing_function.fit([])  # No training data for simple model
    predicted_prices = pricing_function.transform(state_space)

    print("Predicted Prices:", predicted_prices)
