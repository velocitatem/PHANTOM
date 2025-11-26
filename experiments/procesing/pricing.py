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
from pipeline import interaction_pipeline, price_data_pipeline, elasticity_pipeline

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
    interaction_data = interaction_pipeline.fit_transform(None)
    price_data = price_data_pipeline.fit_transform(None)

    elasticity_df = elasticity_pipeline(interaction_data, price_data, window_size="30s")

    # align elasticity with price data by productId, fill missing with 0
    if not price_data.empty and elasticity_df is not None and not elasticity_df.empty:
        # TODO FIX: we might have duplicate productIds
        price_data_merged = price_data.merge(
            elasticity_df[['productId', 'elasticity']],
            on='productId',
            how='left'
        ).fillna({'elasticity': 0.0}) # is it possible we are spilling some elasticities into other products

        prices = price_data_merged['price'].values
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
