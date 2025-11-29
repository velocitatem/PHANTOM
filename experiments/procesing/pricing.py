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
from dotenv import load_dotenv
load_dotenv()
from supabase import create_client, Client

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
        self.price_sensitivity = price_sensitivity

    def fit(self, historical_data):
        return self

    def transform(self, state_space: StateSpace) -> np.ndarray:
        new_prices = state_space.prices + self.price_sensitivity * state_space.demand
        return np.maximum(new_prices, 0)


class ElasticityBasedPricingFunction(PricingFunction):
    """
    Revenue-maximizing pricing using elasticity estimates.

    For each product, optimal price P* maximizes R = P * Q(P)
    where Q(P) follows power law: Q(P) = Q_0 * (P/P_0)^ε

    Taking derivative dR/dP = 0 gives optimal markup:
    P* = P_0 * (1 + 1/ε) if ε < -1 (elastic)

    For inelastic demand (|ε| < 1), we apply bounded markup.
    """

    def __init__(self,
                 cost_floor: float = 0.5,
                 max_markup: float = 2.0,
                 min_markup: float = 1.0,
                 inelastic_markup: float = 1.3):
        super().__init__()
        self.cost_floor = cost_floor  # prices as fraction of base
        self.max_markup = max_markup  # max price = base * max_markup
        self.min_markup = min_markup  # min price = base * min_markup
        self.inelastic_markup = inelastic_markup  # default for |ε| < 1
        self.elasticity_map = {}  # productId -> elasticity

    def fit(self, elasticity_df: pd.DataFrame):
        """
        Args:
            elasticity_df: df with [productId, elasticity, std_error, n_obs]
        """
        if elasticity_df is not None and not elasticity_df.empty:
            self.elasticity_map = dict(zip(
                elasticity_df['productId'],
                elasticity_df['elasticity']
            ))
        return self

    def transform(self, state_space: StateSpace, product_ids: np.ndarray = None) -> np.ndarray:
        """
        Args:
            state_space: current state (prices = base prices)
            product_ids: array of productIds aligned with state_space.prices

        Returns:
            optimized prices P_{t+1}
        """
        base_prices = state_space.prices

        if product_ids is None:
            # fallback: use positional index as productId (not ideal)
            product_ids = np.arange(len(base_prices))

        new_prices = np.zeros_like(base_prices)

        for i, (base_p, pid) in enumerate(zip(base_prices, product_ids)):
            elasticity = self.elasticity_map.get(pid, 0.0)

            if elasticity < -1:  # elastic demand
                # optimal markup: (1 + 1/ε)
                markup = 1 + (1 / elasticity)
                optimal_p = base_p * markup
            elif elasticity > -1 and elasticity < 0:  # inelastic
                # conservative markup
                optimal_p = base_p * self.inelastic_markup
            else:  # ε ≥ 0 (demand increases with price, or no data)
                # no elasticity data or anomalous, keep base price
                optimal_p = base_p

            # apply bounds
            optimal_p = np.clip(
                optimal_p,
                base_p * self.min_markup,
                base_p * self.max_markup
            )
            optimal_p = max(optimal_p, self.cost_floor)

            new_prices[i] = optimal_p

        return new_prices


class ContextualElasticityPricing(PricingFunction):
    """
    Revenue optimization with contextual adjustments based on session features.

    Combines elasticity-based pricing with surge/demand-based multipliers.
    """

    def __init__(self,
                 base_pricer: ElasticityBasedPricingFunction = None,
                 demand_sensitivity: float = 0.1,
                 surge_threshold: float = 0.7):
        super().__init__()
        self.base_pricer = base_pricer or ElasticityBasedPricingFunction()
        self.demand_sensitivity = demand_sensitivity
        self.surge_threshold = surge_threshold

    def fit(self, elasticity_df: pd.DataFrame):
        self.base_pricer.fit(elasticity_df)
        return self

    def transform(self, state_space: StateSpace, product_ids: np.ndarray = None) -> np.ndarray:
        # get base optimal prices from elasticity
        base_optimal = self.base_pricer.transform(state_space, product_ids)

        # compute surge multiplier from demand
        if len(state_space.demand) > 0:
            demand_normalized = state_space.demand / (state_space.demand.max() + 1e-8)
            surge_multiplier = 1 + self.demand_sensitivity * np.maximum(
                demand_normalized - self.surge_threshold, 0
            )
        else:
            surge_multiplier = np.ones_like(base_optimal)

        return base_optimal * surge_multiplier

# Example usage:
if __name__ == "__main__":
    from pipeline import interaction_pipeline, price_data_pipeline, elasticity_pipeline

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
