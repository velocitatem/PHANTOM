import numpy as np
import pandas as pd
from .base import BaseContextStep
from ..pricing import ElasticityBasedPricingFunction

class StateSpace:
    """State representation for pricing functions"""
    def __init__(self,
                 demand: np.ndarray,
                 prices: np.ndarray,
                 session_features: pd.DataFrame = None):
        self.demand = demand
        self.prices = prices
        self.session_features = session_features if session_features is not None else pd.DataFrame()


class BuildStateSpaceStep(BaseContextStep):
    """
    Build state space from elasticity and price data.
    Input: elasticity_df
    Output: StateSpace instance
    """

    def transform(self, elasticity_df: pd.DataFrame):
        products = self.context.products

        # fetch current/base prices from product metadata
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

        return StateSpace(
            demand=merged['elasticity'].values,
            prices=merged['base_price'].values,
            session_features=pd.DataFrame()
        )


class FitPricingFunctionStep(BaseContextStep):
    """
    Fit pricing function using elasticity data.
    Input: elasticity_df
    Output: fitted pricing function instance
    """

    def transform(self, elasticity_df: pd.DataFrame):
        pricing_class = self.context.config.get('pricing_function_class', ElasticityBasedPricingFunction)
        pricing_params = self.context.config.get('pricing_function_params', {})

        pricer = pricing_class(**pricing_params)
        pricer.fit(elasticity_df)

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

        predicted_prices = pricer.transform(state_space, product_ids)

        return pd.DataFrame({
            'productId': product_ids,
            'predicted_price': predicted_prices
        })
