import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from procesing.pricers.simple import StaticPricer
from procesing.steps.base import BaseContextStep
from procesing.pricers import ElasticityBasedPricer

class State:
    def __init__(self,
                 last_action : str,
                 last_productId : str,
                 last_price : float,
                 session_features : np.ndarray
                 ):
        pass



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
