from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class PricingFunction(ABC):
    """
    Abstract base for pricing functions.
    Defines the mapping f: StateSpace -> prices
    """

    @abstractmethod
    def fit(self, historical_data: pd.DataFrame):
        """Train/calibrate the pricing function on historical data"""
        pass

    @abstractmethod
    def predict(self, state_space) -> np.ndarray:
        """
        Generate prices given current state space.

        Args:
            state_space: StateSpace object containing demand, prices, session features

        Returns:
            prices: price vector P_{t+1} in R^n
        """
        pass
