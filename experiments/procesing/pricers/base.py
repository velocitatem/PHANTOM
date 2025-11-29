from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd


class PricingFunction(ABC):
    """
    Abstract base for pricing functions.

    Defines mapping: f(Q_t, P_t, S_t, H_t) -> P_{t+1}

    Where:
        Q_t ∈ R^n: demand vector at time t
        P_t ∈ R^n: price vector at time t
        S_t: session features (behavioral signals, interactions)
        H_t = {Q_{t-k}, P_{t-k}, S_{t-k}}: historical state trajectory

    Objective:
        maximize E[R_T] = E[Σ P_t^T · Q_t]
        subject to:
            Q_t = g(P_t, S_t)  (demand response via elasticity)
            P_t ≥ C  (cost floor)
            minimize L_agent = R_oracle - R_observed
    """

    @abstractmethod
    def fit(self, historical_data: pd.DataFrame, **kwargs):
        """
        Offline training on historical data.

        Args:
            historical_data: DataFrame with elasticity, prices, demand signals
            **kwargs: additional training parameters
        """
        pass

    @abstractmethod
    def predict(self, state_space) -> np.ndarray:
        """
        Generate optimal prices given current state.

        Args:
            state_space: StateSpace object containing Q_t, P_t, S_t, H_t

        Returns:
            P_{t+1}: price vector in R^n
        """
        pass

    def update(self, observation: Dict[str, Any]):
        """
        Online learning update (optional).

        Args:
            observation: dict with {state, action, reward, next_state}
                - state: StateSpace before pricing decision
                - action: prices shown (P_t)
                - reward: revenue/conversion signal
                - next_state: StateSpace after user interaction
        """
        pass  # default: no online learning

    def get_params(self) -> Dict[str, Any]:
        """Return pricing function parameters for serialization."""
        return {}

    def set_params(self, params: Dict[str, Any]):
        """Load pricing function parameters from dict."""
        pass
