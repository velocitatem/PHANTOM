from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd


class PricingFunction(ABC):
    """
    Abstract base for pricing functions.
    Objective:
        maximize E[R_T] = E[Σ P_t^T · Q_t]
        subject to:
            Q_t = g(P_t, S_t)  (demand response via elasticity)
            P_t ≥ C  (cost floor)
            minimize L_agent = R_oracle - R_observed
    """

    @abstractmethod
    def fit(self, *kwargs):
        """
        Offline training on historical data.
        This is where we can think about some maximization of expected revenue
        over historical trajectories to learn parameters of the pricing function.
        (This however we cover move in the RL side of things)

        """
        pass

    @abstractmethod
    def predict(self, *kwargs) -> np.ndarray:
        """
        Generate optimal prices given current state.
        This is an abstract method that transitions from τ -> P*
        which is the mapping from the trajectory to optimal prices under
        some subset of session grouping (so, per sessionId)
        """
        pass

    @abstractmethod
    def _get_features(self, *kwargs) -> np.ndarray:
        """
        Extract features from trajectory for pricing decision.
        Returns:
            np.ndarray of shape (n_products, n_features)
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
