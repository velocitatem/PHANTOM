"""
Standard objective components and penalties.

This module provides common reward terms:
- PnLObjective: Basic profit and loss
- VolatilityPenalty: Penalize price volatility for UX
- HoldingCostPenalty: Inventory holding cost
- LostOpportunityCostPenalty: Stockout/missed fill cost
- InventoryRiskPenalty: Quadratic inventory risk (market making)
- SpreadCaptureReward: Bid-ask spread capture (market making)
"""
from __future__ import annotations
import numpy as np
from .base import BaseObjective
from ..types import Quote, InstrumentSet, StepMetrics, HiddenState, Observation
from ..math_util import inventory_penalty

class PnLObjective(BaseObjective):
    """Profit and loss reward (revenue - cost)."""

    def reward(self, quote: Quote, instruments: InstrumentSet,
               metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> float:
        return metrics.pnl

    def breakdown(self, quote: Quote, instruments: InstrumentSet,
                  metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> dict[str, float]:
        return {'pnl': metrics.pnl, 'revenue': metrics.revenue, 'cost': metrics.cost}

class VolatilityPenalty(BaseObjective):
    """Penalize price volatility for user experience."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def reward(self, quote: Quote, instruments: InstrumentSet,
               metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> float:
        return -self.scale * metrics.volatility

    def breakdown(self, quote: Quote, instruments: InstrumentSet,
                  metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> dict[str, float]:
        return {'volatility_penalty': -self.scale * metrics.volatility}

class HoldingCostPenalty(BaseObjective):
    """Penalty for inventory holding costs."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def reward(self, quote: Quote, instruments: InstrumentSet,
               metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> float:
        return -self.scale * metrics.position_cost

    def breakdown(self, quote: Quote, instruments: InstrumentSet,
                  metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> dict[str, float]:
        return {'holding_cost_penalty': -self.scale * metrics.position_cost}

class LostOpportunityCostPenalty(BaseObjective):
    """Penalty for lost sales due to stockouts or missed fills."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def reward(self, quote: Quote, instruments: InstrumentSet,
               metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> float:
        return -self.scale * metrics.lost_opportunity

    def breakdown(self, quote: Quote, instruments: InstrumentSet,
                  metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> dict[str, float]:
        return {'lost_opportunity_penalty': -self.scale * metrics.lost_opportunity}

class InventoryRiskPenalty(BaseObjective):
    """Quadratic inventory risk penalty (Avellaneda-Stoikov style).

    Penalty = gamma * sigma^2 * q^2 / 2, where q is total position.
    Encourages market makers to keep inventory near zero.
    """

    def __init__(self, gamma: float = 0.1, sigma: float = 1.0):
        self.gamma = gamma
        self.sigma = sigma

    def reward(self, quote: Quote, instruments: InstrumentSet,
               metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> float:
        if obs.position is None: return 0.0
        q = np.sum(obs.position)
        return -inventory_penalty(q, self.gamma, self.sigma)

    def breakdown(self, quote: Quote, instruments: InstrumentSet,
                  metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> dict[str, float]:
        return {'inventory_risk_penalty': self.reward(quote, instruments, metrics, hidden, obs)}

class SpreadCaptureReward(BaseObjective):
    """Reward for capturing bid-ask spread in market making."""

    def reward(self, quote: Quote, instruments: InstrumentSet,
               metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> float:
        return metrics.spread_capture

    def breakdown(self, quote: Quote, instruments: InstrumentSet,
                  metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> dict[str, float]:
        return {'spread_capture': metrics.spread_capture}
