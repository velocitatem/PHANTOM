"""
Base classes for reward objectives.

Objectives compute scalar rewards from step metrics. The CompositeObjective
allows combining multiple objectives with weights for multi-objective optimization.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from ..types import Quote, InstrumentSet, StepMetrics, HiddenState, Observation

class BaseObjective(ABC):
    """Abstract base class for reward objectives.

    Subclasses must implement reward() and breakdown() methods.
    """

    @abstractmethod
    def reward(self, quote: Quote, instruments: InstrumentSet,
               metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> float: ...

    @abstractmethod
    def breakdown(self, quote: Quote, instruments: InstrumentSet,
                  metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> dict[str, float]: ...

class CompositeObjective(BaseObjective):
    """Weighted sum of multiple objectives.

    Allows combining multiple reward terms (e.g., PnL - holding_cost - volatility).

    Args:
        objectives: List of (objective, weight) tuples
    """

    def __init__(self, objectives: list[tuple[BaseObjective, float]]):
        self.objectives = objectives

    def reward(self, quote: Quote, instruments: InstrumentSet,
               metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> float:
        return sum(w * obj.reward(quote, instruments, metrics, hidden, obs)
                   for obj, w in self.objectives)

    def breakdown(self, quote: Quote, instruments: InstrumentSet,
                  metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> dict[str, float]:
        bd = {}
        for obj, w in self.objectives:
            for k, v in obj.breakdown(quote, instruments, metrics, hidden, obs).items():
                bd[k] = w * v
        return bd
