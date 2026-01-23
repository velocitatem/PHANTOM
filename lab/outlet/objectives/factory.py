"""
Factory functions for creating objectives.

Provides:
- make_objective: Create single objective by name
- make_composite: Create weighted combination of objectives
- retail_objective: Default objective for retail pricing
- market_making_objective: Default objective for market making
"""
from __future__ import annotations
from .base import BaseObjective, CompositeObjective
from .penalties import (PnLObjective, VolatilityPenalty, HoldingCostPenalty,
                        LostOpportunityCostPenalty, InventoryRiskPenalty, SpreadCaptureReward)

REGISTRY: dict[str, type[BaseObjective]] = {
    'pnl': PnLObjective,
    'volatility': VolatilityPenalty,
    'holding_cost': HoldingCostPenalty,
    'lost_opportunity': LostOpportunityCostPenalty,
    'inventory_risk': InventoryRiskPenalty,
    'spread_capture': SpreadCaptureReward,
}

def make_objective(name: str, **kwargs) -> BaseObjective:
    """Create an objective by name.

    Args:
        name: Objective name (pnl, volatility, holding_cost, lost_opportunity,
              inventory_risk, spread_capture)
        **kwargs: Passed to objective constructor

    Returns:
        Instantiated objective
    """
    if name not in REGISTRY:
        raise ValueError(f"Unknown objective: {name}. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name](**kwargs)

def make_composite(spec: list[tuple[str, float, dict]] | dict[str, float]) -> CompositeObjective:
    """Create composite objective from specification.

    Args:
        spec: Either:
            - list of (name, weight, kwargs) tuples for full control
            - dict of {name: weight} for simple cases

    Returns:
        CompositeObjective with specified components
    """
    objectives = []
    if isinstance(spec, dict):
        for name, weight in spec.items():
            objectives.append((make_objective(name), weight))
    else:
        for name, weight, kwargs in spec:
            objectives.append((make_objective(name, **kwargs), weight))
    return CompositeObjective(objectives)

def retail_objective(volatility_weight: float = 0.1, holding_weight: float = 0.5,
                     stockout_weight: float = 0.3) -> CompositeObjective:
    """Default objective for retail dynamic pricing.

    Reward = PnL - volatility_weight*volatility - holding_weight*holding_cost
             - stockout_weight*lost_opportunity
    """
    return make_composite({
        'pnl': 1.0,
        'volatility': volatility_weight,
        'holding_cost': holding_weight,
        'lost_opportunity': stockout_weight,
    })

def market_making_objective(gamma: float = 0.1, sigma: float = 1.0) -> CompositeObjective:
    """Default objective for market making.

    Reward = PnL + 0.5*spread_capture - inventory_risk(gamma, sigma)
    """
    return CompositeObjective([
        (PnLObjective(), 1.0),
        (SpreadCaptureReward(), 0.5),
        (InventoryRiskPenalty(gamma=gamma, sigma=sigma), 1.0),
    ])
