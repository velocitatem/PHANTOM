"""
Observation construction with demand censoring.

This module provides the ObservationBuilder that constructs agent observations
from step data. The key invariant is that observations only contain censored
data (fills) and never true demand, ensuring proper research conditions.

The ObservationConfig controls what is included in observations:
- Position visibility
- Market/competitor visibility
- Demand proxy method
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .types import Quote, InstrumentSet, StepLogs, StepMetrics, MarketState, HiddenState, Observation

@dataclass
class ObservationConfig:
    """Configuration for observation construction.

    Attributes:
        include_position: Include current position in observation
        include_market: Include market/competitor state in observation
        mask_true_demand: If True, observation excludes true demand (research mode)
        demand_proxy: Method for demand proxy ('fills', 'exposures', 'weighted')
        exposure_weights: Weights for weighted demand proxy
    """
    include_position: bool = True
    include_market: bool = True
    mask_true_demand: bool = True
    demand_proxy: str = 'fills'
    exposure_weights: dict[str, float] | None = None

class DefaultObservationBuilder:
    """Constructs censored observations for the agent.

    Ensures the key research invariant: observations contain only
    censored fills (realized sales), never true demand. True demand
    is placed in the info dict for research analysis only.
    """

    def __init__(self, cfg: ObservationConfig | None = None):
        self.cfg = cfg or ObservationConfig()

    def build(self, quote: Quote, instruments: InstrumentSet, logs: StepLogs,
              metrics: StepMetrics, market: MarketState | None,
              hidden: HiddenState, mask_demand: bool, t: int) -> Observation:
        n = instruments.n
        cfg = self.cfg

        # always show censored fills
        fills = logs.censored_fills if logs.censored_fills is not None else np.zeros(n)

        # compute exposures from logs
        if logs.events:
            exposures = np.zeros(n)
            for e in logs.events:
                if e.instrument_id is not None:
                    exposures[e.instrument_id] += 1
        else:
            exposures = logs.aggregates.get('exposures', np.zeros(n))

        # position - only if configured and available
        position = None
        if cfg.include_position and instruments.position is not None:
            position = instruments.position.copy()

        # market state - only if configured
        obs_market = market if cfg.include_market else None

        return Observation(
            quotes=quote.prices.copy(),
            position=position,
            fills=fills,
            exposures=exposures,
            market=obs_market,
            t=t
        )

    def make_space(self, n_instruments: int, include_market: bool = True) -> dict:
        """Returns dict describing observation space for gym"""
        space = {
            'quotes': {'shape': (n_instruments,), 'low': 0, 'high': np.inf},
            'fills': {'shape': (n_instruments,), 'low': 0, 'high': np.inf},
            'exposures': {'shape': (n_instruments,), 'low': 0, 'high': np.inf},
        }
        if self.cfg.include_position:
            space['position'] = {'shape': (n_instruments,), 'low': -np.inf, 'high': np.inf}
        if include_market:
            space['competitor_quotes'] = {'shape': (n_instruments,), 'low': 0, 'high': np.inf}
        return space
