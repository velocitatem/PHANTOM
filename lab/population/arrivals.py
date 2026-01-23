"""
Arrival models for generating demand opportunities.

This module provides different arrival processes:
- PoissonArrivalModel: Constant-rate memoryless arrivals
- HawkesArrivalModel: Self-exciting clustered arrivals (market orders)
- SessionArrivalModel: Retail browsing sessions with multi-product views

Each model implements the ArrivalModel protocol and generates Opportunity objects
that flow through the execution pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from uuid import uuid4
from ..outlet.types import Opportunity, InstrumentSet, MarketState, HiddenState
from ..outlet.constants import Side, OpportunityType
from ..outlet.math_util import poisson_arrivals, hawkes_intensity

@dataclass
class PoissonArrivalConfig:
    """Configuration for Poisson arrival process.

    Attributes:
        base_rate: Expected arrivals per unit time (scaled by hidden.true_demand_intensity)
        side_probs: Probability distribution over BUY/SELL sides
    """
    base_rate: float = 10.0
    side_probs: dict[Side, float] = None

    def __post_init__(self):
        if self.side_probs is None:
            self.side_probs = {Side.BUY: 1.0}

class PoissonArrivalModel:
    """Homogeneous Poisson arrival process.

    Generates arrivals at a constant rate (modulated by demand intensity).
    Suitable for stationary demand or as a baseline model.

    The actual arrival count follows Poisson(rate * dt * intensity).
    """

    def __init__(self, cfg: PoissonArrivalConfig | None = None):
        self.cfg = cfg or PoissonArrivalConfig()

    def sample(self, t: float, dt: float, instruments: InstrumentSet,
               market: MarketState | None, hidden: HiddenState,
               rng: np.random.Generator) -> list[Opportunity]:
        n_arrivals = poisson_arrivals(self.cfg.base_rate * hidden.true_demand_intensity, dt, rng)
        opps = []
        for _ in range(n_arrivals):
            inst_id = rng.integers(0, instruments.n)
            side = rng.choice(list(self.cfg.side_probs.keys()),
                              p=list(self.cfg.side_probs.values()))
            opps.append(Opportunity(
                id=str(uuid4())[:8], type=OpportunityType.SESSION,
                side=side, instrument_id=inst_id, size=1.0, t=t,
                context={'segment': 'default'}
            ))
        return opps

@dataclass
class HawkesArrivalConfig:
    """Configuration for Hawkes self-exciting process.

    Attributes:
        base_rate: Baseline arrival intensity
        alpha: Excitation strength (how much each arrival increases intensity)
        beta: Decay rate (how quickly excitation fades)
        side_probs: Probability distribution over BUY/SELL sides
    """
    base_rate: float = 5.0
    alpha: float = 0.5
    beta: float = 1.0
    side_probs: dict[Side, float] = None

    def __post_init__(self):
        if self.side_probs is None:
            self.side_probs = {Side.BUY: 0.5, Side.SELL: 0.5}

class HawkesArrivalModel:
    """Self-exciting Hawkes point process for clustered arrivals.

    Models order flow where arrivals cluster in time (momentum, herding).
    Intensity: lambda(t) = base + alpha * sum(exp(-beta * (t - t_i)))

    Used for market making scenarios where orders arrive in bursts.
    """

    def __init__(self, cfg: HawkesArrivalConfig | None = None):
        self.cfg = cfg or HawkesArrivalConfig()
        self._history: np.ndarray = np.array([])

    def sample(self, t: float, dt: float, instruments: InstrumentSet,
               market: MarketState | None, hidden: HiddenState,
               rng: np.random.Generator) -> list[Opportunity]:
        intensity = hawkes_intensity(
            self.cfg.base_rate * hidden.true_demand_intensity,
            self._history, self.cfg.alpha, self.cfg.beta, t
        )
        n_arrivals = poisson_arrivals(intensity, dt, rng)
        opps = []
        for i in range(n_arrivals):
            arr_t = t + rng.uniform(0, dt)
            self._history = np.append(self._history, arr_t)
            inst_id = rng.integers(0, instruments.n)
            side = rng.choice(list(self.cfg.side_probs.keys()),
                              p=list(self.cfg.side_probs.values()))
            opps.append(Opportunity(
                id=str(uuid4())[:8], type=OpportunityType.MARKET_ORDER,
                side=side, instrument_id=inst_id,
                size=rng.exponential(1.0), t=arr_t,
                context={'intensity': intensity}
            ))
        # decay old history
        self._history = self._history[self._history > t - 10]
        return opps

@dataclass
class SessionArrivalConfig:
    """Configuration for retail session arrivals.

    Attributes:
        sessions_per_step: Number of browsing sessions per step
        views_per_session: (min, max) product views per session
        contamination: Fraction of sessions that are scrapers/bots
    """
    sessions_per_step: int = 20
    views_per_session: tuple[int, int] = (1, 5)
    contamination: float = 0.0

class SessionArrivalModel:
    """Retail browsing session model with multi-product views.

    Each session views multiple products, generating one opportunity per view.
    Scraper sessions (controlled by contamination) view more products
    but convert at lower rates (handled by ExecutionModel).
    """

    def __init__(self, cfg: SessionArrivalConfig | None = None):
        self.cfg = cfg or SessionArrivalConfig()

    def sample(self, t: float, dt: float, instruments: InstrumentSet,
               market: MarketState | None, hidden: HiddenState,
               rng: np.random.Generator) -> list[Opportunity]:
        n_sessions = self.cfg.sessions_per_step
        contamination = hidden.contamination if hidden else self.cfg.contamination
        opps = []

        for _ in range(n_sessions):
            is_scraper = rng.random() < contamination
            n_views = rng.integers(*self.cfg.views_per_session)
            sid = str(uuid4())[:8]

            # scrapers view more products
            if is_scraper:
                n_views = min(instruments.n, n_views * 3)

            viewed = rng.choice(instruments.n, size=min(n_views, instruments.n), replace=False)
            for inst_id in viewed:
                opps.append(Opportunity(
                    id=f"{sid}-{inst_id}", type=OpportunityType.SESSION,
                    side=Side.BUY, instrument_id=int(inst_id), size=1.0, t=t,
                    context={'session_id': sid, 'is_scraper': is_scraper, 'n_views': n_views}
                ))
        return opps
