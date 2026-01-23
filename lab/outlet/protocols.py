"""
Protocol definitions for pluggable simulator components.

This module defines the interfaces (Protocols) that allow swapping different
implementations for each stage of the Quote -> Arrival -> Execution -> Position
pipeline. All protocols use structural subtyping (duck typing).

Protocols:
    Mechanism: How quotes translate to executions (posted price, two-sided, auction)
    ArrivalModel: How opportunities arrive (Poisson, Hawkes, sessions)
    ExecutionModel: Acceptance probability given quote (elasticity, intensity)
    PositionModel: Inventory/position management and censorship
    MarketModel: Competitor/market dynamics
    ObservationBuilder: Constructs agent observations with censoring
    Objective: Computes reward from metrics
"""
from __future__ import annotations
from typing import Protocol, Any, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from .types import (Quote, Opportunity, Execution, InstrumentSet, StepLogs,
                        StepMetrics, HiddenState, Observation, MarketState)
    from .constants import LogLevel

class Mechanism(Protocol):
    """Defines how quotes translate to executions.

    The Mechanism is the core abstraction that differentiates pricing domains:
    - PostedPrice: single price, buyer decides to purchase or not
    - TwoSided: bid/ask spread, execution depends on distance from mid
    - Auction: reserve price affects win probability and clearing price

    Methods:
        apply_quote: Enforce constraints and return valid quote
        process_opportunity: Determine execution given opportunity and quote
    """
    def apply_quote(self, quote: Quote, instruments: InstrumentSet,
                    rng: np.random.Generator) -> Quote:
        """Apply mechanism-specific constraints to a quote.

        Args:
            quote: Raw quote from policy
            instruments: Current instrument set with costs/refs
            rng: Random generator for stochastic constraints

        Returns:
            Constrained quote satisfying mechanism rules (min margin, max delta, etc.)
        """
        ...

    def process_opportunity(self, opp: Opportunity, quote: Quote,
                            instruments: InstrumentSet, market: MarketState | None,
                            rng: np.random.Generator) -> Execution | None:
        """Process an opportunity against the current quote.

        Args:
            opp: Incoming opportunity (session, order, request)
            quote: Current posted quote
            instruments: Instrument set
            market: Current market state (competitor prices, mid-prices)
            rng: Random generator

        Returns:
            Execution if opportunity converts, None otherwise
        """
        ...

class ArrivalModel(Protocol):
    """Generates opportunities (demand arrivals) for each step.

    Different arrival models capture different demand dynamics:
    - Poisson: constant rate, memoryless
    - Hawkes: self-exciting, clustered arrivals
    - Session: retail browsing with multi-product views

    Methods:
        sample: Generate opportunities for a time interval
    """
    def sample(self, t: float, dt: float, instruments: InstrumentSet,
               market: MarketState | None, hidden: HiddenState,
               rng: np.random.Generator) -> list[Opportunity]:
        """Sample opportunities for time interval [t, t+dt).

        Args:
            t: Current time
            dt: Time interval length
            instruments: Available instruments
            market: Current market state
            hidden: Hidden state (contains demand intensity, contamination)
            rng: Random generator

        Returns:
            List of opportunities arriving in this interval
        """
        ...

class ExecutionModel(Protocol):
    """Computes acceptance/execution probability given quote and context.

    Different models capture different demand responses:
    - Elasticity: price sensitivity with competitor cross-effects
    - Intensity: distance-based fill probability (market making)
    - Logit: discrete choice model

    Methods:
        prob: Compute acceptance probability
        uncensor: Estimate true demand from censored fills
    """
    def prob(self, opp: Opportunity, quote: Quote, instruments: InstrumentSet,
             market: MarketState | None, rng: np.random.Generator) -> float:
        """Compute probability that opportunity accepts the quote.

        Args:
            opp: Opportunity to evaluate
            quote: Current quote
            instruments: Instrument set
            market: Market state (competitor prices affect cross-elasticity)
            rng: Random generator

        Returns:
            Probability in [0, 1] that opportunity executes
        """
        ...

    def uncensor(self, fills: np.ndarray, instruments: InstrumentSet,
                 context: dict[str, Any] | None = None) -> np.ndarray:
        """Estimate true demand from censored fills.

        Used for demand estimation research under inventory censorship.

        Args:
            fills: Observed (censored) fill counts
            instruments: Instrument set
            context: Additional context (exposures, prices shown)

        Returns:
            Estimated true demand counts
        """
        ...

class PositionModel(Protocol):
    """Manages inventory (retail) or position (finance).

    Handles:
    - Position constraints and censorship
    - Holding costs (retail) or inventory risk (finance)
    - Replenishment and order receipt

    Methods:
        reset: Initialize position state
        available: Query available capacity for a trade
        apply_execution: Censor execution by available position
        step: Process time-based updates (replenishment, holding cost)

    Properties:
        position: Current position vector
        holding_cost: Cost incurred this step from holding position
    """
    def reset(self, instruments: InstrumentSet, rng: np.random.Generator) -> None:
        """Initialize position state for new episode."""
        ...

    def available(self, instrument_id: int, side: Any) -> float:
        """Query available capacity for a trade.

        Args:
            instrument_id: Which instrument
            side: BUY or SELL

        Returns:
            Maximum tradeable size given current position
        """
        ...

    def apply_execution(self, exe: Execution) -> Execution:
        """Apply position constraints to an execution.

        Args:
            exe: Proposed execution with size_requested

        Returns:
            Censored execution with size_filled <= available capacity
        """
        ...

    def step(self, t: float) -> None:
        """Process time-based position updates.

        Handles replenishment receipt, holding cost calculation, etc.
        """
        ...

    @property
    def position(self) -> np.ndarray:
        """Current position vector (positive=long/inventory, negative=short)."""
        ...

    @property
    def holding_cost(self) -> float:
        """Holding cost incurred this step."""
        ...

class MarketModel(Protocol):
    """Models external market dynamics and competitor behavior.

    For retail: competitor price dynamics (static, reactive, stochastic)
    For finance: mid-price process (GBM, mean-reverting)

    Methods:
        step: Update market state given agent's quotes
    """
    def step(self, t: float, self_quotes: Quote, hidden: HiddenState,
             rng: np.random.Generator) -> MarketState:
        """Update market state for this timestep.

        Args:
            t: Current time
            self_quotes: Agent's current quotes (competitors may react)
            hidden: Hidden state (regime info)
            rng: Random generator

        Returns:
            Updated market state with competitor prices, mid-prices, volatility
        """
        ...

class ObservationBuilder(Protocol):
    """Constructs agent observations with appropriate censoring.

    Critical for research: ensures agent only sees censored fills,
    never true demand (which goes in info dict).

    Methods:
        build: Construct observation from step data
    """
    def build(self, quote: Quote, instruments: InstrumentSet, logs: StepLogs,
              metrics: StepMetrics, market: MarketState | None,
              hidden: HiddenState, mask_demand: bool, t: int) -> Observation:
        """Build observation for agent.

        Args:
            quote: Current quote
            instruments: Instrument set with positions
            logs: Step logs with true_demand and censored_fills
            metrics: Computed metrics
            market: Market state
            hidden: Hidden state (not included in obs)
            mask_demand: If True, exclude true demand from observation
            t: Current timestep

        Returns:
            Observation containing only observable quantities
        """
        ...

class Objective(Protocol):
    """Computes reward from step metrics.

    Supports composite objectives with weighted terms:
    - PnL (profit)
    - Position costs (holding, inventory risk)
    - Lost opportunity (stockouts)
    - Volatility penalty (UX)
    - Spread capture (market making)

    Methods:
        reward: Compute scalar reward
        breakdown: Get per-term contribution for analysis
    """
    def reward(self, quote: Quote, instruments: InstrumentSet,
               metrics: StepMetrics, hidden: HiddenState,
               obs: Observation) -> float:
        """Compute scalar reward for this step.

        Args:
            quote: Current quote
            instruments: Instrument set
            metrics: Step metrics (pnl, costs, etc.)
            hidden: Hidden state
            obs: Agent observation

        Returns:
            Scalar reward value
        """
        ...

    def breakdown(self, quote: Quote, instruments: InstrumentSet,
                  metrics: StepMetrics, hidden: HiddenState,
                  obs: Observation) -> dict[str, float]:
        """Get reward breakdown by component.

        Useful for analyzing which terms dominate the reward.

        Returns:
            Dict mapping term names to their contributions
        """
        ...
