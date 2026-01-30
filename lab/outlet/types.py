"""
Core data types for the Quote-Control simulator.

This module defines the fundamental data structures used throughout the platform:
- Identifiers (InstrumentId, OpportunityId, AgentId)
- Domain objects (Instrument, Quote, Opportunity, Execution)
- Logging structures (StepEvent, StepLogs, StepMetrics)
- State containers (MarketState, HiddenState, Observation, StepResult)

All dataclasses are designed to be serializable and numpy-compatible.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, NewType
import numpy as np
from .constants import Side, InstrumentType, OpportunityType, EventType

InstrumentId = NewType('InstrumentId', int)  # unique instrument index
OpportunityId = NewType('OpportunityId', str)  # unique opportunity/session ID
AgentId = NewType('AgentId', str)  # unique agent/actor ID

@dataclass
class Instrument:
    """Represents a priceable entity in the simulation.

    An instrument can be a retail SKU, financial asset, loan product, or subscription.
    The cost_basis represents the fundamental value (marginal cost for retail,
    mid-price for assets, funding rate for loans).

    Attributes:
        id: Unique identifier for this instrument
        type: Category of instrument (SKU, ASSET, LOAN, SUBSCRIPTION)
        cost_basis: Fundamental cost or value (marginal cost, mid-price, funding rate)
        reference_price: Base or fair price used for action scaling
        attrs: Additional attributes (quality score, category, volatility, etc.)
    """
    id: InstrumentId
    type: InstrumentType
    cost_basis: float
    reference_price: float
    attrs: dict[str, Any] = field(default_factory=dict)

@dataclass
class InstrumentSet:
    """Collection of instruments with optional position tracking.

    Provides vectorized access to instrument properties for efficient computation.
    Position can be positive (long/inventory) or negative (short) for financial assets.

    Attributes:
        instruments: List of Instrument objects
        position: Current position per instrument (None = unlimited capacity)

    Properties:
        n: Number of instruments
        costs: Vector of cost bases
        refs: Vector of reference prices
    """
    instruments: list[Instrument]
    position: np.ndarray | None = None

    @property
    def n(self) -> int: return len(self.instruments)
    @property
    def costs(self) -> np.ndarray: return np.array([i.cost_basis for i in self.instruments], np.float32)
    @property
    def refs(self) -> np.ndarray: return np.array([i.reference_price for i in self.instruments], np.float32)

@dataclass
class Quote:
    """Price quote set by the policy - the action in the MDP.

    Supports multiple quoting mechanisms:
    - Posted price: only `prices` field used
    - Two-sided: `prices` as mid, `spreads` for bid-ask width
    - Auction: `prices` as reserve prices

    The propensity field is critical for off-policy evaluation (OPE).

    Attributes:
        prices: Posted prices (retail) or mid-quotes (market making)
        spreads: Bid-ask spread width for two-sided quoting (None for posted price)
        propensity: P(this quote | behavior policy) for importance sampling
        metadata: Additional info (prev_prices for delta constraints, etc.)

    Properties:
        bids: Computed bid prices (mid - spread/2)
        asks: Computed ask prices (mid + spread/2)
    """
    prices: np.ndarray
    spreads: np.ndarray | None = None
    propensity: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def bids(self) -> np.ndarray | None:
        return self.prices - self.spreads/2 if self.spreads is not None else None
    @property
    def asks(self) -> np.ndarray | None:
        return self.prices + self.spreads/2 if self.spreads is not None else None

@dataclass
class Opportunity:
    """An arrival event that may result in a transaction.

    Opportunities are the demand side of the simulation:
    - Retail: browsing session with purchase intent
    - Market making: incoming market order
    - Lending: loan application

    The context dict carries segment/type information used by execution models.

    Attributes:
        id: Unique identifier for this opportunity
        type: Category (SESSION, MARKET_ORDER, REQUEST)
        side: BUY or SELL intent
        instrument_id: Which instrument the opportunity targets
        size: Requested transaction size (units, shares, principal)
        t: Arrival timestamp
        context: Segment info (is_scraper, credit_score, urgency, etc.)
    """
    id: OpportunityId
    type: OpportunityType
    side: Side
    instrument_id: InstrumentId
    size: float = 1.0
    t: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)

@dataclass
class Execution:
    """A realized transaction after acceptance and position censorship.

    The difference between size_requested and size_filled represents
    censored demand due to inventory/position constraints.

    Attributes:
        opportunity_id: Links back to the originating Opportunity
        instrument_id: Which instrument was traded
        side: BUY or SELL
        size_requested: Original requested size (true demand)
        size_filled: Actual filled size after censorship
        price: Execution price
        propensity: Combined propensity for OPE (quote * acceptance)
        t: Execution timestamp
    """
    opportunity_id: OpportunityId
    instrument_id: InstrumentId
    side: Side
    size_requested: float
    size_filled: float
    price: float
    propensity: float = 1.0
    t: float = 0.0

@dataclass
class StepEvent:
    """Generic logged event"""
    t: float
    type: EventType
    instrument_id: InstrumentId | None = None
    opportunity_id: OpportunityId | None = None
    price: float | None = None
    size: float | None = None
    propensity: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class StepLogs:
    """Container for all logging data from a simulation step.

    Supports both detailed event logging (for OPE) and aggregate-only mode
    (for fast simulation). The true_demand vs censored_fills distinction
    is critical for research on demand estimation under censorship.

    Attributes:
        events: Detailed event log (None if LogLevel != FULL)
        executions: List of executed transactions (None if LogLevel != FULL)
        aggregates: Always-available aggregate statistics
        true_demand: Oracle demand before censorship (for research, not in obs)
        censored_fills: Realized fills after position constraints (observable)
    """
    events: list[StepEvent] | None = None
    executions: list[Execution] | None = None
    aggregates: dict[str, Any] = field(default_factory=dict)
    true_demand: np.ndarray | None = None
    censored_fills: np.ndarray | None = None

@dataclass
class StepMetrics:
    """Computed metrics for a single simulation step.

    Metrics are domain-aware: retail uses revenue/cost/holding_cost,
    market making uses spread_capture and inventory risk.

    Attributes:
        pnl: Profit and loss (revenue - cost for retail, mark-to-market for finance)
        revenue: Gross revenue from sales/executions
        cost: Cost of goods sold or position acquisition cost
        units_traded: Total units/shares transacted
        position_cost: Holding cost (retail) or inventory risk penalty (finance)
        lost_opportunity: Cost of stockouts or missed fills
        spread_capture: Bid-ask spread captured (market making)
        volatility: Price volatility metric for UX consideration
        conversion: Fill rate (executions / opportunities)
        per_instrument: Per-instrument breakdowns (fills, demand, etc.)
    """
    pnl: float = 0.0
    revenue: float = 0.0
    cost: float = 0.0
    units_traded: float = 0.0
    position_cost: float = 0.0
    lost_opportunity: float = 0.0
    spread_capture: float = 0.0
    volatility: float = 0.0
    conversion: float = 0.0
    per_instrument: dict[str, np.ndarray] = field(default_factory=dict)

@dataclass
class MarketState:
    """External market conditions and competitor state.

    For retail: competitor_quotes drives cross-elasticity effects.
    For finance: mid_prices and volatility drive execution dynamics.

    Attributes:
        competitor_quotes: Competitor posted prices (retail)
        mid_prices: Market mid-prices for assets (finance)
        volatility: Per-instrument volatility estimate
        regime: Market regime identifier (normal, price_war, high_vol, etc.)
        t: Timestamp of this market state
    """
    competitor_quotes: np.ndarray | None = None
    mid_prices: np.ndarray | None = None
    volatility: np.ndarray | None = None
    regime: str = 'normal'
    t: float = 0.0

@dataclass
class HiddenState:
    """Internal simulator state not exposed to the agent.

    Contains oracle information for research analysis and
    history needed for non-stationary dynamics.

    Attributes:
        true_demand_intensity: Latent demand multiplier
        contamination: Fraction of arrivals that are adversarial/scraper
        regime: Current market/competitor regime
        quote_history: History of agent quotes for volatility calculation
        market_history: History of market states for analysis
    """
    true_demand_intensity: float = 1.0
    contamination: float = 0.0
    regime: str = 'normal'
    quote_history: list[np.ndarray] = field(default_factory=list)
    market_history: list[MarketState] = field(default_factory=list)

@dataclass
class Observation:
    """Observable state provided to the agent - censored view only.

    Critical invariant: Observation never contains true_demand, only
    censored fills. This enforces the censorship research setting.

    Attributes:
        quotes: Current posted quotes (the agent's last action)
        position: Current inventory/position state
        fills: Censored execution counts per instrument
        exposures: Opportunity exposure counts per instrument
        market: Observable market state (competitor prices, volatility)
        t: Current timestep
        extra: Additional observable features

    Methods:
        to_flat: Flatten to numpy array for gym compatibility
    """
    quotes: np.ndarray
    position: np.ndarray | None
    fills: np.ndarray
    exposures: np.ndarray
    market: MarketState | None
    t: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_flat(self) -> np.ndarray:
        """Flatten observation to 1D numpy array for gym environments."""
        parts = [self.quotes, self.fills, self.exposures]
        if self.position is not None: parts.append(self.position)
        if self.market and self.market.competitor_quotes is not None:
            parts.append(self.market.competitor_quotes)
        return np.concatenate([p.flatten() for p in parts])

@dataclass
class StepResult:
    """Complete result from a simulation step.

    Follows gymnasium convention for obs, reward, terminated, truncated, info.
    Additionally provides metrics, logs, and hidden state for research.

    Attributes:
        obs: Observable state (censored)
        reward: Scalar reward from objective function
        terminated: Episode ended naturally (max_steps reached)
        truncated: Episode ended early (bankruptcy, constraint violation)
        info: Additional info dict (contains true_demand for research)
        metrics: Computed metrics for this step
        logs: Event logs and aggregates
        hidden: Internal simulator state (oracle info)
    """
    obs: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    metrics: StepMetrics
    logs: StepLogs
    hidden: HiddenState
