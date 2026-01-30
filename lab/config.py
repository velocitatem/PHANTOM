"""
Configuration and factory functions for creating pre-configured platforms.

This module provides:
- RetailConfig, MarketMakingConfig: Configuration dataclasses
- make_retail_platform: Factory for retail dynamic pricing scenarios
- make_market_making_platform: Factory for market making scenarios

Example:
    >>> from lab.config import make_retail_platform
    >>> platform = make_retail_platform(RetailConfig(n_instruments=5))
    >>> result = platform.reset(seed=42)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .outlet import (Platform, PlatformConfig, PositionModel, PositionConfig,
                     PostedPriceMechanism, TwoSidedMechanism, make_instruments,
                     InstrumentType, LogLevel)
from .outlet.mechanisms.posted_price import PostedPriceConfig
from .outlet.mechanisms.two_sided import TwoSidedConfig
from .population import (SessionArrivalModel, PoissonArrivalModel, HawkesArrivalModel,
                         ElasticityExecutionModel, IntensityExecutionModel,
                         ReactiveCompetitorModel, GBMMarketModel)
from .population.arrivals import SessionArrivalConfig, PoissonArrivalConfig, HawkesArrivalConfig
from .population.execution import ElasticityConfig, IntensityConfig
from .population.competitors import ReactiveCompetitorConfig, GBMMarketConfig
from .outlet.objectives.factory import retail_objective, market_making_objective

@dataclass
class RetailConfig:
    """Configuration for retail dynamic pricing scenario.

    Attributes:
        n_instruments: Number of products to price
        cost_range: (min, max) for random product costs
        margin_range: (min, max) for random initial margins
        initial_inventory: Starting inventory per product
        holding_cost_rate: Cost per unit per step for holding
        sessions_per_step: Number of browsing sessions per step
        contamination: Fraction of sessions that are scrapers
        max_steps: Maximum episode length
        seed: Random seed for reproducibility
    """
    n_instruments: int = 10
    cost_range: tuple[float, float] = (5.0, 50.0)
    margin_range: tuple[float, float] = (0.2, 0.5)
    initial_inventory: float = 100.0
    holding_cost_rate: float = 0.002
    sessions_per_step: int = 30
    contamination: float = 0.1
    max_steps: int = 500
    seed: int | None = None

def make_retail_platform(cfg: RetailConfig | None = None) -> Platform:
    """Create a pre-configured retail dynamic pricing platform.

    Components:
    - Mechanism: PostedPriceMechanism (single price per product)
    - Arrivals: SessionArrivalModel (browsing sessions with views)
    - Execution: ElasticityExecutionModel (price sensitivity)
    - Market: ReactiveCompetitorModel (can trigger price wars)
    - Objective: PnL - holding_cost - volatility - lost_opportunity

    Args:
        cfg: Configuration (uses defaults if None)

    Returns:
        Configured Platform instance
    """
    cfg = cfg or RetailConfig()
    rng = np.random.default_rng(cfg.seed)

    instruments = make_instruments(cfg.n_instruments, cfg.cost_range, cfg.margin_range,
                                   InstrumentType.SKU, rng)
    instruments.position = np.full(cfg.n_instruments, cfg.initial_inventory)

    mechanism = PostedPriceMechanism(PostedPriceConfig())
    arrival = SessionArrivalModel(SessionArrivalConfig(
        sessions_per_step=cfg.sessions_per_step, contamination=cfg.contamination))
    execution = ElasticityExecutionModel(ElasticityConfig())
    position = PositionModel(PositionConfig(
        initial_position=cfg.initial_inventory,
        holding_cost_rate=cfg.holding_cost_rate))
    market = ReactiveCompetitorModel(ReactiveCompetitorConfig(), refs=instruments.refs)
    objective = retail_objective()

    return Platform(
        instruments=instruments, mechanism=mechanism, arrival=arrival,
        execution=execution, position=position, market=market, objective=objective,
        cfg=PlatformConfig(n_instruments=cfg.n_instruments, max_steps=cfg.max_steps,
                           seed=cfg.seed, log_level=LogLevel.AGG_ONLY)
    )

@dataclass
class MarketMakingConfig:
    """Configuration for market making scenario.

    Attributes:
        n_instruments: Number of assets to quote
        initial_mid: Initial mid-price for assets
        mu: Price drift (expected return)
        sigma: Price volatility
        gamma: Inventory risk aversion parameter
        base_arrival_rate: Order arrival rate (Hawkes baseline)
        max_steps: Maximum episode length
        seed: Random seed for reproducibility
    """
    n_instruments: int = 5
    initial_mid: float = 100.0
    mu: float = 0.0
    sigma: float = 0.02
    gamma: float = 0.1
    base_arrival_rate: float = 20.0
    max_steps: int = 1000
    seed: int | None = None

def make_market_making_platform(cfg: MarketMakingConfig | None = None) -> Platform:
    """Create a pre-configured market making platform.

    Components:
    - Mechanism: TwoSidedMechanism (bid-ask spread quoting)
    - Arrivals: HawkesArrivalModel (clustered order flow)
    - Execution: IntensityExecutionModel (distance-based fills)
    - Market: GBMMarketModel (geometric Brownian motion mid-prices)
    - Objective: PnL + spread_capture - inventory_risk

    Args:
        cfg: Configuration (uses defaults if None)

    Returns:
        Configured Platform instance
    """
    cfg = cfg or MarketMakingConfig()
    rng = np.random.default_rng(cfg.seed)

    instruments = make_instruments(cfg.n_instruments, (cfg.initial_mid*0.9, cfg.initial_mid*1.1),
                                   (0.0, 0.0), InstrumentType.ASSET, rng)
    instruments.position = np.zeros(cfg.n_instruments)

    mechanism = TwoSidedMechanism(TwoSidedConfig())
    arrival = HawkesArrivalModel(HawkesArrivalConfig(base_rate=cfg.base_arrival_rate))
    execution = IntensityExecutionModel(IntensityConfig())
    position = PositionModel(PositionConfig(
        initial_position=0.0, min_position=-500, max_position=500,
        holding_cost_rate=0.0))  # use inventory risk penalty instead
    market = GBMMarketModel(GBMMarketConfig(mu=cfg.mu, sigma=cfg.sigma),
                            initial=instruments.refs)
    objective = market_making_objective(gamma=cfg.gamma, sigma=cfg.sigma)

    return Platform(
        instruments=instruments, mechanism=mechanism, arrival=arrival,
        execution=execution, position=position, market=market, objective=objective,
        cfg=PlatformConfig(n_instruments=cfg.n_instruments, max_steps=cfg.max_steps,
                           seed=cfg.seed, log_level=LogLevel.AGG_ONLY)
    )
