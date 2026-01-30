"""
Quote-Control Simulator: Research-grade platform for dynamic pricing and market making

The platform abstracts pricing as: Quote -> Arrival -> Execution -> Position
Supports multiple mechanisms:
  - PostedPrice: retail dynamic pricing
  - TwoSided: market making with bid-ask spreads
  - Auction: reserve/shading for auction settings

Example usage:
    from lab.config import make_retail_platform
    from lab.experiments import rollout, fixed_price_policy

    platform = make_retail_platform()
    policy = fixed_price_policy(platform.instruments.refs)
    result = rollout(platform, policy, n_steps=100)
    print(f"Total PnL: {result.total_pnl:.2f}")
"""

from .config import make_retail_platform, make_market_making_platform, RetailConfig, MarketMakingConfig
from .outlet import Platform, PlatformConfig, Quote, Observation, StepResult

__all__ = [
    'make_retail_platform', 'make_market_making_platform',
    'RetailConfig', 'MarketMakingConfig',
    'Platform', 'PlatformConfig', 'Quote', 'Observation', 'StepResult',
]
