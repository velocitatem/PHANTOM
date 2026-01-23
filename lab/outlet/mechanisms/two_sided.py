"""
Two-sided quoting mechanism for market making.

In this mechanism, the agent posts both bid and ask prices.
Execution depends on the distance from the market mid-price.
This models liquidity provision in financial markets.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..types import Quote, Opportunity, Execution, InstrumentSet, MarketState
from ..constants import Side
from ..math_util import clamp, intensity_decay

@dataclass
class TwoSidedConfig:
    """Configuration for two-sided quoting mechanism.

    Attributes:
        min_spread: Minimum bid-ask spread
        max_spread: Maximum bid-ask spread
        min_price: Absolute minimum price
        max_price: Absolute maximum price
        fill_kappa: Intensity decay parameter (higher = faster decay with distance)
    """
    min_spread: float = 0.01
    max_spread: float = 0.5
    min_price: float = 0.01
    max_price: float = 10000.0
    fill_kappa: float = 1.5

class TwoSidedMechanism:
    """Two-sided quoting mechanism for market making.

    The agent posts bid (buy) and ask (sell) prices around a mid-point.
    Fill probability decays exponentially with distance from mid-price,
    following the Avellaneda-Stoikov intensity model.

    Both BUY and SELL opportunities are processed:
    - BUY: customer buys at agent's ask price
    - SELL: customer sells at agent's bid price
    """

    def __init__(self, cfg: TwoSidedConfig | None = None):
        self.cfg = cfg or TwoSidedConfig()

    def apply_quote(self, quote: Quote, instruments: InstrumentSet,
                    rng: np.random.Generator) -> Quote:
        prices = quote.prices.copy()
        spreads = quote.spreads.copy() if quote.spreads is not None else np.full_like(prices, 0.02)
        c = self.cfg

        prices = clamp(prices, c.min_price, c.max_price)
        spreads = clamp(spreads, c.min_spread, c.max_spread)

        # ensure bids < asks
        half_spread = spreads / 2
        bids = prices - half_spread
        asks = prices + half_spread
        bids = np.maximum(bids, c.min_price)
        asks = np.minimum(asks, c.max_price)
        spreads = asks - bids
        prices = (bids + asks) / 2

        return Quote(prices=prices, spreads=spreads, propensity=quote.propensity,
                     metadata=quote.metadata)

    def process_opportunity(self, opp: Opportunity, quote: Quote,
                            instruments: InstrumentSet, market: MarketState | None,
                            rng: np.random.Generator) -> Execution | None:
        idx = int(opp.instrument_id)
        mid = market.mid_prices[idx] if market and market.mid_prices is not None else quote.prices[idx]

        if opp.side == Side.BUY:
            price = float(quote.asks[idx]) if quote.asks is not None else float(quote.prices[idx])
            distance = price - mid
        else:
            price = float(quote.bids[idx]) if quote.bids is not None else float(quote.prices[idx])
            distance = mid - price

        # probabilistic fill based on distance from mid
        fill_prob = intensity_decay(abs(distance), self.cfg.fill_kappa)
        if rng.random() > fill_prob: return None

        return Execution(
            opportunity_id=opp.id, instrument_id=opp.instrument_id,
            side=opp.side, size_requested=opp.size, size_filled=opp.size,
            price=price, propensity=quote.propensity * fill_prob, t=opp.t
        )
