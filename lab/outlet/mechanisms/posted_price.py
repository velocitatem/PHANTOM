"""
Posted price mechanism for retail dynamic pricing.

In this mechanism, the agent posts a single price per instrument.
Buyers decide whether to purchase based on the posted price.
This is the standard e-commerce dynamic pricing model.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..types import Quote, Opportunity, Execution, InstrumentSet, MarketState
from ..constants import Side
from ..math_util import clamp

@dataclass
class PostedPriceConfig:
    """Configuration for posted price mechanism.

    Attributes:
        min_price: Absolute minimum price
        max_price: Absolute maximum price
        max_delta_pct: Maximum price change per step as fraction of previous
        min_margin_pct: Minimum margin over cost basis
        round_to: Price rounding granularity (None = no rounding)
    """
    min_price: float = 0.01
    max_price: float = 1000.0
    max_delta_pct: float = 0.2
    min_margin_pct: float = 0.05
    round_to: float | None = 0.01

class PostedPriceMechanism:
    """Posted price mechanism for retail dynamic pricing.

    The agent posts a single price per product. Constraints enforced:
    - Prices within [min_price, max_price]
    - Margin at least min_margin_pct above cost
    - Price changes limited to max_delta_pct per step
    - Prices rounded to round_to granularity

    Only BUY-side opportunities are processed (customers purchasing).
    """

    def __init__(self, cfg: PostedPriceConfig | None = None):
        self.cfg = cfg or PostedPriceConfig()

    def apply_quote(self, quote: Quote, instruments: InstrumentSet,
                    rng: np.random.Generator) -> Quote:
        prices = quote.prices.copy()
        costs = instruments.costs
        refs = instruments.refs
        c = self.cfg

        # enforce min margin
        min_prices = costs * (1 + c.min_margin_pct)
        prices = np.maximum(prices, min_prices)

        # enforce absolute bounds
        prices = clamp(prices, c.min_price, c.max_price)

        # enforce max delta if we have history
        if 'prev_prices' in quote.metadata:
            prev = quote.metadata['prev_prices']
            max_change = prev * c.max_delta_pct
            prices = clamp(prices, prev - max_change, prev + max_change)

        # round prices
        if c.round_to:
            prices = np.round(prices / c.round_to) * c.round_to

        return Quote(prices=prices, propensity=quote.propensity,
                     metadata={**quote.metadata, 'prev_prices': prices})

    def process_opportunity(self, opp: Opportunity, quote: Quote,
                            instruments: InstrumentSet, market: MarketState | None,
                            rng: np.random.Generator) -> Execution | None:
        if opp.side != Side.BUY: return None  # posted price is buy-only
        idx = int(opp.instrument_id)
        price = float(quote.prices[idx])
        return Execution(
            opportunity_id=opp.id, instrument_id=opp.instrument_id,
            side=opp.side, size_requested=opp.size, size_filled=opp.size,
            price=price, propensity=quote.propensity, t=opp.t
        )
