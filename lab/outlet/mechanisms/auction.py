"""
Auction mechanism for reserve pricing and bid shading.

In this mechanism, the agent sets reserve prices that affect
win probability and clearing prices. Used for ad auctions,
marketplace auctions, and similar settings.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..types import Quote, Opportunity, Execution, InstrumentSet, MarketState
from ..constants import Side
from ..math_util import clamp, sigmoid

@dataclass
class AuctionConfig:
    """Configuration for auction mechanism.

    Attributes:
        min_reserve: Minimum reserve price
        max_reserve: Maximum reserve price
        base_win_prob: Baseline win probability at reference reserve
        sensitivity: How much higher reserves reduce win probability
    """
    min_reserve: float = 0.0
    max_reserve: float = 100.0
    base_win_prob: float = 0.3
    sensitivity: float = 2.0

class AuctionMechanism:
    """Auction mechanism for reserve pricing.

    The agent sets reserve prices that affect:
    - Win probability: higher reserves reduce chance of winning
    - Clearing price: bounded between reserve and simulated max bid

    Win probability: base_prob * sigmoid(-sensitivity * (reserve - ref) / ref)
    Clearing price: max(reserve, min(max_bid, reserve + random_increment))

    Only BUY-side opportunities are processed (auction wins).
    """

    def __init__(self, cfg: AuctionConfig | None = None):
        self.cfg = cfg or AuctionConfig()

    def apply_quote(self, quote: Quote, instruments: InstrumentSet,
                    rng: np.random.Generator) -> Quote:
        reserves = clamp(quote.prices, self.cfg.min_reserve, self.cfg.max_reserve)
        return Quote(prices=reserves, propensity=quote.propensity, metadata=quote.metadata)

    def process_opportunity(self, opp: Opportunity, quote: Quote,
                            instruments: InstrumentSet, market: MarketState | None,
                            rng: np.random.Generator) -> Execution | None:
        if opp.side != Side.BUY: return None
        idx = int(opp.instrument_id)
        reserve = float(quote.prices[idx])
        ref = instruments.refs[idx]

        # win probability decreases with higher reserve
        relative_reserve = (reserve - ref) / (ref + 1e-8)
        win_prob = self.cfg.base_win_prob * sigmoid(-self.cfg.sensitivity * relative_reserve)

        if rng.random() > win_prob: return None

        # clearing price is between reserve and some max bid (simulated)
        max_bid = ref * (1 + rng.exponential(0.2))
        clearing = max(reserve, min(max_bid, reserve + rng.exponential(0.1) * ref))

        return Execution(
            opportunity_id=opp.id, instrument_id=opp.instrument_id,
            side=opp.side, size_requested=opp.size, size_filled=opp.size,
            price=clearing, propensity=quote.propensity * win_prob, t=opp.t
        )
