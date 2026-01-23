"""Execution models with divergent H/A behavior using ground truth labels."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
from ...outlet.types import Opportunity, Quote, InstrumentSet, MarketState
from ...outlet.math_util import sigmoid, safe_log, EPS


@dataclass
class HybridExecutionConfig:
    human_base_prob: float = 0.3
    human_elasticity: float = 2.5
    agent_conversion: float = 0.01
    cross_elasticity: float = 0.4
    quality_weight: float = 0.2
    use_separability: bool = False


class HybridExecutionModel:
    """Execution with divergent H/A behavior using ground truth labels."""

    def __init__(self, cfg: HybridExecutionConfig | None = None):
        self.cfg = cfg or HybridExecutionConfig()

    def prob(self, opp: Opportunity, quote: Quote, instruments: InstrumentSet,
             market: MarketState | None, rng: np.random.Generator) -> float:
        cfg, idx = self.cfg, int(opp.instrument_id)
        price, ref, cost = float(quote.prices[idx]), float(instruments.refs[idx]), float(instruments.costs[idx])
        ctx = opp.context
        theta = ctx.get('theta', {})
        is_agent = ctx.get('is_agent', False)

        if is_agent:
            return cfg.agent_conversion * theta.get('base_conversion', 1.0)

        # human logit discrete choice
        sens = theta.get('price_sensitivity', cfg.human_elasticity)
        base = theta.get('base_conversion', cfg.human_base_prob)
        u_price = -sens * safe_log(price / (ref + EPS))
        quality = instruments.instruments[idx].attrs.get('quality', 0.5)
        u_quality = cfg.quality_weight * quality

        u_comp = 0.0
        if market and market.competitor_quotes is not None:
            cp = market.competitor_quotes[idx]
            if cp < price:
                u_comp = -cfg.cross_elasticity * (price - cp) / ref

        utility = safe_log(base / (1 - base + EPS)) + u_price + u_quality + u_comp
        return float(sigmoid(utility))

    def uncensor(self, fills: np.ndarray, instruments: InstrumentSet, context: dict[str, Any] | None = None) -> np.ndarray:
        if context is None:
            return fills / (self.cfg.human_base_prob + EPS)
        agent_frac = context.get('contamination', 0.0)
        return fills / (self.cfg.human_base_prob * (1 - agent_frac) + EPS)


@dataclass
class SeparableExecutionConfig:
    human_funnel: Dict[str, float] = None
    agent_funnel: Dict[str, float] = None

    def __post_init__(self):
        self.human_funnel = self.human_funnel or {'view_to_detail': 0.4, 'detail_to_cart': 0.3, 'cart_to_purchase': 0.6}
        self.agent_funnel = self.agent_funnel or {'view_to_detail': 0.8, 'detail_to_cart': 0.05, 'cart_to_purchase': 0.1}


class SeparableExecutionModel:
    """Execution with Markov funnel kernels using ground truth labels."""

    def __init__(self, cfg: SeparableExecutionConfig | None = None):
        self.cfg = cfg or SeparableExecutionConfig()

    def prob(self, opp: Opportunity, quote: Quote, instruments: InstrumentSet,
             market: MarketState | None, rng: np.random.Generator) -> float:
        is_agent = opp.context.get('is_agent', False)
        probs = self.cfg.agent_funnel if is_agent else self.cfg.human_funnel
        p = probs['view_to_detail'] * probs['detail_to_cart'] * probs['cart_to_purchase']

        if not is_agent:
            idx = int(opp.instrument_id)
            price_ratio = quote.prices[idx] / (instruments.refs[idx] + EPS)
            p *= np.exp(-0.5 * (price_ratio - 1.0))
        return float(np.clip(p, 0, 1))

    def uncensor(self, fills: np.ndarray, instruments: InstrumentSet, context: dict[str, Any] | None = None) -> np.ndarray:
        h = self.cfg.human_funnel
        exp_conv = h['view_to_detail'] * h['detail_to_cart'] * h['cart_to_purchase']
        return fills / (exp_conv + EPS)
