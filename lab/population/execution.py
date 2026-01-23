"""
Execution models for computing acceptance/fill probabilities.

This module provides different models for how opportunities convert to executions:
- ElasticityExecutionModel: Price elasticity with competitor cross-effects (retail)
- IntensityExecutionModel: Distance-based fill intensity (market making)
- LogitExecutionModel: Discrete choice model

Each model implements the ExecutionModel protocol.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from ..outlet.types import Opportunity, Quote, InstrumentSet, MarketState
from ..outlet.constants import Side
from ..outlet.math_util import sigmoid, safe_log, intensity_decay, EPS

@dataclass
class ElasticityConfig:
    """Configuration for price elasticity execution model.

    Attributes:
        base_prob: Baseline purchase probability at reference price
        price_sensitivity: Own-price elasticity coefficient
        cross_elasticity: Competitor price cross-elasticity
        scraper_conversion: Multiplier for scraper conversion (typically << 1)
    """
    base_prob: float = 0.3
    price_sensitivity: float = 2.0
    cross_elasticity: float = 0.5
    scraper_conversion: float = 0.01

class ElasticityExecutionModel:
    """Price elasticity model for retail dynamic pricing.

    P(buy) = base_prob * exp(-sensitivity * log(price/ref)) * cross_effect * scraper_mult

    Higher prices reduce purchase probability exponentially.
    Competitor undercutting shifts demand away from the platform.
    Scrapers convert at a much lower rate (reconnaissance, not purchase).
    """

    def __init__(self, cfg: ElasticityConfig | None = None):
        self.cfg = cfg or ElasticityConfig()

    def prob(self, opp: Opportunity, quote: Quote, instruments: InstrumentSet,
             market: MarketState | None, rng: np.random.Generator) -> float:
        idx = int(opp.instrument_id)
        price = quote.prices[idx]
        ref = instruments.refs[idx]

        # base probability adjusted by price ratio
        log_ratio = safe_log(price / ref)
        prob = self.cfg.base_prob * np.exp(-self.cfg.price_sensitivity * log_ratio)

        # cross-elasticity: competitor undercutting increases their share
        if market and market.competitor_quotes is not None:
            comp_price = market.competitor_quotes[idx]
            if comp_price < price:
                prob *= np.exp(-self.cfg.cross_elasticity * (price - comp_price) / ref)

        # scrapers convert at much lower rate
        if opp.context.get('is_scraper', False):
            prob *= self.cfg.scraper_conversion

        return float(np.clip(prob, 0, 1))

    def uncensor(self, fills: np.ndarray, instruments: InstrumentSet,
                 context: dict[str, Any] | None = None) -> np.ndarray:
        # simple imputation: assume fills = prob * exposures, invert
        exposures = context.get('exposures', fills) if context else fills
        avg_prob = self.cfg.base_prob
        return fills / (avg_prob + EPS)

@dataclass
class IntensityConfig:
    """Configuration for intensity-based execution model.

    Attributes:
        base_intensity: Baseline fill intensity
        kappa: Decay rate with distance from mid-price
        vol_scale: Volatility multiplier for fill intensity
    """
    base_intensity: float = 1.0
    kappa: float = 1.5
    vol_scale: float = 0.5

class IntensityExecutionModel:
    """Avellaneda-Stoikov style fill intensity for market making.

    Fill probability decays exponentially with distance from mid-price:
    P(fill) = base * exp(-kappa * |quote - mid|) * (1 + vol_scale * sigma)

    Tighter spreads (closer to mid) have higher fill probability.
    Higher volatility increases fill probability (more aggressive traders).
    """

    def __init__(self, cfg: IntensityConfig | None = None):
        self.cfg = cfg or IntensityConfig()

    def prob(self, opp: Opportunity, quote: Quote, instruments: InstrumentSet,
             market: MarketState | None, rng: np.random.Generator) -> float:
        idx = int(opp.instrument_id)

        # get mid price from market or use quote price
        if market and market.mid_prices is not None:
            mid = market.mid_prices[idx]
        else:
            mid = quote.prices[idx]

        # compute distance from mid
        if opp.side == Side.BUY:
            exec_price = quote.asks[idx] if quote.asks is not None else quote.prices[idx]
            distance = exec_price - mid
        else:
            exec_price = quote.bids[idx] if quote.bids is not None else quote.prices[idx]
            distance = mid - exec_price

        # intensity decays with distance
        intensity = self.cfg.base_intensity * intensity_decay(abs(distance), self.cfg.kappa)

        # volatility increases fill probability
        if market and market.volatility is not None:
            vol = market.volatility[idx]
            intensity *= (1 + self.cfg.vol_scale * vol)

        return float(np.clip(intensity, 0, 1))

    def uncensor(self, fills: np.ndarray, instruments: InstrumentSet,
                 context: dict[str, Any] | None = None) -> np.ndarray:
        return fills  # market making doesn't have same censorship concept

@dataclass
class LogitConfig:
    """Configuration for logit discrete choice model.

    Attributes:
        beta_0: Intercept (base utility)
        beta_price: Price coefficient (typically negative)
        beta_quality: Quality attribute coefficient
    """
    beta_0: float = 0.5
    beta_price: float = -1.5
    beta_quality: float = 0.3

class LogitExecutionModel:
    """Discrete choice logit model for purchase probability.

    Utility: U = beta_0 + beta_price * (price/ref) + beta_quality * quality
    P(buy) = sigmoid(U)

    Provides a theoretically grounded demand model from economics literature.
    """

    def __init__(self, cfg: LogitConfig | None = None):
        self.cfg = cfg or LogitConfig()

    def prob(self, opp: Opportunity, quote: Quote, instruments: InstrumentSet,
             market: MarketState | None, rng: np.random.Generator) -> float:
        idx = int(opp.instrument_id)
        price = quote.prices[idx]
        ref = instruments.refs[idx]
        quality = instruments.instruments[idx].attrs.get('quality', 0.5)

        # utility
        u = self.cfg.beta_0 + self.cfg.beta_price * (price / ref) + self.cfg.beta_quality * quality

        # choice probability via sigmoid
        return float(sigmoid(u))

    def uncensor(self, fills: np.ndarray, instruments: InstrumentSet,
                 context: dict[str, Any] | None = None) -> np.ndarray:
        return fills / (self.cfg.beta_0 + EPS)
