"""
Market and competitor models for external dynamics.

This module provides models for competitor pricing (retail) and market dynamics (finance):
- StaticCompetitorModel: Fixed competitor prices
- ReactiveCompetitorModel: Competitor reacts to agent's prices, can trigger price wars
- StochasticCompetitorModel: Random walk competitor prices
- GBMMarketModel: Geometric Brownian Motion for asset mid-prices

Each model implements the MarketModel protocol.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..outlet.types import Quote, MarketState, HiddenState
from ..outlet.math_util import clamp, ema

@dataclass
class StaticCompetitorConfig:
    """Configuration for static competitor.

    Attributes:
        markup: Fixed percentage markup over reference prices
    """
    markup: float = 0.1

class StaticCompetitorModel:
    """Static competitor with fixed markup pricing.

    Competitor prices = reference * (1 + markup).
    Useful as a baseline or for testing without competitor dynamics.
    """

    def __init__(self, cfg: StaticCompetitorConfig | None = None, refs: np.ndarray | None = None):
        self.cfg = cfg or StaticCompetitorConfig()
        self.refs = refs

    def step(self, t: float, self_quotes: Quote, hidden: HiddenState,
             rng: np.random.Generator) -> MarketState:
        refs = self.refs if self.refs is not None else self_quotes.prices
        comp_prices = refs * (1 + self.cfg.markup)
        return MarketState(competitor_quotes=comp_prices, regime='static', t=t)

@dataclass
class ReactiveCompetitorConfig:
    """Configuration for reactive competitor.

    Attributes:
        follow_weight: Smoothing weight for price following (0=ignore, 1=instant)
        band_pct: Maximum deviation from reference prices
        war_threshold: Relative price diff that triggers price war
        war_aggression: How much competitor cuts prices during war
    """
    follow_weight: float = 0.3
    band_pct: float = 0.1
    war_threshold: float = -0.15
    war_aggression: float = 0.2

class ReactiveCompetitorModel:
    """Competitor that reacts to agent's prices with price war dynamics.

    The competitor follows the agent's prices with smoothing.
    If the agent undercuts significantly (beyond war_threshold),
    a price war is triggered where the competitor becomes more aggressive.

    This creates non-stationary dynamics that test policy robustness.
    """

    def __init__(self, cfg: ReactiveCompetitorConfig | None = None, refs: np.ndarray | None = None):
        self.cfg = cfg or ReactiveCompetitorConfig()
        self.refs = refs
        self._prices: np.ndarray | None = None
        self._in_war: bool = False

    def step(self, t: float, self_quotes: Quote, hidden: HiddenState,
             rng: np.random.Generator) -> MarketState:
        refs = self.refs if self.refs is not None else self_quotes.prices
        c = self.cfg

        if self._prices is None:
            self._prices = refs.copy()

        # check for price war trigger
        relative_diff = (self_quotes.prices - self._prices) / (self._prices + 1e-8)
        if np.any(relative_diff < c.war_threshold):
            self._in_war = True
        elif np.all(relative_diff > -c.war_threshold / 2):
            self._in_war = False

        # update prices
        if self._in_war:
            target = self_quotes.prices * (1 - c.war_aggression)
            hidden.regime = 'price_war'
        else:
            target = self_quotes.prices * (1 + c.follow_weight * 0.05)
            hidden.regime = 'normal'

        # follow with smoothing
        new_prices = np.array([ema(old, new, c.follow_weight)
                               for old, new in zip(self._prices, target)])

        # stay within band
        new_prices = clamp(new_prices, refs * (1 - c.band_pct), refs * (1 + c.band_pct))
        self._prices = new_prices

        return MarketState(competitor_quotes=new_prices, regime=hidden.regime, t=t)

@dataclass
class StochasticCompetitorConfig:
    """Configuration for stochastic competitor.

    Attributes:
        drift: Price drift per step
        volatility: Price volatility (std of random shocks)
        mean_revert: Mean reversion strength toward reference
    """
    drift: float = 0.0
    volatility: float = 0.02
    mean_revert: float = 0.1

class StochasticCompetitorModel:
    """Ornstein-Uhlenbeck style stochastic competitor prices.

    Prices follow: dP = drift + mean_revert*(ref - P) + volatility*P*dW

    Provides non-stationary competitor dynamics independent of agent actions.
    Useful for testing robustness to market noise.
    """

    def __init__(self, cfg: StochasticCompetitorConfig | None = None, refs: np.ndarray | None = None):
        self.cfg = cfg or StochasticCompetitorConfig()
        self.refs = refs
        self._prices: np.ndarray | None = None

    def step(self, t: float, self_quotes: Quote, hidden: HiddenState,
             rng: np.random.Generator) -> MarketState:
        refs = self.refs if self.refs is not None else self_quotes.prices
        c = self.cfg

        if self._prices is None:
            self._prices = refs.copy()

        # Ornstein-Uhlenbeck style dynamics
        n = len(self._prices)
        noise = rng.normal(0, c.volatility, n)
        reversion = c.mean_revert * (refs - self._prices)
        self._prices = self._prices + c.drift + reversion + noise * self._prices
        self._prices = np.maximum(self._prices, refs * 0.5)

        return MarketState(competitor_quotes=self._prices.copy(), regime='stochastic', t=t)

@dataclass
class GBMMarketConfig:
    """Configuration for GBM market model.

    Attributes:
        mu: Price drift (expected return)
        sigma: Price volatility
        dt: Time step size
    """
    mu: float = 0.0
    sigma: float = 0.1
    dt: float = 1.0

class GBMMarketModel:
    """Geometric Brownian Motion model for asset mid-prices.

    Standard Black-Scholes dynamics: dS = mu*S*dt + sigma*S*dW

    Used for market making scenarios where the underlying asset price
    follows a random walk. The agent quotes around this moving mid-price.
    """

    def __init__(self, cfg: GBMMarketConfig | None = None, initial: np.ndarray | None = None):
        self.cfg = cfg or GBMMarketConfig()
        self._mids = initial

    def step(self, t: float, self_quotes: Quote, hidden: HiddenState,
             rng: np.random.Generator) -> MarketState:
        if self._mids is None:
            self._mids = self_quotes.prices.copy()

        c = self.cfg
        n = len(self._mids)
        z = rng.standard_normal(n)
        self._mids = self._mids * np.exp((c.mu - 0.5*c.sigma**2)*c.dt + c.sigma*np.sqrt(c.dt)*z)

        vol = np.full(n, c.sigma)
        return MarketState(mid_prices=self._mids.copy(), volatility=vol, regime='gbm', t=t)
