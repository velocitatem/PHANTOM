"""
Gymnasium-compatible wrapper for the Quote-Control platform.

Provides a standard Gym interface for RL training:
- observation_space: Box space with flattened observation
- action_space: Box space with price multipliers [0.5, 2.0]
- reset(), step(), render(), close() methods

Example:
    >>> from lab.config import make_retail_platform
    >>> from lab.outlet.gym_wrapper import QuoteGymEnv
    >>> env = QuoteGymEnv(make_retail_platform())
    >>> obs, info = env.reset()
    >>> obs, reward, done, truncated, info = env.step(env.action_space.sample())
"""
from __future__ import annotations
from typing import Any
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

from .platform import Platform, PlatformConfig
from .types import Quote, InstrumentSet, StepResult

class QuoteGymEnv:
    """Gymnasium-compatible environment wrapper.

    Wraps a Platform instance with standard Gym interface.
    Actions are price multipliers in [0.5, 2.0] applied to reference prices.
    Observations are flattened numpy arrays containing quotes, fills, exposures.
    """

    def __init__(self, platform: Platform):
        if not HAS_GYM:
            raise ImportError("gymnasium required for QuoteGymEnv")
        self.platform = platform
        self.n = platform.instruments.n
        self._last_result: StepResult | None = None

        # action space: price adjustments as multipliers [0.5, 2.0]
        self.action_space = spaces.Box(low=0.5, high=2.0, shape=(self.n,), dtype=np.float32)

        # observation space
        obs_dim = self.n * 4  # quotes + fills + exposures + position
        if platform.market:
            obs_dim += self.n  # competitor quotes
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        result = self.platform.reset(seed)
        self._last_result = result
        return result.obs.to_flat().astype(np.float32), result.info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # convert action (multipliers) to absolute prices
        refs = self.platform.instruments.refs
        prices = refs * action
        result = self.platform.step(prices)
        self._last_result = result
        return (result.obs.to_flat().astype(np.float32), result.reward,
                result.terminated, result.truncated, result.info)

    def render(self) -> None:
        if self._last_result:
            m = self._last_result.metrics
            print(f"t={self.platform._t} pnl={m.pnl:.2f} units={m.units_traded:.0f} "
                  f"conv={m.conversion:.3f} vol={m.volatility:.3f}")

    def close(self) -> None:
        pass

def make_env(platform: Platform) -> QuoteGymEnv:
    return QuoteGymEnv(platform)

if HAS_GYM:
    # register if gymnasium available
    try:
        gym.register(id='QuoteControl-v0', entry_point='outlet.gym_wrapper:QuoteGymEnv')
    except:
        pass  # already registered or other issue
