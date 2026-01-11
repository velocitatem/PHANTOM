from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class SurgePricingConfig:
    high_threshold: int
    surge_multiplier: float
    window_size: int  # seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "high_threshold": self.high_threshold,
            "surge_multiplier": self.surge_multiplier,
            "window_size": self.window_size,
        }


@dataclass(frozen=True)
class SessionAwareConfig:
    velocity_threshold: float
    view_depth_threshold: int
    cart_ratio_threshold: float
    agent_multiplier: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "velocity_threshold": self.velocity_threshold,
            "view_depth_threshold": self.view_depth_threshold,
            "cart_ratio_threshold": self.cart_ratio_threshold,
            "agent_multiplier": self.agent_multiplier,
        }


# Test-optimized configurations for deterministic results
AGGRESSIVE_SURGE = SurgePricingConfig(
    high_threshold=3,
    surge_multiplier=1.5,
    window_size=10,
)

MODERATE_SURGE = SurgePricingConfig(
    high_threshold=5,
    surge_multiplier=1.3,
    window_size=15,
)

STRICT_SESSION_AWARE = SessionAwareConfig(
    velocity_threshold=2.0,  # 2 events/sec is agent-like
    view_depth_threshold=5,
    cart_ratio_threshold=0.8,
    agent_multiplier=1.4,
)
