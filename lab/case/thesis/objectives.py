"""
Thesis-specific objectives implementing robust pricing under contamination.

Implements the Maximin objective from Eq 23:
π* = argmax_π min_{Q ∈ U_ε} E_d~Q[R(p,d) - λ·COI(p)]

Key components:
- COIObjective: Cost of Information penalty (Definition 1)
- RobustStackelbergObjective: Full maximin objective with Wasserstein robustness
- UXPenalty: User experience degradation from volatility
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ...outlet.objectives.base import BaseObjective, CompositeObjective
from ...outlet.types import Quote, InstrumentSet, StepMetrics, HiddenState, Observation
from ...outlet.math_util import safe_log, EPS

class COIObjective(BaseObjective):
    """Cost of Information penalty from Definition 1.

    COI(π) = E[P] - p_min

    The expected price premium over marginal cost represents the platform's
    pricing power. Agent reconnaissance erodes this by revealing price
    distribution to buyers.

    We implement COI_leakage = f(τ') · InfoValue(p, τ')
    where f(τ') is the estimated agent probability.
    """

    def __init__(self, lambda_coi: float = 1.0, use_revelation: bool = False):
        """
        Args:
            lambda_coi: Weight on COI penalty
            use_revelation: If True, use -log(π(p)) as info value (penalizes rare prices)
        """
        self.lambda_coi = lambda_coi
        self.use_revelation = use_revelation

    def reward(self, quote: Quote, instruments: InstrumentSet,
               metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> float:
        # COI_leakage = α · InfoValue
        alpha = hidden.contamination

        if self.use_revelation:
            # revelation surrogate: rare prices reveal more about policy
            # InfoValue = -log(π(p|τ')) ≈ surprise of the price
            price_surprise = np.mean(np.abs(quote.prices - instruments.refs) / (instruments.refs + EPS))
            info_value = price_surprise
        else:
            # query-tax surrogate: each agent query incurs constant leakage
            info_value = 1.0

        leakage = alpha * info_value
        return -self.lambda_coi * leakage

    def breakdown(self, quote: Quote, instruments: InstrumentSet,
                  metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> dict[str, float]:
        alpha = hidden.contamination
        margins = (quote.prices - instruments.costs) / (instruments.costs + EPS)
        return {
            'coi_penalty': self.reward(quote, instruments, metrics, hidden, obs),
            'contamination': alpha,
            'avg_margin': float(np.mean(margins)),
        }

@dataclass
class RobustObjectiveConfig:
    """Configuration for robust Stackelberg objective.

    Attributes:
        lambda_coi: Weight on COI penalty (λ in Eq 23)
        lambda_ux: Weight on UX penalty
        lambda_volatility: Weight on price volatility penalty
        gamma_inventory: Inventory risk aversion
        wasserstein_epsilon: Ambiguity set radius (ε in Eq 21)
    """
    lambda_coi: float = 0.5
    lambda_ux: float = 0.1
    lambda_volatility: float = 0.2
    gamma_inventory: float = 0.1
    wasserstein_epsilon: float = 0.1

class RobustStackelbergObjective(BaseObjective):
    """Implements the Maximin Objective from thesis Eq 23.

    π* = argmax_π min_{Q ∈ U_ε(P̂_N)} E_d~Q[R(p,d) - λ·COI(p)]

    The objective balances:
    1. Revenue R(p,d) from human purchases
    2. COI penalty for information leakage to agents
    3. UX penalty for price volatility
    4. Inventory/holding costs

    The min over ambiguity set U_ε is approximated by penalizing
    high contamination scenarios more heavily.
    """

    def __init__(self, cfg: RobustObjectiveConfig | None = None):
        self.cfg = cfg or RobustObjectiveConfig()

    def reward(self, quote: Quote, instruments: InstrumentSet,
               metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> float:
        cfg = self.cfg

        # 1. base revenue (R(p,d))
        revenue = metrics.revenue
        cost = metrics.cost
        profit = revenue - cost

        # 2. COI penalty: scales with contamination and margin extraction
        # high margins + high contamination = high leakage
        alpha = hidden.contamination
        margins = quote.prices - instruments.costs
        avg_margin = float(np.mean(margins))
        coi_penalty = cfg.lambda_coi * avg_margin * alpha

        # 3. UX penalty: price volatility harms legitimate users
        volatility_penalty = cfg.lambda_volatility * metrics.volatility

        # 4. inventory/position cost
        position_penalty = cfg.gamma_inventory * metrics.position_cost

        # 5. lost opportunity cost (stockouts)
        lost_penalty = 0.1 * metrics.lost_opportunity

        # robust adjustment: under adversarial distribution Q,
        # expect lower revenue and higher costs
        # approximate via worst-case contamination within ε-ball
        worst_case_alpha = min(alpha + cfg.wasserstein_epsilon, 1.0)
        robustness_penalty = cfg.wasserstein_epsilon * avg_margin * worst_case_alpha

        total = profit - coi_penalty - volatility_penalty - position_penalty - lost_penalty - robustness_penalty

        return total

    def breakdown(self, quote: Quote, instruments: InstrumentSet,
                  metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> dict[str, float]:
        cfg = self.cfg
        alpha = hidden.contamination
        margins = quote.prices - instruments.costs
        avg_margin = float(np.mean(margins))

        return {
            'revenue': metrics.revenue,
            'cost': metrics.cost,
            'profit': metrics.revenue - metrics.cost,
            'coi_penalty': -cfg.lambda_coi * avg_margin * alpha,
            'volatility_penalty': -cfg.lambda_volatility * metrics.volatility,
            'position_penalty': -cfg.gamma_inventory * metrics.position_cost,
            'lost_penalty': -0.1 * metrics.lost_opportunity,
            'robustness_penalty': -cfg.wasserstein_epsilon * avg_margin * min(alpha + cfg.wasserstein_epsilon, 1.0),
            'contamination': alpha,
            'avg_margin_pct': avg_margin / (float(np.mean(instruments.costs)) + EPS),
        }

class UXPenalty(BaseObjective):
    """User experience penalty from price volatility.

    High price volatility degrades UX for legitimate human users.
    This term ensures the defense doesn't harm real customers while
    protecting against agent reconnaissance.
    """

    def __init__(self, scale: float = 1.0, max_acceptable_volatility: float = 0.1):
        self.scale = scale
        self.max_vol = max_acceptable_volatility

    def reward(self, quote: Quote, instruments: InstrumentSet,
               metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> float:
        # penalty increases quadratically beyond threshold
        excess_vol = max(0, metrics.volatility - self.max_vol)
        return -self.scale * (excess_vol ** 2)

    def breakdown(self, quote: Quote, instruments: InstrumentSet,
                  metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> dict[str, float]:
        return {
            'ux_penalty': self.reward(quote, instruments, metrics, hidden, obs),
            'volatility': metrics.volatility,
        }

class AdaptiveObjective(BaseObjective):
    """Objective that adapts weights based on estimated contamination.

    When contamination is low, focus on revenue maximization.
    When contamination is high, increase COI defense weight.
    """

    def __init__(self, base_lambda_coi: float = 0.3, max_lambda_coi: float = 2.0,
                 adaptation_rate: float = 2.0):
        self.base_lambda = base_lambda_coi
        self.max_lambda = max_lambda_coi
        self.rate = adaptation_rate

    def _adaptive_lambda(self, alpha: float) -> float:
        # sigmoid scaling: λ(α) = base + (max-base) * sigmoid(rate*(α-0.5))
        from ...outlet.math_util import sigmoid
        scale = sigmoid(self.rate * (alpha - 0.3))
        return self.base_lambda + (self.max_lambda - self.base_lambda) * scale

    def reward(self, quote: Quote, instruments: InstrumentSet,
               metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> float:
        alpha = hidden.contamination
        lambda_coi = self._adaptive_lambda(alpha)

        profit = metrics.revenue - metrics.cost
        margins = quote.prices - instruments.costs
        coi_penalty = lambda_coi * float(np.mean(margins)) * alpha

        return profit - coi_penalty

    def breakdown(self, quote: Quote, instruments: InstrumentSet,
                  metrics: StepMetrics, hidden: HiddenState, obs: Observation) -> dict[str, float]:
        alpha = hidden.contamination
        return {
            'profit': metrics.revenue - metrics.cost,
            'adaptive_lambda': self._adaptive_lambda(alpha),
            'contamination': alpha,
        }

def make_thesis_objective(lambda_coi: float = 0.5, lambda_ux: float = 0.1,
                          lambda_vol: float = 0.2) -> CompositeObjective:
    """Create the standard thesis objective composition."""
    return CompositeObjective([
        (RobustStackelbergObjective(RobustObjectiveConfig(
            lambda_coi=lambda_coi, lambda_ux=lambda_ux, lambda_volatility=lambda_vol)), 1.0),
    ])
