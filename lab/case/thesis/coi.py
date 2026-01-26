"""Cost of Information (COI) computation for thesis pricing simulation.

Implements the corrected COI formulation:

    COI = E[p] - p

where:
- E[p] = expected price BEFORE information revelation (window start price)
- p = actual transaction price (price at which sales occur)

The fundamental insight is that COI should measure PRICE EROSION over time,
not instantaneous margin leakage. When agents explore across sessions:
1. They reveal demand signals that drive platform price adjustments
2. Coordinated agents can find the minimum price across their session pool
3. The price path from window start to transaction captures information leakage

Key components:
- COIWindow: Windowed price erosion measurement over K steps
- compute_coi_window: Per-episode COI from session-level transactions
- coi_erosion: Order statistic erosion (Theorem 1: N agents -> min price)

This fixes the fundamental error of treating COI as instantaneous margin × alpha.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .simplified import Session

EPS = 1e-10


@dataclass
class COIWindow:
    """Windowed COI measurement capturing price erosion over time.

    Attributes:
        policy: Platform's intended COI (prices at window start - cost)
        agent: Realized COI for agents (prices at transaction - cost)
        leak: COI leakage = policy - agent (price erosion due to exploration)
        survival_ratio: Fraction of intended COI that survives (agent/policy)
        policy_by_product: Per-product policy COI
        agent_by_product: Per-product agent COI
        demand_weights: Demand weights used for aggregation
    """
    policy: float = 0.0          # E[p] - c at window start
    agent: float = 0.0           # p_transaction - c
    leak: float = 0.0            # policy - agent = price erosion
    survival_ratio: float = 1.0  # agent / policy
    policy_by_product: np.ndarray = field(default_factory=lambda: np.zeros(1))
    agent_by_product: np.ndarray = field(default_factory=lambda: np.zeros(1))
    demand_weights: np.ndarray = field(default_factory=lambda: np.zeros(1))

    def to_dict(self) -> Dict[str, float]:
        return {
            'coi_policy': self.policy,
            'coi_agent': self.agent,
            'coi_leak': self.leak,
            'coi_survival': self.survival_ratio,
        }


def compute_coi_window(
    sessions: List["Session"],
    costs: np.ndarray,
    demand_mapping: Dict[str, float] = None,
    window_prices: np.ndarray = None,
) -> COIWindow:
    """Compute COI from session data using the corrected formulation.

    COI = E[p_start] - p_transaction

    This measures how much the platform's pricing power eroded during the window.
    Price at window start represents E[p] (what we expected to charge).
    Transaction prices represent p (what we actually charged).

    Args:
        sessions: List of sessions with events containing price_seen and purchases
        costs: Product costs array
        demand_mapping: Optional session_id -> demand proxy mapping
        window_prices: Optional explicit window start prices (otherwise use first seen)

    Returns:
        COIWindow with erosion metrics
    """
    if not sessions:
        n = len(costs)
        zeros = np.zeros(n)
        return COIWindow(policy=0.0, agent=0.0, leak=0.0, survival_ratio=1.0,
                        policy_by_product=zeros, agent_by_product=zeros, demand_weights=zeros)

    n = len(costs)
    demand_mapping = demand_mapping or {}

    # Track prices seen at start (E[p]) and transaction prices (p)
    first_prices = np.zeros(n)  # first price seen per product (window start proxy)
    transaction_prices = np.zeros(n)  # prices at which purchases occurred
    transaction_counts = np.zeros(n)
    view_counts = np.zeros(n)
    demand_weights = np.zeros(n)

    for sess in sessions:
        sid = sess.sid
        sess_demand = demand_mapping.get(sid, 1.0)

        for e in sess.events:
            pidx = e.product_idx
            if pidx < 0 or pidx >= n:
                continue

            price_seen = float(e.price_seen)

            # Track first price seen (proxy for E[p] at window start)
            if view_counts[pidx] == 0:
                first_prices[pidx] = price_seen
            view_counts[pidx] += 1

            # Track transaction prices
            if e.action == "purchase":
                transaction_prices[pidx] += price_seen
                transaction_counts[pidx] += 1
                demand_weights[pidx] += sess_demand

    # Compute per-product COI
    # Policy COI: what we intended to charge (first seen price - cost)
    policy_by_product = np.zeros(n)
    agent_by_product = np.zeros(n)

    for i in range(n):
        if view_counts[i] > 0:
            # Use explicit window prices if provided, else first seen
            start_price = window_prices[i] if window_prices is not None else first_prices[i]
            policy_by_product[i] = max(0, start_price - costs[i])

        if transaction_counts[i] > 0:
            avg_transaction = transaction_prices[i] / transaction_counts[i]
            agent_by_product[i] = max(0, avg_transaction - costs[i])

    # Aggregate with demand weighting
    total_demand = np.sum(demand_weights) + EPS
    weights = demand_weights / total_demand

    # Only count products with transactions for fair comparison
    active_mask = transaction_counts > 0
    if np.any(active_mask):
        policy = float(np.sum(policy_by_product[active_mask] * weights[active_mask]) /
                      (np.sum(weights[active_mask]) + EPS))
        agent = float(np.sum(agent_by_product[active_mask] * weights[active_mask]) /
                     (np.sum(weights[active_mask]) + EPS))
    else:
        # No transactions - use view-weighted policy COI
        view_weights = view_counts / (np.sum(view_counts) + EPS)
        policy = float(np.sum(policy_by_product * view_weights))
        agent = policy  # No erosion without transactions

    # Leak = price erosion due to information revelation
    leak = max(0, policy - agent)
    survival = agent / (policy + EPS) if policy > EPS else 1.0

    return COIWindow(
        policy=policy,
        agent=agent,
        leak=leak,
        survival_ratio=float(np.clip(survival, 0, 1)),
        policy_by_product=policy_by_product,
        agent_by_product=agent_by_product,
        demand_weights=demand_weights,
    )


def coi_erosion(policy_coi: float, agent_coi: float) -> float:
    """Compute COI erosion rate: (policy - agent) / policy.

    Returns the fraction of intended COI that was lost to information leakage.
    0 = no erosion, 1 = complete erosion.
    """
    if policy_coi < EPS:
        return 0.0
    return float(np.clip((policy_coi - agent_coi) / policy_coi, 0, 1))


def order_statistic_erosion(n_agents: int, price_std: float, base_margin: float = 1.0) -> float:
    """Compute COI erosion from order statistic effect (Theorem 1).

    When N agents independently query prices:
    - Each sees a price p_i ~ N(μ, σ²)
    - They coordinate to buy at min(p_1, ..., p_N)
    - Expected minimum: μ - σ * E[order_stat]

    As N -> ∞, E[min] -> p_min, so COI -> 0.

    This quantifies the price discovery benefit of multiple sessions.

    Args:
        n_agents: Number of independent agent sessions
        price_std: Standard deviation of price distribution
        base_margin: Expected margin (μ - cost)

    Returns:
        Erosion rate in [0, 1]
    """
    if n_agents <= 1 or price_std < EPS:
        return 0.0

    # For standard normal order statistics, E[min of N] ≈ -Φ^{-1}(1/(N+1))
    # For large N, this grows like sqrt(2 * log(N))
    log_n = np.log(n_agents)
    if log_n < 0.1:
        return 0.0

    # Extreme value theory: expected min shift
    shift = price_std * (np.sqrt(2 * log_n) -
                        (np.log(log_n) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * log_n) + EPS))

    # Erosion = shift / base_margin, capped at 1
    return float(np.clip(shift / (base_margin + EPS), 0, 1))


@dataclass
class COITracker:
    """Track COI over multiple windows for temporal analysis.

    This addresses the user's insight: compute COI over K episodes to see
    how prices change from window start to end.

    If at start of window price is A and by end it's B, the difference
    A - B represents COI leakage from exploratory sessions.
    """
    window_size: int = 10  # K episodes per window
    _price_history: List[np.ndarray] = field(default_factory=list)
    _transaction_history: List[np.ndarray] = field(default_factory=list)
    _coi_history: List[float] = field(default_factory=list)

    def add_step(self, prices: np.ndarray, transactions: np.ndarray = None):
        """Record price observation for current step."""
        self._price_history.append(prices.copy())
        if transactions is not None:
            self._transaction_history.append(transactions.copy())

    def compute_window_coi(self, costs: np.ndarray) -> float:
        """Compute COI over the current window.

        COI = E[p_start] - E[p_end] for the window.
        This captures price erosion due to information revelation.
        """
        if len(self._price_history) < 2:
            return 0.0

        # Get prices at window boundaries
        window_start = max(0, len(self._price_history) - self.window_size)
        start_prices = self._price_history[window_start]
        end_prices = self._price_history[-1]

        # COI = (start_price - cost) - (end_price - cost) = start_price - end_price
        start_margin = np.mean(start_prices - costs)
        end_margin = np.mean(end_prices - costs)

        coi = max(0, start_margin - end_margin)
        self._coi_history.append(coi)
        return coi

    def get_cumulative_erosion(self, costs: np.ndarray) -> float:
        """Compute total COI erosion from first observation to now."""
        if len(self._price_history) < 2:
            return 0.0

        initial = np.mean(self._price_history[0] - costs)
        current = np.mean(self._price_history[-1] - costs)
        return max(0, initial - current)

    def get_erosion_trend(self) -> float:
        """Get average COI per window (erosion rate)."""
        if not self._coi_history:
            return 0.0
        return float(np.mean(self._coi_history))

    def reset(self):
        """Reset tracker for new episode."""
        self._price_history.clear()
        self._transaction_history.clear()
        self._coi_history.clear()


def compute_multi_session_coi(
    sessions: List["Session"],
    costs: np.ndarray,
    alpha: float,
    initial_prices: np.ndarray,
) -> Dict[str, float]:
    """Compute COI accounting for multi-session agent behavior.

    This is the key fix for the fundamental error:
    - Agents use different sessions to gather information
    - Each session reveals price information
    - Coordinated agents find the minimum across their session pool

    The COI is computed as:
    1. What platform intended to charge: initial_prices - costs
    2. What agents actually paid: min(prices seen across sessions) - costs
    3. Leak = (1) - (2)

    Args:
        sessions: All sessions in the episode
        costs: Product costs
        alpha: Contamination level (fraction of agent sessions)
        initial_prices: Prices at episode start (E[p])

    Returns:
        Dictionary with COI metrics
    """
    n = len(costs)

    # Separate agent and human sessions by ground truth label
    agent_sessions = [s for s in sessions if s.actor == "A"]
    human_sessions = [s for s in sessions if s.actor == "H"]

    # Track prices seen by agents per product (for min finding)
    agent_prices_seen: Dict[int, List[float]] = {i: [] for i in range(n)}
    human_prices_paid: Dict[int, List[float]] = {i: [] for i in range(n)}

    for sess in agent_sessions:
        for e in sess.events:
            if 0 <= e.product_idx < n:
                agent_prices_seen[e.product_idx].append(e.price_seen)

    for sess in human_sessions:
        for e in sess.events:
            if 0 <= e.product_idx < n and e.action == "purchase":
                human_prices_paid[e.product_idx].append(e.price_seen)

    # Compute COI components
    policy_coi = float(np.mean(initial_prices - costs))  # E[p] - c

    # Agent COI: they find the minimum price via exploration
    agent_coi_by_product = np.zeros(n)
    for i in range(n):
        if agent_prices_seen[i]:
            min_price = min(agent_prices_seen[i])
            agent_coi_by_product[i] = max(0, min_price - costs[i])
        else:
            agent_coi_by_product[i] = initial_prices[i] - costs[i]

    agent_coi = float(np.mean(agent_coi_by_product))

    # Human COI: they pay whatever price is offered
    human_coi_by_product = np.zeros(n)
    for i in range(n):
        if human_prices_paid[i]:
            avg_price = np.mean(human_prices_paid[i])
            human_coi_by_product[i] = max(0, avg_price - costs[i])
        else:
            human_coi_by_product[i] = initial_prices[i] - costs[i]

    human_coi = float(np.mean(human_coi_by_product))

    # Total leak: weighted by contamination
    # Agents erode COI, humans pay full price
    realized_coi = (1 - alpha) * human_coi + alpha * agent_coi
    leak = policy_coi - realized_coi

    # Order statistic effect: more agents = more erosion
    n_agents = len(agent_sessions)
    price_std = float(np.std(initial_prices))
    order_erosion = order_statistic_erosion(n_agents, price_std, policy_coi)

    return {
        'policy_coi': policy_coi,
        'agent_coi': agent_coi,
        'human_coi': human_coi,
        'realized_coi': realized_coi,
        'leak': leak,
        'order_stat_erosion': order_erosion,
        'n_agent_sessions': n_agents,
        'n_human_sessions': len(human_sessions),
        'survival_ratio': realized_coi / (policy_coi + EPS) if policy_coi > EPS else 1.0,
    }
