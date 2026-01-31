"""Cost of Information (COI) computation for thesis pricing system.

Core KPI: COI = E[p_shown] - p_min measures pricing power from information asymmetry.
Theorem 1 shows COI erodes as agent queries increase: as N->inf, p^(1)->p_min.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .simplified import Session


@dataclass(frozen=True)
class COIWindow:
    """Windowed COI metrics computed from realized price exposures.

    policy: E[p_shown] - cost, the definition-level KPI
    agent: E[p^(1)] - cost where p^(1) is min price under agent querying
    leak: max(policy - agent, 0), observable gap from reconnaissance
    survival_ratio: agent/policy, fraction of pricing power retained
    """
    policy: float
    agent: float
    leak: float
    survival_ratio: float
    policy_by_product: np.ndarray
    agent_by_product: np.ndarray
    demand_weights: np.ndarray


def aggregate_prices(sessions: List["Session"], mode: str = "all") -> Dict[int, List[float] | float]:
    """Unified price aggregation across sessions.

    mode: "all" returns all prices per product, "min_per_session" returns min price per session per product,
          "min_across" returns single min price per product
    """
    if mode == "min_across":
        mins: Dict[int, float] = {}
        for s in sessions:
            for e in s.events:
                pidx, price = int(e.product_idx), float(e.price_seen)
                mins[pidx] = min(mins.get(pidx, price), price)
        return mins
    elif mode == "min_per_session":
        result: Dict[int, List[float]] = {}
        for s in sessions:
            by_p: Dict[int, float] = {}
            for e in s.events:
                pidx, price = int(e.product_idx), float(e.price_seen)
                by_p[pidx] = min(by_p.get(pidx, price), price)
            for pidx, pmin in by_p.items():
                result.setdefault(pidx, []).append(pmin)
        return result
    else:  # "all"
        prices: Dict[int, List[float]] = {}
        for s in sessions:
            for e in s.events:
                prices.setdefault(e.product_idx, []).append(float(e.price_seen))
        return prices


def demand_weights_by_product(sessions: List["Session"], demand_mapping: Dict[str, float], n_products: int) -> np.ndarray:
    """Compute demand-weighted importance per product."""
    w = np.zeros(n_products, dtype=float)
    sessions_by_id = {s.sid: s for s in sessions}
    for sid, q in demand_mapping.items():
        sess = sessions_by_id.get(sid)
        if sess and sess.events:
            w[int(sess.events[0].product_idx)] += float(q)
    total = float(np.sum(w))
    return (w / total) if total > 0 else w


def compute_coi_window(sessions: List["Session"], costs: np.ndarray, demand_mapping: Dict[str, float] | None = None) -> COIWindow:
    """Compute COI metrics over session window.

    Aggregates price exposures and computes policy-level vs agent-realized COI.
    """
    n = int(len(costs))
    prices = aggregate_prices(sessions, mode="all")
    agent_sessions = [s for s in sessions if s.actor == "A"]
    agent_min = aggregate_prices(agent_sessions, mode="min_across") if agent_sessions else {}

    policy_by = np.zeros(n, dtype=float)
    agent_by = np.zeros(n, dtype=float)
    seen = np.array([(i in prices) for i in range(n)], dtype=bool)
    agent_seen = np.array([(i in agent_min) for i in range(n)], dtype=bool)

    for pidx, ps in prices.items():
        if 0 <= pidx < n and ps:
            policy_by[pidx] = float(np.mean(ps) - float(costs[pidx]))
    for pidx, pmin in agent_min.items():
        if 0 <= pidx < n:
            agent_by[pidx] = float(pmin - float(costs[pidx]))

    agent_by[seen & ~agent_seen] = policy_by[seen & ~agent_seen]  # no erosion if no agent exposure

    demand_w = demand_weights_by_product(sessions, demand_mapping, n) if demand_mapping else np.zeros(n, dtype=float)
    has_weights = float(np.sum(demand_w)) > 0

    if has_weights:
        policy, agent = float(np.dot(demand_w, policy_by)), float(np.dot(demand_w, agent_by))
    elif np.any(seen):
        policy, agent = float(np.mean(policy_by[seen])), float(np.mean(agent_by[seen]))
    else:
        policy, agent = 0.0, 0.0

    leak = float(max(policy - agent, 0.0))
    survival = float(np.clip(agent / policy, 0.0, 1.0)) if policy > 0 else 0.0

    return COIWindow(policy=policy, agent=agent, leak=leak, survival_ratio=survival,
                     policy_by_product=policy_by, agent_by_product=agent_by, demand_weights=demand_w)


def coi_erosion(coi_policy: float, coi_agent: float, eps: float = 1e-9) -> float:
    """Thesis-consistent COI erosion: fraction of pricing power destroyed by agent queries.

    erosion = 1 - (COI_agent / COI_policy)
    When agents find low prices, COI_agent -> 0, erosion -> 1.
    """
    if coi_policy <= eps:
        return 0.0
    return float(np.clip(1.0 - (coi_agent / (coi_policy + eps)), 0.0, 1.0))
