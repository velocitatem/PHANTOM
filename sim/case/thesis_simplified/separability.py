"""Behavioral separability for human/agent detection.

Computes divergence signals delta_H, delta_A from session trajectories using
transition kernel estimation and KL divergence to prototype behavioral profiles.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .simplified import Event, Session


# prototype behavioral kernels for human vs agent sessions
TRANS_H = {
    "start": {"view": 0.85, "end": 0.15},
    "view": {"detail": 0.4, "cart": 0.3, "view": 0.2, "end": 0.1},
    "detail": {"cart": 0.5, "view": 0.3, "end": 0.2},
    "cart": {"purchase": 0.6, "view": 0.25, "end": 0.15},
    "purchase": {"end": 1.0},
}

TRANS_A = {
    "start": {"view": 0.95, "end": 0.05},
    "view": {"detail": 0.6, "view": 0.25, "cart": 0.1, "end": 0.05},
    "detail": {"view": 0.5, "cart": 0.15, "detail": 0.3, "end": 0.05},
    "cart": {"view": 0.4, "purchase": 0.2, "end": 0.4},
    "purchase": {"end": 1.0},
}


def kl_div(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-10) -> float:
    """KL divergence D_KL(p || q) for discrete distributions."""
    keys = set(p.keys()) | set(q.keys())
    return sum(p.get(k, eps) * np.log((p.get(k, eps) + eps) / (q.get(k, eps) + eps)) for k in keys)


def build_kernel(events: List["Event"]) -> Dict[str, Dict[str, float]]:
    """Build empirical transition kernel T' from trajectory events."""
    trans: Dict[str, Dict[str, int]] = {}
    prev = "start"
    for e in events:
        curr = e.action
        trans.setdefault(prev, {})
        trans[prev][curr] = trans[prev].get(curr, 0) + 1
        prev = curr
    return {s: {d: c / sum(dsts.values()) for d, c in dsts.items()} for s, dsts in trans.items() if sum(dsts.values()) > 0}


def compute_divergence(session: "Session") -> Tuple[float, float]:
    """Compute divergence signals delta_H, delta_A for session.

    delta_H = mean KL(T' || T_H) across states, measures distance to human prototype
    delta_A = mean KL(T' || T_A) across states, measures distance to agent prototype
    """
    kernel = build_kernel(session.events)
    if not kernel:
        return 0.5, 0.5
    delta_h = sum(kl_div(kernel.get(s, {}), TRANS_H.get(s, {})) for s in kernel) / len(kernel)
    delta_a = sum(kl_div(kernel.get(s, {}), TRANS_A.get(s, {})) for s in kernel) / len(kernel)
    return delta_h, delta_a


def estimate_alpha(session: "Session", beta: float = 2.0) -> float:
    """Per-session contamination estimate alpha_hat = sigma(beta*(delta_H - delta_A)).

    Returns probability session is agent-generated based on behavioral divergence.
    """
    dh, da = compute_divergence(session)
    if (dh + da) <= 0:
        return 0.5
    return 1.0 / (1.0 + np.exp(-beta * (dh - da)))
