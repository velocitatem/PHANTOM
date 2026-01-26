"""Behavioral separability for thesis human/agent classification.

Implements KL-divergence based separability scoring (Eq 20-21):
- Δ_H = D_KL(T̂' || T̄_H): divergence from human reference kernel
- Δ_A = D_KL(T̂' || T̄_A): divergence from agent reference kernel
- α̂(τ') = σ(β(Δ_H - Δ_A)): per-session contamination estimate
"""
from __future__ import annotations
from typing import Dict, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .simplified import Session


# Reference transition kernels T̄_H, T̄_A estimated from real data (Eq 19)
TRANS_H = {
    "start": {"view": 0.85, "end": 0.15},
    "view": {"detail": 0.4, "add_to_cart": 0.3, "view": 0.2, "end": 0.1},
    "detail": {"add_to_cart": 0.5, "view": 0.3, "end": 0.2},
    "add_to_cart": {"purchase": 0.6, "view": 0.25, "end": 0.15},
    "purchase": {"end": 1.0},
    "checkout": {"purchase": 0.8, "end": 0.2},
    "hover": {"view": 0.5, "detail": 0.3, "end": 0.2},
}

TRANS_A = {
    "start": {"view": 0.95, "end": 0.05},
    "view": {"detail": 0.6, "view": 0.25, "add_to_cart": 0.1, "end": 0.05},
    "detail": {"view": 0.5, "add_to_cart": 0.15, "detail": 0.3, "end": 0.05},
    "add_to_cart": {"view": 0.4, "purchase": 0.2, "end": 0.4},
    "purchase": {"end": 1.0},
    "checkout": {"purchase": 0.3, "end": 0.7},
    "hover": {"view": 0.6, "detail": 0.35, "end": 0.05},
}


def kl_div(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-10) -> float:
    """Compute KL(p || q) with smoothing."""
    if not p or not q:
        return 0.0
    all_keys = set(p.keys()) | set(q.keys())
    total = 0.0
    for k in all_keys:
        pk = p.get(k, eps)
        qk = q.get(k, eps)
        if pk > eps:
            total += pk * np.log(pk / max(qk, eps))
    return max(0.0, total)


def build_kernel(events: List) -> Dict[str, Dict[str, float]]:
    """Build empirical transition kernel from event sequence."""
    trans: Dict[str, Dict[str, int]] = {}
    prev = "start"
    for e in events:
        curr = getattr(e, 'action', None) or e.get('action', 'end') if isinstance(e, dict) else 'end'
        trans.setdefault(prev, {})
        trans[prev][curr] = trans[prev].get(curr, 0) + 1
        prev = curr
    # add terminal transition
    trans.setdefault(prev, {})
    trans[prev]["end"] = trans[prev].get("end", 0) + 1

    # normalize to probabilities
    kernel = {}
    for s, dests in trans.items():
        total = sum(dests.values())
        kernel[s] = {d: c / total for d, c in dests.items()} if total > 0 else {"end": 1.0}
    return kernel


def compute_divergence(kernel: Dict[str, Dict[str, float]], ref_h: Dict = None, ref_a: Dict = None) -> tuple[float, float]:
    """Compute Δ_H, Δ_A divergence from reference kernels (Eq 20-21)."""
    ref_h = ref_h or TRANS_H
    ref_a = ref_a or TRANS_A
    delta_h = sum(kl_div(kernel.get(s, {}), ref_h.get(s, {})) for s in kernel) / max(len(kernel), 1)
    delta_a = sum(kl_div(kernel.get(s, {}), ref_a.get(s, {})) for s in kernel) / max(len(kernel), 1)
    return delta_h, delta_a


def estimate_alpha(session: "Session", beta: float = 2.0) -> float:
    """Estimate per-session contamination α̂(τ') = σ(β(Δ_H - Δ_A)).

    High Δ_H (far from human) and low Δ_A (close to agent) -> high α̂ (likely agent).
    """
    if not session.events:
        return 0.5
    kernel = build_kernel(session.events)
    delta_h, delta_a = compute_divergence(kernel)

    if delta_h + delta_a < 1e-6:
        return 0.5

    # sigmoid: high when trajectory is more divergent from human than agent
    return 1.0 / (1.0 + np.exp(-beta * (delta_h - delta_a)))


def batch_estimate_alpha(sessions: List["Session"]) -> tuple[float, List[float]]:
    """Estimate aggregate and per-session contamination."""
    if not sessions:
        return 0.0, []
    alphas = [estimate_alpha(s) for s in sessions]
    return float(np.mean(alphas)), alphas
