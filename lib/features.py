"""Unified featurization utilities for trajectory -> feature vector conversion
Used by both experiments/ml/ and sim/rl/ components
"""
import numpy as np
from collections import defaultdict
from typing import List, Dict, Callable, Optional, Any, Set
from datetime import datetime


def transition_histogram(events: List, state_fn: Callable, max_states: int = 50) -> np.ndarray:
    """compute normalized histogram of state transitions in trajectory
    events: list of event objects/dicts
    state_fn: function mapping event -> state string
    max_states: maximum dimensions for histogram
    """
    if len(events) < 2:
        return np.zeros(max_states, dtype=np.float32)
    states = [state_fn(e) for e in events]
    trans_counts = defaultdict(int)
    for s, s_next in zip(states, states[1:]):
        trans_counts[(s, s_next)] += 1
    total = sum(trans_counts.values())
    hist = np.array(list(trans_counts.values())[:max_states], dtype=np.float32)
    hist = np.pad(hist, (0, max(0, max_states - len(hist))))
    return hist / (total + 1e-10)


def temporal_signature(events: List, ts_fn: Callable) -> np.ndarray:
    """extract temporal features: mean/std/skew of inter-event times plus count
    events: list of event objects/dicts
    ts_fn: function mapping event -> timestamp (float seconds)
    returns: [mean_dt, std_dt, skew, n_intervals] array
    """
    if len(events) < 2:
        return np.zeros(4, dtype=np.float32)
    times = sorted([ts_fn(e) for e in events])
    diffs = np.diff(times).astype(np.float32)
    if len(diffs) == 0:
        return np.zeros(4, dtype=np.float32)
    mean_dt, std_dt = np.mean(diffs), np.std(diffs) + 1e-10
    skew = np.mean(((diffs - mean_dt) / std_dt) ** 3) if std_dt > 1e-8 else 0.0
    return np.array([mean_dt, std_dt, skew, len(diffs)], dtype=np.float32)


def state_coverage(events: List, state_fn: Callable, mdp_states: Set[str]) -> float:
    """fraction of MDP states visited by trajectory
    events: list of event objects/dicts
    state_fn: function mapping event -> state string
    mdp_states: set of all possible MDP states
    """
    if not mdp_states:
        return 0.0
    visited = set(state_fn(e) for e in events)
    return len(visited & mdp_states) / len(mdp_states)


def transition_entropy(events: List, state_fn: Callable) -> float:
    """compute entropy of transition distribution (randomness of navigation)
    higher entropy = more random browsing pattern
    """
    if len(events) < 2:
        return 0.0
    states = [state_fn(e) for e in events]
    trans_counts = defaultdict(int)
    for s, s_next in zip(states, states[1:]):
        trans_counts[(s, s_next)] += 1
    total = sum(trans_counts.values())
    probs = [c / total for c in trans_counts.values()]
    return -sum(p * np.log(p + 1e-10) for p in probs)


def event_type_distribution(events: List, event_name_fn: Callable) -> np.ndarray:
    """compute proportions of different event type categories
    returns: [page_view_ratio, hover_ratio, cart_ratio, purchase_ratio]
    """
    if not events:
        return np.zeros(4, dtype=np.float32)
    n = len(events)
    names = [event_name_fn(e).lower() for e in events]
    return np.array([
        sum(1 for nm in names if 'page' in nm or 'view' in nm) / n,
        sum(1 for nm in names if 'hover' in nm) / n,
        sum(1 for nm in names if 'cart' in nm) / n,
        sum(1 for nm in names if 'purchase' in nm or 'checkout' in nm) / n
    ], dtype=np.float32)


def featurize_trajectory(events: List, state_fn: Callable, ts_fn: Callable,
                         event_name_fn: Callable, mdp_states: Optional[Set[str]] = None,
                         output_dim: int = 64) -> np.ndarray:
    """convert trajectory to fixed-dimension feature vector
    events: list of event objects/dicts
    state_fn: function mapping event -> state string
    ts_fn: function mapping event -> timestamp (float)
    event_name_fn: function mapping event -> event name string
    mdp_states: optional set of all MDP states for coverage calculation
    output_dim: desired output dimension (will pad/truncate)
    """
    feats = []
    feats.extend(transition_histogram(events, state_fn, max_states=40))  # 40 dims
    feats.extend(temporal_signature(events, ts_fn))  # 4 dims
    feats.append(state_coverage(events, state_fn, mdp_states or set()))  # 1 dim
    feats.append(transition_entropy(events, state_fn))  # 1 dim
    feats.append(float(len(events)))  # trajectory length
    feats.append(float(len(set(state_fn(e) for e in events))))  # unique states
    feats.extend(event_type_distribution(events, event_name_fn))  # 4 dims

    feats = np.array(feats[:output_dim], dtype=np.float32)
    if len(feats) < output_dim:
        feats = np.pad(feats, (0, output_dim - len(feats)))
    return feats


def parse_timestamp(ts: Any) -> float:
    """parse various timestamp formats to float seconds"""
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
        except ValueError:
            return 0.0
    return 0.0
