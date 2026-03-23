import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

try:
    from sim.rl.behavior_loader.models import (
        BehaviorModel,
        AgentBehaviorModel,
        aggregate_event_transitions,
    )
except ImportError:
    BehaviorModel = None
    AgentBehaviorModel = None
    aggregate_event_transitions = None
import pandas as pd
import numpy as np
from .demand import generate_demand_for_actor

base_dir = Path(__file__).parents[2] / "experiments"
human_dir = str(base_dir / "collected_data")
agent_dir = str(base_dir / "agents" / "collected_data")

_cache = {}  # lazy cache for models and base pivots
# cache keyed by (human: bool, condition_tuple) so we skip Kronecker re-expansion
# for repeated calls with the same demand condition inside the robustness inner loop
_transition_cache: dict = {}


def _get_base_pivot(human: bool):
    if (
        BehaviorModel is None
        or AgentBehaviorModel is None
        or aggregate_event_transitions is None
    ):
        raise ImportError("behavior loader dependencies are unavailable")
    key = "human" if human else "agent"
    if key not in _cache:
        model = BehaviorModel(human_dir) if human else AgentBehaviorModel(agent_dir)
        mdp = model.build_MDP()
        _cache[key] = pd.DataFrame(aggregate_event_transitions(mdp)).fillna(0.0)
    return _cache[key]


def get_transition_models():
    """load human and agent transition models for agent probability calculation

    returns:
        tuple: (human_transitions, agent_transitions) as dicts of event->event->prob
    """
    if (
        BehaviorModel is None
        or AgentBehaviorModel is None
        or aggregate_event_transitions is None
    ):
        raise ImportError("behavior loader dependencies are unavailable")

    human_model = BehaviorModel(human_dir)
    agent_model = AgentBehaviorModel(agent_dir)

    human_mdp = human_model.build_MDP()
    agent_mdp = agent_model.build_MDP()

    human_trans = aggregate_event_transitions(human_mdp)
    agent_trans = aggregate_event_transitions(agent_mdp)

    return human_trans, agent_trans


def trajectory_to_events(trajectory: list) -> list:
    """extract event names from trajectory for KL divergence calculation

    trajectories are in format 'eventName_product0', extract just eventName
    """
    return [s.rsplit("_product", 1)[0] if "_product" in s else s for s in trajectory]


class _TransitionTable:
    """numpy-backed transition table; replaces per-step pandas .loc[] indexing.

    the profiling hotspot was DataFrame.xs called ~4-16k times per outer step.
    converting once to a dense float32 array with an int-keyed state index map
    reduces each row lookup to a single array slice with no pandas overhead.
    rows are pre-normalized so sampling requires no per-step division.
    """

    __slots__ = ("matrix", "states", "state_index", "n_states")

    def __init__(self, df: pd.DataFrame):
        self.states: list[str] = df.index.tolist()
        self.state_index: dict[str, int] = {s: i for i, s in enumerate(self.states)}
        # float64 throughout: float32 row-sums can drift enough to break np.random.choice
        mat = np.nan_to_num(
            df.values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        mat = np.clip(mat, 0.0, None)
        row_sums = mat.sum(axis=1)
        # dead rows (all zero) get uniform distribution so sampling never receives NaN
        dead = row_sums <= 0
        mat[dead] = 1.0
        row_sums[dead] = float(mat.shape[1])
        mat = mat / row_sums[:, np.newaxis]
        # final nan guard in case fp still drifts
        np.nan_to_num(mat, nan=0.0, copy=False)
        row_sums2 = mat.sum(axis=1, keepdims=True)
        row_sums2[row_sums2 <= 0] = 1.0
        self.matrix: np.ndarray = mat / row_sums2
        self.n_states: int = len(self.states)


def adjust_behavior_to_condition(condition, transition_matrix):
    # expand NxN transition matrix to (N*P)x(N*P) weighted by demand condition
    condition = np.asarray(condition, dtype=float)
    condition = np.nan_to_num(condition, nan=0.0, posinf=0.0, neginf=0.0)
    condition = np.clip(condition, 0.0, None)
    s = float(np.sum(condition))
    cond_norm = (
        condition / s
        if np.isfinite(s) and s > 0
        else np.full(len(condition), 1.0 / max(len(condition), 1), dtype=float)
    )
    n_products = len(condition)
    base_vals = transition_matrix.values
    base_cols, base_rows = (
        transition_matrix.columns.tolist(),
        transition_matrix.index.tolist(),
    )
    expanded = np.kron(base_vals, np.outer(cond_norm, cond_norm))
    new_cols = [f"{c}_product{p}" for c in base_cols for p in range(n_products)]
    new_rows = [f"{r}_product{p}" for r in base_rows for p in range(n_products)]
    return pd.DataFrame(expanded, index=new_rows, columns=new_cols)


def get_adjusted_transitions(condition, human=True) -> _TransitionTable:
    """return a _TransitionTable for the given demand condition.

    results are cached by (human, rounded-condition) so that repeated calls with
    the same condition inside the robustness inner loop (K candidates, same prices)
    skip the Kronecker expansion entirely.
    """
    condition = np.asarray(condition, dtype=float)
    # round to 4 significant digits for cache key stability
    cache_key = (human, tuple(np.round(condition, 4).tolist()))
    if cache_key in _transition_cache:
        return _transition_cache[cache_key]

    # prevent OOM by capping cache size
    if len(_transition_cache) > 100:
        _transition_cache.clear()

    base_pivot = _get_base_pivot(human)
    df = adjust_behavior_to_condition(condition, base_pivot)
    table = _TransitionTable(df)
    _transition_cache[cache_key] = table
    return table


def clear_transition_cache():
    """drop cached transition tables; call between episodes if condition space is large."""
    _transition_cache.clear()


def sample_behavior_from_transitions(table, max_len=40):
    """sample a Markov trajectory.

    accepts _TransitionTable (fast path) or a legacy pandas DataFrame so existing
    call sites that pass a DataFrame directly continue to work unchanged.
    """
    if isinstance(table, pd.DataFrame):
        table = _TransitionTable(table)

    idx = np.random.randint(table.n_states)
    trajectory = [table.states[idx]]
    while len(trajectory) < max_len and "checkout" not in trajectory[-1]:
        row = table.matrix[table.state_index[trajectory[-1]]]
        idx = int(np.random.choice(table.n_states, p=row))
        trajectory.append(table.states[idx])
    return trajectory


def sample_behavior(condition, human=True, max_len=40):
    table = get_adjusted_transitions(condition, human=human)
    return sample_behavior_from_transitions(table, max_len=max_len)


if __name__ == "__main__":
    t = sample_behavior(generate_demand_for_actor(np.array([10, 20, 30])), human=True)
    print(t)
    t = sample_behavior(generate_demand_for_actor(np.array([10, 20, 30])), human=False)
    print(t)
