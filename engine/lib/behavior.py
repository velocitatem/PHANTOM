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

    args:
        trajectory: list like ['view_product0', 'add_to_cart_product1', 'checkout_product1']

    returns:
        list: event names like ['view', 'add_to_cart', 'checkout']
    """
    events = []
    for state in trajectory:
        # state format from sample_behavior: 'eventName_productX'
        if "_product" in state:
            event = state.rsplit("_product", 1)[0]
        else:
            event = state
        events.append(event)
    return events


def adjust_behavior_to_condition(condition, transition_matrix):
    # expand NxN transition matrix to (N*P)x(N*P) weighted by demand condition
    condition = np.asarray(condition, dtype=float)
    condition = np.nan_to_num(condition, nan=0.0, posinf=0.0, neginf=0.0)
    condition = np.clip(condition, 0.0, None)
    s = float(np.sum(condition))
    if not np.isfinite(s) or s <= 0:
        cond_norm = np.full(len(condition), 1.0 / max(len(condition), 1), dtype=float)
    else:
        cond_norm = condition / s
    n_products = len(condition)
    base_vals = transition_matrix.values
    base_cols, base_rows = (
        transition_matrix.columns.tolist(),
        transition_matrix.index.tolist(),
    )

    # expand via kronecker-like tiling: each cell becomes a P*P block weighted by outer product of cond_norm
    expanded = np.kron(base_vals, np.outer(cond_norm, cond_norm))
    new_cols = [f"{c}_product{p}" for c in base_cols for p in range(n_products)]
    new_rows = [f"{r}_product{p}" for r in base_rows for p in range(n_products)]
    return pd.DataFrame(expanded, index=new_rows, columns=new_cols)


def get_adjusted_transitions(condition, human=True):
    base_pivot = _get_base_pivot(human)
    return adjust_behavior_to_condition(condition, base_pivot)


def sample_behavior_from_transitions(adjusted_transitions, max_len=40):
    trajectory = [np.random.choice(adjusted_transitions.index)]
    while len(trajectory) < max_len and "checkout" not in trajectory[-1]:
        probs = np.asarray(adjusted_transitions.loc[trajectory[-1]].values, dtype=float)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 0.0, None)
        s = float(np.sum(probs))
        sample = np.random.choice(
            adjusted_transitions.columns, p=(probs / s) if s > 0 else None
        )
        trajectory.append(sample)
    return trajectory


def sample_behavior(condition, human=True, max_len=40):
    adjusted_transitions = get_adjusted_transitions(condition, human=human)
    return sample_behavior_from_transitions(adjusted_transitions, max_len=max_len)


if __name__ == "__main__":
    t = sample_behavior(generate_demand_for_actor(np.array([10, 20, 30])), human=True)
    print(t)
    t = sample_behavior(generate_demand_for_actor(np.array([10, 20, 30])), human=False)
    print(t)
