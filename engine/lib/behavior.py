from sim.rl.behavior_loader.models import BehaviorModel, AgentBehaviorModel, aggregate_event_transitions
import pandas as pd
import numpy as np
from .demand import generate_demand_for_actor

base_dir = "/home/velocitatem/Documents/Projects/PHANTOM/experiments"
human_dir, agent_dir = f"{base_dir}/collected_data/", f"{base_dir}/agents/collected_data/"

_cache = {}  # lazy cache for models and base pivots

def _get_base_pivot(human: bool):
    key = 'human' if human else 'agent'
    if key not in _cache:
        model = BehaviorModel(human_dir) if human else AgentBehaviorModel(agent_dir)
        mdp = model.build_MDP()
        _cache[key] = pd.DataFrame(aggregate_event_transitions(mdp)).fillna(0.0)
    return _cache[key]

def adjust_behavior_to_condition(condition, transition_matrix):
    # expand NxN transition matrix to (N*P)x(N*P) weighted by demand condition
    cond_norm = condition / np.sum(condition)
    n_products = len(condition)
    base_vals = transition_matrix.values
    base_cols, base_rows = transition_matrix.columns.tolist(), transition_matrix.index.tolist()

    # expand via kronecker-like tiling: each cell becomes a P*P block weighted by outer product of cond_norm
    expanded = np.kron(base_vals, np.outer(cond_norm, cond_norm))
    new_cols = [f"{c}_product{p}" for c in base_cols for p in range(n_products)]
    new_rows = [f"{r}_product{p}" for r in base_rows for p in range(n_products)]
    return pd.DataFrame(expanded, index=new_rows, columns=new_cols)

def sample_behavior(condition, human=True, max_len=40):
    base_pivot = _get_base_pivot(human)
    adjusted_transitions = adjust_behavior_to_condition(condition, base_pivot)

    trajectory = [np.random.choice(adjusted_transitions.index)]
    while len(trajectory) < max_len or 'checkout' in trajectory[-1]:
        probs = adjusted_transitions.loc[trajectory[-1]].values
        sample = np.random.choice(adjusted_transitions.columns, p=probs/np.sum(probs) if np.sum(probs) > 0 else None)
        trajectory.append(sample)
    return trajectory

if __name__ == "__main__":
    t=sample_behavior(generate_demand_for_actor(np.array([10,20,30])), human=True)
    print(t)
    t=sample_behavior(generate_demand_for_actor(np.array([10,20,30])), human=False)
    print(t)
