from sim.rl.behavior_loader.models import BehaviorModel, AgentBehaviorModel, aggregate_event_transitions
import pandas as pd
import numpy as np
from .demand import generate_demand

base_dir = "/home/velocitatem/Documents/Projects/PHANTOM/experiments"
human_dir, agent_dir = f"{base_dir}/collected_data/", f"{base_dir}/agents/collected_data/"


def adjust_behavior_to_condition(condition, transition_matrix):
    # transition matrix just maps probability of eventA to eventB
    # we enhance this that eventA-productI to eventB-productJ... based on the condition of interest
    # this is to simulate the impact of demand onto the behavior
    # NxN -> (N*(P + 1))x(N*(P + 1)) where P is number of products
    new_transitions = transition_matrix.copy()
    for col in new_transitions.columns:
        for product in range(len(condition)):
            # adjust the transition probability based on the demand condition
            newname = f"{col}_product{product}"
            new_transitions[newname] = new_transitions[col] * (condition[product] / np.sum(condition))
    for row in transition_matrix.index:
        for product in range(len(condition)):
            newname = f"{row}_product{product}"
            new_transitions.loc[newname] = new_transitions.loc[row] * (condition[product] / np.sum(condition))

    return new_transitions.fillna(0.0)

def sample_behavior(condition, human=True, max_len=40):
    model = BehaviorModel(human_dir) if human else AgentBehaviorModel(agent_dir)
    mdp = model.build_MDP()
    raw_events = aggregate_event_transitions(mdp) # this gets us transtition between events (blind to products or prices)
    # staet: {state: p} is raw_events we needc a matrix a pivot table
    events_pivot = pd.DataFrame(raw_events).fillna(0.0)
    # now adjust the transition matrix based on the condition to get a more informed transition matrix
    adjusted_transitions = adjust_behavior_to_condition(condition, events_pivot)

    trajectory = [np.random.choice(adjusted_transitions.index)]
    while len(trajectory) < max_len or 'checkout' in trajectory[-1]:
        probs = adjusted_transitions.loc[trajectory[-1]].values
        sample = np.random.choice(adjusted_transitions.columns, p=probs/np.sum(probs) if np.sum(probs) > 0 else None)
        trajectory.append(sample)
    return trajectory

if __name__ == "__main__":
    t=sample_behavior(generate_demand(np.array([10,20,30])), human=True)
    print(t)
    t=sample_behavior(generate_demand(np.array([10,20,30])), human=False)
    print(t)
