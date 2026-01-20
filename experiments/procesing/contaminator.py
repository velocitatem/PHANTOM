import pandas as pd
import random
from sim.rl.behavior_loader import AgentBehaviorModel

base_dir = "/home/velocitatem/Documents/Projects/PHANTOM/experiments"
human_dir, agent_dir = f"{base_dir}/collected_data/", f"{base_dir}/agents/collected_data/"



def remap_schema(df : pd.DataFrame, mapping: dict, on: str = "event_type"):
    df = df.copy()
    df[on] = df[on].map(mapping).fillna(df[on])
    return df


def contaminate_dataset(df : pd.DataFrame, on : str = "event_type",
                        contamination_rate: float = 0.1) -> pd.DataFrame:
    model = AgentBehaviorModel(agent_dir)
    target_df_schema = df[on].unique().tolist()
    mapping = {
        'view': 'view_page'
        # TODO: define properly for the given dataset
    }
    OG_event_distribution = df[on].value_counts(normalize=True).to_dict()
    # normalize to weights
    OG_event_distribution = {k: v / sum(OG_event_distribution.values()) for k, v in OG_event_distribution.items()}
    mapped_df = remap_schema(df, mapping, on=on)
    N = len(df)
    N_final = N / (1 - contamination_rate) # TODO: explain this in paper
    N_contaminate = int(N_final - N)
    start_event_types = random.choices(list(OG_event_distribution.keys()),
                                    weights=list(OG_event_distribution.values()), k=N_contaminate)
    # it makes sense
    new_trajectories = []
    for start_event in start_event_types:
        # sample from og start
        start = None # TODO: defin start accoding to dataset (randomly sample with weights of event distr)
        trajectory = model.sample_trajectory(start) # TODO: explain this method in paper
        new_trajectories.extend(trajectory)

    # TODO: make sure the new trajctories schema conforms with dataset
    contaminate_df = pd.DataFrame(new_trajectories)
    df = pd.concat([df, contaminate_df], ignore_index=True)
    return df
