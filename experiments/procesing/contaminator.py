import pandas as pd
import random
import os
from pathlib import Path

# use relative import when in package context, fallback for standalone
try:
    from sim.rl.behavior_loader.models import AgentBehaviorModel
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sim" / "rl" / "behavior_loader"))
    from models import AgentBehaviorModel

# paths should be configurable via environment or relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
AGENT_DATA_DIR = Path(os.getenv('PHANTOM_AGENT_DATA_DIR', PROJECT_ROOT / "experiments" / "agents" / "collected_data"))


def remap_schema(df: pd.DataFrame, mapping: dict, on: str = "event_type") -> pd.DataFrame:
    """remap column values according to mapping dict, preserving unmapped values"""
    df = df.copy()
    df[on] = df[on].map(mapping).fillna(df[on])
    return df


def contaminate_dataset(df: pd.DataFrame, on: str = "event_type",
                        contamination_rate: float = 0.1,
                        agent_data_dir: Path = None) -> pd.DataFrame:
    """inject synthetic agent trajectories into a dataset
    contamination_rate: fraction of final dataset that should be agent data (0.1 = 10% agents)
    """
    data_dir = agent_data_dir or AGENT_DATA_DIR
    model = AgentBehaviorModel(str(data_dir))
    model.build_MDP()  # ensure MDP is built before sampling

    # compute event distribution from original data
    event_dist = df[on].value_counts(normalize=True).to_dict()
    total = sum(event_dist.values())
    event_dist = {k: v / total for k, v in event_dist.items()}

    # calculate how many synthetic events to add
    N = len(df)
    N_final = N / (1 - contamination_rate)
    N_contaminate = int(N_final - N)

    # sample start states weighted by original distribution
    start_events = random.choices(list(event_dist.keys()), weights=list(event_dist.values()), k=N_contaminate)

    # generate synthetic trajectories
    new_rows = []
    for start_event in start_events:
        # sample trajectory from agent model, using a state that contains the event type
        mdp_states = model.mdp.get('states', []) if model.mdp else []
        matching_starts = [s for s in mdp_states if start_event in s]
        if not matching_starts:
            continue  # skip if no matching start state
        start_state = random.choice(matching_starts)
        trajectory = model.sample_traj(start_state, max_len=20)
        for state in trajectory:
            parts = state.split('|')  # page|productId|eventName format
            new_rows.append({on: parts[-1] if parts else start_event, 'source': 'synthetic_agent'})

    if new_rows:
        contaminate_df = pd.DataFrame(new_rows)
        df = pd.concat([df, contaminate_df], ignore_index=True)
    return df
