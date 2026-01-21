from sim.rl.behavior_loader.loader import AgentLoader, Loader, JointLoader
from sim.rl.behavior_loader.loader import PayloadModel
from arch import WeakClassifier

agent_dir = "/home/velocitatem/Documents/Projects/PHANTOM/experiments/agents/collected_data/"
human_dir = "/home/velocitatem/Documents/Projects/PHANTOM/experiments/collected_data/"

def augment_trajectory(trajectory : list[PayloadModel], augmentation_rate: float = 0.1) -> list[PayloadModel]:
    # augmentations possible:
    # return a sub-trajectory window of the original trajectory
    # insert random noise events
    # shuffle a few events (find a few indices and swap them with i+1 neighbor)
    # adjust metadata
    return trajectory


def train():
    pass



if __name__ == "__main__":
    joint_loader = JointLoader(human_dir, agent_dir)
    data = joint_loader.get_data()
    entries, num_entries = joint_loader.get_entries()
    print(f"Loaded {num_entries} entries")
    # TODO: augment
    # fit model
    model = WeakClassifier()
    model.fit(data)
