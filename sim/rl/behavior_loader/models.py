from loader import Loader
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import numpy as np
import graphviz

DIR = "/home/velocitatem/Documents/Projects/PHANTOM/experiments/collected_data/"

class BehaviorModel:
    def __init__(self, src_dir: str = DIR):
        self.loader = Loader(src_dir)
        self.data = self.loader.get_data()
        self.entries, self.num_entries = self.loader.get_entries()
        self.mdp = None

    def _state_repr(self, evt) -> str:
        p = evt.value.payload
        return f"{p.page or 'unk'}|{p.productId or 'none'}|{p.eventName}"

    def _extract_sessions(self):
        # transform raw events into sequential state trajectories per session
        trajectories = []
        for sid, evts in self.data.items():
            if len(evts) < 2: continue
            states = [self._state_repr(e) for e in sorted(evts, key=lambda x: x.timestamp)]
            trajectories.append(states)
        return trajectories

    def _calc_transitions(self, trajectories: List[List[str]]) -> Tuple[Dict, Set]:
        trans = defaultdict(lambda: defaultdict(int))
        states = set()
        for traj in trajectories:
            for i in range(len(traj) - 1):
                s, s_next = traj[i], traj[i+1]
                trans[s][s_next] += 1
                states.update([s, s_next])
        return trans, states

    def _calc_rewards(self, trajectories: List[List[str]]) -> Dict:
        # reward based on session progression depth
        rwd = defaultdict(list)
        for traj in trajectories:
            n = len(traj)
            for i, s in enumerate(traj):
                rwd[s].append(i / n)
        return rwd

    def _normalize_trans(self, counts: Dict) -> Dict:
        return {s: {s_n: cnt/sum(nxt.values()) for s_n, cnt in nxt.items()}
                for s, nxt in counts.items()}

    def build_MDP(self) -> Dict:
        trajs = self._extract_sessions()
        trans_cnt, states = self._calc_transitions(trajs)
        trans_prob = self._normalize_trans(trans_cnt)
        state_rwd = self._calc_rewards(trajs)
        state_val = {s: np.mean(r) for s, r in state_rwd.items()}

        self.mdp = {
            'states': sorted(list(states)),
            'num_states': len(states),
            'transitions': trans_prob,
            'state_values': state_val,
            'state_rewards': state_rwd,
            'trans_counts': trans_cnt,
        }
        return self.mdp

    def transition_prob(self, s: str, s_next: str) -> float:
        if not self.mdp: raise ValueError("build MDP first")
        return self.mdp['transitions'].get(s, {}).get(s_next, 0.0)

    def state_value(self, s: str) -> float:
        if not self.mdp: raise ValueError("build MDP first")
        return self.mdp['state_values'].get(s, 0.0)

    def sample_traj(self, start: str, max_len: int = 50) -> List[str]:
        if not self.mdp: raise ValueError("build MDP first")
        path = [start]
        curr = start
        for _ in range(max_len):
            nxt = self.mdp['transitions'].get(curr, {})
            if not nxt: break
            curr = np.random.choice(list(nxt.keys()), p=list(nxt.values()))
            path.append(curr)
        return path

def visualize_mdp(model: BehaviorModel, threshold: float = 0.05, output: str = "mdp_graph", fmt: str = "svg", view: bool = False):
    """visualize MDP as directed graph using graphviz, aggregated by event type"""
    if not model.mdp: raise ValueError("build MDP first")

    # aggregate transitions by event type
    evt_trans = defaultdict(lambda: defaultdict(float))
    for s, trans in model.mdp['transitions'].items():
        evt_src = s.split('|')[2]
        for s_next, prob in trans.items():
            evt_dst = s_next.split('|')[2]
            evt_trans[evt_src][evt_dst] += prob

    # normalize aggregated transitions
    for evt_src in evt_trans:
        total = sum(evt_trans[evt_src].values())
        if total > 0:
            for evt_dst in evt_trans[evt_src]:
                evt_trans[evt_src][evt_dst] /= total

    g = graphviz.Digraph(format=fmt)
    g.attr(rankdir='LR', size='30')
    g.attr('node', shape='circle', width='1', height='1')

    # collect all event types
    events = set(evt_trans.keys())
    for trans in evt_trans.values():
        events.update(trans.keys())

    # add nodes for each event type
    for evt in events:
        g.node(evt)

    # add edges above threshold
    for evt_src in evt_trans:
        for evt_dst, prob in evt_trans[evt_src].items():
            if prob > threshold:
                g.edge(evt_src, evt_dst, label=f'{prob:.2f}')

    g.render(output, view=view, cleanup=True)
    print(f"Saved MDP graph to {output}.{fmt}")
    return g

if __name__ == "__main__":
    model = BehaviorModel(DIR)
    mdp = model.build_MDP()
    print(f"Built MDP: {mdp['num_states']} states, {sum(len(t) for t in mdp['transitions'].values())} transitions")
    if not mdp['states']:
        print("No states found")
        exit(1)
    visualize_mdp(model, threshold=0.05, output="mdp_viz", fmt="svg")
