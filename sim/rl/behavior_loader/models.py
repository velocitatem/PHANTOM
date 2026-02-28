try:
    from loader import Loader, AgentLoader, JointLoader
except ImportError:
    from sim.rl.behavior_loader.loader import Loader, AgentLoader, JointLoader
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import numpy as np
import graphviz
import sys
from pathlib import Path

# import lib utilities for optional use - models keep their own _state_repr for backwards compat
# with the specific event structure (evt.value.payload)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "lib"))
try:
    from lib.state import make_state_repr as lib_make_state_repr
    from lib.features import transition_histogram as lib_transition_histogram
except ImportError:
    lib_make_state_repr = None
    lib_transition_histogram = None


class BehaviorModel:
    def __init__(self, src_dir: str, loader_cls=Loader):
        self.loader = loader_cls(src_dir)
        self.data = self.loader.get_data()
        self.entries, self.num_entries = self.loader.get_entries()
        self.mdp = None

    def _state_repr(self, evt) -> str:
        p = evt.value.payload
        return f"{p.page or 'unk'}|{p.productId or 'none'}|{p.eventName}"

    def _sort_key(self, evt):
        return evt.timestamp

    def _extract_sessions(self) -> List[List[str]]:
        trajs = []
        for evts in self.data.values():
            if len(evts) < 2:
                continue
            states = [self._state_repr(e) for e in sorted(evts, key=self._sort_key)]
            trajs.append(states)
        return trajs

    def _calc_transitions(self, trajs: List[List[str]]) -> Tuple[Dict, Set]:
        trans, states = defaultdict(lambda: defaultdict(int)), set()
        for traj in trajs:
            for s, s_next in zip(traj, traj[1:]):
                trans[s][s_next] += 1
                states.update([s, s_next])
        return trans, states

    def _calc_rewards(self, trajs: List[List[str]]) -> Dict:
        rwd = defaultdict(list)
        for traj in trajs:
            n = len(traj)
            for i, s in enumerate(traj):
                rwd[s].append(i / n)
        return rwd

    def _normalize_trans(self, cnts: Dict) -> Dict:
        return {
            s: {s_n: cnt / sum(nxt.values()) for s_n, cnt in nxt.items()}
            for s, nxt in cnts.items()
        }

    def build_MDP(self) -> Dict:
        trajs = self._extract_sessions()
        trans_cnt, states = self._calc_transitions(trajs)
        trans_prob = self._normalize_trans(trans_cnt)
        state_rwd = self._calc_rewards(trajs)

        self.mdp = {
            "states": sorted(states),
            "num_states": len(states),
            "transitions": trans_prob,
            "state_values": {s: np.mean(r) for s, r in state_rwd.items()},
            "state_rewards": state_rwd,
            "trans_counts": trans_cnt,
        }
        return self.mdp

    def transition_prob(self, s: str, s_next: str) -> float:
        if not self.mdp:
            raise ValueError("build MDP first")
        return self.mdp["transitions"].get(s, {}).get(s_next, 0.0)

    def state_value(self, s: str) -> float:
        if not self.mdp:
            raise ValueError("build MDP first")
        return self.mdp["state_values"].get(s, 0.0)

    def sample_traj(self, start: str, max_len: int = 50) -> List[str]:
        if not self.mdp:
            raise ValueError("build MDP first")
        path, curr = [start], start
        for _ in range(max_len):
            nxt = self.mdp["transitions"].get(curr, {})
            if not nxt:
                break
            curr = np.random.choice(list(nxt.keys()), p=list(nxt.values()))
            path.append(curr)
        return path

    def extract_trajectory_features(
        self, events: List, max_trans_dim: int = 50
    ) -> np.ndarray:
        """Convert trajectory to feature vector using MDP structure for contrastive learning"""
        if not self.mdp:
            self.build_MDP()

        states = [self._state_repr(e) for e in sorted(events, key=self._sort_key)]
        features = []

        # transition histogram over MDP state space
        trans_counts = defaultdict(int)
        for s, s_next in zip(states, states[1:]):
            trans_counts[(s, s_next)] += 1
        all_trans = [
            (s, t)
            for s in self.mdp["states"]
            for t in self.mdp["transitions"].get(s, {}).keys()
        ]
        trans_vec = [trans_counts.get(tr, 0) for tr in all_trans[:max_trans_dim]]
        trans_vec = trans_vec + [0] * (max_trans_dim - len(trans_vec))  # pad
        total_trans = sum(trans_counts.values()) or 1
        features.extend([v / total_trans for v in trans_vec])

        # state coverage ratio
        visited = set(states)
        features.append(len(visited) / max(self.mdp["num_states"], 1))

        # temporal entropy of transitions
        if len(states) > 1:
            trans_probs = [
                self.transition_prob(s, s_n) for s, s_n in zip(states, states[1:])
            ]
            entropy = -sum(p * np.log(p + 1e-10) for p in trans_probs if p > 0)
            features.append(entropy / max(len(states), 1))
        else:
            features.append(0.0)

        # trajectory length and unique state count
        features.append(len(states))
        features.append(len(visited))

        # state value statistics along trajectory
        vals = [self.state_value(s) for s in states]
        if vals:
            features.extend([np.mean(vals), np.std(vals), np.min(vals), np.max(vals)])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)


class AgentBehaviorModel(BehaviorModel):
    def __init__(self, src_dir: str):
        super().__init__(src_dir, AgentLoader)

    def _state_repr(self, evt) -> str:
        return f"{evt.page or 'unk'}|{evt.productId or 'none'}|{evt.eventName}"

    def _sort_key(self, evt):
        return evt.ts


class JointBehaviorModel(BehaviorModel):
    def __init__(self, human_dir: str, agent_dir: str):
        self.loader = JointLoader(human_dir, agent_dir)
        self.data = self.loader.get_data()
        self.entries, self.num_entries = self.loader.get_entries()
        self.mdp = None

    def _state_repr(self, evt) -> str:
        return f"{evt.page or 'unk'}|{evt.productId or 'none'}|{evt.eventName}"

    def _sort_key(self, evt):
        return evt.ts


def aggregate_event_transitions(mdp: Dict) -> Dict[str, Dict[str, float]]:
    evt_trans = defaultdict(lambda: defaultdict(float))
    for s, trans in mdp["transitions"].items():
        src = s.split("|")[2]
        for s_next, prob in trans.items():
            dst = s_next.split("|")[2]
            evt_trans[src][dst] += prob

    for src in evt_trans:
        total = sum(evt_trans[src].values())
        if total > 0:
            evt_trans[src] = {dst: p / total for dst, p in evt_trans[src].items()}
    return dict(evt_trans)


def visualize_mdp(
    model: BehaviorModel,
    threshold: float = 0.05,
    output: str = "mdp_graph",
    fmt: str = "svg",
    view: bool = False,
    export_dot: bool = False,
):
    if not model.mdp:
        raise ValueError("build MDP first")

    evt_trans = aggregate_event_transitions(model.mdp)
    g = graphviz.Digraph(format=fmt)
    g.attr(rankdir="LR", size="30")
    g.attr("node", shape="circle", width="1", height="1")

    events = set(evt_trans.keys()) | {
        e for trans in evt_trans.values() for e in trans.keys()
    }
    for evt in events:
        g.node(evt)

    for src, dsts in evt_trans.items():
        for dst, prob in dsts.items():
            if prob > threshold:
                g.edge(src, dst, label=f"{prob:.2f}")

    g.render(output, view=view, cleanup=True)
    print(f"Saved MDP graph to {output}.{fmt}")

    if export_dot:
        with open(f"{output}.dot", "w") as f:
            f.write(g.source)
        print(f"Exported DOT source to {output}.dot")

    return g


def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    eps = 1e-10
    # p + log(p / q) summed over all keys in P
    return sum((p[k] + eps) * np.log((p[k] + eps) / (q.get(k, 0.0) + eps)) for k in p)


def _build_subset_mdp(model: BehaviorModel, session_ids: List) -> Dict:
    trajs = []
    for sid in session_ids:
        evts = model.data.get(sid, [])
        if len(evts) < 2:
            continue
        states = [model._state_repr(e) for e in sorted(evts, key=model._sort_key)]
        trajs.append(states)
    trans_cnt, _ = model._calc_transitions(trajs)
    return {"transitions": model._normalize_trans(trans_cnt)}


def _avg_event_kl(
    src_evt: Dict[str, Dict[str, float]], dst_evt: Dict[str, Dict[str, float]]
) -> float:
    common = set(src_evt.keys()) & set(dst_evt.keys())
    if not common:
        return 0.0
    return float(np.mean([kl_divergence(src_evt[e], dst_evt[e]) for e in common]))


def bootstrap_intra_class_divergence(
    model: BehaviorModel,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    session_ids = list(model.data.keys())
    n = len(session_ids)
    if n < 2:
        return {
            "mean": 0.0,
            "std": 0.0,
            "q05": 0.0,
            "q95": 0.0,
            "n_bootstrap": 0,
            "scores": [],
            "available": False,
            "num_sessions": int(n),
        }

    half = n // 2
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_bootstrap):
        perm = rng.permutation(session_ids)
        split_a, split_b = perm[:half], perm[half:]
        mdp_a = _build_subset_mdp(model, list(split_a))
        mdp_b = _build_subset_mdp(model, list(split_b))
        score = _avg_event_kl(
            aggregate_event_transitions(mdp_a),
            aggregate_event_transitions(mdp_b),
        )
        scores.append(score)

    arr = np.array(scores, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q95": float(np.quantile(arr, 0.95)),
        "n_bootstrap": int(n_bootstrap),
        "scores": arr.tolist(),
        "available": True,
        "num_sessions": int(n),
    }


if __name__ == "__main__":
    base_dir = "/home/velocitatem/Documents/Projects/PHANTOM/experiments"
    human_dir, agent_dir = (
        f"{base_dir}/collected_data/",
        f"{base_dir}/agents/collected_data/",
    )

    human_model = BehaviorModel(human_dir)
    human_mdp = human_model.build_MDP()
    print(
        f"Built MDP: {human_mdp['num_states']} states, "
        f"{sum(len(t) for t in human_mdp['transitions'].values())} transitions"
    )
    if not human_mdp["states"]:
        exit("No states found")
    visualize_mdp(
        human_model, threshold=0.05, output="human_mdp_viz", fmt="pdf", export_dot=True
    )

    agent_model = AgentBehaviorModel(agent_dir)
    agent_mdp = agent_model.build_MDP()

    print(
        f"AGENT... Built MDP: {agent_mdp['num_states']} states, "
        f"{sum(len(t) for t in agent_mdp['transitions'].values())} transitions"
    )
    if not agent_mdp["states"]:
        exit("No states found")
    visualize_mdp(
        agent_model, threshold=0.05, output="agent_mdp_viz", fmt="pdf", export_dot=True
    )

    human_evt = aggregate_event_transitions(human_mdp)
    agent_evt = aggregate_event_transitions(agent_mdp)

    common = set(human_evt.keys()) & set(agent_evt.keys())

    if not common:
        exit("No common event types for KL divergence analysis")

    kl_divs = sorted(
        [(e, kl_divergence(human_evt[e], agent_evt[e])) for e in common],
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"Average KL divergence: {np.mean([kl for _, kl in kl_divs]):.4f}")
    print("\nMost divergent event types:")
    for evt, kl in kl_divs:
        print(f"  {evt}: {kl:.4f}")

    print("\n=== Joint Model (Human + Agent Combined) ===")
    joint_model = JointBehaviorModel(human_dir, agent_dir)
    joint_mdp = joint_model.build_MDP()
    print(
        f"Built joint MDP: {joint_mdp['num_states']} states, "
        f"{sum(len(t) for t in joint_mdp['transitions'].values())} transitions"
    )
    if joint_mdp["states"]:
        visualize_mdp(
            joint_model,
            threshold=0.05,
            output="joint_mdp_viz",
            fmt="pdf",
            export_dot=True,
        )

    inter_class_avg = float(np.mean([kl for _, kl in kl_divs]))
    human_intra = bootstrap_intra_class_divergence(
        human_model, n_bootstrap=100, seed=42
    )
    agent_intra = bootstrap_intra_class_divergence(
        agent_model, n_bootstrap=100, seed=43
    )
    pooled_scores = human_intra["scores"] + agent_intra["scores"]
    if not pooled_scores:
        pooled_scores = [0.0]
    pooled_null = np.array(pooled_scores, dtype=float)
    p_empirical = float(
        (np.sum(pooled_null >= inter_class_avg) + 1) / (len(pooled_null) + 1)
    )

    print("\nIntra-class KL bootstrap baseline:")
    if human_intra["available"]:
        print(
            f"  Human split KL: {human_intra['mean']:.4f} +- {human_intra['std']:.4f} "
            f"(5-95%: {human_intra['q05']:.4f}-{human_intra['q95']:.4f}, n_sessions={human_intra['num_sessions']})"
        )
    else:
        print(
            f"  Human split KL: unavailable (need >=2 sessions, got {human_intra['num_sessions']})"
        )
    if agent_intra["available"]:
        print(
            f"  Agent split KL: {agent_intra['mean']:.4f} +- {agent_intra['std']:.4f} "
            f"(5-95%: {agent_intra['q05']:.4f}-{agent_intra['q95']:.4f}, n_sessions={agent_intra['num_sessions']})"
        )
    else:
        print(
            f"  Agent split KL: unavailable (need >=2 sessions, got {agent_intra['num_sessions']})"
        )
    print(f"  Between-class KL: {inter_class_avg:.4f}")
    print(
        f"  Lift vs pooled intra mean: {inter_class_avg / max(float(np.mean(pooled_null)), 1e-10):.2f}x"
    )
    print(f"  Empirical p-value (inter > intra): {p_empirical:.4f}")
