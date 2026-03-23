import numpy as np
from typing import Dict

from lib.agent_probability import DEFAULT_AGENT_PRIOR, estimate_agent_probability


def compute_agent_probability(
    trajectory: list,
    human_transitions: Dict,
    agent_transitions: Dict,
    temperature: float = 1.0,
    prior_agent: float = DEFAULT_AGENT_PRIOR,
) -> float:
    """estimate agent probability via KL divergence between trajectory transitions and reference models

    compares empirical trajectory transition distribution to human/agent prototypes

    args:
        trajectory: list of state/event strings from session
        human_transitions: reference transition dict from human MDP (event->event->prob)
        agent_transitions: reference transition dict from agent MDP (event->event->prob)

    returns:
        agent probability in [0, 1] via sigma((delta_h - delta_a) / T)
    """
    if len(trajectory) < 2:
        return float(prior_agent)

    # build empirical transition distribution from trajectory
    trans_counts = {}
    for s, s_next in zip(trajectory[:-1], trajectory[1:]):
        if s not in trans_counts:
            trans_counts[s] = {}
        trans_counts[s][s_next] = trans_counts[s].get(s_next, 0) + 1

    # normalize to probabilities
    empirical = {}
    for s, nxt in trans_counts.items():
        total = sum(nxt.values())
        empirical[s] = {s_n: cnt / total for s_n, cnt in nxt.items()}

    # compute KL divergence to each prototype
    def kl_div(p_dist: Dict, q_dist: Dict) -> float:
        eps = 1e-10
        # aggregate over all source states in empirical dist
        kl = 0.0
        for s in p_dist:
            if s not in q_dist:
                continue  # skip states not in reference
            p_trans, q_trans = p_dist[s], q_dist[s]
            for k in p_trans:
                p_val = p_trans[k] + eps
                q_val = q_trans.get(k, 0.0) + eps
                kl += p_val * np.log(p_val / q_val)
        return kl

    kl_human = kl_div(empirical, human_transitions)
    kl_agent = kl_div(empirical, agent_transitions)

    return estimate_agent_probability(
        delta_h=kl_human,
        delta_a=kl_agent,
        temperature=temperature,
        prior_agent=prior_agent,
    )


def extract_purchases(trajectories: list) -> Dict[int, int]:
    purchases: Dict[int, int] = {}
    for traj in trajectories:
        if traj and "checkout" in traj[-1] and "_product" in traj[-1]:
            prod_id = int(traj[-1].rsplit("_product", 1)[1])
            purchases[prod_id] = purchases.get(prod_id, 0) + 1
    return purchases


def compute_uplift_coi(
    prices: np.ndarray, purchases: Dict[int, int], baseline_prices: np.ndarray
) -> float:
    # TODO: consider view-weighted fractional purchase for denser signal
    return float(
        sum(max(0.0, prices[k] - baseline_prices[k]) * n for k, n in purchases.items())
    )
