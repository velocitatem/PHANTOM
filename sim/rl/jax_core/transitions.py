"""Dense transition matrices for JAX Markov chain sampling."""
from dataclasses import dataclass
import numpy as np

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jnp, JAX_AVAILABLE = np, False

STATES = ["session_start", "view_item_page", "learn_more_about_item", "add_item_to_cart", "purchase_complete", "session_end"]
S2I = {s: i for i, s in enumerate(STATES)}
N_STATES, TERM_IDX, PURCHASE_IDX, CART_IDX = len(STATES), 5, 4, 3

@dataclass
class TransitionData:
    human_T: np.ndarray   # (6,6) transition probs
    agent_T: np.ndarray   # (6,6)
    human_dwell: np.ndarray  # (6,2) shape,scale
    agent_dwell: np.ndarray  # (6,2)

    def to_jax(self):
        if not JAX_AVAILABLE: return self
        return TransitionData(*[jnp.array(x) for x in [self.human_T, self.agent_T, self.human_dwell, self.agent_dwell]])

def dict_to_dense(d):
    m = np.zeros((N_STATES, N_STATES), dtype=np.float32)
    for src, dsts in d.items():
        if (i := S2I.get(src)) is not None:
            for dst, p in dsts.items():
                if (j := S2I.get(dst)) is not None: m[i,j] = p
    m /= np.maximum(m.sum(1, keepdims=True), 1e-8)
    m[TERM_IDX] = 0; m[TERM_IDX, TERM_IDX] = 1.0
    return m

def compile_transitions(human_profile, agent_profile):
    def dwell_arr(params): return np.array([[params.get(s, (2.0, 1.0)) for s in STATES]], dtype=np.float32).reshape(N_STATES, 2)
    return TransitionData(dict_to_dense(human_profile.transitions), dict_to_dense(agent_profile.transitions),
                          dwell_arr(human_profile.dwell_params), dwell_arr(agent_profile.dwell_params))

def fallback_transitions():
    H = {"session_start": {"view_item_page": .85, "session_end": .15}, "view_item_page": {"learn_more_about_item": .4, "add_item_to_cart": .3, "view_item_page": .2, "session_end": .1},
         "learn_more_about_item": {"add_item_to_cart": .5, "view_item_page": .3, "session_end": .2}, "add_item_to_cart": {"purchase_complete": .6, "view_item_page": .25, "session_end": .15}, "purchase_complete": {"session_end": 1.0}}
    A = {"session_start": {"view_item_page": .9, "session_end": .1}, "view_item_page": {"learn_more_about_item": .5, "add_item_to_cart": .25, "view_item_page": .15, "session_end": .1},
         "learn_more_about_item": {"add_item_to_cart": .4, "view_item_page": .4, "session_end": .2}, "add_item_to_cart": {"purchase_complete": .5, "view_item_page": .3, "session_end": .2}, "purchase_complete": {"session_end": 1.0}}
    dwell = np.full((N_STATES, 2), [2.0, 1.0], dtype=np.float32)
    return TransitionData(dict_to_dense(H), dict_to_dense(A), dwell.copy(), dwell.copy())
