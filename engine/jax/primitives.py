"""JAX-compatible primitives for PHANTOM session simulation and separability."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Mapping, Sequence

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = np  # type: ignore[assignment]
    JAX_AVAILABLE = False


STATE_START_KEYS = ("session_start", "start")
TERMINAL_EVENT_TOKENS = (
    "session_end",
    "end",
    "purchase_complete",
    "checkout_start",
    "checkout",
)
PURCHASE_EVENT_TOKENS = (
    "purchase_complete",
    "purchase",
    "checkout_start",
    "checkout",
)

CATEGORY_WEIGHTS = {"cart": 4.0, "dwell": 2.0, "nav": 1.0, "filter": 0.5}
ACTION_CATEGORIES = {
    "cart": {"add_item", "add_to_cart", "remove", "checkout", "purchase"},
    "dwell": {
        "hover_title",
        "hover_paragraph",
        "hover_link",
        "hover_over_title",
        "hover_over_paragraph",
        "hover_over_link",
        "hover_over_button",
    },
    "nav": {
        "page_view",
        "view_item",
        "view",
        "learn_more",
        "learn_more_about_item",
        "view_item_page",
        "session_start",
    },
    "filter": {
        "search",
        "filter_date",
        "filter_price",
        "sort",
        "filter_for_date",
        "filter_for_price",
        "filter_for_amenities",
        "sort_change",
    },
}
DEFAULT_ACTION_WEIGHTS = {
    action: CATEGORY_WEIGHTS[group]
    for group, actions in ACTION_CATEGORIES.items()
    for action in actions
}


@dataclass(frozen=True)
class TransitionData:
    """Dense transition kernels and per-state metadata."""

    human_T: np.ndarray
    agent_T: np.ndarray
    terminal_mask: np.ndarray
    purchase_mask: np.ndarray
    event_weights: np.ndarray
    event_names: tuple[str, ...]
    start_idx: int
    term_idx: int

    def to_jax(self) -> "TransitionData":
        if not JAX_AVAILABLE:
            return self
        return TransitionData(
            human_T=jnp.asarray(self.human_T),
            agent_T=jnp.asarray(self.agent_T),
            terminal_mask=jnp.asarray(self.terminal_mask),
            purchase_mask=jnp.asarray(self.purchase_mask),
            event_weights=jnp.asarray(self.event_weights),
            event_names=self.event_names,
            start_idx=int(self.start_idx),
            term_idx=int(self.term_idx),
        )


@dataclass(frozen=True)
class SessionBatch:
    states: np.ndarray
    products: np.ndarray
    actors: np.ndarray
    lengths: np.ndarray


def _event_weight(name: str) -> float:
    if name in DEFAULT_ACTION_WEIGHTS:
        return float(DEFAULT_ACTION_WEIGHTS[name])
    if name.startswith("hover"):
        return float(CATEGORY_WEIGHTS["dwell"])
    if name.startswith("filter") or name in {"search", "sort", "sort_change"}:
        return float(CATEGORY_WEIGHTS["filter"])
    if name.startswith("add") or name in {
        "checkout",
        "checkout_start",
        "purchase",
        "remove_item",
        "purchase_complete",
    }:
        return float(CATEGORY_WEIGHTS["cart"])
    if any(token in name for token in TERMINAL_EVENT_TOKENS):
        return 0.0
    return float(CATEGORY_WEIGHTS["nav"])


def _is_terminal(name: str) -> bool:
    return any(token in name for token in TERMINAL_EVENT_TOKENS)


def _is_purchase(name: str) -> bool:
    return any(token in name for token in PURCHASE_EVENT_TOKENS)


def _collect_events(*transitions: Mapping[str, Mapping[str, float]]) -> tuple[str, ...]:
    names: set[str] = set()
    for trans in transitions:
        for src, dsts in trans.items():
            names.add(src)
            names.update(dsts.keys())
    names.discard("__terminal__")
    return tuple(sorted(names))


def _normalize_rows(matrix: np.ndarray, term_idx: int) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    dead_rows = np.isclose(row_sums.squeeze(-1), 0.0)
    if np.any(dead_rows):
        matrix[dead_rows] = 0.0
        matrix[dead_rows, term_idx] = 1.0
        row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / np.maximum(row_sums, 1e-8)


def _dense_from_dict(
    transitions: Mapping[str, Mapping[str, float]],
    event_to_idx: Mapping[str, int],
    term_idx: int,
) -> np.ndarray:
    n_states = len(event_to_idx)
    matrix = np.zeros((n_states, n_states), dtype=np.float32)
    for src, dsts in transitions.items():
        i = event_to_idx.get(src)
        if i is None:
            continue
        for dst, prob in dsts.items():
            j = event_to_idx.get(dst)
            if j is None:
                continue
            matrix[i, j] += float(prob)
    return _normalize_rows(matrix, term_idx)


def compile_transition_data(
    human_transitions: Mapping[str, Mapping[str, float]],
    agent_transitions: Mapping[str, Mapping[str, float]],
) -> TransitionData:
    event_names = _collect_events(human_transitions, agent_transitions)
    if not event_names:
        return fallback_transition_data()

    event_names = tuple([*event_names, "__terminal__"])
    term_idx = len(event_names) - 1
    event_to_idx = {name: i for i, name in enumerate(event_names)}

    human_T = _dense_from_dict(human_transitions, event_to_idx, term_idx)
    agent_T = _dense_from_dict(agent_transitions, event_to_idx, term_idx)

    terminal_mask = np.array([_is_terminal(name) for name in event_names], dtype=bool)
    purchase_mask = np.array([_is_purchase(name) for name in event_names], dtype=bool)
    event_weights = np.array(
        [_event_weight(name) for name in event_names], dtype=np.float32
    )

    terminal_mask[term_idx] = True

    for idx, is_term in enumerate(terminal_mask):
        if not is_term:
            continue
        human_T[idx] = 0.0
        agent_T[idx] = 0.0
        human_T[idx, idx] = 1.0
        agent_T[idx, idx] = 1.0

    start_idx = 0
    for key in STATE_START_KEYS:
        if key in event_to_idx:
            start_idx = int(event_to_idx[key])
            break

    return TransitionData(
        human_T=human_T,
        agent_T=agent_T,
        terminal_mask=terminal_mask,
        purchase_mask=purchase_mask,
        event_weights=event_weights,
        event_names=event_names,
        start_idx=start_idx,
        term_idx=term_idx,
    )


def fallback_transition_data() -> TransitionData:
    human = {
        "session_start": {
            "page_view": 0.80,
            "view_item_page": 0.15,
            "session_end": 0.05,
        },
        "page_view": {"view_item_page": 0.55, "search": 0.25, "session_end": 0.20},
        "view_item_page": {
            "learn_more_about_item": 0.40,
            "add_item_to_cart": 0.28,
            "session_end": 0.32,
        },
        "learn_more_about_item": {
            "add_item_to_cart": 0.50,
            "view_item_page": 0.30,
            "session_end": 0.20,
        },
        "add_item_to_cart": {
            "checkout_start": 0.58,
            "view_item_page": 0.24,
            "session_end": 0.18,
        },
        "checkout_start": {"purchase_complete": 0.70, "session_end": 0.30},
        "purchase_complete": {"session_end": 1.0},
    }
    agent = {
        "session_start": {
            "page_view": 0.90,
            "view_item_page": 0.08,
            "session_end": 0.02,
        },
        "page_view": {"view_item_page": 0.40, "search": 0.35, "session_end": 0.25},
        "view_item_page": {
            "learn_more_about_item": 0.55,
            "add_item_to_cart": 0.15,
            "session_end": 0.30,
        },
        "learn_more_about_item": {
            "view_item_page": 0.45,
            "add_item_to_cart": 0.20,
            "session_end": 0.35,
        },
        "add_item_to_cart": {
            "checkout_start": 0.42,
            "view_item_page": 0.28,
            "session_end": 0.30,
        },
        "checkout_start": {"purchase_complete": 0.52, "session_end": 0.48},
        "purchase_complete": {"session_end": 1.0},
    }
    return compile_transition_data(human, agent)


def load_transition_data(prefer_data: bool = True) -> TransitionData:
    if not prefer_data:
        return fallback_transition_data()
    try:
        from ..lib.behavior import get_transition_models

        human_trans, agent_trans = get_transition_models()
        return compile_transition_data(human_trans, agent_trans)
    except Exception:
        return fallback_transition_data()


if JAX_AVAILABLE:

    @partial(jax.jit, static_argnums=(8, 9, 10))
    def _sample_sessions_jax(
        key: jax.Array,
        human_T: jax.Array,
        agent_T: jax.Array,
        terminal_mask: jax.Array,
        start_idx: int,
        term_idx: int,
        alpha: float,
        n_products: int,
        n_sessions: int,
        max_steps: int,
        n_states: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        k_actor, k_product, k_step = jax.random.split(key, 3)
        actor_draw = jax.random.uniform(k_actor, (n_sessions,))
        actors = (actor_draw < alpha).astype(jnp.int32)
        products = jax.random.randint(
            k_product, (n_sessions,), 0, n_products, dtype=jnp.int32
        )

        active_init = jnp.ones((n_sessions,), dtype=jnp.bool_)
        state_init = jnp.full((n_sessions,), int(start_idx), dtype=jnp.int32)

        def _scan_step(carry, _):
            states, active, rng = carry
            rng, k = jax.random.split(rng)
            probs_h = human_T[states]
            probs_a = agent_T[states]
            probs = jnp.where(actors[:, None] == 0, probs_h, probs_a)
            next_state = jax.random.categorical(k, jnp.log(probs + 1e-10), axis=-1)
            next_state = jnp.where(active, next_state, int(term_idx))
            emitted = jnp.where(active, next_state, -1)
            is_terminal = terminal_mask[jnp.clip(next_state, 0, n_states - 1)]
            next_active = active & (~is_terminal)
            carry_states = jnp.where(next_active, next_state, int(term_idx))
            return (carry_states, next_active, rng), emitted

        _, state_t = jax.lax.scan(
            _scan_step, (state_init, active_init, k_step), None, length=max_steps
        )
        states = state_t.T
        lengths = jnp.sum(states >= 0, axis=1, dtype=jnp.int32)
        return states, products, actors, lengths


def sample_sessions(
    key,
    transition_data: TransitionData,
    alpha: float,
    n_products: int,
    n_sessions: int,
    max_steps: int,
) -> SessionBatch:
    if JAX_AVAILABLE:
        td = transition_data.to_jax()
        states, products, actors, lengths = _sample_sessions_jax(
            key,
            td.human_T,
            td.agent_T,
            td.terminal_mask,
            int(td.start_idx),
            int(td.term_idx),
            float(alpha),
            int(n_products),
            int(n_sessions),
            int(max_steps),
            int(td.human_T.shape[0]),
        )
        return SessionBatch(
            states=states, products=products, actors=actors, lengths=lengths
        )

    rng = np.random.default_rng(int(np.asarray(key).reshape(-1)[0]))
    n_states = transition_data.human_T.shape[0]
    products = rng.integers(0, n_products, size=n_sessions, dtype=np.int32)
    actors = (rng.random(size=n_sessions) < alpha).astype(np.int32)
    states = np.full((n_sessions, max_steps), -1, dtype=np.int32)
    lengths = np.zeros((n_sessions,), dtype=np.int32)
    for i in range(n_sessions):
        current = int(transition_data.start_idx)
        mat = transition_data.agent_T if actors[i] == 1 else transition_data.human_T
        for t in range(max_steps):
            nxt = int(rng.choice(n_states, p=mat[current]))
            states[i, t] = nxt
            if transition_data.terminal_mask[nxt]:
                lengths[i] = t + 1
                break
            current = nxt
        if lengths[i] == 0:
            lengths[i] = max_steps
    return SessionBatch(
        states=states, products=products, actors=actors, lengths=lengths
    )


if JAX_AVAILABLE:

    @partial(jax.jit, static_argnums=(2,))
    def compute_session_transitions(states, lengths, n_states: int):
        src = states[:, :-1]
        dst = states[:, 1:]
        time_idx = jnp.arange(src.shape[1])[None, :]
        valid = (src >= 0) & (dst >= 0) & (time_idx < (lengths[:, None] - 1))
        src_clip = jnp.clip(src, 0, n_states - 1)
        dst_clip = jnp.clip(dst, 0, n_states - 1)
        src_oh = jax.nn.one_hot(src_clip, n_states)
        dst_oh = jax.nn.one_hot(dst_clip, n_states)
        counts = jnp.einsum(
            "nti,ntj,nt->nij", src_oh, dst_oh, valid.astype(jnp.float32)
        )
        row_sums = jnp.sum(counts, axis=-1, keepdims=True)
        return counts / (row_sums + 1e-10)


else:

    def compute_session_transitions(states, lengths, n_states: int):
        trans = np.zeros((states.shape[0], n_states, n_states), dtype=np.float32)
        for i in range(states.shape[0]):
            for t in range(max(int(lengths[i]) - 1, 0)):
                s = int(states[i, t])
                d = int(states[i, t + 1])
                if s >= 0 and d >= 0:
                    trans[i, s, d] += 1.0
        row_sums = trans.sum(axis=-1, keepdims=True)
        return trans / (row_sums + 1e-10)


def batch_kl(P, Q_human, Q_agent, eps: float = 1e-10):
    p = P + eps
    p = p / jnp.sum(p, axis=-1, keepdims=True)
    qh = Q_human[None, ...] + eps
    qa = Q_agent[None, ...] + eps
    delta_h = jnp.sum(p * jnp.log(p / qh), axis=(1, 2))
    delta_a = jnp.sum(p * jnp.log(p / qa), axis=(1, 2))
    return delta_h, delta_a


if JAX_AVAILABLE:
    batch_kl = jax.jit(batch_kl)


def agent_probability_from_kl(delta_h, delta_a, temperature: float = 1.0):
    t = jnp.maximum(float(temperature), 1e-6)
    exp_h = jnp.exp(-delta_h / t)
    exp_a = jnp.exp(-delta_a / t)
    return exp_a / (exp_h + exp_a + 1e-10)


def estimate_alpha_from_kl(delta_h, delta_a, beta: float = 2.0):
    logits = beta * (delta_h - delta_a)
    return 1.0 / (1.0 + jnp.exp(-logits))


def weighted_demand(states, products, n_products: int, event_weights):
    valid = states >= 0
    state_clip = jnp.clip(states, 0, event_weights.shape[0] - 1)
    weights = event_weights[state_clip] * valid
    per_session = jnp.sum(weights, axis=1)
    demand = jnp.zeros((n_products,), dtype=jnp.float32)
    demand = demand.at[products].add(per_session)
    total = jnp.sum(demand)
    return jnp.where(total > 0.0, (demand / total) * 100.0, demand)


if JAX_AVAILABLE:
    weighted_demand = jax.jit(weighted_demand, static_argnums=(2,))


def purchase_flags(states, purchase_mask):
    state_clip = jnp.clip(states, 0, purchase_mask.shape[0] - 1)
    hits = purchase_mask[state_clip] & (states >= 0)
    return jnp.any(hits, axis=1)


if JAX_AVAILABLE:
    purchase_flags = jax.jit(purchase_flags)


def revenue_from_demand(prices, demand):
    return jnp.dot(prices, demand)


if JAX_AVAILABLE:
    revenue_from_demand = jax.jit(revenue_from_demand)


def reward_with_coi_penalty(
    revenue, agent_prob: float, lambda_coi: float, info_value: float
):
    leakage = agent_prob * info_value
    discount = jnp.clip(1.0 - lambda_coi * leakage, 0.0, 1.0)
    return revenue * discount, leakage, discount


if JAX_AVAILABLE:
    reward_with_coi_penalty = jax.jit(reward_with_coi_penalty)
