"""Vectorized Markov chain session sampling with JAX."""
from typing import NamedTuple, Tuple
import numpy as np
from functools import partial

try:
    import jax, jax.numpy as jnp
    from jax import lax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from .transitions import TransitionData, N_STATES, TERM_IDX, PURCHASE_IDX, CART_IDX

class SessionBatch(NamedTuple):
    states: np.ndarray      # (n_sess, max_len) state indices, -1=padding
    dwells: np.ndarray      # (n_sess, max_len) dwell times
    products: np.ndarray    # (n_sess,) product index per session
    actors: np.ndarray      # (n_sess,) 0=human, 1=agent
    lengths: np.ndarray     # (n_sess,) actual session length

class SimResult(NamedTuple):
    demand_human: np.ndarray
    demand_agent: np.ndarray
    revenue: float
    n_human_purchases: int
    n_agent_purchases: int
    sessions: SessionBatch

if JAX_AVAILABLE:
    @partial(jax.jit, static_argnums=(5,6,7))
    def _sample_sessions_jax(key, T_human, T_agent, dwell_human, dwell_agent, n_human, n_agent, max_steps):
        n = n_human + n_agent
        k1, k2, k3, k4 = jax.random.split(key, 4)
        actors = jnp.concatenate([jnp.zeros(n_human, dtype=jnp.int32), jnp.ones(n_agent, dtype=jnp.int32)])
        T = jnp.where(actors[:,None,None]==0, T_human[None], T_agent[None])  # (n,6,6)
        dwell_p = jnp.where(actors[:,None,None]==0, dwell_human[None], dwell_agent[None])  # (n,6,2)

        def step(carry, _):
            s, active, k = carry
            k, k1, k2 = jax.random.split(k, 3)
            probs = T[jnp.arange(n), s]  # (n,6)
            nxt = jax.random.categorical(k1, jnp.log(probs + 1e-10))
            nxt = jnp.where(active, nxt, -1)
            shape = dwell_p[jnp.arange(n), s, 0]
            scale = dwell_p[jnp.arange(n), s, 1]
            dwell = jnp.maximum(0.3, jax.random.gamma(k2, shape) * scale)
            still = active & (nxt != TERM_IDX) & (nxt >= 0)
            return (nxt, still, k), (nxt, dwell)

        init = (jnp.zeros(n, dtype=jnp.int32), jnp.ones(n, dtype=jnp.bool_), k3)
        _, (states, dwells) = lax.scan(step, init, None, length=max_steps)
        states, dwells = states.T, dwells.T  # (n, max_steps)
        is_term = (states == -1) | (states == TERM_IDX)
        lengths = jnp.argmax(is_term, axis=1) + 1
        lengths = jnp.where(jnp.any(is_term, axis=1), lengths, max_steps)
        return states, dwells, actors, lengths

def sample_sessions(key, trans: TransitionData, n_human: int, n_agent: int, n_products: int, max_steps: int = 40) -> SessionBatch:
    if JAX_AVAILABLE:
        k1, k2 = jax.random.split(key)
        states, dwells, actors, lengths = _sample_sessions_jax(k1, trans.human_T, trans.agent_T, trans.human_dwell, trans.agent_dwell, n_human, n_agent, max_steps)
        products = jax.random.randint(k2, (n_human + n_agent,), 0, n_products)
        return SessionBatch(np.asarray(states), np.asarray(dwells), np.asarray(products), np.asarray(actors), np.asarray(lengths))
    # numpy fallback
    rng = np.random.default_rng(int(key[0]) if hasattr(key, '__getitem__') else 42)
    n = n_human + n_agent
    actors = np.concatenate([np.zeros(n_human, dtype=np.int32), np.ones(n_agent, dtype=np.int32)])
    products = rng.integers(0, n_products, size=n)
    states, dwells = np.full((n, max_steps), -1, dtype=np.int32), np.zeros((n, max_steps), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int32)
    for i in range(n):
        T = trans.human_T if actors[i] == 0 else trans.agent_T
        dp = trans.human_dwell if actors[i] == 0 else trans.agent_dwell
        s, t = 0, 0
        while t < max_steps and s != TERM_IDX:
            states[i, t] = s
            dwells[i, t] = max(0.3, rng.gamma(dp[s, 0], dp[s, 1]))
            s = rng.choice(N_STATES, p=T[s])
            t += 1
        lengths[i] = t
    return SessionBatch(states, dwells, products, actors, lengths)

def compute_metrics(batch: SessionBatch, prices: np.ndarray, unit_cost: np.ndarray) -> SimResult:
    purchased = np.any(batch.states == PURCHASE_IDX, axis=1)
    human_mask, agent_mask = batch.actors == 0, batch.actors == 1
    human_purch = purchased & human_mask
    agent_purch = purchased & agent_mask
    demand_h = np.bincount(batch.products[human_purch], minlength=len(prices)).astype(np.float32)
    demand_a = np.bincount(batch.products[agent_purch], minlength=len(prices)).astype(np.float32)
    revenue = float(np.sum(prices[batch.products[purchased]]))
    return SimResult(demand_h, demand_a, revenue, int(human_purch.sum()), int(agent_purch.sum()), batch)
