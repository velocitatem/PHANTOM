"""Vectorized session feature extraction."""
import numpy as np
from .transitions import N_STATES, PURCHASE_IDX, CART_IDX
from .simulation import SessionBatch

try:
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except ImportError:
    jnp, JAX_AVAILABLE = np, False
    def jit(f): return f

@jit
def extract_features(states, dwells, lengths):
    """Extract per-session features. Returns (n_sess, 9) array."""
    n, max_len = states.shape
    mask = jnp.arange(max_len)[None,:] < lengths[:,None]
    duration = jnp.sum(dwells * mask, axis=1)
    total = lengths.astype(jnp.float32)
    count = lambda idx: jnp.sum((states == idx) & mask, axis=1).astype(jnp.float32)
    views, learn, carts, purchases = count(1), count(2), count(3), count(4)
    velocity = total / (duration + 1e-6)
    conversion = purchases / (views + 1e-6)
    avg_dwell = duration / (total + 1e-6)
    return jnp.stack([duration, avg_dwell, total, velocity, views, carts, purchases, learn, conversion], axis=1)

def session_features(batch: SessionBatch) -> np.ndarray:
    if JAX_AVAILABLE:
        return np.asarray(extract_features(jnp.array(batch.states), jnp.array(batch.dwells), jnp.array(batch.lengths)))
    # numpy fallback
    n, max_len = batch.states.shape
    mask = np.arange(max_len)[None,:] < batch.lengths[:,None]
    duration = np.sum(batch.dwells * mask, axis=1)
    total = batch.lengths.astype(np.float32)
    count = lambda idx: np.sum((batch.states == idx) & mask, axis=1).astype(np.float32)
    views, learn, carts, purchases = count(1), count(2), count(3), count(4)
    return np.stack([duration, duration/(total+1e-6), total, total/(duration+1e-6), views, carts, purchases, learn, purchases/(views+1e-6)], axis=1)

@jit
def session_transitions(states, lengths, n_states=N_STATES):
    """Compute empirical transition counts per session. Returns (n_sess, n_states, n_states)."""
    n, max_len = states.shape
    mask = jnp.arange(max_len - 1)[None,:] < (lengths[:,None] - 1)
    src, dst = states[:, :-1], states[:, 1:]
    # handle -1 padding by clamping to valid range
    src_c, dst_c = jnp.clip(src, 0, n_states-1), jnp.clip(dst, 0, n_states-1)
    valid = mask & (src >= 0) & (dst >= 0)
    def per_session(i):
        s, d, v = src_c[i], dst_c[i], valid[i]
        trans = (jnp.eye(n_states)[s,:,None] * jnp.eye(n_states)[d,None,:]).sum(0) * v[:,None,None]
        return trans.sum(0)
    # vmap not ideal here, use manual loop for clarity
    trans = jnp.stack([per_session(i) for i in range(n)])
    row_sums = trans.sum(axis=-1, keepdims=True)
    return trans / (row_sums + 1e-10)

def compute_session_transitions(batch: SessionBatch) -> np.ndarray:
    if JAX_AVAILABLE:
        return np.asarray(session_transitions(jnp.array(batch.states), jnp.array(batch.lengths)))
    # numpy fallback
    n, max_len = batch.states.shape
    trans = np.zeros((n, N_STATES, N_STATES), dtype=np.float32)
    for i in range(n):
        for t in range(batch.lengths[i] - 1):
            s, d = batch.states[i, t], batch.states[i, t+1]
            if s >= 0 and d >= 0: trans[i, s, d] += 1
    row_sums = trans.sum(axis=-1, keepdims=True)
    return trans / (row_sums + 1e-10)
