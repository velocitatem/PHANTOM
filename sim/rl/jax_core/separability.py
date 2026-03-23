"""Vectorized KL divergence for separability scoring."""

import numpy as np
from typing import Tuple

from lib.agent_probability import (
    DEFAULT_AGENT_PRIOR,
    estimate_agent_probability_batch,
)

try:
    import jax.numpy as jnp
    from jax import jit

    JAX_AVAILABLE = True
except ImportError:
    jnp, JAX_AVAILABLE = np, False

    def jit(f):
        return f


@jit
def batch_kl(P, Q_human, Q_agent, eps=1e-10):
    """Compute KL(P||Q) for batched P. P:(n,s,s), Q:(s,s). Returns (delta_h, delta_a) each (n,)."""
    p = P + eps
    p = p / p.sum(axis=-1, keepdims=True)
    qh, qa = Q_human[None] + eps, Q_agent[None] + eps
    delta_h = jnp.sum(p * jnp.log(p / qh), axis=(1, 2))
    delta_a = jnp.sum(p * jnp.log(p / qa), axis=(1, 2))
    return delta_h, delta_a


def compute_divergences(
    session_trans: np.ndarray, ref_human: np.ndarray, ref_agent: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute KL divergence of each session from human/agent prototypes."""
    if JAX_AVAILABLE:
        dh, da = batch_kl(
            jnp.array(session_trans), jnp.array(ref_human), jnp.array(ref_agent)
        )
        return np.asarray(dh), np.asarray(da)
    # numpy fallback
    eps = 1e-10
    p = session_trans + eps
    p = p / p.sum(axis=-1, keepdims=True)
    qh, qa = ref_human[None] + eps, ref_agent[None] + eps
    delta_h = np.sum(p * np.log(p / qh), axis=(1, 2))
    delta_a = np.sum(p * np.log(p / qa), axis=(1, 2))
    return delta_h, delta_a


def estimate_alpha_batch(
    prob_agent: np.ndarray,
    delta_h: np.ndarray,
    delta_a: np.ndarray,
    temp: float = 1.0,
    prior_agent: float = DEFAULT_AGENT_PRIOR,
) -> np.ndarray:
    """Vectorized alpha estimation using divergence gap mapping."""
    _ = prob_agent
    return estimate_agent_probability_batch(
        delta_h=np.asarray(delta_h, dtype=float),
        delta_a=np.asarray(delta_a, dtype=float),
        temperature=temp,
        prior_agent=prior_agent,
    )
