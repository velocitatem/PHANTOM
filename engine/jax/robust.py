"""JAX-accelerated robust inner loop for PHANTOM.

provides a drop-in replacement for the sequential alpha-candidate evaluation in
wrapper.py::_select_adversarial_alpha.  the demand generation and reward
computation are vmapped over the K candidate alpha values so all candidates are
evaluated in a single vectorized pass instead of K sequential Python calls.

public surface:
    select_adversarial_alpha_jax(candidates, prices, human_params, agent_params,
                                  noise_std, n_sessions, n_products,
                                  baseline_prices, lambda_coi, info_value,
                                  reward_profit_weight, rng_key)
        -> (best_alpha: float, rewards: np.ndarray)

falls back gracefully when JAX is unavailable.
"""

from __future__ import annotations

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import vmap, jit

    _JAX_OK = True
except ImportError:
    _JAX_OK = False


def _demand_for_actor_jax(prices, mean, std, noise_std, key):
    """d(p;theta) = max(0, val - price + noise), normalized to sum 100."""
    k1, k2 = jax.random.split(key)
    val = jax.random.normal(k1, shape=prices.shape) * std + mean
    noise = jax.random.normal(k2, shape=prices.shape) * noise_std
    demand = jnp.maximum(0.0, val - prices + noise)
    total = demand.sum()
    return jnp.where(total > 0, demand / total * 100.0, demand)


def _reward_for_candidate(
    alpha,
    prices,
    human_mean,
    human_std,
    agent_mean,
    agent_std,
    noise_std,
    baseline_prices,
    lambda_coi,
    info_value,
    reward_profit_weight,
    key,
):
    """compute a scalar reward for a single alpha candidate (pure JAX, vmappable)."""
    k_h, k_a = jax.random.split(key)
    # mixed demand proxy: weighted sum of human and agent demand signals
    demand_h = _demand_for_actor_jax(prices, human_mean, human_std, noise_std, k_h)
    demand_a = _demand_for_actor_jax(prices, agent_mean, agent_std, noise_std, k_a)
    demand = (1.0 - alpha) * demand_h + alpha * demand_a

    revenue = jnp.dot(prices, demand)
    floor_cost = jnp.dot(baseline_prices, demand)
    profit = revenue - floor_cost

    # agent_prob proxy: use alpha directly (no trajectory available in vectorized path)
    coi_leakage = alpha * info_value
    info_budget = jnp.maximum(floor_cost, 1.0)
    coi_penalty = lambda_coi * coi_leakage * info_budget

    return reward_profit_weight * profit - coi_penalty


if _JAX_OK:
    # compile once; retracing only happens on shape/dtype changes
    # 12 args: alpha, prices, h_mean, h_std, a_mean, a_std, noise_std,
    #          baseline_prices, lambda_coi, info_value, reward_profit_weight, key
    _reward_batched = jit(
        vmap(
            _reward_for_candidate,
            in_axes=(0, None, None, None, None, None, None, None, None, None, None, 0),
        )
    )


def select_adversarial_alpha_jax(
    candidates: np.ndarray,
    prices: np.ndarray,
    human_params: tuple,
    agent_params: tuple,
    noise_std: float,
    baseline_prices: np.ndarray,
    lambda_coi: float,
    info_value: float,
    reward_profit_weight: float,
    rng_seed: int = 0,
) -> tuple[float, np.ndarray]:
    """evaluate all alpha candidates in a single vmapped pass.

    returns (best_alpha, rewards_array) where best_alpha minimizes reward
    (worst case for the platform, driving robust policy training).

    falls back to a pure-numpy sequential loop when JAX is unavailable so the
    wrapper can call this function unconditionally.
    """
    if not _JAX_OK:
        return _fallback(
            candidates,
            prices,
            human_params,
            agent_params,
            noise_std,
            baseline_prices,
            lambda_coi,
            info_value,
            reward_profit_weight,
        )

    k = len(candidates)
    key = jax.random.PRNGKey(rng_seed)
    keys = jax.random.split(key, k)

    rewards = np.asarray(
        _reward_batched(
            jnp.asarray(candidates, dtype=jnp.float32),
            jnp.asarray(prices, dtype=jnp.float32),
            float(human_params[0]),
            float(human_params[1]),
            float(agent_params[0]),
            float(agent_params[1]),
            float(noise_std),
            jnp.asarray(baseline_prices, dtype=jnp.float32),
            float(lambda_coi),
            float(info_value),
            float(reward_profit_weight),
            keys,
        )
    )
    best_idx = int(np.argmin(rewards))
    return float(candidates[best_idx]), rewards


def _fallback(
    candidates,
    prices,
    human_params,
    agent_params,
    noise_std,
    baseline_prices,
    lambda_coi,
    info_value,
    reward_profit_weight,
):
    """numpy fallback matching the reward formula above."""
    rewards = []
    for alpha in candidates:
        rng = np.random.default_rng()
        val_h = rng.normal(*human_params, size=len(prices))
        val_a = rng.normal(*agent_params, size=len(prices))
        noise_h = rng.normal(0, noise_std, len(prices))
        noise_a = rng.normal(0, noise_std, len(prices))
        d_h = np.maximum(0, val_h - prices + noise_h)
        d_a = np.maximum(0, val_a - prices + noise_a)
        s_h, s_a = d_h.sum(), d_a.sum()
        d_h = d_h / s_h * 100 if s_h > 0 else d_h
        d_a = d_a / s_a * 100 if s_a > 0 else d_a
        demand = (1.0 - alpha) * d_h + alpha * d_a
        revenue = float(np.dot(prices, demand))
        floor_cost = float(np.dot(baseline_prices, demand))
        profit = revenue - floor_cost
        coi_penalty = lambda_coi * alpha * info_value * max(floor_cost, 1.0)
        rewards.append(reward_profit_weight * profit - coi_penalty)
    rewards = np.array(rewards)
    best_idx = int(np.argmin(rewards))
    return float(candidates[best_idx]), rewards
