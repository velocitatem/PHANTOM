"""JAX-native PHANTOM environment with robust contamination step."""

from __future__ import annotations

from typing import NamedTuple

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    raise ImportError("engine.jax.env requires JAX") from exc

from .primitives import (
    _sample_sessions_jax,
    agent_probability_from_kl,
    batch_kl,
    compute_session_transitions,
    load_transition_data,
    purchase_flags,
    reward_with_coi_penalty,
    revenue_from_demand,
    weighted_demand,
)


class EnvParams(NamedTuple):
    n_products: int
    n_sessions: int
    max_episode_steps: int
    max_session_steps: int
    price_low: float
    price_high: float
    lambda_coi: float
    info_value: float
    eta_ux: float
    robust_radius: float
    margin_floor: float
    margin_floor_patience: int
    action_scales: jax.Array
    alpha_nominal: float
    alpha_candidates: jax.Array
    human_T: jax.Array
    agent_T: jax.Array
    terminal_mask: jax.Array
    purchase_mask: jax.Array
    event_weights: jax.Array
    start_idx: int
    term_idx: int


class EnvState(NamedTuple):
    prices: jax.Array
    demand: jax.Array
    step_count: jax.Array
    low_margin_streak: jax.Array
    last_agent_prob: jax.Array
    last_alpha_adv: jax.Array


class CandidateEval(NamedTuple):
    reward: jax.Array
    revenue: jax.Array
    demand: jax.Array
    agent_prob: jax.Array
    leakage: jax.Array
    discount: jax.Array
    ux_penalty: jax.Array
    n_purchases: jax.Array
    n_agents: jax.Array


def make_env_params(
    *,
    n_products: int,
    alpha: float,
    n_sessions: int,
    lambda_coi: float,
    robust_radius: float,
    robust_points: int,
    info_value: float,
    eta_ux: float = 0.5,
    action_levels: int,
    action_scale_low: float,
    action_scale_high: float,
    price_low: float,
    price_high: float,
    max_episode_steps: int,
    max_session_steps: int = 40,
    margin_floor: float = 0.05,
    margin_floor_patience: int = 5,
    prefer_behavior_data: bool = True,
) -> EnvParams:
    transition = load_transition_data(prefer_data=prefer_behavior_data).to_jax()
    if robust_radius <= 0.0 or robust_points <= 1:
        alpha_candidates = jnp.asarray([float(alpha)], dtype=jnp.float32)
    else:
        lo = max(0.0, float(alpha) - float(robust_radius))
        hi = min(1.0, float(alpha) + float(robust_radius))
        alpha_candidates = jnp.linspace(lo, hi, int(robust_points), dtype=jnp.float32)

    action_scales = jnp.linspace(
        float(action_scale_low),
        float(action_scale_high),
        int(action_levels),
        dtype=jnp.float32,
    )
    return EnvParams(
        n_products=int(n_products),
        n_sessions=int(n_sessions),
        max_episode_steps=int(max_episode_steps),
        max_session_steps=int(max_session_steps),
        price_low=float(price_low),
        price_high=float(price_high),
        lambda_coi=float(lambda_coi),
        info_value=float(info_value),
        eta_ux=float(eta_ux),
        robust_radius=float(robust_radius),
        margin_floor=float(margin_floor),
        margin_floor_patience=int(margin_floor_patience),
        action_scales=action_scales,
        alpha_nominal=float(alpha),
        alpha_candidates=alpha_candidates,
        human_T=jnp.asarray(transition.human_T),
        agent_T=jnp.asarray(transition.agent_T),
        terminal_mask=jnp.asarray(transition.terminal_mask),
        purchase_mask=jnp.asarray(transition.purchase_mask),
        event_weights=jnp.asarray(transition.event_weights),
        start_idx=int(transition.start_idx),
        term_idx=int(transition.term_idx),
    )


def _flatten_obs(demand: jax.Array, prices: jax.Array) -> jax.Array:
    return jnp.concatenate([demand.astype(jnp.float32), prices.astype(jnp.float32)])


def _decode_action(
    prices: jax.Array, action: jax.Array, params: EnvParams
) -> jax.Array:
    idx = jnp.clip(action.astype(jnp.int32), 0, params.action_scales.shape[0] - 1)
    scale = params.action_scales[idx]
    next_prices = prices * scale
    return jnp.clip(next_prices, params.price_low, params.price_high)


def _evaluate_candidate(
    key: jax.Array,
    alpha_candidate: jax.Array,
    prices: jax.Array,
    ux_volatility: jax.Array,
    params: EnvParams,
) -> CandidateEval:
    states, products, actors, lengths = _sample_sessions_jax(
        key,
        params.human_T,
        params.agent_T,
        params.terminal_mask,
        params.start_idx,
        params.term_idx,
        alpha_candidate,
        params.n_products,
        params.n_sessions,
        params.max_session_steps,
        int(params.human_T.shape[0]),
    )
    session_trans = compute_session_transitions(
        states, lengths, int(params.human_T.shape[0])
    )
    delta_h, delta_a = batch_kl(session_trans, params.human_T, params.agent_T)
    agent_probs = agent_probability_from_kl(delta_h, delta_a)
    agent_prob = jnp.mean(agent_probs)

    demand = weighted_demand(states, products, params.n_products, params.event_weights)
    revenue = revenue_from_demand(prices, demand)
    reward, leakage, discount, ux_penalty = reward_with_coi_penalty(
        revenue,
        agent_prob,
        params.lambda_coi,
        params.info_value,
        params.eta_ux,
        ux_volatility,
    )
    purchases = purchase_flags(states, params.purchase_mask)
    return CandidateEval(
        reward=reward,
        revenue=revenue,
        demand=demand,
        agent_prob=agent_prob,
        leakage=leakage,
        discount=discount,
        ux_penalty=ux_penalty,
        n_purchases=jnp.sum(purchases.astype(jnp.float32)),
        n_agents=jnp.sum(actors.astype(jnp.float32)),
    )


def reset_env(key: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:
    prices = jax.random.uniform(
        key,
        shape=(params.n_products,),
        minval=params.price_low,
        maxval=params.price_high,
    )
    demand = jnp.zeros((params.n_products,), dtype=jnp.float32)
    state = EnvState(
        prices=prices,
        demand=demand,
        step_count=jnp.asarray(0, dtype=jnp.int32),
        low_margin_streak=jnp.asarray(0, dtype=jnp.int32),
        last_agent_prob=jnp.asarray(params.alpha_nominal, dtype=jnp.float32),
        last_alpha_adv=jnp.asarray(params.alpha_nominal, dtype=jnp.float32),
    )
    return _flatten_obs(demand, prices), state


def step_env(
    key: jax.Array,
    state: EnvState,
    action: jax.Array,
    params: EnvParams,
) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[str, jax.Array]]:
    prices = _decode_action(state.prices, action, params)

    baseline = jnp.maximum(state.prices, 1.0)
    ux_volatility = jnp.where(
        state.step_count > 0, jnp.mean(jnp.abs(prices - state.prices) / baseline), 0.0
    )

    n_candidates = params.alpha_candidates.shape[0]
    cand_keys = jax.random.split(key, n_candidates)
    evals = jax.vmap(
        lambda k, a: _evaluate_candidate(k, a, prices, ux_volatility, params),
        in_axes=(0, 0),
    )(cand_keys, params.alpha_candidates)
    idx = jnp.argmin(evals.reward)

    demand = evals.demand[idx]
    reward = evals.reward[idx]
    revenue = evals.revenue[idx]
    agent_prob = evals.agent_prob[idx]
    leakage = evals.leakage[idx]
    discount = evals.discount[idx]
    ux_penalty = evals.ux_penalty[idx]
    n_purchases = evals.n_purchases[idx]
    n_agents = evals.n_agents[idx]
    alpha_adv = params.alpha_candidates[idx]

    step_count = state.step_count + 1
    avg_price = jnp.maximum(jnp.mean(prices), 1e-6)
    avg_margin = (avg_price - params.price_low) / avg_price
    next_streak = jnp.where(
        avg_margin < params.margin_floor, state.low_margin_streak + 1, 0
    )

    margin_collapsed = next_streak >= params.margin_floor_patience
    done = (step_count >= params.max_episode_steps) | margin_collapsed

    next_state = EnvState(
        prices=prices,
        demand=demand,
        step_count=step_count,
        low_margin_streak=next_streak,
        last_agent_prob=agent_prob,
        last_alpha_adv=alpha_adv,
    )
    obs = _flatten_obs(demand, prices)
    info = {
        "revenue": revenue,
        "agent_prob": agent_prob,
        "alpha_adv": alpha_adv,
        "coi_leakage": leakage,
        "coi_discount": discount,
        "ux_penalty": ux_penalty,
        "volatility": ux_volatility,
        "n_purchases": n_purchases,
        "n_agents": n_agents,
        "avg_margin": avg_margin,
    }
    return obs, next_state, reward, done, info


class PHANTOMJAXEnv:
    def __init__(self, params: EnvParams):
        self.params = params

    def reset(self, key: jax.Array, params: EnvParams | None = None):
        return reset_env(key, self.params if params is None else params)

    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams | None = None,
    ):
        return step_env(key, state, action, self.params if params is None else params)

    def action_space_n(self, params: EnvParams | None = None) -> int:
        p = self.params if params is None else params
        return int(p.action_scales.shape[0])

    def observation_dim(self, params: EnvParams | None = None) -> int:
        p = self.params if params is None else params
        return int(p.n_products * 2)
