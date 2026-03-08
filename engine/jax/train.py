"""Pure JAX trainers for PHANTOM environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import signal
import threading

import numpy as np

_stop_requested = threading.Event()
_jax_dist_initialized = False


def _init_jax_distributed() -> None:
    """Initialize JAX distributed if running on a multi-host TPU pod.
    Safe to call multiple times; no-op after first successful init or when JAX unavailable."""
    global _jax_dist_initialized
    if _jax_dist_initialized:
        return
    _jax_dist_initialized = True
    try:
        import jax as _jax

        _jax.distributed.initialize()
    except Exception:
        pass


try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from ..wandb_checkpoint import (  # noqa: E402
    checkpoint_artifact_name,
    download_latest_checkpoint,
    log_checkpoint_bytes,
)

try:
    import jax
    import jax.numpy as jnp
    import distrax
    import flax.linen as nn
    import optax
    from flax import serialization
    from flax.linen.initializers import constant, orthogonal
    from flax.training.train_state import TrainState

    HAS_JAX_STACK = True
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    distrax = None  # type: ignore[assignment]
    optax = None  # type: ignore[assignment]
    serialization = None  # type: ignore[assignment]

    class _ModuleStub:
        pass

    class _NNStub:
        Module = _ModuleStub

        @staticmethod
        def compact(fn):
            return fn

    nn = _NNStub()  # type: ignore[assignment]

    def constant(*_args, **_kwargs):  # type: ignore[override]
        return None

    def orthogonal(*_args, **_kwargs):  # type: ignore[override]
        return None

    class TrainState:  # type: ignore[override]
        pass

    HAS_JAX_STACK = False

from .env import PHANTOMJAXEnv, make_env_params  # noqa: E402


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation_fn = nn.relu if self.activation == "relu" else nn.tanh

        actor = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
        )(x)
        actor = activation_fn(actor)
        actor = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
        )(actor)
        actor = activation_fn(actor)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)

        critic = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
        )(x)
        critic = activation_fn(critic)
        critic = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation_fn(critic)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        return distrax.Categorical(logits=logits), jnp.squeeze(value, axis=-1)


class QNetwork(nn.Module):
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        activation_fn = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Dense(
            128,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
        )(x)
        x = activation_fn(x)
        x = nn.Dense(
            128,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
        )(x)
        x = activation_fn(x)
        q_values = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(x)
        return q_values


class Transition(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    info: dict[str, jax.Array]


class ReplayBatch(NamedTuple):
    obs: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_obs: jax.Array
    dones: jax.Array


class ReplayBuffer(NamedTuple):
    obs: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_obs: jax.Array
    dones: jax.Array
    ptr: jax.Array
    size: jax.Array


def _jax_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    out = {
        "algo": str(cfg.get("algo", "ppo")).lower(),
        "seed": int(cfg.get("seed", 42)),
        "learning_rate": float(cfg.get("learning_rate", 3e-4)),
        "gamma": float(cfg.get("gamma", 0.99)),
        "gae_lambda": float(cfg.get("gae_lambda", 0.95)),
        "clip_range": float(cfg.get("clip_range", 0.2)),
        "ent_coef": float(cfg.get("ent_coef", 0.01)),
        "vf_coef": float(cfg.get("vf_coef", 0.5)),
        "max_grad_norm": float(cfg.get("max_grad_norm", 0.5)),
        "activation": str(cfg.get("activation", "relu")),
        "total_timesteps": int(cfg.get("total_timesteps", 50_000)),
        "eval_episodes": int(cfg.get("eval_episodes", 5)),
        "model_dir": str(cfg.get("model_dir", "engine/models")),
        "log_freq": int(cfg.get("log_freq", 100)),
        "n_products": int(cfg.get("n_products", 10)),
        "N": int(cfg.get("N", 100)),
        "alpha": float(cfg.get("alpha", 0.3)),
        "lambda_coi": float(cfg.get("lambda_coi", 0.2)),
        "robust_radius": float(cfg.get("robust_radius", 0.15)),
        "robust_points": int(cfg.get("robust_points", 5)),
        "info_value": float(cfg.get("info_value", 1.0)),
        "price_low": float(cfg.get("price_low", 10.0)),
        "price_high": float(cfg.get("price_high", 150.0)),
        "action_levels": int(cfg.get("action_levels", 9)),
        "action_scale_low": float(cfg.get("action_scale_low", 0.8)),
        "action_scale_high": float(cfg.get("action_scale_high", 1.2)),
        "max_episode_steps": int(cfg.get("max_steps", 100)),
        "max_session_steps": int(cfg.get("max_session_steps", 40)),
        "margin_floor": float(cfg.get("margin_floor", 0.05)),
        "margin_floor_patience": int(cfg.get("margin_floor_patience", 5)),
        "prefer_behavior_data": bool(cfg.get("prefer_behavior_data", True)),
        "num_envs": int(cfg.get("jax_num_envs", 16)),
        "num_steps": int(cfg.get("jax_num_steps", 128)),
        "num_minibatches": int(cfg.get("jax_num_minibatches", 4)),
        "update_epochs": int(cfg.get("jax_update_epochs", 4)),
        "anneal_lr": bool(cfg.get("jax_anneal_lr", True)),
        "checkpoint_interval": int(cfg.get("checkpoint_interval", 10_000)),
        "buffer_size": int(cfg.get("buffer_size", 50_000)),
        "batch_size": int(cfg.get("batch_size", 256)),
        "train_freq": int(cfg.get("train_freq", 1)),
        "learning_starts": int(cfg.get("learning_starts", 1_000)),
        "target_update_interval": int(cfg.get("target_update_interval", 1_000)),
        "exploration_fraction": float(cfg.get("exploration_fraction", 0.2)),
        "exploration_final_eps": float(cfg.get("exploration_final_eps", 0.05)),
        "eps_start": float(cfg.get("eps_start", 1.0)),
        "eps_end": float(cfg.get("eps_end", 0.05)),
        "eps_decay": float(cfg.get("eps_decay", 0.9995)),
        "q_lr": float(cfg.get("q_lr", 0.1)),
        "q_bins": int(cfg.get("q_bins", 6)),
    }
    rollout = out["num_envs"] * out["num_steps"]
    out["num_updates"] = max(1, out["total_timesteps"] // max(rollout, 1))
    out["minibatch_size"] = max(1, rollout // max(out["num_minibatches"], 1))
    return out


def _scalar(value: Any) -> float:
    return float(np.asarray(value))


def _scalar_int(value: Any) -> int:
    return int(np.asarray(value))


def _make_env(cfg: dict[str, Any]) -> PHANTOMJAXEnv:
    env_params = make_env_params(
        n_products=cfg["n_products"],
        alpha=cfg["alpha"],
        n_sessions=cfg["N"],
        lambda_coi=cfg["lambda_coi"],
        robust_radius=cfg["robust_radius"],
        robust_points=cfg["robust_points"],
        info_value=cfg["info_value"],
        action_levels=cfg["action_levels"],
        action_scale_low=cfg["action_scale_low"],
        action_scale_high=cfg["action_scale_high"],
        price_low=cfg["price_low"],
        price_high=cfg["price_high"],
        max_episode_steps=cfg["max_episode_steps"],
        max_session_steps=cfg["max_session_steps"],
        margin_floor=cfg["margin_floor"],
        margin_floor_patience=cfg["margin_floor_patience"],
        prefer_behavior_data=cfg["prefer_behavior_data"],
    )
    return PHANTOMJAXEnv(env_params)


def _select_env_state(done: jax.Array, keep: jax.Array, reset: jax.Array) -> jax.Array:
    mask = done
    while mask.ndim < keep.ndim:
        mask = mask[..., None]
    return jnp.where(mask, reset, keep)


def _epsilon_by_fraction(step: int, cfg: dict[str, Any]) -> float:
    start = float(cfg["eps_start"])
    end = float(cfg["exploration_final_eps"])
    frac = float(cfg["exploration_fraction"])
    total = max(1, int(cfg["total_timesteps"]))
    decay_steps = max(1, int(total * frac))
    if step >= decay_steps:
        return end
    slope = (end - start) / decay_steps
    return float(start + slope * step)


def _digitize_scalar(value: jax.Array, bins: jax.Array) -> jax.Array:
    return jnp.sum(value > bins).astype(jnp.int32)


def _encode_qtable_state(
    obs: jax.Array,
    *,
    n_products: int,
    demand_bins: jax.Array,
    price_bins: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    demand = obs[:n_products]
    prices = obs[n_products : 2 * n_products]
    d_mean = jnp.mean(demand)
    d_std = jnp.std(demand)
    p_mean = jnp.mean(prices)
    return (
        _digitize_scalar(d_mean, demand_bins),
        _digitize_scalar(d_std, demand_bins),
        _digitize_scalar(p_mean, price_bins),
    )


def _init_replay_buffer(capacity: int, obs_dim: int) -> ReplayBuffer:
    cap = max(1, int(capacity))
    return ReplayBuffer(
        obs=jnp.zeros((cap, obs_dim), dtype=jnp.float32),
        actions=jnp.zeros((cap,), dtype=jnp.int32),
        rewards=jnp.zeros((cap,), dtype=jnp.float32),
        next_obs=jnp.zeros((cap, obs_dim), dtype=jnp.float32),
        dones=jnp.zeros((cap,), dtype=jnp.float32),
        ptr=jnp.asarray(0, dtype=jnp.int32),
        size=jnp.asarray(0, dtype=jnp.int32),
    )


def _replay_size(buffer: ReplayBuffer) -> int:
    return _scalar_int(buffer.size)


def _replay_add(
    buffer: ReplayBuffer,
    obs: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    next_obs: jax.Array,
    done: jax.Array,
) -> ReplayBuffer:
    capacity = int(buffer.obs.shape[0])
    idx = buffer.ptr % capacity
    return ReplayBuffer(
        obs=buffer.obs.at[idx].set(obs.astype(jnp.float32)),
        actions=buffer.actions.at[idx].set(action.astype(jnp.int32)),
        rewards=buffer.rewards.at[idx].set(reward.astype(jnp.float32)),
        next_obs=buffer.next_obs.at[idx].set(next_obs.astype(jnp.float32)),
        dones=buffer.dones.at[idx].set(done.astype(jnp.float32)),
        ptr=buffer.ptr + 1,
        size=jnp.minimum(buffer.size + 1, jnp.asarray(capacity, dtype=jnp.int32)),
    )


def _replay_sample(
    buffer: ReplayBuffer, key: jax.Array, batch_size: int
) -> ReplayBatch:
    size = jnp.maximum(buffer.size, 1)
    idx = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=size)
    return ReplayBatch(
        obs=buffer.obs[idx],
        actions=buffer.actions[idx],
        rewards=buffer.rewards[idx],
        next_obs=buffer.next_obs[idx],
        dones=buffer.dones[idx],
    )


def _make_actor_critic_train(
    config: dict[str, Any], *, algo: str, use_pmap: bool = False
):
    cfg = dict(config)
    cfg["algo"] = algo
    env = _make_env(cfg)
    network = ActorCritic(env.action_space_n(), activation=cfg["activation"])

    def linear_schedule(count: jax.Array) -> jax.Array:
        updates_done = count // (cfg["num_minibatches"] * cfg["update_epochs"])
        frac = 1.0 - updates_done / max(cfg["num_updates"], 1)
        return cfg["learning_rate"] * frac

    if cfg["anneal_lr"]:
        tx = optax.chain(
            optax.clip_by_global_norm(cfg["max_grad_norm"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(cfg["max_grad_norm"]),
            optax.adam(cfg["learning_rate"], eps=1e-5),
        )

    def init_runner_state(rng: jax.Array):
        rng, init_key = jax.random.split(rng)
        init_obs = jnp.zeros((env.observation_dim(),), dtype=jnp.float32)
        params = network.init(init_key, init_obs)
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key, cfg["num_envs"])
        obs, env_state = jax.vmap(env.reset)(reset_keys)
        return train_state, env_state, obs, rng

    def _update_step(runner_state, _):
        def _env_step(runner_state, _):
            train_state, env_state, last_obs, rng = runner_state
            rng, action_key = jax.random.split(rng)
            policy, value = network.apply(train_state.params, last_obs)
            action = policy.sample(seed=action_key)
            log_prob = policy.log_prob(action)

            rng, step_key = jax.random.split(rng)
            step_keys = jax.random.split(step_key, cfg["num_envs"])
            nxt_obs, nxt_state, reward, done, info = jax.vmap(
                env.step,
                in_axes=(0, 0, 0),
            )(step_keys, env_state, action)

            rng, reset_key = jax.random.split(rng)
            reset_keys = jax.random.split(reset_key, cfg["num_envs"])
            rst_obs, rst_state = jax.vmap(env.reset)(reset_keys)
            obs_next = jnp.where(done[:, None], rst_obs, nxt_obs)
            env_next = jax.tree_util.tree_map(
                lambda keep, reset: _select_env_state(done, keep, reset),
                nxt_state,
                rst_state,
            )
            transition = Transition(
                done=done,
                action=action,
                value=value,
                reward=reward,
                log_prob=log_prob,
                obs=last_obs,
                info=info,
            )
            return (train_state, env_next, obs_next, rng), transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step,
            runner_state,
            None,
            length=cfg["num_steps"],
        )

        train_state, env_state, last_obs, rng = runner_state
        _, last_value = network.apply(train_state.params, last_obs)

        def _compute_gae(traj_batch, last_value):
            def _gae_step(carry, transition):
                gae, next_value = carry
                delta = (
                    transition.reward
                    + cfg["gamma"] * next_value * (1.0 - transition.done)
                    - transition.value
                )
                gae = (
                    delta
                    + cfg["gamma"] * cfg["gae_lambda"] * (1.0 - transition.done) * gae
                )
                return (gae, transition.value), gae

            _, advantages = jax.lax.scan(
                _gae_step,
                (jnp.zeros_like(last_value), last_value),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            targets = advantages + traj_batch.value
            return advantages, targets

        advantages, targets = _compute_gae(traj_batch, last_value)

        def _update_epoch(update_state, _):
            def _update_minibatch(train_state, batch_info):
                traj_b, adv_b, tgt_b = batch_info

                def _loss_fn(params, traj_b, adv_b, tgt_b):
                    policy, value = network.apply(params, traj_b.obs)
                    log_prob = policy.log_prob(traj_b.action)
                    adv_norm = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                    if algo == "ppo":
                        value_clipped = traj_b.value + (value - traj_b.value).clip(
                            -cfg["clip_range"], cfg["clip_range"]
                        )
                        value_loss = (
                            0.5
                            * jnp.maximum(
                                jnp.square(value - tgt_b),
                                jnp.square(value_clipped - tgt_b),
                            ).mean()
                        )
                        ratio = jnp.exp(log_prob - traj_b.log_prob)
                        policy_loss = -jnp.minimum(
                            ratio * adv_norm,
                            jnp.clip(
                                ratio,
                                1.0 - cfg["clip_range"],
                                1.0 + cfg["clip_range"],
                            )
                            * adv_norm,
                        ).mean()
                    else:
                        value_loss = 0.5 * jnp.mean(jnp.square(value - tgt_b))
                        policy_loss = -(log_prob * adv_norm).mean()

                    entropy = policy.entropy().mean()
                    total_loss = (
                        policy_loss
                        + cfg["vf_coef"] * value_loss
                        - cfg["ent_coef"] * entropy
                    )
                    return total_loss, (value_loss, policy_loss, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (_, _), grads = grad_fn(train_state.params, traj_b, adv_b, tgt_b)
                if use_pmap:
                    grads = jax.lax.pmean(grads, axis_name="devices")
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, jnp.asarray(0.0, dtype=jnp.float32)

            train_state, traj_batch, advantages, targets, rng = update_state
            rng, perm_key = jax.random.split(rng)
            batch_size = cfg["num_envs"] * cfg["num_steps"]
            permutation = jax.random.permutation(perm_key, batch_size)

            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]),
                batch,
            )
            shuffled = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0),
                batch,
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape(
                    (cfg["num_minibatches"], cfg["minibatch_size"]) + x.shape[1:]
                ),
                shuffled,
            )
            train_state, _ = jax.lax.scan(_update_minibatch, train_state, minibatches)
            return (train_state, traj_batch, advantages, targets, rng), None

        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, _ = jax.lax.scan(
            _update_epoch,
            update_state,
            None,
            length=cfg["update_epochs"],
        )
        train_state = update_state[0]
        rng = update_state[-1]

        metric = {
            "reward": jnp.mean(traj_batch.reward),
            "revenue": jnp.mean(traj_batch.info["revenue"]),
            "agent_prob": jnp.mean(traj_batch.info["agent_prob"]),
            "alpha_adv": jnp.mean(traj_batch.info["alpha_adv"]),
            "coi_leakage": jnp.mean(traj_batch.info["coi_leakage"]),
        }
        next_runner_state = (train_state, env_state, last_obs, rng)
        return next_runner_state, metric

    def run_updates(runner_state, num_updates: int):
        updates = max(1, int(num_updates))
        runner_state, metric = jax.lax.scan(
            _update_step,
            runner_state,
            None,
            length=updates,
        )
        return {
            "runner_state": runner_state,
            "metrics": metric,
        }

    return init_runner_state, run_updates, network, env, cfg


def make_train(config: dict[str, Any]):
    cfg = _jax_cfg(config)
    algo = cfg["algo"]
    if algo not in {"ppo", "a2c"}:
        raise ValueError(f"make_train supports actor-critic algos only, got '{algo}'")
    return _make_actor_critic_train(cfg, algo=algo)


def evaluate_policy(
    *,
    network: ActorCritic,
    params: Any,
    env: PHANTOMJAXEnv,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    rewards: list[float] = []
    revenues: list[float] = []
    key = jax.random.PRNGKey(seed)

    for _ in range(int(episodes)):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        ep_reward = 0.0
        ep_revenue = 0.0
        done = False
        steps = 0

        while not done and steps < int(env.params.max_episode_steps):
            policy, _ = network.apply(params, obs)
            action = jnp.argmax(policy.logits)
            key, step_key = jax.random.split(key)
            obs, state, reward, done_flag, info = env.step(step_key, state, action)
            ep_reward += _scalar(reward)
            ep_revenue += _scalar(info["revenue"])
            done = bool(np.asarray(done_flag))
            steps += 1

        rewards.append(ep_reward)
        revenues.append(ep_revenue)

    return {
        "eval/reward_mean": float(np.mean(rewards)),
        "eval/revenue_mean": float(np.mean(revenues)),
        "eval/reward_std": float(np.std(rewards)),
        "eval/revenue_std": float(np.std(revenues)),
    }


def _evaluate_q_network(
    *,
    network: QNetwork,
    params: Any,
    env: PHANTOMJAXEnv,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    rewards: list[float] = []
    revenues: list[float] = []
    key = jax.random.PRNGKey(seed)

    for _ in range(int(episodes)):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        ep_reward = 0.0
        ep_revenue = 0.0
        done = False
        steps = 0

        while not done and steps < int(env.params.max_episode_steps):
            q_values = network.apply(params, obs)
            action = jnp.argmax(q_values)
            key, step_key = jax.random.split(key)
            obs, state, reward, done_flag, info = env.step(step_key, state, action)
            ep_reward += _scalar(reward)
            ep_revenue += _scalar(info["revenue"])
            done = bool(np.asarray(done_flag))
            steps += 1

        rewards.append(ep_reward)
        revenues.append(ep_revenue)

    return {
        "eval/reward_mean": float(np.mean(rewards)),
        "eval/revenue_mean": float(np.mean(revenues)),
        "eval/reward_std": float(np.std(rewards)),
        "eval/revenue_std": float(np.std(revenues)),
    }


def _evaluate_q_table(
    *,
    q_table: jax.Array,
    env: PHANTOMJAXEnv,
    episodes: int,
    seed: int,
    n_products: int,
    demand_bins: jax.Array,
    price_bins: jax.Array,
) -> dict[str, float]:
    rewards: list[float] = []
    revenues: list[float] = []
    key = jax.random.PRNGKey(seed)

    for _ in range(int(episodes)):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        ep_reward = 0.0
        ep_revenue = 0.0
        done = False
        steps = 0

        while not done and steps < int(env.params.max_episode_steps):
            s0, s1, s2 = _encode_qtable_state(
                obs,
                n_products=n_products,
                demand_bins=demand_bins,
                price_bins=price_bins,
            )
            action = jnp.argmax(q_table[s0, s1, s2])
            key, step_key = jax.random.split(key)
            obs, state, reward, done_flag, info = env.step(step_key, state, action)
            ep_reward += _scalar(reward)
            ep_revenue += _scalar(info["revenue"])
            done = bool(np.asarray(done_flag))
            steps += 1

        rewards.append(ep_reward)
        revenues.append(ep_revenue)

    return {
        "eval/reward_mean": float(np.mean(rewards)),
        "eval/revenue_mean": float(np.mean(revenues)),
        "eval/reward_std": float(np.std(rewards)),
        "eval/revenue_std": float(np.std(revenues)),
    }


def _train_actor_critic(
    cfg: dict[str, Any],
    *,
    algo: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    num_devices = jax.local_device_count()
    use_pmap = num_devices > 1
    global_devices = max(1, int(jax.device_count()))
    process_idx = int(jax.process_index())

    init_runner_state, run_updates_raw, network, env, run_cfg = (
        _make_actor_critic_train(cfg, algo=algo, use_pmap=use_pmap)
    )

    if use_pmap:
        run_fn = jax.pmap(
            run_updates_raw,
            axis_name="devices",
            static_broadcasted_argnums=(1,),
            devices=jax.local_devices(),
        )
    else:
        run_fn = jax.jit(run_updates_raw, static_argnames=("num_updates",))

    rollout_steps = int(run_cfg["num_steps"] * run_cfg["num_envs"])
    rollout_steps_global = rollout_steps * (global_devices if use_pmap else 1)
    total_updates = int(run_cfg["num_updates"])
    checkpoint_interval = max(1, int(run_cfg.get("checkpoint_interval", 10_000)))
    segment_updates = max(1, checkpoint_interval // max(rollout_steps_global, 1))

    base_rng = jax.random.PRNGKey(run_cfg["seed"])
    base_rng = jax.random.fold_in(base_rng, process_idx)
    if use_pmap:
        init_keys = jax.random.split(base_rng, num_devices)
        runner_state = jax.vmap(init_runner_state)(init_keys)
        single_runner_state = jax.tree_util.tree_map(lambda x: x[0], runner_state)
    else:
        single_runner_state = init_runner_state(base_rng)
        runner_state = single_runner_state
    updates_done = 0
    restored_train_state = None

    is_primary = process_idx == 0
    artifact_name = None
    if HAS_WANDB and wandb.run is not None:
        sweep_id = getattr(wandb.run, "sweep_id", None)
        artifact_name = checkpoint_artifact_name(
            run_cfg,
            backend="jax",
            sweep_id=sweep_id,
        )
        restored = download_latest_checkpoint(
            artifact_name,
            file_name=f"jax_{algo}_runner_state.msgpack",
        )
        if restored is not None:
            checkpoint_path, metadata = restored
            template = {"runner_state": single_runner_state, "updates_done": 0}
            payload = serialization.from_bytes(template, checkpoint_path.read_bytes())
            single_runner_state = payload["runner_state"]
            restored_train_state = payload["runner_state"][0]
            updates_done = int(payload.get("updates_done", 0))
            if updates_done <= 0:
                updates_done = int(metadata.get("updates_done", 0))
            updates_done = max(0, min(updates_done, total_updates))

    if use_pmap and restored_train_state is not None:
        runner_state = (
            jax.device_put_replicated(restored_train_state, jax.local_devices()),
            runner_state[1],
            runner_state[2],
            runner_state[3],
        )
    elif not use_pmap:
        runner_state = single_runner_state

    metric_keys = ["reward", "revenue", "agent_prob", "alpha_adv", "coi_leakage"]
    metric_sums = {k: 0.0 for k in metric_keys}
    metric_count = 0

    while updates_done < total_updates:
        updates_this_segment = min(segment_updates, total_updates - updates_done)
        if use_pmap:
            out = run_fn(runner_state, updates_this_segment)
        else:
            out = run_fn(runner_state, updates_this_segment)
        runner_state = out["runner_state"]
        metric = out["metrics"]

        if use_pmap:
            segment_values = {
                key: np.asarray(metric[key], dtype=np.float64).reshape(-1)
                for key in metric_keys
            }
        else:
            segment_values = {
                key: np.asarray(metric[key], dtype=np.float64).reshape(-1)
                for key in metric_keys
            }

        segment_count = int(segment_values["reward"].shape[0]) if segment_values else 0
        metric_count += segment_count
        for key in metric_keys:
            metric_sums[key] += float(segment_values[key].sum())

        updates_done += int(updates_this_segment)
        global_step = int(updates_done * rollout_steps_global)

        if is_primary and HAS_WANDB and wandb.run is not None:
            wandb.log(
                {
                    "train/reward_mean": float(segment_values["reward"].mean()),
                    "train/revenue_mean": float(segment_values["revenue"].mean()),
                    "train/agent_prob": float(segment_values["agent_prob"].mean()),
                    "train/alpha_adv": float(segment_values["alpha_adv"].mean()),
                    "train/coi_leakage": float(segment_values["coi_leakage"].mean()),
                    "train/global_step": global_step,
                },
                step=global_step,
            )
            if artifact_name is not None:
                # extract device-0 state for checkpoint portability
                state_to_save = (
                    jax.tree_util.tree_map(lambda x: x[0], runner_state)
                    if use_pmap
                    else runner_state
                )
                checkpoint_payload = serialization.to_bytes(
                    {"runner_state": state_to_save, "updates_done": updates_done}
                )
                log_checkpoint_bytes(
                    artifact_name,
                    file_name=f"jax_{algo}_runner_state.msgpack",
                    payload=checkpoint_payload,
                    metadata={
                        "step": global_step,
                        "updates_done": updates_done,
                        "rollout_steps": rollout_steps_global,
                        "algo": algo,
                    },
                )
        if _stop_requested.is_set():
            break

    # extract device-0 params for eval and save
    final_runner = (
        jax.tree_util.tree_map(lambda x: x[0], runner_state)
        if use_pmap
        else runner_state
    )
    train_state = final_runner[0]
    denom = float(metric_count) if metric_count > 0 else 1.0
    metrics = {
        "train/reward_mean": float(metric_sums["reward"] / denom),
        "train/revenue_mean": float(metric_sums["revenue"] / denom),
        "train/agent_prob": float(metric_sums["agent_prob"] / denom),
        "train/alpha_adv": float(metric_sums["alpha_adv"] / denom),
        "train/coi_leakage": float(metric_sums["coi_leakage"] / denom),
        "train/global_step": int(updates_done * rollout_steps_global),
    }

    eval_metrics = evaluate_policy(
        network=network,
        params=train_state.params,
        env=env,
        episodes=run_cfg["eval_episodes"],
        seed=run_cfg["seed"] + 7,
    )
    metrics.update(eval_metrics)

    if is_primary:
        model_dir = Path(run_cfg["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"phantom_{algo}_jax.msgpack"
        model_path.write_bytes(serialization.to_bytes(train_state.params))
        metrics["model/path"] = str(model_path)

    return {"params": train_state.params}, metrics


def _train_dqn(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    run_cfg = dict(cfg)
    env = _make_env(run_cfg)
    action_dim = env.action_space_n()
    obs_dim = env.observation_dim()
    q_net = QNetwork(action_dim=action_dim, activation=run_cfg["activation"])

    init_obs = jnp.zeros((obs_dim,), dtype=jnp.float32)
    rng = jax.random.PRNGKey(run_cfg["seed"])
    rng, init_key = jax.random.split(rng)
    params = q_net.init(init_key, init_obs)
    tx = optax.adam(run_cfg["learning_rate"])
    train_state = TrainState.create(apply_fn=q_net.apply, params=params, tx=tx)
    target_params = train_state.params

    buffer = _init_replay_buffer(run_cfg["buffer_size"], obs_dim)

    rng, reset_key = jax.random.split(rng)
    obs, env_state = env.reset(reset_key)

    start_step = 0
    epsilon_value = float(run_cfg["eps_start"])
    artifact_name = None

    if HAS_WANDB and wandb.run is not None:
        sweep_id = getattr(wandb.run, "sweep_id", None)
        artifact_name = checkpoint_artifact_name(
            run_cfg,
            backend="jax",
            sweep_id=sweep_id,
        )
        restored = download_latest_checkpoint(
            artifact_name,
            file_name="jax_dqn_state.msgpack",
        )
        if restored is not None:
            checkpoint_path, metadata = restored
            template = {
                "params": train_state.params,
                "target_params": target_params,
                "opt_state": train_state.opt_state,
                "global_step": 0,
                "epsilon": epsilon_value,
            }
            payload = serialization.from_bytes(template, checkpoint_path.read_bytes())
            train_state = train_state.replace(
                params=payload["params"],
                opt_state=payload["opt_state"],
            )
            target_params = payload["target_params"]
            start_step = int(payload.get("global_step", metadata.get("step", 0)))
            start_step = max(0, min(start_step, int(run_cfg["total_timesteps"])))
            epsilon_value = float(payload.get("epsilon", epsilon_value))

    @jax.jit
    def dqn_update(
        state: TrainState,
        target: Any,
        batch: ReplayBatch,
    ) -> tuple[TrainState, jax.Array]:
        def loss_fn(model_params):
            q_values = q_net.apply(model_params, batch.obs)
            chosen = jnp.take_along_axis(
                q_values,
                batch.actions[:, None],
                axis=1,
            ).squeeze(-1)
            next_q = q_net.apply(target, batch.next_obs)
            next_max = jnp.max(next_q, axis=1)
            td_target = (
                batch.rewards + run_cfg["gamma"] * (1.0 - batch.dones) * next_max
            )
            td_error = chosen - jax.lax.stop_gradient(td_target)
            return jnp.mean(jnp.square(td_error))

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        next_state = state.apply_gradients(grads=grads)
        return next_state, loss

    metric_sums = {
        "reward": 0.0,
        "revenue": 0.0,
        "agent_prob": 0.0,
        "alpha_adv": 0.0,
        "coi_leakage": 0.0,
        "loss": 0.0,
    }
    metric_count = 0
    loss_count = 0

    total_steps = int(run_cfg["total_timesteps"])
    checkpoint_interval = max(1, int(run_cfg["checkpoint_interval"]))
    batch_size = max(1, int(run_cfg["batch_size"]))

    for global_step in range(start_step + 1, total_steps + 1):
        epsilon_value = _epsilon_by_fraction(global_step - 1, run_cfg)

        rng, eps_key, action_key, step_key, reset_key, sample_key = jax.random.split(
            rng, 6
        )
        do_explore = bool(np.asarray(jax.random.uniform(eps_key) < epsilon_value))
        if do_explore:
            action = jax.random.randint(
                action_key, shape=(), minval=0, maxval=action_dim
            )
        else:
            q_values = q_net.apply(train_state.params, obs)
            action = jnp.argmax(q_values)

        next_obs, next_state, reward, done, info = env.step(step_key, env_state, action)
        buffer = _replay_add(
            buffer,
            obs,
            action,
            reward,
            next_obs,
            done.astype(jnp.float32),
        )

        metric_count += 1
        metric_sums["reward"] += _scalar(reward)
        metric_sums["revenue"] += _scalar(info["revenue"])
        metric_sums["agent_prob"] += _scalar(info["agent_prob"])
        metric_sums["alpha_adv"] += _scalar(info["alpha_adv"])
        metric_sums["coi_leakage"] += _scalar(info["coi_leakage"])

        if bool(np.asarray(done)):
            obs, env_state = env.reset(reset_key)
        else:
            obs, env_state = next_obs, next_state

        ready = (
            global_step >= int(run_cfg["learning_starts"])
            and global_step % int(run_cfg["train_freq"]) == 0
            and _replay_size(buffer) >= batch_size
        )
        if ready:
            batch = _replay_sample(buffer, sample_key, batch_size)
            train_state, loss = dqn_update(train_state, target_params, batch)
            metric_sums["loss"] += _scalar(loss)
            loss_count += 1

        if global_step % int(run_cfg["target_update_interval"]) == 0:
            target_params = train_state.params

        if (
            HAS_WANDB
            and wandb.run is not None
            and global_step % int(run_cfg["log_freq"]) == 0
        ):
            wandb.log(
                {
                    "train/reward_mean": metric_sums["reward"] / max(metric_count, 1),
                    "train/revenue_mean": metric_sums["revenue"] / max(metric_count, 1),
                    "train/agent_prob": metric_sums["agent_prob"]
                    / max(metric_count, 1),
                    "train/alpha_adv": metric_sums["alpha_adv"] / max(metric_count, 1),
                    "train/coi_leakage": metric_sums["coi_leakage"]
                    / max(metric_count, 1),
                    "train/loss": metric_sums["loss"] / max(loss_count, 1),
                    "train/epsilon": epsilon_value,
                    "train/global_step": global_step,
                },
                step=global_step,
            )

        if artifact_name is not None and global_step % checkpoint_interval == 0:
            payload = serialization.to_bytes(
                {
                    "params": train_state.params,
                    "target_params": target_params,
                    "opt_state": train_state.opt_state,
                    "global_step": global_step,
                    "epsilon": epsilon_value,
                }
            )
            log_checkpoint_bytes(
                artifact_name,
                file_name="jax_dqn_state.msgpack",
                payload=payload,
                metadata={
                    "step": global_step,
                    "algo": "dqn",
                },
            )
        if _stop_requested.is_set():
            break

    denom = float(metric_count) if metric_count > 0 else 1.0
    metrics = {
        "train/reward_mean": float(metric_sums["reward"] / denom),
        "train/revenue_mean": float(metric_sums["revenue"] / denom),
        "train/agent_prob": float(metric_sums["agent_prob"] / denom),
        "train/alpha_adv": float(metric_sums["alpha_adv"] / denom),
        "train/coi_leakage": float(metric_sums["coi_leakage"] / denom),
        "train/loss": float(metric_sums["loss"] / max(loss_count, 1)),
        "train/global_step": total_steps,
    }

    eval_metrics = _evaluate_q_network(
        network=q_net,
        params=train_state.params,
        env=env,
        episodes=run_cfg["eval_episodes"],
        seed=run_cfg["seed"] + 7,
    )
    metrics.update(eval_metrics)

    model_dir = Path(run_cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "phantom_dqn_jax.msgpack"
    model_path.write_bytes(serialization.to_bytes(train_state.params))
    metrics["model/path"] = str(model_path)
    return {
        "params": train_state.params,
        "target_params": target_params,
    }, metrics


def _train_qtable(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    run_cfg = dict(cfg)
    env = _make_env(run_cfg)
    action_dim = env.action_space_n()
    n_bins = max(2, int(run_cfg["q_bins"]))
    n_products = int(run_cfg["n_products"])

    q_table = jnp.zeros((n_bins, n_bins, n_bins, action_dim), dtype=jnp.float32)
    demand_bins = jnp.linspace(0.0, 100.0, n_bins + 1, dtype=jnp.float32)[1:-1]
    price_bins = jnp.linspace(
        float(run_cfg["price_low"]),
        float(run_cfg["price_high"]),
        n_bins + 1,
        dtype=jnp.float32,
    )[1:-1]

    rng = jax.random.PRNGKey(run_cfg["seed"])
    rng, reset_key = jax.random.split(rng)
    obs, env_state = env.reset(reset_key)

    epsilon_value = float(run_cfg["eps_start"])
    start_step = 0
    artifact_name = None

    if HAS_WANDB and wandb.run is not None:
        sweep_id = getattr(wandb.run, "sweep_id", None)
        artifact_name = checkpoint_artifact_name(
            run_cfg,
            backend="jax",
            sweep_id=sweep_id,
        )
        restored = download_latest_checkpoint(
            artifact_name,
            file_name="jax_qtable_state.msgpack",
        )
        if restored is not None:
            checkpoint_path, metadata = restored
            template = {
                "q_table": q_table,
                "global_step": 0,
                "epsilon": epsilon_value,
            }
            payload = serialization.from_bytes(template, checkpoint_path.read_bytes())
            q_table = payload["q_table"]
            start_step = int(payload.get("global_step", metadata.get("step", 0)))
            start_step = max(0, min(start_step, int(run_cfg["total_timesteps"])))
            epsilon_value = float(payload.get("epsilon", epsilon_value))

    metric_sums = {
        "reward": 0.0,
        "revenue": 0.0,
        "agent_prob": 0.0,
        "alpha_adv": 0.0,
        "coi_leakage": 0.0,
    }
    metric_count = 0

    total_steps = int(run_cfg["total_timesteps"])
    checkpoint_interval = max(1, int(run_cfg["checkpoint_interval"]))

    for global_step in range(start_step + 1, total_steps + 1):
        s0, s1, s2 = _encode_qtable_state(
            obs,
            n_products=n_products,
            demand_bins=demand_bins,
            price_bins=price_bins,
        )
        state_q = q_table[s0, s1, s2]

        rng, eps_key, action_key, step_key, reset_key = jax.random.split(rng, 5)
        do_explore = bool(np.asarray(jax.random.uniform(eps_key) < epsilon_value))
        if do_explore:
            action = jax.random.randint(
                action_key, shape=(), minval=0, maxval=action_dim
            )
        else:
            action = jnp.argmax(state_q)

        next_obs, next_state, reward, done, info = env.step(step_key, env_state, action)
        ns0, ns1, ns2 = _encode_qtable_state(
            next_obs,
            n_products=n_products,
            demand_bins=demand_bins,
            price_bins=price_bins,
        )

        best_next = jnp.max(q_table[ns0, ns1, ns2])
        done_f = done.astype(jnp.float32)
        td_target = reward + run_cfg["gamma"] * (1.0 - done_f) * best_next
        old_value = q_table[s0, s1, s2, action]
        new_value = old_value + run_cfg["q_lr"] * (td_target - old_value)
        q_table = q_table.at[s0, s1, s2, action].set(new_value)

        epsilon_value = max(
            float(run_cfg["eps_end"]),
            epsilon_value * float(run_cfg["eps_decay"]),
        )

        metric_count += 1
        metric_sums["reward"] += _scalar(reward)
        metric_sums["revenue"] += _scalar(info["revenue"])
        metric_sums["agent_prob"] += _scalar(info["agent_prob"])
        metric_sums["alpha_adv"] += _scalar(info["alpha_adv"])
        metric_sums["coi_leakage"] += _scalar(info["coi_leakage"])

        if bool(np.asarray(done)):
            obs, env_state = env.reset(reset_key)
        else:
            obs, env_state = next_obs, next_state

        if (
            HAS_WANDB
            and wandb.run is not None
            and global_step % int(run_cfg["log_freq"]) == 0
        ):
            wandb.log(
                {
                    "train/reward_mean": metric_sums["reward"] / max(metric_count, 1),
                    "train/revenue_mean": metric_sums["revenue"] / max(metric_count, 1),
                    "train/agent_prob": metric_sums["agent_prob"]
                    / max(metric_count, 1),
                    "train/alpha_adv": metric_sums["alpha_adv"] / max(metric_count, 1),
                    "train/coi_leakage": metric_sums["coi_leakage"]
                    / max(metric_count, 1),
                    "train/epsilon": epsilon_value,
                    "train/global_step": global_step,
                },
                step=global_step,
            )

        if artifact_name is not None and global_step % checkpoint_interval == 0:
            payload = serialization.to_bytes(
                {
                    "q_table": q_table,
                    "global_step": global_step,
                    "epsilon": epsilon_value,
                }
            )
            log_checkpoint_bytes(
                artifact_name,
                file_name="jax_qtable_state.msgpack",
                payload=payload,
                metadata={
                    "step": global_step,
                    "algo": "qtable",
                },
            )

    denom = float(metric_count) if metric_count > 0 else 1.0
    metrics = {
        "train/reward_mean": float(metric_sums["reward"] / denom),
        "train/revenue_mean": float(metric_sums["revenue"] / denom),
        "train/agent_prob": float(metric_sums["agent_prob"] / denom),
        "train/alpha_adv": float(metric_sums["alpha_adv"] / denom),
        "train/coi_leakage": float(metric_sums["coi_leakage"] / denom),
        "train/global_step": total_steps,
    }

    eval_metrics = _evaluate_q_table(
        q_table=q_table,
        env=env,
        episodes=run_cfg["eval_episodes"],
        seed=run_cfg["seed"] + 7,
        n_products=n_products,
        demand_bins=demand_bins,
        price_bins=price_bins,
    )
    metrics.update(eval_metrics)

    model_dir = Path(run_cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "phantom_qtable_jax.msgpack"
    model_path.write_bytes(serialization.to_bytes(q_table))
    metrics["model/path"] = str(model_path)
    return {"q_table": q_table}, metrics


def train_jax(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    if not HAS_JAX_STACK:
        raise ImportError(
            "JAX path requires jax, flax, optax, and distrax. "
            "Install engine/jax/requirements.txt on this machine first."
        )

    _init_jax_distributed()
    _stop_requested.clear()
    run_cfg = _jax_cfg(cfg)
    algo = run_cfg["algo"]
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda *_: _stop_requested.set())

    if algo in {"ppo", "a2c"}:
        return _train_actor_critic(run_cfg, algo=algo)
    if algo == "dqn":
        return _train_dqn(run_cfg)
    if algo == "qtable":
        return _train_qtable(run_cfg)
    raise ValueError(f"Unsupported JAX algo '{algo}'")
