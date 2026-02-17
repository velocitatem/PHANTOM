"""Pure JAX PPO trainer for the PHANTOM environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from ..wandb_checkpoint import (
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

from .env import PHANTOMJAXEnv, make_env_params


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


class Transition(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    info: dict[str, jax.Array]


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
    }
    rollout = out["num_envs"] * out["num_steps"]
    out["num_updates"] = max(1, out["total_timesteps"] // max(rollout, 1))
    out["minibatch_size"] = max(1, rollout // max(out["num_minibatches"], 1))
    return out


def _select_env_state(done: jax.Array, keep: jax.Array, reset: jax.Array) -> jax.Array:
    mask = done
    while mask.ndim < keep.ndim:
        mask = mask[..., None]
    return jnp.where(mask, reset, keep)


def make_train(config: dict[str, Any]):
    cfg = _jax_cfg(config)
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
    env = PHANTOMJAXEnv(env_params)
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

                    adv_norm = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
                    ratio = jnp.exp(log_prob - traj_b.log_prob)
                    loss_actor = -jnp.minimum(
                        ratio * adv_norm,
                        jnp.clip(
                            ratio,
                            1.0 - cfg["clip_range"],
                            1.0 + cfg["clip_range"],
                        )
                        * adv_norm,
                    ).mean()
                    entropy = policy.entropy().mean()
                    total_loss = (
                        loss_actor
                        + cfg["vf_coef"] * value_loss
                        - cfg["ent_coef"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (_, _), grads = grad_fn(train_state.params, traj_b, adv_b, tgt_b)
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

    def run_updates(runner_state, *, num_updates: int):
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
            ep_reward += float(np.asarray(reward))
            ep_revenue += float(np.asarray(info["revenue"]))
            done = bool(np.asarray(done_flag))
            steps += 1

        rewards.append(ep_reward)
        revenues.append(ep_revenue)

    return {
        "eval/reward": float(np.mean(rewards)),
        "eval/revenue": float(np.mean(revenues)),
        "eval/reward_std": float(np.std(rewards)),
        "eval/revenue_std": float(np.std(revenues)),
    }


def train_jax(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    if not HAS_JAX_STACK:
        raise ImportError(
            "JAX PPO path requires jax, flax, optax, and distrax. "
            "Install engine/jax/requirements.txt on this machine first."
        )

    run_cfg = _jax_cfg(cfg)
    if run_cfg["algo"] != "ppo":
        raise ValueError(
            f"JAX backend currently supports algo='ppo' only, got '{run_cfg['algo']}'"
        )

    init_runner_state, run_updates, network, env, run_cfg = make_train(run_cfg)
    run_updates_jit = jax.jit(run_updates, static_argnames=("num_updates",))
    rollout_steps = int(run_cfg["num_steps"] * run_cfg["num_envs"])
    total_updates = int(run_cfg["num_updates"])
    checkpoint_interval = max(1, int(run_cfg.get("checkpoint_interval", 10_000)))
    segment_updates = max(1, checkpoint_interval // max(rollout_steps, 1))

    rng = jax.random.PRNGKey(run_cfg["seed"])
    runner_state = init_runner_state(rng)
    updates_done = 0

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
            file_name="jax_runner_state.msgpack",
        )
        if restored is not None:
            checkpoint_path, metadata = restored
            template = {
                "runner_state": runner_state,
                "updates_done": 0,
            }
            payload = serialization.from_bytes(template, checkpoint_path.read_bytes())
            runner_state = payload["runner_state"]
            updates_done = int(payload.get("updates_done", 0))
            if updates_done <= 0:
                updates_done = int(metadata.get("updates_done", 0))
            updates_done = max(0, min(updates_done, total_updates))

    metric_keys = ["reward", "revenue", "agent_prob", "alpha_adv", "coi_leakage"]
    metric_sums = {k: 0.0 for k in metric_keys}
    metric_count = 0

    while updates_done < total_updates:
        updates_this_segment = min(segment_updates, total_updates - updates_done)
        out = run_updates_jit(runner_state, num_updates=updates_this_segment)
        runner_state = out["runner_state"]
        metric = out["metrics"]

        segment_values = {
            k: np.asarray(metric[k], dtype=np.float64) for k in metric_keys
        }
        segment_count = int(segment_values["reward"].shape[0]) if segment_values else 0
        metric_count += segment_count
        for key in metric_keys:
            metric_sums[key] += float(segment_values[key].sum())

        updates_done += int(updates_this_segment)
        global_step = int(updates_done * rollout_steps)

        if HAS_WANDB and wandb.run is not None:
            wandb.log(
                {
                    "train/reward": float(segment_values["reward"].mean()),
                    "train/revenue": float(segment_values["revenue"].mean()),
                    "train/agent_prob": float(segment_values["agent_prob"].mean()),
                    "train/alpha_adv": float(segment_values["alpha_adv"].mean()),
                    "train/coi_leakage": float(segment_values["coi_leakage"].mean()),
                    "train/global_step": global_step,
                },
                step=global_step,
            )
            if artifact_name is not None:
                checkpoint_payload = serialization.to_bytes(
                    {
                        "runner_state": runner_state,
                        "updates_done": updates_done,
                    }
                )
                log_checkpoint_bytes(
                    artifact_name,
                    file_name="jax_runner_state.msgpack",
                    payload=checkpoint_payload,
                    metadata={
                        "step": global_step,
                        "updates_done": updates_done,
                        "rollout_steps": rollout_steps,
                        "algo": "ppo",
                    },
                )

    train_state = runner_state[0]
    denom = float(metric_count) if metric_count > 0 else 1.0
    metrics = {
        "train/reward": float(metric_sums["reward"] / denom),
        "train/revenue": float(metric_sums["revenue"] / denom),
        "train/agent_prob": float(metric_sums["agent_prob"] / denom),
        "train/alpha_adv": float(metric_sums["alpha_adv"] / denom),
        "train/coi_leakage": float(metric_sums["coi_leakage"] / denom),
        "train/global_step": int(updates_done * rollout_steps),
    }

    eval_metrics = evaluate_policy(
        network=network,
        params=train_state.params,
        env=env,
        episodes=run_cfg["eval_episodes"],
        seed=run_cfg["seed"] + 7,
    )
    metrics.update(eval_metrics)

    model_dir = Path(run_cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "phantom_ppo_jax.msgpack"
    model_path.write_bytes(serialization.to_bytes(train_state.params))
    metrics["model/path"] = str(model_path)
    return {"params": train_state.params}, metrics
