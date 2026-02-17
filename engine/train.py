from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import numpy as np

from .wandb_checkpoint import checkpoint_artifact_name, download_latest_checkpoint

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

from .jax import JAX_AVAILABLE


DEFAULT_CFG = {
    "project": "phantom-pricing",
    "algo": "ppo",
    "seed": 42,
    "total_timesteps": 50_000,
    "eval_episodes": 5,
    "eval_freq": 1_000,
    "log_freq": 100,
    "revenue_weight": 0.01,
    "n_products": 10,
    "N": 100,
    "alpha": 0.3,
    "lambda_coi": 0.2,
    "robust_radius": 0.15,
    "robust_points": 5,
    "info_value": 1.0,
    "price_low": 10.0,
    "price_high": 150.0,
    "action_levels": 9,
    "action_scale_low": 0.8,
    "action_scale_high": 1.2,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "buffer_size": 50_000,
    "batch_size": 256,
    "tau": 0.005,
    "train_freq": 1,
    "learning_starts": 1_000,
    "target_update_interval": 1_000,
    "exploration_fraction": 0.2,
    "exploration_final_eps": 0.05,
    "n_steps": 2_048,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "q_lr": 0.1,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay": 0.9995,
    "model_dir": "engine/models",
    "arch": "small",
    "activation": "relu",
    "q_bins": 6,
    "max_steps": 100,
    "margin_floor": 0.05,
    "margin_floor_patience": 5,
    "use_jax": False,
    "jax_num_envs": 16,
    "jax_num_steps": 128,
    "jax_num_minibatches": 4,
    "jax_update_epochs": 4,
    "jax_anneal_lr": True,
    "checkpoint_interval": 10_000,
}


def _truthy(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _cfg(raw: dict | None = None) -> dict:
    cfg = dict(DEFAULT_CFG)
    if raw:
        cfg.update({k: v for k, v in raw.items() if v is not None})
    cfg["algo"] = str(cfg["algo"]).lower()
    cfg["use_jax"] = _truthy(cfg.get("use_jax")) or _truthy(
        os.environ.get("PHANTOM_USE_JAX")
    )
    return cfg


def _wandb_cfg_dict() -> dict:
    return (
        {k: wandb.config[k] for k in wandb.config.keys()}
        if HAS_WANDB and wandb.run
        else {}
    )


def make_env(cfg: dict):
    from gymnasium.wrappers import FlattenObservation

    from .wrapper import PHANTOM
    from .lib.wrappers import EconomicMetricsWrapper

    env = PHANTOM(
        n_products=int(cfg["n_products"]),
        alpha=float(cfg["alpha"]),
        N=int(cfg["N"]),
        price_bounds=(float(cfg["price_low"]), float(cfg["price_high"])),
        lambda_coi=float(cfg["lambda_coi"]),
        robust_radius=float(cfg["robust_radius"]),
        robust_points=int(cfg["robust_points"]),
        info_value=float(cfg["info_value"]),
        action_levels=int(cfg["action_levels"]),
        action_scale_low=float(cfg["action_scale_low"]),
        action_scale_high=float(cfg["action_scale_high"]),
        max_steps=int(cfg.get("max_steps", 100)),
        margin_floor=float(cfg.get("margin_floor", 0.05)),
        margin_floor_patience=int(cfg.get("margin_floor_patience", 5)),
        render_mode=None,
    )
    env = EconomicMetricsWrapper(env)
    env = FlattenObservation(env)
    return env


def _net_arch(name) -> list[int]:
    presets = {
        "tiny": [32, 32],
        "small": [64, 64],
        "medium": [128, 128],
        "large": [256, 256],
    }
    if isinstance(name, (list, tuple)):
        return [int(v) for v in name]
    s = str(name).lower().strip()
    if s in presets:
        return presets[s]
    if "x" in s:
        try:
            vals = [int(v) for v in s.split("x") if v]
            return vals if vals else presets["small"]
        except ValueError:
            return presets["small"]
    return presets["small"]


def _activation(name):
    try:
        import torch.nn as nn
    except ImportError:
        return None
    return {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }.get(str(name).lower().strip(), nn.ReLU)


def _policy_kwargs(cfg: dict) -> dict:
    kw = {"net_arch": _net_arch(cfg.get("arch", "small"))}
    act = _activation(cfg.get("activation", "relu"))
    if act is not None:
        kw["activation_fn"] = act
    return kw


def _action(agent, obs, deterministic: bool = True):
    out = agent.predict(obs, deterministic=deterministic)
    a = out[0] if isinstance(out, tuple) else out
    if isinstance(a, np.ndarray) and a.size == 1:
        return int(a.reshape(-1)[0])
    return a


def evaluate(agent, env, episodes: int) -> dict:
    rewards, revenues = [], []
    for _ in range(int(episodes)):
        obs, _ = env.reset()
        done, ep_r, ep_rev = False, 0.0, 0.0
        while not done:
            obs, reward, term, trunc, info = env.step(_action(agent, obs, True))
            done = term or trunc
            ep_r += float(reward)
            ep_rev += float(
                info.get("economics", {}).get("revenue", info.get("revenue", 0.0))
            )
        rewards.append(ep_r)
        revenues.append(ep_rev)
    return {
        "eval/reward": float(np.mean(rewards)),
        "eval/revenue": float(np.mean(revenues)),
        "eval/reward_std": float(np.std(rewards)),
        "eval/revenue_std": float(np.std(revenues)),
    }


def build_model(cfg: dict, env):
    algo = cfg["algo"]
    policy_kwargs = _policy_kwargs(cfg)
    if algo == "sac":
        raise ValueError("sac is not supported with the discrete core env")
    if algo == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            seed=int(cfg["seed"]),
            learning_rate=float(cfg["learning_rate"]),
            n_steps=int(cfg["n_steps"]),
            batch_size=int(cfg["batch_size"]),
            n_epochs=int(cfg["n_epochs"]),
            gamma=float(cfg["gamma"]),
            gae_lambda=float(cfg["gae_lambda"]),
            clip_range=float(cfg["clip_range"]),
            ent_coef=float(cfg["ent_coef"]),
        )
    if algo == "a2c":
        return A2C(
            "MlpPolicy",
            env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            seed=int(cfg["seed"]),
            learning_rate=float(cfg["learning_rate"]),
            n_steps=max(5, int(cfg["n_steps"]) // 32),
            gamma=float(cfg["gamma"]),
            gae_lambda=float(cfg["gae_lambda"]),
            ent_coef=float(cfg["ent_coef"]),
        )
    if algo == "dqn":
        return DQN(
            "MlpPolicy",
            env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            seed=int(cfg["seed"]),
            learning_rate=float(cfg["learning_rate"]),
            buffer_size=int(cfg["buffer_size"]),
            batch_size=int(cfg["batch_size"]),
            gamma=float(cfg["gamma"]),
            train_freq=int(cfg["train_freq"]),
            learning_starts=int(cfg["learning_starts"]),
            target_update_interval=int(cfg["target_update_interval"]),
            exploration_fraction=float(cfg["exploration_fraction"]),
            exploration_final_eps=float(cfg["exploration_final_eps"]),
        )
    raise ValueError(f"unsupported algo '{algo}'")


def _sb3_model_cls(algo: str):
    if algo == "ppo":
        return PPO
    if algo == "a2c":
        return A2C
    if algo == "dqn":
        return DQN
    raise ValueError(f"unsupported algo '{algo}'")


def train_qtable(cfg: dict) -> tuple[EventQTable, dict]:
    from .lib.discrete import EventQTable

    np.random.seed(int(cfg["seed"]))
    env = make_env(cfg)
    eval_env = make_env(cfg)
    agent = EventQTable(
        env.action_space.n,
        int(cfg["n_products"]),
        (float(cfg["price_low"]), float(cfg["price_high"])),
        lr=float(cfg["q_lr"]),
        gamma=float(cfg["gamma"]),
        n_bins=int(cfg["q_bins"]),
    )
    eps = float(cfg["eps_start"])
    obs, _ = env.reset(seed=int(cfg["seed"]))
    for t in range(int(cfg["total_timesteps"])):
        a, s = agent.act(obs, eps)
        nxt, reward, term, trunc, info = env.step(a)
        done = term or trunc
        agent.update(s, a, float(reward), agent.encode(nxt), done)
        eps = max(float(cfg["eps_end"]), eps * float(cfg["eps_decay"]))
        if HAS_WANDB and wandb.run and (t + 1) % int(cfg["log_freq"]) == 0:
            econ = info.get("economics", {})
            wandb.log(
                {
                    "train/reward": float(reward),
                    "train/revenue": float(econ.get("revenue", 0.0)),
                    "train/epsilon": float(eps),
                },
                step=t + 1,
            )
        obs = env.reset()[0] if done else nxt
    metrics = evaluate(agent, eval_env, int(cfg["eval_episodes"]))
    metrics["train/global_step"] = int(cfg["total_timesteps"])
    env.close()
    eval_env.close()
    return agent, metrics


def train_sb3(cfg: dict) -> tuple[object, dict]:
    if not HAS_SB3:
        raise ImportError("stable-baselines3 is required for SB3 models")
    from .lib.callbacks import CheckpointArtifactCallback, MetricsCallback

    env = make_env(cfg)
    eval_env = make_env(cfg)
    env = Monitor(env)
    eval_env = Monitor(eval_env)
    model = build_model(cfg, env)
    resume_step = 0
    if HAS_WANDB and wandb.run is not None:
        sweep_id = getattr(wandb.run, "sweep_id", None)
        artifact_name = checkpoint_artifact_name(cfg, backend="sb3", sweep_id=sweep_id)
        checkpoint_file = f"phantom_{cfg['algo']}_checkpoint.zip"
        restored = download_latest_checkpoint(artifact_name, file_name=checkpoint_file)
        if restored is not None:
            checkpoint_path, metadata = restored
            model = _sb3_model_cls(cfg["algo"]).load(
                checkpoint_path.as_posix(), env=env
            )
            resume_step = int(metadata.get("step", getattr(model, "num_timesteps", 0)))
            model.num_timesteps = max(
                int(getattr(model, "num_timesteps", 0)), resume_step
            )

    cbs = [MetricsCallback(log_histograms=True, log_freq=int(cfg["log_freq"]))]
    cbs.append(
        CheckpointArtifactCallback(
            cfg,
            interval=int(cfg.get("checkpoint_interval", 10_000)),
        )
    )
    cbs.append(
        EvalCallback(
            eval_env,
            eval_freq=int(cfg["eval_freq"]),
            n_eval_episodes=int(cfg["eval_episodes"]),
            deterministic=True,
            verbose=0,
        )
    )
    target_steps = int(cfg["total_timesteps"])
    remaining_steps = max(0, target_steps - int(getattr(model, "num_timesteps", 0)))
    if remaining_steps > 0:
        model.learn(
            total_timesteps=remaining_steps,
            callback=cbs,
            reset_num_timesteps=False,
        )

    model_path = Path(cfg["model_dir"])
    model_path.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path / f"phantom_{cfg['algo']}"))
    metrics = evaluate(model, eval_env, int(cfg["eval_episodes"]))
    metrics["train/global_step"] = int(model.num_timesteps)
    env.close()
    eval_env.close()
    return model, metrics


def train_once(cfg: dict) -> dict:
    algo = cfg["algo"]
    if cfg.get("use_jax"):
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX backend requested but JAX is not installed. "
                "Install engine/jax/requirements.txt and jax[tpu] for TPU runs."
            )
        if algo == "qtable":
            raise ValueError("qtable is not supported in JAX backend")
        try:
            from .jax.train import train_jax
        except Exception as exc:  # pragma: no cover
            raise ImportError(f"Failed to import JAX trainer: {exc}") from exc
        _, metrics = train_jax(cfg)
    elif algo == "qtable":
        _, metrics = train_qtable(cfg)
    else:
        _, metrics = train_sb3(cfg)
    metrics["sweep/score"] = float(
        metrics["eval/reward"] + float(cfg["revenue_weight"]) * metrics["eval/revenue"]
    )
    return metrics


def run_wandb(
    project: str, overrides: dict, mode: str = "online", sweep_mode: bool = False
) -> dict:
    if not HAS_WANDB:
        raise ImportError("wandb is required for sweep runs")
    init_kwargs = {"mode": mode}
    if sweep_mode:
        run = wandb.init(**init_kwargs)
        cfg = _cfg(_wandb_cfg_dict())
        for k, v in overrides.items():
            if k not in wandb.config:
                cfg[k] = v
    else:
        run = wandb.init(project=project, config=overrides, **init_kwargs)
        cfg = _cfg(_wandb_cfg_dict())
    metrics = train_once(cfg)
    step = int(metrics.get("train/global_step", cfg["total_timesteps"]))
    wandb.log(metrics, step=step)
    for k, v in metrics.items():
        run.summary[k] = v
    wandb.finish()
    return metrics


def run_local(overrides: dict) -> dict:
    cfg = _cfg(overrides)
    metrics = train_once(cfg)
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    p = argparse.ArgumentParser(description="PHANTOM training and W&B sweeps")
    p.add_argument("--project", default=DEFAULT_CFG["project"])
    p.add_argument("--algo", choices=["ppo", "a2c", "dqn", "qtable"])
    p.add_argument("--total-timesteps", type=int)
    p.add_argument("--alpha", type=float)
    p.add_argument("--n-products", type=int)
    p.add_argument("--lambda-coi", type=float)
    p.add_argument("--robust-radius", type=float)
    p.add_argument("--robust-points", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--gamma", type=float)
    p.add_argument("--revenue-weight", type=float)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--margin-floor", type=float)
    p.add_argument("--margin-floor-patience", type=int)
    p.add_argument("--arch", type=str)
    p.add_argument("--activation", type=str)
    p.add_argument("--jax", action="store_true")
    p.add_argument("--jax-num-envs", type=int)
    p.add_argument("--jax-num-steps", type=int)
    p.add_argument("--jax-num-minibatches", type=int)
    p.add_argument("--jax-update-epochs", type=int)
    p.add_argument("--jax-anneal-lr", type=str)
    p.add_argument("--checkpoint-interval", type=int)
    p.add_argument("--sweep-agent", action="store_true")
    p.add_argument("--sweep-id", type=str)
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    overrides = {
        "algo": args.algo,
        "total_timesteps": args.total_timesteps,
        "alpha": args.alpha,
        "n_products": args.n_products,
        "lambda_coi": args.lambda_coi,
        "robust_radius": args.robust_radius,
        "robust_points": args.robust_points,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "revenue_weight": args.revenue_weight,
        "max_steps": args.max_steps,
        "margin_floor": args.margin_floor,
        "margin_floor_patience": args.margin_floor_patience,
        "arch": args.arch,
        "activation": args.activation,
        "use_jax": args.jax,
        "jax_num_envs": args.jax_num_envs,
        "jax_num_steps": args.jax_num_steps,
        "jax_num_minibatches": args.jax_num_minibatches,
        "jax_update_epochs": args.jax_update_epochs,
        "checkpoint_interval": args.checkpoint_interval,
        "jax_anneal_lr": _truthy(args.jax_anneal_lr)
        if args.jax_anneal_lr is not None
        else None,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    if args.sweep_agent:
        if args.no_wandb:
            raise ValueError("sweep agent requires wandb")
        if not args.sweep_id:
            raise ValueError("--sweep-id is required with --sweep-agent")
        mode = "offline" if args.offline else "online"
        wandb.agent(
            args.sweep_id,
            function=lambda: run_wandb(
                args.project, overrides, mode=mode, sweep_mode=True
            ),
            count=args.count if args.count > 0 else None,
        )
        return

    if args.no_wandb or not HAS_WANDB:
        run_local(overrides)
        return

    run_wandb(args.project, overrides, mode="offline" if args.offline else "online")


if __name__ == "__main__":
    main()
