from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from ..lib.callbacks import EvalMetricsCallback, MetricsCallback
from ..wandb_checkpoint import checkpoint_artifact_name, log_checkpoint_file
from .common import evaluate, make_env


def _net_arch(name: Any) -> list[int]:
    presets = {
        "tiny": [32, 32],
        "small": [64, 64],
        "medium": [128, 128],
        "large": [256, 256],
    }
    if isinstance(name, (list, tuple)):
        return [int(v) for v in name]
    raw = str(name).lower().strip()
    if raw in presets:
        return presets[raw]
    if "x" in raw:
        try:
            parsed = [int(v) for v in raw.split("x") if v]
            return parsed if parsed else presets["small"]
        except ValueError:
            return presets["small"]
    return presets["small"]


def _activation(name: Any):
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


def _policy_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"net_arch": _net_arch(cfg.get("arch", "small"))}
    activation = _activation(cfg.get("activation", "relu"))
    if activation is not None:
        kwargs["activation_fn"] = activation
    return kwargs


def build_model(cfg: Mapping[str, Any], env: Any):
    try:
        from stable_baselines3 import A2C, DQN, PPO
    except ImportError as exc:
        raise ImportError("stable-baselines3 is required for SB3 algorithms") from exc

    algo = str(cfg["algo"])
    policy_kwargs = _policy_kwargs(cfg)
    device = str(cfg.get("device", "auto"))
    seed = int(cfg["seed"])

    if algo == "sac":
        raise ValueError("sac is not supported with the discrete core env")
    if algo == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            policy_kwargs=policy_kwargs,
            seed=seed,
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
            device=device,
            policy_kwargs=policy_kwargs,
            seed=seed,
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
            device=device,
            policy_kwargs=policy_kwargs,
            seed=seed,
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


def train_sb3(cfg: Mapping[str, Any]) -> tuple[object, dict[str, Any]]:
    try:
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise ImportError("stable-baselines3 is required for SB3 models") from exc

    env = Monitor(make_env(cfg))
    eval_env = Monitor(make_env(cfg))
    model = build_model(cfg, env)

    try:
        import torch

        print(
            "PHANTOM_DEVICE: "
            + json.dumps(
                {
                    "requested": str(cfg.get("device", "auto")),
                    "torch_cuda_available": bool(torch.cuda.is_available()),
                    "torch_device_count": int(torch.cuda.device_count()),
                    "sb3_device": str(getattr(model, "device", "unknown")),
                }
            )
        )
    except Exception:
        pass

    metrics_callback = MetricsCallback(
        log_histograms=True,
        log_freq=int(cfg["log_freq"]),
        hist_freq=int(cfg.get("hist_freq", 500)),
        step_offset=int(cfg.get("wandb_step_offset", 0)),
    )
    eval_callback = EvalMetricsCallback(
        eval_env,
        eval_freq=int(cfg["eval_freq"]),
        n_eval_episodes=int(cfg["eval_episodes"]),
        step_offset=int(cfg.get("wandb_step_offset", 0)),
        deterministic=True,
        verbose=0,
    )
    callbacks = [metrics_callback, eval_callback]

    target_steps = int(cfg["total_timesteps"])
    remaining_steps = max(0, target_steps - int(getattr(model, "num_timesteps", 0)))
    if remaining_steps > 0:
        model.learn(
            total_timesteps=remaining_steps,
            callback=callbacks,
            reset_num_timesteps=False,
        )

    model_dir = Path(str(cfg["model_dir"]))
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"phantom_{cfg['algo']}"
    model.save(str(model_path))

    artifact_name = checkpoint_artifact_name(
        cfg,
        backend="sb3",
        sweep_id=os.getenv("WANDB_SWEEP_ID"),
    )
    artifact_logged = False
    try:
        artifact_logged = bool(
            log_checkpoint_file(
                artifact_name,
                file_path=model_path.with_suffix(".zip"),
                artifact_file_name="model.zip",
                metadata={
                    "algo": str(cfg.get("algo", "ppo")),
                    "backend": "sb3",
                    "seed": int(cfg.get("seed", 0)),
                    "step": int(getattr(model, "num_timesteps", 0)),
                },
            )
        )
    except Exception:
        artifact_logged = False

    metrics: dict[str, Any] = evaluate(
        model,
        eval_env,
        int(cfg["eval_episodes"]),
        cfg=cfg,
    )
    metrics["train/global_step"] = int(model.num_timesteps)
    metrics["model/path"] = str(model_path.with_suffix(".zip"))
    metrics["model/artifact_name"] = str(artifact_name)
    metrics["model/artifact_logged"] = float(artifact_logged)
    metrics["_train_events"] = sorted(
        [*metrics_callback.events, *eval_callback.events],
        key=lambda event: int(event.get("train/global_step", 0)),
    )

    env.close()
    eval_env.close()
    return model, metrics
