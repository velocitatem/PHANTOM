from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Mapping, Sequence


def _truthy(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_keys(raw: Mapping[str, Any]) -> dict[str, Any]:
    alias_map = {
        "algorithm": "algo",
        "algorithm.name": "algo",
        "env.n_products": "n_products",
        "env.action_levels": "action_levels",
        "env.action_scale_low": "action_scale_low",
        "env.action_scale_high": "action_scale_high",
        "env.price_low": "price_low",
        "env.price_high": "price_high",
        "env.max_steps": "max_steps",
        "env.margin_floor": "margin_floor",
        "env.margin_floor_patience": "margin_floor_patience",
        "env.n_sessions": "N",
        "study.alpha": "alpha",
        "study.lambda_coi": "lambda_coi",
        "study.robust_radius": "robust_radius",
        "study.robust_points": "robust_points",
        "study.info_value": "info_value",
        "study.revenue_weight": "revenue_weight",
        "optimizer.learning_rate": "learning_rate",
        "optimizer.gamma": "gamma",
        "optimizer.batch_size": "batch_size",
        "optimizer.n_steps": "n_steps",
        "runtime.backend": "backend",
        "runtime.device": "device",
        "runtime.seed": "seed",
        "runtime.total_timesteps": "total_timesteps",
        "runtime.checkpoint_interval": "checkpoint_interval",
        "eval.eval_freq": "eval_freq",
        "eval.eval_episodes": "eval_episodes",
    }
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        canonical = alias_map.get(str(key), str(key))
        normalized[canonical] = value
    return normalized


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str = "ppo"


@dataclass(frozen=True)
class EnvSpec:
    n_products: int = 10
    n_sessions: int = 100
    price_low: float = 10.0
    price_high: float = 150.0
    action_levels: int = 9
    action_scale_low: float = 0.8
    action_scale_high: float = 1.2
    max_steps: int = 100
    margin_floor: float = 0.05
    margin_floor_patience: int = 5


@dataclass(frozen=True)
class StudySpec:
    alpha: float = 0.3
    lambda_coi: float = 0.2
    robust_radius: float = 0.15
    robust_points: int = 5
    info_value: float = 1.0
    revenue_weight: float = 0.01
    no_robust: bool = False


@dataclass(frozen=True)
class OptimizerSpec:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    buffer_size: int = 50_000
    batch_size: int = 256
    tau: float = 0.005
    train_freq: int = 1
    learning_starts: int = 1_000
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.2
    exploration_final_eps: float = 0.05
    n_steps: int = 2_048
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    q_lr: float = 0.1
    q_bins: int = 6
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.9995
    arch: str = "small"
    activation: str = "relu"
    jax_num_envs: int = 16
    jax_num_steps: int = 128
    jax_num_minibatches: int = 4
    jax_update_epochs: int = 4
    jax_anneal_lr: bool = True
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class RuntimeSpec:
    project: str = "capstone"
    backend: str = "sb3"
    device: str = "auto"
    seed: int = 42
    total_timesteps: int = 50_000
    checkpoint_interval: int = 200_000
    model_dir: str = "engine/models"
    log_freq: int = 100
    use_jax: bool = False


@dataclass(frozen=True)
class EvalSpec:
    eval_freq: int = 1_000
    eval_episodes: int = 5
    robust_eval_enabled: bool = True


@dataclass(frozen=True)
class TrainSpec:
    algorithm: AlgorithmSpec = field(default_factory=AlgorithmSpec)
    env: EnvSpec = field(default_factory=EnvSpec)
    study: StudySpec = field(default_factory=StudySpec)
    optimizer: OptimizerSpec = field(default_factory=OptimizerSpec)
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)
    eval: EvalSpec = field(default_factory=EvalSpec)

    def to_flat_dict(self) -> dict[str, Any]:
        return {
            "project": self.runtime.project,
            "algo": self.algorithm.name,
            "seed": self.runtime.seed,
            "total_timesteps": self.runtime.total_timesteps,
            "eval_episodes": self.eval.eval_episodes,
            "eval_freq": self.eval.eval_freq,
            "log_freq": self.runtime.log_freq,
            "model_dir": self.runtime.model_dir,
            "backend": self.runtime.backend,
            "device": self.runtime.device,
            "use_jax": self.runtime.use_jax,
            "checkpoint_interval": self.runtime.checkpoint_interval,
            "n_products": self.env.n_products,
            "N": self.env.n_sessions,
            "price_low": self.env.price_low,
            "price_high": self.env.price_high,
            "action_levels": self.env.action_levels,
            "action_scale_low": self.env.action_scale_low,
            "action_scale_high": self.env.action_scale_high,
            "max_steps": self.env.max_steps,
            "margin_floor": self.env.margin_floor,
            "margin_floor_patience": self.env.margin_floor_patience,
            "alpha": self.study.alpha,
            "lambda_coi": self.study.lambda_coi,
            "robust_radius": self.study.robust_radius,
            "robust_points": self.study.robust_points,
            "info_value": self.study.info_value,
            "revenue_weight": self.study.revenue_weight,
            "no_robust": self.study.no_robust,
            "learning_rate": self.optimizer.learning_rate,
            "gamma": self.optimizer.gamma,
            "buffer_size": self.optimizer.buffer_size,
            "batch_size": self.optimizer.batch_size,
            "tau": self.optimizer.tau,
            "train_freq": self.optimizer.train_freq,
            "learning_starts": self.optimizer.learning_starts,
            "target_update_interval": self.optimizer.target_update_interval,
            "exploration_fraction": self.optimizer.exploration_fraction,
            "exploration_final_eps": self.optimizer.exploration_final_eps,
            "n_steps": self.optimizer.n_steps,
            "n_epochs": self.optimizer.n_epochs,
            "gae_lambda": self.optimizer.gae_lambda,
            "clip_range": self.optimizer.clip_range,
            "ent_coef": self.optimizer.ent_coef,
            "q_lr": self.optimizer.q_lr,
            "q_bins": self.optimizer.q_bins,
            "eps_start": self.optimizer.eps_start,
            "eps_end": self.optimizer.eps_end,
            "eps_decay": self.optimizer.eps_decay,
            "arch": self.optimizer.arch,
            "activation": self.optimizer.activation,
            "jax_num_envs": self.optimizer.jax_num_envs,
            "jax_num_steps": self.optimizer.jax_num_steps,
            "jax_num_minibatches": self.optimizer.jax_num_minibatches,
            "jax_update_epochs": self.optimizer.jax_update_epochs,
            "jax_anneal_lr": self.optimizer.jax_anneal_lr,
            "vf_coef": self.optimizer.vf_coef,
            "max_grad_norm": self.optimizer.max_grad_norm,
            "robust_eval_enabled": self.eval.robust_eval_enabled,
        }

    @classmethod
    def from_flat(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        env_vars: Mapping[str, str] | None = None,
    ) -> "TrainSpec":
        base = cls().to_flat_dict()
        incoming = _normalize_keys(raw or {})
        base.update({k: v for k, v in incoming.items() if v is not None})

        runtime_env = os.environ if env_vars is None else env_vars
        base["device"] = str(
            base.get("device", runtime_env.get("PHANTOM_DEVICE", "auto"))
        )

        requested_jax = _truthy(base.get("use_jax")) or _truthy(
            runtime_env.get("PHANTOM_USE_JAX")
        )
        backend = str(base.get("backend", "jax" if requested_jax else "sb3")).lower()
        if backend == "auto":
            backend = "jax" if requested_jax else "sb3"
        if backend == "jax":
            requested_jax = True

        no_robust = _truthy(base.get("no_robust"))
        if no_robust:
            base["lambda_coi"] = 0.0
            base["robust_radius"] = 0.0
            base["robust_points"] = 1

        return cls(
            algorithm=AlgorithmSpec(name=str(base["algo"]).lower().strip()),
            env=EnvSpec(
                n_products=int(base["n_products"]),
                n_sessions=int(base["N"]),
                price_low=float(base["price_low"]),
                price_high=float(base["price_high"]),
                action_levels=int(base["action_levels"]),
                action_scale_low=float(base["action_scale_low"]),
                action_scale_high=float(base["action_scale_high"]),
                max_steps=int(base["max_steps"]),
                margin_floor=float(base["margin_floor"]),
                margin_floor_patience=int(base["margin_floor_patience"]),
            ),
            study=StudySpec(
                alpha=float(base["alpha"]),
                lambda_coi=float(base["lambda_coi"]),
                robust_radius=float(base["robust_radius"]),
                robust_points=int(base["robust_points"]),
                info_value=float(base["info_value"]),
                revenue_weight=float(base["revenue_weight"]),
                no_robust=no_robust,
            ),
            optimizer=OptimizerSpec(
                learning_rate=float(base["learning_rate"]),
                gamma=float(base["gamma"]),
                buffer_size=int(base["buffer_size"]),
                batch_size=int(base["batch_size"]),
                tau=float(base["tau"]),
                train_freq=int(base["train_freq"]),
                learning_starts=int(base["learning_starts"]),
                target_update_interval=int(base["target_update_interval"]),
                exploration_fraction=float(base["exploration_fraction"]),
                exploration_final_eps=float(base["exploration_final_eps"]),
                n_steps=int(base["n_steps"]),
                n_epochs=int(base["n_epochs"]),
                gae_lambda=float(base["gae_lambda"]),
                clip_range=float(base["clip_range"]),
                ent_coef=float(base["ent_coef"]),
                q_lr=float(base["q_lr"]),
                q_bins=int(base["q_bins"]),
                eps_start=float(base["eps_start"]),
                eps_end=float(base["eps_end"]),
                eps_decay=float(base["eps_decay"]),
                arch=str(base["arch"]),
                activation=str(base["activation"]),
                jax_num_envs=int(base["jax_num_envs"]),
                jax_num_steps=int(base["jax_num_steps"]),
                jax_num_minibatches=int(base["jax_num_minibatches"]),
                jax_update_epochs=int(base["jax_update_epochs"]),
                jax_anneal_lr=_truthy(base.get("jax_anneal_lr")),
                vf_coef=float(base["vf_coef"]),
                max_grad_norm=float(base["max_grad_norm"]),
            ),
            runtime=RuntimeSpec(
                project=str(base["project"]),
                backend=backend,
                device=str(base["device"]),
                seed=int(base["seed"]),
                total_timesteps=int(base["total_timesteps"]),
                checkpoint_interval=int(base["checkpoint_interval"]),
                model_dir=str(base["model_dir"]),
                log_freq=int(base["log_freq"]),
                use_jax=requested_jax,
            ),
            eval=EvalSpec(
                eval_freq=int(base["eval_freq"]),
                eval_episodes=int(base["eval_episodes"]),
                robust_eval_enabled=_truthy(base.get("robust_eval_enabled", True)),
            ),
        )


def run_name(spec: TrainSpec, *, kind: str, scenario: str) -> str:
    return (
        f"{kind}/{spec.algorithm.name}/{spec.runtime.backend}/"
        f"{spec.runtime.device}/{scenario}/s{spec.runtime.seed}"
    )


def run_metadata(
    spec: TrainSpec,
    *,
    kind: str,
    scenario: str,
    group: str | None = None,
    tags: Sequence[str] = (),
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "run.kind": str(kind),
        "run.algo": spec.algorithm.name,
        "run.backend": spec.runtime.backend,
        "run.device": spec.runtime.device,
        "run.scenario": str(scenario),
        "run.seed": spec.runtime.seed,
        "run.tags": list(tags),
    }
    if group:
        metadata["run.group"] = group
    return metadata
