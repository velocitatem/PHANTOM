from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .spec import TrainSpec
from .telemetry.metrics import canonicalize_metrics


@dataclass(frozen=True)
class TrainResult:
    spec: TrainSpec
    metrics: dict[str, Any]
    artifacts: dict[str, str]


def run_train(spec: TrainSpec) -> TrainResult:
    cfg = spec.to_flat_dict()
    algo = spec.algorithm.name

    if spec.runtime.use_jax or spec.runtime.backend == "jax":
        from .backends.jax import train_jax_backend

        _, raw_metrics = train_jax_backend(cfg)
    elif algo == "qtable":
        from .backends.qtable import train_qtable

        _, raw_metrics = train_qtable(cfg)
    else:
        from .backends.sb3 import train_sb3

        _, raw_metrics = train_sb3(cfg)

    metrics = canonicalize_metrics(raw_metrics, spec)
    artifacts: dict[str, str] = {}
    model_path = raw_metrics.get("model/path")
    if isinstance(model_path, str):
        artifacts["model/path"] = model_path

    return TrainResult(spec=spec, metrics=metrics, artifacts=artifacts)
