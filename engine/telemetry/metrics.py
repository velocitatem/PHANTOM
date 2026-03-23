from __future__ import annotations

from typing import Any, Mapping

from ..spec import TrainSpec


_ALIASES = {
    "train/reward": "train/reward_mean",
    "train/revenue": "train/revenue_mean",
    "train/dqn_loss": "train/loss",
    "eval/reward": "eval/reward_mean",
    "eval/revenue": "eval/revenue_mean",
    "train/steps_per_second": "runtime/steps_per_second",
}


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def canonicalize_metrics(raw: Mapping[str, Any], spec: TrainSpec) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, value in raw.items():
        canonical = _ALIASES.get(str(key), str(key))
        if canonical in metrics and canonical != key:
            continue
        metrics[canonical] = value

    metrics.setdefault("train/global_step", spec.runtime.total_timesteps)

    eval_reward = (
        _as_float(
            metrics.get(
                "eval/stress_reward_worst",
                metrics.get(
                    "eval/robust_reward_worst", metrics.get("eval/reward_mean")
                ),
            ),
            0.0,
        )
        or 0.0
    )
    metrics["objective/score"] = eval_reward

    margin_mean = _as_float(metrics.get("eval/margin_mean"), None)
    if margin_mean is not None:
        metrics["objective/constraint_margin"] = margin_mean - spec.env.margin_floor

    coi_level = _as_float(metrics.get("eval/coi_level_mean"), None)
    metrics["objective/coi_preserved"] = 0.0 if coi_level is None else coi_level

    metrics["study/alpha"] = spec.study.alpha
    metrics["study/mode"] = "baseline" if bool(spec.study.no_robust) else "defended"
    metrics["study/baseline_mode"] = float(bool(spec.study.no_robust))
    metrics["study/lambda_coi"] = spec.study.lambda_coi
    metrics["study/ambiguity_radius"] = spec.study.robust_radius
    metrics["study/info_value"] = spec.study.info_value
    metrics["tiers"] = spec.algorithm.name

    metrics["runtime/backend"] = spec.runtime.backend
    metrics["runtime/device"] = spec.runtime.device
    metrics["runtime/seed"] = spec.runtime.seed

    return metrics
