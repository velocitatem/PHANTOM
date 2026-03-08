from __future__ import annotations

import json
from typing import Any, Sequence

from ..spec import TrainSpec, run_metadata, run_name
from ..telemetry.wandb import (
    finish_run,
    get_wandb_module,
    init_run,
    log_metrics,
    update_run_config,
    update_summary,
)
from ..train_core import run_train


def _tags_for_run(spec: TrainSpec, kind: str, extra_tags: Sequence[str]) -> list[str]:
    tags = [
        kind,
        spec.algorithm.name,
        spec.runtime.backend,
        "vanilla" if spec.study.no_robust else "robust",
    ]
    tags.extend([tag for tag in extra_tags if tag])
    return tags


def _print_local_metrics(metrics: dict[str, Any]) -> None:
    print(json.dumps(metrics, indent=2))
    print("PHANTOM_METRICS:" + json.dumps(metrics))


def _should_print_local(spec: TrainSpec) -> bool:
    if not spec.runtime.use_jax:
        return True
    try:
        import jax

        return int(jax.process_index()) == 0
    except Exception:
        return True


def _is_non_primary_jax_worker(spec: TrainSpec) -> bool:
    if not spec.runtime.use_jax:
        return False
    try:
        import jax

        return int(jax.process_count()) > 1 and int(jax.process_index()) != 0
    except Exception:
        return False


def run_train_once(
    spec: TrainSpec,
    *,
    project: str,
    offline: bool,
    no_wandb: bool,
    kind: str,
    scenario: str,
    group: str | None,
    extra_tags: Sequence[str],
) -> dict[str, Any]:
    wandb = get_wandb_module()
    if no_wandb or wandb is None or _is_non_primary_jax_worker(spec):
        result = run_train(spec)
        if _should_print_local(spec):
            _print_local_metrics(result.metrics)
        return result.metrics

    mode = "offline" if offline else "online"
    tags = _tags_for_run(spec, kind, extra_tags)
    metadata = run_metadata(
        spec,
        kind=kind,
        scenario=scenario,
        group=group,
        tags=tags,
    )
    config = spec.to_flat_dict()
    config.update(metadata)
    name = run_name(spec, kind=kind, scenario=scenario)
    init_run(
        mode=mode,
        project=project,
        config=config,
        name=name,
        tags=tags,
        group=group,
        sweep_mode=False,
    )

    try:
        result = run_train(spec)
        metrics = result.metrics
        step = int(metrics.get("train/global_step", spec.runtime.total_timesteps))
        log_metrics(metrics, step=step)
        update_summary(metrics)
        return metrics
    finally:
        finish_run()


def run_with_active_sweep_run(
    spec: TrainSpec,
    *,
    kind: str,
    scenario: str,
    group: str | None,
    extra_tags: Sequence[str],
) -> dict[str, Any]:
    tags = _tags_for_run(spec, kind, extra_tags)
    metadata = run_metadata(
        spec,
        kind=kind,
        scenario=scenario,
        group=group,
        tags=tags,
    )
    update_run_config({**spec.to_flat_dict(), **metadata})
    result = run_train(spec)
    metrics = result.metrics
    step = int(metrics.get("train/global_step", spec.runtime.total_timesteps))
    log_metrics(metrics, step=step)
    update_summary(metrics)
    return metrics
