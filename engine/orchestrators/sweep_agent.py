from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..spec import TrainSpec, run_name
from ..telemetry.wandb import (
    current_config,
    finish_run,
    get_wandb_module,
    init_run,
    run_agent,
)
from .train import run_with_active_sweep_run


def run_sweep_agent(
    *,
    project: str,
    sweep_id: str,
    count: int,
    offline: bool,
    no_wandb: bool,
    base_overrides: Mapping[str, Any],
    kind: str,
    scenario: str,
    group: str | None,
    extra_tags: Sequence[str],
) -> None:
    if no_wandb:
        raise ValueError("sweep agent requires wandb")
    if not sweep_id:
        raise ValueError("--sweep-id is required with --sweep-agent")
    if get_wandb_module() is None:
        raise ImportError("wandb is required for sweep runs")

    mode = "offline" if offline else "online"

    def _sweep_trial() -> None:
        run = init_run(mode=mode, project=project, group=group, sweep_mode=True)
        try:
            merged = dict(base_overrides)
            merged.update(current_config())
            spec = TrainSpec.from_flat(merged)
            if run is not None:
                run.name = run_name(spec, kind=kind, scenario=scenario)
            run_with_active_sweep_run(
                spec,
                kind=kind,
                scenario=scenario,
                group=group,
                extra_tags=extra_tags,
            )
        finally:
            finish_run()

    run_agent(
        sweep_id,
        _sweep_trial,
        count=count if count > 0 else None,
    )
