from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping


def get_wandb_module():
    try:
        import wandb

        return wandb
    except ImportError:
        return None


def _require_wandb():
    wandb = get_wandb_module()
    if wandb is None:
        raise ImportError("wandb is required for this workflow")
    return wandb


def init_run(
    *,
    mode: str,
    project: str | None = None,
    config: Mapping[str, Any] | None = None,
    name: str | None = None,
    tags: Iterable[str] | None = None,
    group: str | None = None,
    sweep_mode: bool = False,
):
    wandb = _require_wandb()
    kwargs: dict[str, Any] = {"mode": mode}
    if group:
        kwargs["group"] = group
    if sweep_mode:
        run = wandb.init(**kwargs)
        if name and run is not None:
            run.name = name
        return run

    init_kwargs = dict(kwargs)
    init_kwargs["project"] = project
    if config is not None:
        init_kwargs["config"] = dict(config)
    if name:
        init_kwargs["name"] = name
    if tags:
        init_kwargs["tags"] = list(tags)
    return wandb.init(**init_kwargs)


def finish_run() -> None:
    wandb = get_wandb_module()
    if wandb is not None and wandb.run is not None:
        wandb.finish()


def current_config() -> dict[str, Any]:
    wandb = get_wandb_module()
    if wandb is None or wandb.run is None:
        return {}
    return {key: wandb.config[key] for key in wandb.config.keys()}


def update_run_config(config: Mapping[str, Any]) -> None:
    wandb = get_wandb_module()
    if wandb is None or wandb.run is None:
        return
    try:
        wandb.config.update(dict(config), allow_val_change=True)
    except TypeError:
        wandb.config.update(dict(config))


def log_metrics(metrics: Mapping[str, Any], *, step: int) -> None:
    wandb = get_wandb_module()
    if wandb is None or wandb.run is None:
        return
    wandb.log(dict(metrics), step=step)


def update_summary(metrics: Mapping[str, Any]) -> None:
    wandb = get_wandb_module()
    if wandb is None or wandb.run is None:
        return
    for key, value in metrics.items():
        wandb.run.summary[key] = value


def run_agent(
    sweep_id: str,
    fn: Callable[[], None],
    *,
    count: int | None = None,
) -> None:
    wandb = _require_wandb()
    wandb.agent(sweep_id, function=fn, count=count)
