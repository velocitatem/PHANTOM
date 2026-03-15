from __future__ import annotations

import os
import time
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


def _warn(message: str) -> None:
    print(f"PHANTOM_WANDB_WARNING: {message}")


def _sanitize_key(raw_key: str) -> str | None:
    key = str(raw_key)
    replacements = {
        "no_robust": "baseline_mode",
        "study/no_robust": "study/baseline_mode",
        "study/robust_radius": "study/ambiguity_radius",
        "robust_radius": "ambiguity_radius",
        "robust_points": "ambiguity_points",
        "robust_rollouts": "ambiguity_rollouts",
        "robust_eval_enabled": "stress_eval_enabled",
        "eval/robust_alpha_high": "eval/stress_alpha_high",
        "eval/robust_alpha_low": "eval/stress_alpha_low",
        "eval/robust_reward_worst": "eval/stress_reward_worst",
        "eval/robust_revenue_worst": "eval/stress_revenue_worst",
        "eval/robust_coi_leakage_worst": "eval/stress_coi_leakage_worst",
    }
    key = replacements.get(key, key)
    if "robust" in key.lower():
        return None
    return key


def _sanitize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in payload.items():
        clean_key = _sanitize_key(str(key))
        if clean_key is None:
            continue
        sanitized[clean_key] = value
    return sanitized


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
        try:
            run = wandb.init(**kwargs)
        except Exception as exc:
            _warn(f"init failed in sweep mode ({exc})")
            return None
        if name and run is not None:
            run.name = name
        return run

    init_kwargs = dict(kwargs)
    init_kwargs["project"] = project
    if config is not None:
        init_kwargs["config"] = _sanitize_payload(dict(config))
    if name:
        init_kwargs["name"] = name
    if tags:
        init_kwargs["tags"] = list(tags)
    try:
        return wandb.init(**init_kwargs)
    except Exception as exc:
        _warn(f"init failed ({exc})")
        return None


def finish_run() -> None:
    wandb = get_wandb_module()
    if wandb is not None and wandb.run is not None:
        try:
            wandb.finish()
        except Exception as exc:
            _warn(f"finish failed ({exc})")


def current_config() -> dict[str, Any]:
    wandb = get_wandb_module()
    if wandb is None or wandb.run is None:
        return {}
    return {key: wandb.config[key] for key in wandb.config.keys()}


def update_run_config(config: Mapping[str, Any]) -> None:
    wandb = get_wandb_module()
    if wandb is None or wandb.run is None:
        return
    payload = _sanitize_payload(dict(config))
    if not payload:
        return
    try:
        wandb.config.update(payload, allow_val_change=True)
    except TypeError:
        try:
            wandb.config.update(payload)
        except Exception as exc:
            _warn(f"config update failed ({exc})")
    except Exception as exc:
        _warn(f"config update failed ({exc})")


def log_metrics(metrics: Mapping[str, Any], *, step: int) -> None:
    wandb = get_wandb_module()
    if wandb is None or wandb.run is None:
        return
    payload = _sanitize_payload(dict(metrics))
    if not payload:
        return
    try:
        wandb.log(payload, step=step)
    except Exception as exc:
        _warn(f"log failed at step {step} ({exc})")


def update_summary(metrics: Mapping[str, Any]) -> None:
    wandb = get_wandb_module()
    if wandb is None or wandb.run is None:
        return
    payload = _sanitize_payload(dict(metrics))
    if not payload:
        return
    try:
        for key, value in payload.items():
            wandb.run.summary[key] = value
    except Exception as exc:
        _warn(f"summary update failed ({exc})")


def run_agent(
    sweep_id: str,
    fn: Callable[[], None],
    *,
    count: int | None = None,
) -> None:
    wandb = _require_wandb()
    retry_max = max(0, int(os.getenv("PHANTOM_WANDB_AGENT_RETRIES", "8")))
    retry_delay = max(1.0, float(os.getenv("PHANTOM_WANDB_AGENT_RETRY_DELAY", "5")))
    retry_backoff = max(
        1.0, float(os.getenv("PHANTOM_WANDB_AGENT_RETRY_BACKOFF", "1.5"))
    )
    retry_max_delay = max(
        retry_delay,
        float(os.getenv("PHANTOM_WANDB_AGENT_MAX_RETRY_DELAY", "60")),
    )

    target = None if count is None else max(0, int(count))
    completed = 0

    def _wrapped() -> None:
        nonlocal completed
        fn()
        completed += 1

    attempt = 0
    while True:
        remaining = None if target is None else max(0, int(target - completed))
        if target is not None and remaining == 0:
            return
        try:
            wandb.agent(sweep_id, function=_wrapped, count=remaining)
            return
        except Exception as exc:
            attempt += 1
            if attempt > retry_max:
                raise
            wait = min(retry_max_delay, retry_delay * (retry_backoff ** (attempt - 1)))
            _warn(
                f"agent disconnected (attempt {attempt}/{retry_max}, "
                f"completed={completed}, remaining={remaining}): {exc}"
            )
            time.sleep(wait)
