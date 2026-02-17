"""Orbax checkpoint helpers for JAX training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import orbax.checkpoint as ocp

    HAS_ORBAX = True
except ImportError:
    HAS_ORBAX = False


def _require_orbax() -> None:
    if not HAS_ORBAX:
        raise ImportError(
            "orbax-checkpoint is required for checkpoint support. "
            "Install engine/jax/requirements.txt first."
        )


def create_manager(directory: str | Path, max_to_keep: int = 5):
    _require_orbax()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max(1, int(max_to_keep)), create=True
    )
    return ocp.CheckpointManager(root.as_posix(), ocp.PyTreeCheckpointer(), options)


def save(manager, *, step: int, payload: Any) -> bool:
    _require_orbax()
    return bool(manager.save(int(step), payload))


def latest_step(manager) -> int | None:
    _require_orbax()
    return manager.latest_step()


def restore(manager, *, target: Any, step: int | None = None) -> Any:
    _require_orbax()
    step_to_restore = manager.latest_step() if step is None else int(step)
    if step_to_restore is None:
        return target
    return manager.restore(step_to_restore, items=target)
