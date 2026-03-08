from __future__ import annotations

from typing import Any, Mapping

from ..jax import JAX_AVAILABLE


def train_jax_backend(
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float | int | str]]:
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX backend requested but JAX is not installed. "
            "Install engine/jax/requirements.txt and jax[tpu] for TPU runs."
        )
    from ..jax.train import train_jax

    return train_jax(dict(cfg))
