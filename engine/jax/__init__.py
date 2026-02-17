"""JAX-compatible training and environment modules for PHANTOM."""

from __future__ import annotations

try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

__all__ = ["JAX_AVAILABLE"]
