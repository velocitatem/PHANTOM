"""JAX-accelerated simulation core for PHANTOM environment."""
from .transitions import TransitionData, compile_transitions, fallback_transitions, JAX_AVAILABLE
from .simulation import SessionBatch, SimResult, sample_sessions, compute_metrics
from .features import session_features, compute_session_transitions
from .separability import compute_divergences, estimate_alpha_batch

__all__ = [
    "JAX_AVAILABLE", "TransitionData", "compile_transitions", "fallback_transitions",
    "SessionBatch", "SimResult", "sample_sessions", "compute_metrics",
    "session_features", "compute_session_transitions", "compute_divergences", "estimate_alpha_batch",
]
