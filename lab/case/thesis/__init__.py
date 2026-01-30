"""
Thesis-specific implementation of the PHANTOM pricing defense framework.

This module implements the mathematical models from the thesis:
- ContaminatedArrivalModel: Mixture demand Q(p) = (1-α)d_H + αd_A (Eq 3)
- HybridExecutionModel: Divergent H/A behavior with separability (Section 2.1)
- RobustStackelbergObjective: Maximin objective with COI penalty (Eq 23)
- COIMetrics: Cost of Information tracking (Definition 1)

The platform configuration creates a research environment that directly
maps to the thesis mathematical framework for DR-RL experiments.
"""
from .arrivals import ContaminatedArrivalModel, ContaminatedArrivalConfig
from .execution import HybridExecutionModel, HybridExecutionConfig
from .objectives import RobustStackelbergObjective, COIObjective
from .platform import make_thesis_platform, ThesisConfig
from .metrics import COIMetrics, compute_coi, compute_separability

__all__ = [
    'ContaminatedArrivalModel', 'ContaminatedArrivalConfig',
    'HybridExecutionModel', 'HybridExecutionConfig',
    'RobustStackelbergObjective', 'COIObjective',
    'make_thesis_platform', 'ThesisConfig',
    'COIMetrics', 'compute_coi', 'compute_separability',
]
