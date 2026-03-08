from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "estimate_demand": (".demand", "estimate_demand"),
    "estimate_weighted_demand": (".demand", "estimate_weighted_demand"),
    "generate_demand_for_actor": (".demand", "generate_demand_for_actor"),
    "sample_behavior": (".behavior", "sample_behavior"),
    "get_transition_models": (".behavior", "get_transition_models"),
    "trajectory_to_events": (".behavior", "trajectory_to_events"),
    "DashboardRenderer": (".render", "DashboardRenderer"),
    "style_axis": (".render", "style_axis"),
    "EconomicMetricsWrapper": (".wrappers", "EconomicMetricsWrapper"),
    "MetricsCallback": (".callbacks", "MetricsCallback"),
    "EvalMetricsCallback": (".callbacks", "EvalMetricsCallback"),
    "ProviderBenchmark": (".providers", "ProviderBenchmark"),
    "ProviderResult": (".providers", "ProviderResult"),
    "BenchmarkConfig": (".providers", "BenchmarkConfig"),
    "RandomBaseline": (".providers", "RandomBaseline"),
    "SurgeBaseline": (".providers", "SurgeBaseline"),
    "compute_uplift_coi": (".coi", "compute_uplift_coi"),
    "extract_purchases": (".coi", "extract_purchases"),
    "compute_agent_probability": (".coi", "compute_agent_probability"),
    "EventQTable": (".discrete", "EventQTable"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
