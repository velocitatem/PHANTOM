"""Thesis metrics for COI and behavioral analysis using ground truth labels."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import numpy as np
from ...outlet.types import StepLogs, StepMetrics, Quote, InstrumentSet
from ...outlet.math_util import safe_log, EPS


@dataclass
class COIMetrics:
    coi_level: float = 0.0
    coi_leakage: float = 0.0
    realized_premium: float = 0.0
    theoretical_max: float = 0.0
    erosion_rate: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in ['coi_level', 'coi_leakage', 'realized_premium', 'theoretical_max', 'erosion_rate']}


def compute_coi(quote: Quote, instruments: InstrumentSet, metrics: StepMetrics, contamination: float) -> COIMetrics:
    prices, costs, refs = quote.prices, instruments.costs, instruments.refs
    margins = prices - costs
    coi_level = float(np.mean(margins))
    theoretical_max = float(np.mean(costs))
    realized_premium = (metrics.revenue - metrics.cost) / metrics.units_traded if metrics.units_traded > 0 else 0.0
    price_var = float(np.var(prices / refs))
    coi_leakage = contamination * (coi_level + price_var)
    erosion_rate = contamination * coi_level / (theoretical_max + EPS)
    return COIMetrics(coi_level=coi_level, coi_leakage=coi_leakage, realized_premium=realized_premium,
                      theoretical_max=theoretical_max, erosion_rate=erosion_rate)


@dataclass
class SeparabilityMetrics:
    classification_accuracy: float = 0.0
    estimated_alpha: float = 0.0
    n_human_sessions: int = 0
    n_agent_sessions: int = 0


def compute_separability(logs: StepLogs, true_alpha: float) -> SeparabilityMetrics:
    """Compute separability using ground truth labels only."""
    if logs.events is None or len(logs.events) == 0:
        return SeparabilityMetrics(estimated_alpha=true_alpha)

    sessions: Dict[str, bool] = {}
    for evt in logs.events:
        sid = evt.metadata.get('session_id', evt.opportunity_id)
        if sid not in sessions:
            sessions[sid] = evt.metadata.get('is_agent', False)

    n_agent = sum(1 for is_agent in sessions.values() if is_agent)
    n_human = len(sessions) - n_agent
    est_alpha = n_agent / len(sessions) if sessions else 0.0

    return SeparabilityMetrics(
        classification_accuracy=1.0,  # ground truth is always correct
        estimated_alpha=est_alpha,
        n_human_sessions=n_human,
        n_agent_sessions=n_agent)


@dataclass
class RevenueAttribution:
    total_revenue: float = 0.0
    human_revenue: float = 0.0
    agent_revenue: float = 0.0
    human_conversion: float = 0.0
    agent_conversion: float = 0.0


def compute_attribution(logs: StepLogs, metrics: StepMetrics) -> RevenueAttribution:
    if logs.executions is None:
        return RevenueAttribution(total_revenue=metrics.revenue)

    human_rev, agent_rev, human_cnt, agent_cnt = 0.0, 0.0, 0, 0
    for exe in logs.executions:
        if exe.propensity < 0.05:
            agent_rev += exe.price * exe.size_filled
            agent_cnt += 1
        else:
            human_rev += exe.price * exe.size_filled
            human_cnt += 1

    total_exp = logs.aggregates.get('n_arrivals', 1)
    return RevenueAttribution(
        total_revenue=metrics.revenue, human_revenue=human_rev, agent_revenue=agent_rev,
        human_conversion=human_cnt / (total_exp * 0.8 + EPS),
        agent_conversion=agent_cnt / (total_exp * 0.2 + EPS))


def order_statistic_erosion(n_agents: int, price_variance: float) -> float:
    """COI erosion from Theorem 1: as N->inf, min(p_1..p_N)->p_min."""
    if n_agents <= 1:
        return 0.0
    sigma, log_n = np.sqrt(price_variance), safe_log(n_agents)
    if log_n < 1:
        return 0.0
    shift = sigma * (np.sqrt(2 * log_n) - (safe_log(log_n) + safe_log(4 * np.pi)) / (2 * np.sqrt(2 * log_n) + EPS))
    return float(min(shift / (sigma * 2 + EPS), 1.0))
