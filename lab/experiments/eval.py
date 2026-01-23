"""
Evaluation utilities for policy testing and off-policy evaluation.

This module provides:
- rollout: Run a policy on the platform for multiple steps
- compare_policies: Compare multiple policies with statistics
- Baseline policies: fixed_price, cost_plus_margin, random_walk, epsilon_greedy
- OPE estimators: IPS and SNIPS for off-policy evaluation

Example:
    >>> from lab.config import make_retail_platform
    >>> from lab.experiments.eval import rollout, fixed_price_policy
    >>> platform = make_retail_platform()
    >>> policy = fixed_price_policy(platform.instruments.refs)
    >>> result = rollout(platform, policy, n_steps=100)
    >>> print(f"Total PnL: {result.total_pnl:.2f}")
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any
import numpy as np
from ..outlet.platform import Platform
from ..outlet.types import StepResult, StepLogs, Quote

# Policy signature: takes (observation_flat, timestep) -> (action_prices, propensity)
Policy = Callable[[np.ndarray, int], tuple[np.ndarray, float]]

@dataclass
class RolloutResult:
    """Results from a policy rollout.

    Attributes:
        rewards: Per-step rewards
        metrics: Per-step StepMetrics objects
        logs: Per-step StepLogs objects
        total_reward: Sum of rewards
        total_pnl: Sum of PnL from metrics
        avg_conversion: Average conversion rate
    """
    rewards: list[float]
    metrics: list[Any]
    logs: list[StepLogs]
    total_reward: float
    total_pnl: float
    avg_conversion: float

def rollout(platform: Platform, policy: Policy, n_steps: int, seed: int | None = None) -> RolloutResult:
    """Execute a policy on the platform for n_steps.

    Args:
        platform: The simulation platform
        policy: Function (obs, t) -> (action, propensity)
        n_steps: Number of steps to run
        seed: Random seed for reproducibility

    Returns:
        RolloutResult with rewards, metrics, and summary statistics
    """
    result = platform.reset(seed)
    rewards, metrics, logs = [], [], []

    for t in range(n_steps):
        obs_flat = result.obs.to_flat()
        action, propensity = policy(obs_flat, t)
        result = platform.step(action, propensity)
        rewards.append(result.reward)
        metrics.append(result.metrics)
        logs.append(result.logs)
        if result.terminated or result.truncated:
            break

    return RolloutResult(
        rewards=rewards, metrics=metrics, logs=logs,
        total_reward=sum(rewards),
        total_pnl=sum(m.pnl for m in metrics),
        avg_conversion=np.mean([m.conversion for m in metrics])
    )

# Baseline policies for comparison

def fixed_price_policy(refs: np.ndarray) -> Policy:
    """Policy that always quotes at reference prices."""
    def policy(obs: np.ndarray, t: int) -> tuple[np.ndarray, float]:
        return refs.copy(), 1.0
    return policy

def cost_plus_margin_policy(costs: np.ndarray, margin: float = 0.3) -> Policy:
    """Policy that quotes at cost * (1 + margin)."""
    prices = costs * (1 + margin)
    def policy(obs: np.ndarray, t: int) -> tuple[np.ndarray, float]:
        return prices.copy(), 1.0
    return policy

def random_walk_policy(refs: np.ndarray, volatility: float = 0.05,
                       rng: np.random.Generator | None = None) -> Policy:
    """Policy that performs a random walk around reference prices."""
    rng = rng or np.random.default_rng()
    prices = refs.copy()
    def policy(obs: np.ndarray, t: int) -> tuple[np.ndarray, float]:
        nonlocal prices
        delta = rng.normal(0, volatility, len(prices))
        prices = prices * (1 + delta)
        prices = np.clip(prices, refs * 0.5, refs * 2.0)
        return prices.copy(), 1.0
    return policy

def epsilon_greedy_policy(base_policy: Policy, refs: np.ndarray,
                          epsilon: float = 0.1, rng: np.random.Generator | None = None) -> Policy:
    """Wrap a policy with epsilon-greedy exploration."""
    rng = rng or np.random.default_rng()
    def policy(obs: np.ndarray, t: int) -> tuple[np.ndarray, float]:
        if rng.random() < epsilon:
            action = refs * rng.uniform(0.8, 1.2, len(refs))
            return action, epsilon / len(refs)
        else:
            action, _ = base_policy(obs, t)
            return action, 1 - epsilon
    return policy

# Off-Policy Evaluation (OPE)

@dataclass
class OPEResult:
    """Results from off-policy evaluation.

    Attributes:
        ips_estimate: Inverse Propensity Scoring estimate
        snips_estimate: Self-normalized IPS estimate (more stable)
        n_samples: Number of samples used
        effective_samples: Effective sample size (accounts for variance)
    """
    ips_estimate: float
    snips_estimate: float
    n_samples: int
    effective_samples: float

def compute_ips(logs: list[StepLogs], rewards: list[float],
                target_policy: Policy, behavior_propensities: list[float] | None = None) -> OPEResult:
    """Compute IPS and SNIPS estimators for off-policy evaluation.

    Uses logged propensities to estimate expected reward under a target
    policy from data collected under a behavior policy.

    Args:
        logs: Step logs containing propensities
        rewards: Observed rewards from behavior policy
        target_policy: Policy to evaluate (not currently used, assumes deterministic)
        behavior_propensities: Override propensities if not in logs

    Returns:
        OPEResult with IPS, SNIPS estimates and sample statistics
    """
    if behavior_propensities is None:
        # extract from logs
        behavior_propensities = []
        for log in logs:
            if log.executions:
                avg_prop = np.mean([e.propensity for e in log.executions])
            else:
                avg_prop = 1.0
            behavior_propensities.append(avg_prop)

    # compute importance weights
    weights = []
    for i, (log, bp) in enumerate(zip(logs, behavior_propensities)):
        # target propensity would need obs reconstruction - simplified here
        tp = 1.0  # assume deterministic target
        w = tp / (bp + 1e-8)
        weights.append(w)

    weights = np.array(weights)
    rewards = np.array(rewards)

    # IPS estimate
    ips = np.sum(weights * rewards) / len(rewards)

    # SNIPS (self-normalized)
    snips = np.sum(weights * rewards) / (np.sum(weights) + 1e-8)

    # effective sample size
    ess = (np.sum(weights) ** 2) / (np.sum(weights ** 2) + 1e-8)

    return OPEResult(ips_estimate=ips, snips_estimate=snips,
                     n_samples=len(rewards), effective_samples=ess)

def compare_policies(platform: Platform, policies: dict[str, Policy],
                     n_steps: int = 100, n_runs: int = 5, seed: int = 42) -> dict[str, dict]:
    """Compare multiple policies with statistical summary.

    Args:
        platform: Simulation platform
        policies: Dict mapping policy names to policy functions
        n_steps: Steps per rollout
        n_runs: Number of rollouts per policy (different seeds)
        seed: Base random seed

    Returns:
        Dict mapping policy names to result dicts with mean/std statistics
    """
    results = {}
    for name, policy in policies.items():
        run_results = []
        for i in range(n_runs):
            r = rollout(platform, policy, n_steps, seed=seed + i)
            run_results.append(r)

        results[name] = {
            'mean_reward': np.mean([r.total_reward for r in run_results]),
            'std_reward': np.std([r.total_reward for r in run_results]),
            'mean_pnl': np.mean([r.total_pnl for r in run_results]),
            'mean_conversion': np.mean([r.avg_conversion for r in run_results]),
        }
    return results
