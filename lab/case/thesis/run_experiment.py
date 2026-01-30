#!/usr/bin/env python
"""Thesis simulation experiments with real MDP behavioral models."""
from __future__ import annotations
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from lab.case.thesis.platform import make_thesis_platform, ThesisConfig
from lab.case.thesis.metrics import compute_coi, compute_separability
from lab.experiments.eval import compare_policies
import numpy as np


def demo_basic_simulation():
    print("=" * 70)
    print("THESIS SIMULATION: Contaminated Dynamic Pricing (Real MDP Kernels)")
    print("=" * 70)

    cfg = ThesisConfig(n_instruments=5, alpha_contamination=0.3, lambda_coi=0.5,
                       max_steps=100, seed=42, use_real_behavior=True)
    platform = make_thesis_platform(cfg)

    print(f"\nInstruments: {platform.instruments.n}")
    print(f"Reference prices: {platform.instruments.refs.round(2)}")
    print(f"Costs: {platform.instruments.costs.round(2)}")
    print(f"Initial contamination alpha={cfg.alpha_contamination}")
    print(f"Using real behavior: {cfg.use_real_behavior}")

    result = platform.reset(seed=42)
    total_reward, coi_history = 0, []

    print(f"\n{'Step':>5} {'Reward':>10} {'PnL':>10} {'COI':>8} {'alpha':>6} {'Conv':>8}")
    print("-" * 55)

    for t in range(cfg.max_steps):
        action = platform.instruments.refs * np.random.uniform(0.95, 1.15, size=platform.instruments.n)
        result = platform.step(action)
        total_reward += result.reward
        coi = compute_coi(platform._quote, platform.instruments, result.metrics, result.hidden.contamination)
        coi_history.append(coi.coi_level)

        if t % 20 == 0:
            print(f"{t:5d} {result.reward:10.2f} {result.metrics.pnl:10.2f} "
                  f"{coi.coi_level:8.2f} {result.hidden.contamination:6.2f} {result.metrics.conversion:8.3f}")

    print("-" * 55)
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Average COI: {np.mean(coi_history):.2f}")
    print(f"COI Trend: {coi_history[-1] - coi_history[0]:+.2f}")


def demo_contamination_sweep():
    print("\n" + "=" * 70)
    print("EXPERIMENT: COI Erosion vs Contamination (Theorem 1)")
    print("=" * 70)

    from lab.case.thesis.platform import sweep_contamination
    trials = 20
    alpha_values = [i/trials for i in range(trials)]
    results = sweep_contamination(alpha_values, n_steps=100, seed=42)

    print(f"\n{'alpha':>6} {'Reward':>12} {'PnL':>12} {'Conv':>10}")
    print("-" * 45)
    for alpha, m in sorted(results.items()):
        print(f"{alpha:6.2f} {m['total_reward']:12.2f} {m['total_pnl']:12.2f} {m['avg_conversion']:10.3f}")

    rewards = [results[a]['total_reward'] for a in sorted(results.keys())]
    dataset = np.array([[a, r] for a, r in zip(alpha_values, rewards)])
    trend = np.corrcoef(dataset[:, 0], dataset[:, 1])[0, 1]
    print(f"Trend (alpha~reward correlation): {trend:.3f}")


def demo_policy_comparison():
    print("\n" + "=" * 70)
    print("EXPERIMENT: Policy Comparison under Contamination")
    print("=" * 70)

    cfg = ThesisConfig(n_instruments=5, alpha_contamination=0.25, max_steps=100, seed=42)
    platform = make_thesis_platform(cfg)

    def fixed_policy(obs, t): return platform.instruments.refs.copy(), 1.0
    def aggressive_policy(obs, t): return platform.instruments.refs * 1.3, 1.0
    def conservative_policy(obs, t): return platform.instruments.refs * 1.05, 1.0
    def adaptive_policy(obs, t):
        fills = obs[platform.instruments.n:2*platform.instruments.n]
        exp = obs[2*platform.instruments.n:3*platform.instruments.n]
        conv = np.sum(fills) / (np.sum(exp) + 1e-8)
        return platform.instruments.refs * (1.0 + 0.2 * conv), 1.0

    policies = {'fixed': fixed_policy, 'aggressive': aggressive_policy,
                'conservative': conservative_policy, 'adaptive': adaptive_policy}
    results = compare_policies(platform, policies, n_steps=100, n_runs=3, seed=42)

    print(f"\n{'Policy':>15} {'Reward':>12} {'Std':>10} {'PnL':>12} {'Conv':>10}")
    print("-" * 65)
    for name, r in sorted(results.items(), key=lambda x: -x[1]['mean_reward']):
        print(f"{name:>15} {r['mean_reward']:12.2f} {r['std_reward']:10.2f} "
              f"{r['mean_pnl']:12.2f} {r['mean_conversion']:10.3f}")


def demo_session_analysis():
    """Analyze session-level behavior from MDP trajectories."""
    print("\n" + "=" * 70)
    print("EXPERIMENT: Session Analysis (Ground Truth)")
    print("=" * 70)

    from lab.outlet.constants import LogLevel
    cfg = ThesisConfig(n_instruments=5, alpha_contamination=0.3, max_steps=50,
                       log_level=LogLevel.FULL, seed=42, use_real_behavior=True)
    platform = make_thesis_platform(cfg)

    result = platform.reset(seed=42)
    human_sessions, agent_sessions = 0, 0

    for t in range(cfg.max_steps):
        action = platform.instruments.refs * 1.1
        result = platform.step(action)
        sep = compute_separability(result.logs, result.hidden.contamination)
        human_sessions += sep.n_human_sessions
        agent_sessions += sep.n_agent_sessions

    total = human_sessions + agent_sessions
    print(f"\nTotal sessions: {total}")
    print(f"Human sessions: {human_sessions} ({100*human_sessions/total:.1f}%)")
    print(f"Agent sessions: {agent_sessions} ({100*agent_sessions/total:.1f}%)")
    print(f"True contamination: {cfg.alpha_contamination:.1%}")
    print(f"Observed contamination: {agent_sessions/total:.1%}")


if __name__ == '__main__':
    demo_basic_simulation()
    demo_contamination_sweep()
    # demo_policy_comparison()
    # demo_session_analysis()
