#!/usr/bin/env python
"""Example script demonstrating the Quote-Control platform"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from lab.config import make_retail_platform, make_market_making_platform
from lab.experiments.eval import (rollout, compare_policies, fixed_price_policy,
                                   cost_plus_margin_policy, random_walk_policy)

def demo_retail():
    print("=" * 60)
    print("RETAIL DYNAMIC PRICING DEMO")
    print("=" * 60)

    platform = make_retail_platform()
    print(f"Instruments: {platform.instruments.n}")
    print(f"Reference prices: {platform.instruments.refs[:5].round(2)}...")

    # compare policies
    policies = {
        'fixed': fixed_price_policy(platform.instruments.refs),
        'cost_plus_30%': cost_plus_margin_policy(platform.instruments.costs, 0.3),
        'cost_plus_50%': cost_plus_margin_policy(platform.instruments.costs, 0.5),
        'random_walk': random_walk_policy(platform.instruments.refs, 0.03),
    }

    results = compare_policies(platform, policies, n_steps=100, n_runs=3)

    print("\nPolicy Comparison (100 steps, 3 runs):")
    print("-" * 50)
    for name, r in sorted(results.items(), key=lambda x: -x[1]['mean_pnl']):
        print(f"{name:20s} PnL={r['mean_pnl']:8.1f} +/- {r['std_reward']:6.1f}  "
              f"conv={r['mean_conversion']:.3f}")

def demo_market_making():
    print("\n" + "=" * 60)
    print("MARKET MAKING DEMO")
    print("=" * 60)

    platform = make_market_making_platform()
    print(f"Instruments: {platform.instruments.n}")
    print(f"Initial mids: {platform.instruments.refs.round(2)}")

    # simple policy: quote at mid with fixed spread
    def mm_policy(obs: np.ndarray, t: int):
        mids = platform.instruments.refs  # would use obs in real policy
        return mids, 1.0

    result = rollout(platform, mm_policy, n_steps=200, seed=42)
    print(f"\nRollout (200 steps):")
    print(f"  Total PnL: {result.total_pnl:.2f}")
    print(f"  Avg conversion: {result.avg_conversion:.3f}")
    print(f"  Total spread capture: {sum(m.spread_capture for m in result.metrics):.2f}")

if __name__ == '__main__':
    demo_retail()
    demo_market_making()
