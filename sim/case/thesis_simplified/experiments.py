"""COI leakage experiments and policy comparisons.

Demonstrates the core thesis contribution: COI erosion under agent contamination
and recovery via robust pricing policies.

Generates TensorBoard logs for:
- COI erosion curves across contamination levels
- Policy comparison (fixed vs adaptive vs RL)
- Revenue/margin trade-offs
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

from .simplified_env import PricingEnv, EnvConfig, make_env
from .simplified import System


@dataclass
class ExperimentResult:
    """Container for experiment metrics."""
    name: str
    alpha: float
    reward_mean: float
    reward_std: float
    coi_erosion: float
    alpha_error: float
    revenue: float
    margin: float

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def theoretical_coi_erosion_curve(alphas: np.ndarray, n_sessions: int = 1000) -> np.ndarray:
    """Theoretical COI erosion from Theorem 1 using order statistic model.

    For N i.i.d. uniform queries on [p_min, p_max]:
    E[p^(1)] = p_min + (p_max - p_min)/(N+1), so erosion = 1 - 2/(N+1)
    """
    erosions = []
    for a in alphas:
        n_agents = max(1, int(a * n_sessions))
        erosions.append(1.0 - 2.0 / (n_agents + 1))
    return np.array(erosions)


def run_policy_episode(
    env: PricingEnv,
    policy_fn,
    n_episodes: int = 10
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Run policy and collect per-step metrics."""
    rewards, coi_erosions, alpha_errors, revenues = [], [], [], []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action = policy_fn(obs, env.n)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            if 'coi_erosion' in info:
                coi_erosions.append(info['coi_erosion'])
            if 'alpha_true' in info and 'alpha_est' in info:
                alpha_errors.append(abs(info['alpha_true'] - info['alpha_est']))
            if 'revenue' in info:
                revenues.append(info['revenue'])

    return rewards, coi_erosions, alpha_errors, revenues


class PolicyRegistry:
    """Registry of baseline policies."""

    @staticmethod
    def fixed(obs: np.ndarray, n: int, margin: float = 0.15) -> np.ndarray:
        return np.ones(n, dtype=np.float32) * (1.0 + margin)

    @staticmethod
    def random(obs: np.ndarray, n: int, rng: np.random.Generator = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.uniform(0.7, 1.3, n).astype(np.float32)

    @staticmethod
    def adaptive(obs: np.ndarray, n: int, base_margin: float = 0.15) -> np.ndarray:
        """Reduce margins when alpha estimate is high."""
        alpha_est = obs[2 * n] if len(obs) > 2 * n else 0.2
        margin_scale = 1.0 - 0.4 * alpha_est
        return np.ones(n, dtype=np.float32) * (1.0 + base_margin * margin_scale)

    @staticmethod
    def aggressive(obs: np.ndarray, n: int) -> np.ndarray:
        """High margins, ignores contamination."""
        return np.ones(n, dtype=np.float32) * 1.4

    @staticmethod
    def defensive(obs: np.ndarray, n: int) -> np.ndarray:
        """Low margins, always cautious."""
        return np.ones(n, dtype=np.float32) * 1.05

    @staticmethod
    def alpha_proportional(obs: np.ndarray, n: int, max_margin: float = 0.3) -> np.ndarray:
        """Margin inversely proportional to estimated alpha."""
        alpha_est = obs[2 * n] if len(obs) > 2 * n else 0.2
        margin = max_margin * (1.0 - alpha_est)
        return np.ones(n, dtype=np.float32) * (1.0 + margin)


def run_contamination_sweep(
    alphas: List[float],
    policies: Dict[str, callable],
    n_products: int = 10,
    max_steps: int = 200,
    n_episodes: int = 10,
    seed: int = 42,
    log_dir: str = None
) -> Dict[str, List[ExperimentResult]]:
    """Run policies across contamination levels."""

    results = {name: [] for name in policies}
    writer = SummaryWriter(Path(log_dir) / "sweep") if log_dir and HAS_TB else None

    for alpha in alphas:
        print(f"  alpha={alpha:.2f}", end=" ")
        env_cfg = EnvConfig(
            n_products=n_products, max_steps=max_steps,
            alpha_true=alpha, reward_mode="robust", seed=seed)
        env = make_env(env_cfg)

        for name, policy_fn in policies.items():
            rewards, coi_vals, alpha_errs, revenues = run_policy_episode(env, policy_fn, n_episodes)

            result = ExperimentResult(
                name=name, alpha=alpha,
                reward_mean=float(np.mean(rewards)),
                reward_std=float(np.std(rewards)),
                coi_erosion=float(np.mean(coi_vals)) if coi_vals else 0.0,
                alpha_error=float(np.mean(alpha_errs)) if alpha_errs else 0.0,
                revenue=float(np.mean(revenues)) if revenues else 0.0,
                margin=float(np.mean([policy_fn(np.zeros(3 * n_products + 3), n_products)]) - 1.0))

            results[name].append(result)

            if writer:
                step = int(alpha * 100)
                writer.add_scalar(f'{name}/reward', result.reward_mean, step)
                writer.add_scalar(f'{name}/coi_erosion', result.coi_erosion, step)
                writer.add_scalar(f'{name}/alpha_error', result.alpha_error, step)
                writer.add_scalar(f'{name}/revenue', result.revenue, step)

        print(f"done")

    # add theoretical curve
    if writer:
        theo = theoretical_coi_erosion_curve(np.array(alphas))
        for i, (a, e) in enumerate(zip(alphas, theo)):
            writer.add_scalar('theoretical/coi_erosion', e, int(a * 100))
        writer.close()

    return results


def run_coi_demonstration(log_dir: str = "sim/case/thesis_simplified/runs", seed: int = 42) -> Dict:
    """Main COI demonstration experiment."""
    print("=== COI Leakage Demonstration ===\n")

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(Path(log_dir) / "coi_demo") if HAS_TB else None

    # theoretical erosion curve
    print("1. Theoretical COI erosion (Theorem 1)")
    alphas = np.linspace(0.0, 0.6, 13)
    theo_erosion = theoretical_coi_erosion_curve(alphas, n_sessions=1000)

    for a, e in zip(alphas, theo_erosion):
        print(f"   alpha={a:.2f} -> erosion={e:.3f}")
        if writer:
            writer.add_scalar('theory/coi_erosion', e, int(a * 100))

    # policy comparison
    print("\n2. Policy comparison across contamination levels")
    policies = {
        'fixed': lambda obs, n: PolicyRegistry.fixed(obs, n),
        'aggressive': PolicyRegistry.aggressive,
        'defensive': PolicyRegistry.defensive,
        'adaptive': PolicyRegistry.adaptive,
        'alpha_proportional': PolicyRegistry.alpha_proportional,
    }

    sweep_alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = run_contamination_sweep(
        sweep_alphas, policies, n_products=10, max_steps=100,
        n_episodes=5, seed=seed, log_dir=log_dir)

    # summarize
    print("\n3. Summary by policy")
    for name, res_list in results.items():
        avg_reward = np.mean([r.reward_mean for r in res_list])
        avg_coi = np.mean([r.coi_erosion for r in res_list])
        print(f"   {name:20s}: avg_reward={avg_reward:.2f}, avg_coi={avg_coi:.3f}")

    # save results
    output = {
        'theoretical': {'alphas': alphas.tolist(), 'erosion': theo_erosion.tolist()},
        'empirical': {name: [r.to_dict() for r in res_list] for name, res_list in results.items()}}

    with open(Path(log_dir) / "coi_demo_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    if writer:
        writer.close()

    print(f"\nResults saved to {log_dir}/coi_demo_results.json")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")

    return output


def run_reward_mode_comparison(log_dir: str = "sim/case/thesis_simplified/runs", seed: int = 42) -> Dict:
    """Compare different reward modes."""
    print("=== Reward Mode Comparison ===\n")

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(Path(log_dir) / "reward_modes") if HAS_TB else None

    reward_modes = ["revenue", "profit", "robust", "coi_aware"]
    alpha = 0.3  # moderate contamination

    results = {}
    for mode in reward_modes:
        print(f"  mode={mode}", end=" ")
        env_cfg = EnvConfig(
            n_products=10, max_steps=200, alpha_true=alpha,
            reward_mode=mode, seed=seed)
        env = make_env(env_cfg)

        rewards, coi_vals, _, revenues = run_policy_episode(
            env, PolicyRegistry.adaptive, n_episodes=10)

        results[mode] = {
            'reward_mean': float(np.mean(rewards)),
            'reward_std': float(np.std(rewards)),
            'coi_erosion': float(np.mean(coi_vals)) if coi_vals else 0.0,
            'revenue': float(np.mean(revenues)) if revenues else 0.0}

        if writer:
            for k, v in results[mode].items():
                writer.add_scalar(f'{mode}/{k}', v, 0)

        print(f"reward={results[mode]['reward_mean']:.2f}, coi={results[mode]['coi_erosion']:.3f}")

    if writer:
        writer.close()

    with open(Path(log_dir) / "reward_mode_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_alpha_drift_experiment(log_dir: str = "sim/case/thesis_simplified/runs", seed: int = 42) -> Dict:
    """Test policy robustness under non-stationary contamination."""
    print("=== Alpha Drift Experiment ===\n")

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(Path(log_dir) / "alpha_drift") if HAS_TB else None

    drift_rates = [0.0, 0.01, 0.02, 0.05]
    results = {}

    for drift in drift_rates:
        print(f"  drift={drift:.2f}", end=" ")
        env_cfg = EnvConfig(
            n_products=10, max_steps=200, alpha_true=0.2,
            alpha_drift=drift, reward_mode="robust", seed=seed)
        env = make_env(env_cfg)

        rewards, coi_vals, alpha_errs, _ = run_policy_episode(
            env, PolicyRegistry.adaptive, n_episodes=10)

        results[f'drift_{drift}'] = {
            'reward_mean': float(np.mean(rewards)),
            'coi_erosion': float(np.mean(coi_vals)) if coi_vals else 0.0,
            'alpha_tracking_error': float(np.mean(alpha_errs)) if alpha_errs else 0.0}

        if writer:
            for k, v in results[f'drift_{drift}'].items():
                writer.add_scalar(f'drift_{drift}/{k}', v, 0)

        print(f"reward={results[f'drift_{drift}']['reward_mean']:.2f}, "
              f"alpha_err={results[f'drift_{drift}']['alpha_tracking_error']:.3f}")

    if writer:
        writer.close()

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run COI experiments")
    parser.add_argument("--exp", type=str, default="coi", choices=["coi", "reward", "drift", "all"])
    parser.add_argument("--log-dir", type=str, default="sim/case/thesis_simplified/runs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.exp == "coi" or args.exp == "all":
        run_coi_demonstration(args.log_dir, args.seed)

    if args.exp == "reward" or args.exp == "all":
        run_reward_mode_comparison(args.log_dir, args.seed)

    if args.exp == "drift" or args.exp == "all":
        run_alpha_drift_experiment(args.log_dir, args.seed)
