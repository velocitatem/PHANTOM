"""RL training for thesis pricing system with thesis-aligned metrics.

Trains pricing policies using stable-baselines3 with TensorBoard logging.
Tracks COI erosion, alpha estimation error, and economic KPIs per thesis formulation.

Usage:
    python -m lab.case.thesis.train --algo ppo --alpha 0.3 --steps 100000
    python -m lab.case.thesis.train --algo adaptive --sweep  # run alpha sweep
    tensorboard --logdir lab/case/thesis/runs
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Callable, Any
import numpy as np

try:
    from stable_baselines3 import PPO, SAC, A2C
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

from .simplified_env import PricingEnv, EnvConfig, make_env, adaptive_policy, fixed_price_policy, random_policy
from .simplified import coi_erosion


# thesis-aligned KPIs tracked per episode
@dataclass
class EpisodeMetrics:
    reward: float = 0.0
    revenue: float = 0.0
    profit: float = 0.0
    coi_erosion: float = 0.0      # theorem 1: order statistic erosion
    coi_leakage: float = 0.0      # per-step leakage penalty
    alpha_error: float = 0.0      # |α - α̂|
    avg_margin: float = 0.0
    n_agents: int = 0
    steps: int = 0


@dataclass
class ExperimentConfig:
    """Full experiment specification for reproducibility."""
    algo: str = "ppo"
    total_timesteps: int = 100_000
    n_envs: int = 4
    eval_freq: int = 5000
    n_eval_episodes: int = 10
    log_dir: str = "lab/case/thesis/runs"
    seed: int = 42
    n_products: int = 10
    max_steps: int = 200
    alpha_true: float = 0.2
    reward_mode: str = "robust"
    experiment_name: str | None = None

    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"{self.algo}_a{self.alpha_true:.2f}_{self.reward_mode}"


# unified policy interface wrapping all baselines
class Policy:
    """Unified policy interface for baselines and trained models."""

    def __init__(self, policy_fn: Callable[[np.ndarray, int], np.ndarray], name: str):
        self._fn = policy_fn
        self.name = name

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        n = (len(obs) - 3) // 3
        return self._fn(obs, n), None

    @staticmethod
    def fixed(margin: float = 0.15) -> "Policy":
        return Policy(lambda obs, n: fixed_price_policy(np.ones(n), margin), f"fixed_{margin:.2f}")

    @staticmethod
    def adaptive(base_margin: float = 0.15) -> "Policy":
        return Policy(lambda obs, n: adaptive_policy(obs, n, base_margin), f"adaptive_{base_margin:.2f}")

    @staticmethod
    def random() -> "Policy":
        return Policy(lambda obs, n: random_policy(n), "random")

    @staticmethod
    def myopic(greed: float = 0.3) -> "Policy":
        """Myopic: maximize immediate margin, ignore alpha."""
        def _fn(obs: np.ndarray, n: int) -> np.ndarray:
            demand_norm = obs[n:2*n] if len(obs) > 2*n else np.ones(n) * 0.5
            mult = 1.0 + greed * (1 + np.mean(demand_norm))
            return np.ones(n, dtype=np.float32) * np.clip(mult, 0.5, 1.5)
        return Policy(_fn, f"myopic_{greed:.1f}")


class MetricsCallback(BaseCallback):
    """Tracks thesis-aligned metrics during RL training."""

    def __init__(self, writer: SummaryWriter | None, verbose: int = 0):
        super().__init__(verbose)
        self._writer = writer
        self._ep = EpisodeMetrics()
        self._buffer: List[EpisodeMetrics] = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            self._ep.steps += 1
            self._ep.reward += info.get('reward', 0)
            self._ep.revenue += info.get('revenue', 0)
            self._ep.profit += info.get('profit', 0)
            self._ep.coi_erosion += info.get('coi_erosion', 0)
            self._ep.coi_leakage += info.get('coi_leakage', 0)
            self._ep.alpha_error += abs(info.get('alpha_true', 0) - info.get('alpha_est', 0))
            self._ep.avg_margin += info.get('avg_margin', 0)
            self._ep.n_agents += info.get('n_agents', 0)
        return True

    def _on_rollout_end(self) -> None:
        if self._ep.steps == 0 or self._writer is None:
            return
        s, step = self._ep.steps, self.num_timesteps
        self._writer.add_scalar('economics/revenue', self._ep.revenue / s, step)
        self._writer.add_scalar('economics/profit', self._ep.profit / s, step)
        self._writer.add_scalar('economics/margin', self._ep.avg_margin / s, step)
        self._writer.add_scalar('coi/erosion', self._ep.coi_erosion / s, step)
        self._writer.add_scalar('coi/leakage', self._ep.coi_leakage / s, step)
        self._writer.add_scalar('alpha/estimation_error', self._ep.alpha_error / s, step)
        self._writer.add_scalar('agents/count', self._ep.n_agents / s, step)
        self._buffer.append(self._ep)
        self._ep = EpisodeMetrics()


def make_vec_env(cfg: ExperimentConfig, n_envs: int = 1) -> DummyVecEnv:
    def _make():
        env_cfg = EnvConfig(n_products=cfg.n_products, max_steps=cfg.max_steps,
                            alpha_true=cfg.alpha_true, reward_mode=cfg.reward_mode, seed=cfg.seed)
        return Monitor(make_env(env_cfg))
    return DummyVecEnv([_make for _ in range(n_envs)])


def evaluate_policy(policy: Policy | Any, cfg: ExperimentConfig, n_episodes: int = 20) -> Dict[str, float]:
    """Evaluate policy and return thesis-aligned metrics."""
    env_cfg = EnvConfig(n_products=cfg.n_products, max_steps=cfg.max_steps,
                        alpha_true=cfg.alpha_true, reward_mode=cfg.reward_mode, seed=cfg.seed + 999)
    env = make_env(env_cfg)
    metrics = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep = EpisodeMetrics()
        done = False
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep.reward += reward
            ep.revenue += info.get('revenue', 0)
            ep.profit += info.get('profit', 0)
            ep.coi_erosion += info.get('coi_erosion', 0)
            ep.coi_leakage += info.get('coi_leakage', 0)
            ep.alpha_error += abs(info['alpha_true'] - info['alpha_est'])
            ep.avg_margin += info.get('avg_margin', 0)
            ep.steps += 1
        metrics.append(ep)

    n = len(metrics)
    return {
        'reward_mean': np.mean([m.reward for m in metrics]),
        'reward_std': np.std([m.reward for m in metrics]),
        'revenue_mean': np.mean([m.revenue / m.steps for m in metrics]),
        'profit_mean': np.mean([m.profit / m.steps for m in metrics]),
        'coi_erosion_mean': np.mean([m.coi_erosion / m.steps for m in metrics]),
        'coi_leakage_mean': np.mean([m.coi_leakage / m.steps for m in metrics]),
        'alpha_error_mean': np.mean([m.alpha_error / m.steps for m in metrics]),
        'margin_mean': np.mean([m.avg_margin / m.steps for m in metrics]),
    }


def train(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Train RL agent or evaluate baseline policy."""
    is_baseline = cfg.algo.lower() in ["fixed", "adaptive", "random", "myopic"]
    if not HAS_SB3 and not is_baseline:
        raise ImportError("stable-baselines3 required: pip install stable-baselines3[extra]")

    log_path = Path(cfg.log_dir) / cfg.experiment_name
    log_path.mkdir(parents=True, exist_ok=True)
    with open(log_path / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    writer = SummaryWriter(log_path) if HAS_TB else None
    train_env = make_vec_env(cfg, cfg.n_envs)
    eval_env = make_vec_env(cfg, 1)

    if is_baseline:
        policy_map = {"fixed": Policy.fixed(), "adaptive": Policy.adaptive(),
                      "random": Policy.random(), "myopic": Policy.myopic()}
        policy = policy_map[cfg.algo.lower()]
        run_baseline(policy, train_env, cfg.total_timesteps, writer)
        final_metrics = evaluate_policy(policy, cfg)
    else:
        algo_cls = {"ppo": PPO, "sac": SAC, "a2c": A2C}.get(cfg.algo.lower())
        if algo_cls is None:
            raise ValueError(f"unknown algo: {cfg.algo}")
        common = dict(verbose=1, seed=cfg.seed, tensorboard_log=str(log_path), device="auto")
        if cfg.algo.lower() == "ppo":
            model = PPO("MlpPolicy", train_env, learning_rate=3e-4, n_steps=2048,
                        batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                        clip_range=0.2, ent_coef=0.01, **common)
        elif cfg.algo.lower() == "sac":
            model = SAC("MlpPolicy", train_env, learning_rate=3e-4, buffer_size=100_000,
                        batch_size=256, tau=0.005, gamma=0.99, **common)
        else:
            model = A2C("MlpPolicy", train_env, learning_rate=7e-4, n_steps=5, gamma=0.99, **common)

        cb = MetricsCallback(writer)
        eval_cb = EvalCallback(eval_env, best_model_save_path=str(log_path / "best"),
                               log_path=str(log_path), eval_freq=cfg.eval_freq,
                               n_eval_episodes=cfg.n_eval_episodes, deterministic=True)
        model.learn(cfg.total_timesteps, callback=[cb, eval_cb], progress_bar=True)
        model.save(log_path / "final_model")
        policy = model
        final_metrics = evaluate_policy(model, cfg)

    if writer:
        for k, v in final_metrics.items():
            writer.add_scalar(f'final/{k}', v, cfg.total_timesteps)
        writer.close()

    train_env.close()
    eval_env.close()
    with open(log_path / "results.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    return {"path": str(log_path), "metrics": final_metrics}


def run_baseline(policy: Policy, vec_env: DummyVecEnv, total_steps: int, writer: SummaryWriter | None):
    """Run baseline policy through environment with logging."""
    obs = vec_env.reset()
    n_envs = vec_env.num_envs
    ep_rewards = np.zeros(n_envs)
    all_rewards, coi_buf, alpha_buf = [], [], []

    for step in range(0, total_steps, n_envs):
        actions = np.array([policy.predict(obs[i])[0] for i in range(n_envs)])
        obs, rewards, dones, infos = vec_env.step(actions)
        ep_rewards += rewards
        for i, info in enumerate(infos):
            coi_buf.append(info.get('coi_erosion', 0))
            alpha_buf.append(abs(info.get('alpha_true', 0) - info.get('alpha_est', 0)))
            if dones[i]:
                all_rewards.append(ep_rewards[i])
                ep_rewards[i] = 0
        if writer and step % 1000 < n_envs and all_rewards:
            writer.add_scalar('rollout/ep_rew_mean', np.mean(all_rewards[-20:]), step)
            writer.add_scalar('coi/erosion', np.mean(coi_buf[-100:]), step)
            writer.add_scalar('alpha/estimation_error', np.mean(alpha_buf[-100:]), step)


def run_sweep(cfg: ExperimentConfig, alphas: List[float] | None = None) -> Dict[str, Dict]:
    """Run experiment across contamination levels for scientific comparison."""
    alphas = alphas or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = {}
    for alpha in alphas:
        sweep_cfg = ExperimentConfig(**{**asdict(cfg), "alpha_true": alpha,
                                         "experiment_name": f"{cfg.algo}_a{alpha:.2f}_{cfg.reward_mode}"})
        print(f"\n=== α={alpha:.2f} ===")
        out = train(sweep_cfg)
        results[f"alpha_{alpha:.2f}"] = out["metrics"]
    summary_path = Path(cfg.log_dir) / f"sweep_{cfg.algo}_{cfg.reward_mode}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep results saved to {summary_path}")
    return results


def compare_policies(cfg: ExperimentConfig, policies: List[str] | None = None) -> Dict[str, Dict]:
    """Compare multiple policies at same contamination level."""
    policies = policies or ["fixed", "adaptive", "myopic", "random"]
    results = {}
    for algo in policies:
        cmp_cfg = ExperimentConfig(**{**asdict(cfg), "algo": algo,
                                       "experiment_name": f"cmp_{algo}_a{cfg.alpha_true:.2f}"})
        print(f"\n=== {algo} ===")
        out = train(cmp_cfg)
        results[algo] = out["metrics"]
    cmp_path = Path(cfg.log_dir) / f"compare_a{cfg.alpha_true:.2f}.json"
    with open(cmp_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison saved to {cmp_path}")
    for algo, m in results.items():
        print(f"  {algo:12s}: reward={m['reward_mean']:.2f} coi_erosion={m['coi_erosion_mean']:.4f} "
              f"alpha_err={m['alpha_error_mean']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train RL pricing policies")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "sac", "a2c", "fixed", "adaptive", "random", "myopic"])
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--reward-mode", default="robust", choices=["revenue", "profit", "robust", "coi_aware"])
    parser.add_argument("--n-products", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="lab/case/thesis/runs")
    parser.add_argument("--sweep", action="store_true", help="run contamination sweep")
    parser.add_argument("--compare", action="store_true", help="compare all baselines")
    args = parser.parse_args()

    cfg = ExperimentConfig(algo=args.algo, total_timesteps=args.steps, alpha_true=args.alpha,
                           reward_mode=args.reward_mode, n_products=args.n_products,
                           n_envs=args.n_envs, seed=args.seed, log_dir=args.log_dir)

    if args.sweep:
        run_sweep(cfg)
    elif args.compare:
        compare_policies(cfg)
    else:
        result = train(cfg)
        print(f"\nTraining complete: {result['path']}")
        print(f"Metrics: {json.dumps(result['metrics'], indent=2)}")


if __name__ == "__main__":
    main()
