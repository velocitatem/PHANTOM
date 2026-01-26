"""RL training for thesis pricing system with thesis-aligned metrics.

Trains pricing policies using stable-baselines3 with TensorBoard logging.
Tracks COI erosion, alpha estimation error, and economic KPIs per thesis formulation.
"""
from __future__ import annotations
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
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


@dataclass
class EpisodeMetrics:
    reward: float = 0.0
    revenue: float = 0.0
    profit: float = 0.0
    coi_erosion: float = 0.0
    coi_leakage: float = 0.0
    alpha_error: float = 0.0
    avg_margin: float = 0.0
    n_agents: int = 0
    steps: int = 0

    def accumulate(self, info: Dict[str, Any]) -> None:
        self.steps += 1
        self.reward += info.get('reward', 0)
        self.revenue += info.get('revenue', 0)
        self.profit += info.get('profit', 0)
        self.coi_erosion += info.get('coi_erosion', 0)
        self.coi_leakage += info.get('coi_leakage', 0)
        self.alpha_error += abs(info.get('alpha_true', 0) - info.get('alpha_est', 0))
        self.avg_margin += info.get('avg_margin', 0)
        self.n_agents += info.get('n_agents', 0)

    def normalized(self) -> Dict[str, float]:
        s = max(self.steps, 1)
        return {k: getattr(self, k) / s for k in ['revenue', 'profit', 'coi_erosion', 'coi_leakage', 'alpha_error', 'avg_margin', 'n_agents']}


@dataclass
class ExperimentConfig:
    algo: str = "ppo"
    total_timesteps: int = 100_000
    n_envs: int = 4
    eval_freq: int = 5000
    n_eval_episodes: int = 10
    log_dir: str = "sim/case/thesis_simplified/runs"
    seed: int = 42
    n_products: int = 10
    max_steps: int = 200
    alpha_true: float = 0.2
    reward_mode: str = "robust"
    experiment_name: str | None = None

    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"{self.algo}_a{self.alpha_true:.2f}_{self.reward_mode}"


class Policy:
    """Unified policy interface for baselines and trained models."""

    def __init__(self, policy_fn: Callable[[np.ndarray, int], np.ndarray], name: str):
        self._fn, self.name = policy_fn, name

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        return self._fn(obs, (len(obs) - 3) // 3), None

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
        def _fn(obs: np.ndarray, n: int) -> np.ndarray:
            demand_norm = obs[n:2*n] if len(obs) > 2*n else np.ones(n) * 0.5
            return np.ones(n, dtype=np.float32) * np.clip(1.0 + greed * (1 + np.mean(demand_norm)), 0.5, 1.5)
        return Policy(_fn, f"myopic_{greed:.1f}")


def log_metrics(writer: SummaryWriter | None, metrics: Dict[str, float], prefix: str, step: int) -> None:
    if writer is None:
        return
    for k, v in metrics.items():
        writer.add_scalar(f'{prefix}/{k}', v, step)


class MetricsCallback(BaseCallback):
    def __init__(self, writer: SummaryWriter | None, verbose: int = 0):
        super().__init__(verbose)
        self._writer = writer

    def _on_step(self) -> bool:
        if self._writer is None:
            return True
        for info in self.locals.get('infos', []):
            t = self.num_timesteps
            self._writer.add_scalar('economics/revenue', info.get('revenue', 0), t)
            self._writer.add_scalar('economics/profit', info.get('profit', 0), t)
            self._writer.add_scalar('economics/margin', info.get('avg_margin', 0), t)
            self._writer.add_scalar('coi/erosion', info.get('coi_erosion', 0), t)
            self._writer.add_scalar('coi/leakage', info.get('coi_leakage', 0), t)
            self._writer.add_scalar('alpha/estimation_error', abs(info.get('alpha_true', 0) - info.get('alpha_est', 0)), t)
            self._writer.add_scalar('agents/count', info.get('n_agents', 0), t)
        return True


def make_vec_env(cfg: ExperimentConfig, n_envs: int = 1) -> DummyVecEnv:
    def _make():
        return Monitor(make_env(EnvConfig(n_products=cfg.n_products, max_steps=cfg.max_steps,
                                          alpha_true=cfg.alpha_true, reward_mode=cfg.reward_mode, seed=cfg.seed)))
    return DummyVecEnv([_make for _ in range(n_envs)])


def run_episodes(policy: Policy | Any, env: PricingEnv, n_episodes: int) -> List[EpisodeMetrics]:
    """Run policy for n episodes and collect metrics."""
    metrics = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep, done = EpisodeMetrics(), False
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep.accumulate(info)
            ep.reward += reward
        metrics.append(ep)
    return metrics


def evaluate_policy(policy: Policy | Any, cfg: ExperimentConfig, n_episodes: int = 20) -> Dict[str, float]:
    env = make_env(EnvConfig(n_products=cfg.n_products, max_steps=cfg.max_steps,
                             alpha_true=cfg.alpha_true, reward_mode=cfg.reward_mode, seed=cfg.seed + 999))
    metrics = run_episodes(policy, env, n_episodes)
    return {
        'reward_mean': np.mean([m.reward for m in metrics]), 'reward_std': np.std([m.reward for m in metrics]),
        **{f'{k}_mean': np.mean([m.normalized()[k] for m in metrics])
           for k in ['revenue', 'profit', 'coi_erosion', 'coi_leakage', 'alpha_error', 'avg_margin']},
    }


def run_baseline(policy: Policy, vec_env: DummyVecEnv, total_steps: int, writer: SummaryWriter | None):
    obs, n_envs = vec_env.reset(), vec_env.num_envs
    ep_rewards = np.zeros(n_envs)

    for step in range(0, total_steps, n_envs):
        actions = np.array([policy.predict(obs[i])[0] for i in range(n_envs)])
        obs, rewards, dones, infos = vec_env.step(actions)
        ep_rewards += rewards
        for i, info in enumerate(infos):
            if writer:
                writer.add_scalar('economics/revenue', info.get('revenue', 0), step)
                writer.add_scalar('economics/profit', info.get('profit', 0), step)
                writer.add_scalar('economics/margin', info.get('avg_margin', 0), step)
                writer.add_scalar('coi/erosion', info.get('coi_erosion', 0), step)
                writer.add_scalar('coi/leakage', info.get('coi_leakage', 0), step)
                writer.add_scalar('alpha/estimation_error', abs(info.get('alpha_true', 0) - info.get('alpha_est', 0)), step)
                writer.add_scalar('agents/count', info.get('n_agents', 0), step)
            if dones[i]:
                if writer:
                    writer.add_scalar('rollout/ep_reward', ep_rewards[i], step)
                ep_rewards[i] = 0


def train(cfg: ExperimentConfig) -> Dict[str, Any]:
    is_baseline = cfg.algo.lower() in ["fixed", "adaptive", "random", "myopic"]
    if not HAS_SB3 and not is_baseline:
        raise ImportError("stable-baselines3 required: pip install stable-baselines3[extra]")

    log_path = Path(cfg.log_dir) / cfg.experiment_name
    log_path.mkdir(parents=True, exist_ok=True)
    with open(log_path / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    writer = SummaryWriter(log_path) if HAS_TB else None
    train_env, eval_env = make_vec_env(cfg, cfg.n_envs), make_vec_env(cfg, 1)

    if is_baseline:
        policy = {"fixed": Policy.fixed, "adaptive": Policy.adaptive, "random": Policy.random, "myopic": Policy.myopic}[cfg.algo.lower()]()
        run_baseline(policy, train_env, cfg.total_timesteps, writer)
        final_metrics = evaluate_policy(policy, cfg)
    else:
        algo_cls = {"ppo": PPO, "sac": SAC, "a2c": A2C}[cfg.algo.lower()]
        common = dict(verbose=1, seed=cfg.seed, tensorboard_log=str(log_path), device="auto")
        model = {
            "ppo": lambda: PPO("MlpPolicy", train_env, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, **common),
            "sac": lambda: SAC("MlpPolicy", train_env, learning_rate=1e-4, buffer_size=50_000, batch_size=512, tau=0.02, gamma=0.99, learning_starts=1000, ent_coef="auto_0.1", train_freq=4, **common),
            "a2c": lambda: A2C("MlpPolicy", train_env, learning_rate=7e-4, n_steps=5, gamma=0.99, **common),
        }[cfg.algo.lower()]()

        cb = MetricsCallback(writer)
        eval_cb = EvalCallback(eval_env, best_model_save_path=str(log_path / "best"), log_path=str(log_path),
                               eval_freq=cfg.eval_freq, n_eval_episodes=cfg.n_eval_episodes, deterministic=True)
        model.learn(cfg.total_timesteps, callback=[cb, eval_cb], progress_bar=True)
        model.save(log_path / "final_model")
        policy = model
        final_metrics = evaluate_policy(model, cfg)

    if writer:
        log_metrics(writer, final_metrics, 'final', cfg.total_timesteps)
        writer.close()

    train_env.close(); eval_env.close()
    with open(log_path / "results.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    return {"path": str(log_path), "metrics": final_metrics}


def _train_alpha(args: tuple) -> tuple[str, Dict]:
    """Worker for parallel sweep - must be top-level for pickling."""
    cfg_dict, alpha = args
    cfg_dict["alpha_true"] = alpha
    cfg_dict["experiment_name"] = f"{cfg_dict['algo']}_a{alpha:.2f}_{cfg_dict['reward_mode']}"
    sweep_cfg = ExperimentConfig(**cfg_dict)
    print(f"[alpha={alpha:.2f}] starting")
    metrics = train(sweep_cfg)["metrics"]
    print(f"[alpha={alpha:.2f}] done")
    return f"alpha_{alpha:.2f}", metrics


def run_sweep(cfg: ExperimentConfig, alphas: List[float] | None = None, max_workers: int | None = None) -> Dict[str, Dict]:
    alphas = alphas or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cfg_dict = asdict(cfg)

    if max_workers == 1:  # sequential fallback
        results = dict(_train_alpha((cfg_dict.copy(), a)) for a in alphas)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_train_alpha, (cfg_dict.copy(), a)): a for a in alphas}
            results = {}
            for fut in as_completed(futures):
                key, metrics = fut.result()
                results[key] = metrics

    summary_path = Path(cfg.log_dir) / f"sweep_{cfg.algo}_{cfg.reward_mode}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep results saved to {summary_path}")
    return results


def _train_policy(args: tuple) -> tuple[str, Dict]:
    """Worker for parallel policy comparison."""
    cfg_dict, algo = args
    cfg_dict["algo"] = algo
    cfg_dict["experiment_name"] = f"cmp_{algo}_a{cfg_dict['alpha_true']:.2f}"
    cmp_cfg = ExperimentConfig(**cfg_dict)
    print(f"[{algo}] starting")
    metrics = train(cmp_cfg)["metrics"]
    print(f"[{algo}] done")
    return algo, metrics


def compare_policies(cfg: ExperimentConfig, policies: List[str] | None = None, max_workers: int | None = None) -> Dict[str, Dict]:
    policies = policies or ["fixed", "adaptive", "myopic", "random"]
    cfg_dict = asdict(cfg)

    if max_workers == 1:
        results = dict(_train_policy((cfg_dict.copy(), p)) for p in policies)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_train_policy, (cfg_dict.copy(), p)): p for p in policies}
            results = {}
            for fut in as_completed(futures):
                algo, metrics = fut.result()
                results[algo] = metrics

    cmp_path = Path(cfg.log_dir) / f"compare_a{cfg.alpha_true:.2f}.json"
    with open(cmp_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison saved to {cmp_path}")
    for algo, m in results.items():
        print(f"  {algo:12s}: reward={m['reward_mean']:.2f} coi_erosion={m['coi_erosion_mean']:.4f} alpha_err={m['alpha_error_mean']:.4f}")
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
    parser.add_argument("--log-dir", default="sim/case/thesis_simplified/runs")
    parser.add_argument("--sweep", action="store_true", help="run contamination sweep")
    parser.add_argument("--compare", action="store_true", help="compare all baselines")
    parser.add_argument("--workers", type=int, default=None, help="max parallel workers for sweep (None=auto, 1=sequential)")
    args = parser.parse_args()

    cfg = ExperimentConfig(algo=args.algo, total_timesteps=args.steps, alpha_true=args.alpha,
                           reward_mode=args.reward_mode, n_products=args.n_products,
                           n_envs=args.n_envs, seed=args.seed, log_dir=args.log_dir)

    if args.sweep:
        run_sweep(cfg, max_workers=args.workers)
    elif args.compare:
        compare_policies(cfg, max_workers=args.workers)
    else:
        result = train(cfg)
        print(f"\nTraining complete: {result['path']}")
        print(f"Metrics: {json.dumps(result['metrics'], indent=2)}")


if __name__ == "__main__":
    main()
