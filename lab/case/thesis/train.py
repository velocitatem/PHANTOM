"""RL training for thesis pricing system with COI tracking.

Trains pricing policies using stable-baselines3 with TensorBoard logging.
Demonstrates COI leakage under different contamination levels and policies.

Usage:
    python -m lab.case.thesis.train --algo ppo --alpha 0.3 --steps 100000
    tensorboard --logdir lab/case/thesis/runs
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Callable, Any
import numpy as np

try:
    from stable_baselines3 import PPO, SAC, A2C
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure
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


class BaselinePolicy:
    """Wrapper to make baseline policies compatible with SB3 interface."""

    def __init__(self, policy_fn, name: str):
        self.policy_fn = policy_fn
        self.name = name
        self.num_timesteps = 0

    def predict(self, obs, deterministic: bool = True):
        n = (len(obs) - 3) // 3  # infer n_products from obs shape
        action = self.policy_fn(obs, n)
        return action, None

    def learn(self, total_timesteps: int, callback=None, progress_bar: bool = False):
        self.num_timesteps = total_timesteps
        return self

    def save(self, path):
        pass  # no-op for baselines

    @staticmethod
    def load(path):
        raise NotImplementedError("baselines cannot be loaded")


def myopic_policy(obs: np.ndarray, n: int, greed: float = 0.3) -> np.ndarray:
    """Myopic: maximize immediate margin, ignore alpha and future COI erosion.

    Greedy short-term optimizer that sets high prices when demand looks good,
    completely ignoring the alpha estimate and long-term consequences.
    """
    demand_norm = obs[n:2*n] if len(obs) > 2*n else np.ones(n) * 0.5
    avg_demand = np.mean(demand_norm)
    multiplier = 1.0 + greed * (1 + avg_demand)
    return np.ones(n, dtype=np.float32) * np.clip(multiplier, 0.5, 1.5)


def random_myopic_policy(obs: np.ndarray, n: int) -> np.ndarray:
    """Random myopic: iid random prices each step, no state awareness.

    Represents worst-case baseline where pricing has no strategy at all.
    """
    return np.random.uniform(0.8, 1.4, n).astype(np.float32)


@dataclass
class TrainConfig:
    """Training configuration."""
    algo: str = "ppo"  # ppo | sac | a2c
    total_timesteps: int = 100_000
    n_envs: int = 4
    eval_freq: int = 5000
    n_eval_episodes: int = 10
    log_dir: str = "lab/case/thesis/runs"
    seed: int = 42
    # env config
    n_products: int = 10
    max_steps: int = 200
    alpha_true: float = 0.2
    reward_mode: str = "robust"
    # baseline sweep
    run_baselines: bool = True
    alpha_sweep: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])


class COICallback(BaseCallback):
    """Custom callback for tracking COI metrics in TensorBoard."""

    def __init__(self, writer: Any = None, verbose: int = 0):
        super().__init__(verbose)
        self._writer = writer
        self._episode_coi_leak = []
        self._episode_alpha_err = []
        self._episode_revenues = []
        self._episode_margins = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'alpha_true' in info and 'alpha_est' in info:
                self._episode_alpha_err.append(abs(info['alpha_true'] - info['alpha_est']))
            if 'coi_erosion' in info:
                self._episode_coi_leak.append(info['coi_erosion'])
            if 'revenue' in info:
                self._episode_revenues.append(info['revenue'])
            if 'avg_margin' in info:
                self._episode_margins.append(info['avg_margin'])
        return True

    def _on_rollout_end(self) -> None:
        if self._writer is None:
            return
        step = self.num_timesteps
        if self._episode_coi_leak:
            self._writer.add_scalar('coi/erosion_mean', np.mean(self._episode_coi_leak), step)
            self._writer.add_scalar('coi/erosion_max', np.max(self._episode_coi_leak), step)
        if self._episode_alpha_err:
            self._writer.add_scalar('alpha/estimation_error', np.mean(self._episode_alpha_err), step)
        if self._episode_revenues:
            self._writer.add_scalar('economics/revenue_mean', np.mean(self._episode_revenues), step)
        if self._episode_margins:
            self._writer.add_scalar('economics/margin_mean', np.mean(self._episode_margins), step)
        self._episode_coi_leak.clear()
        self._episode_alpha_err.clear()
        self._episode_revenues.clear()
        self._episode_margins.clear()


def run_baseline_with_logging(model: BaselinePolicy, vec_env, total_timesteps: int, writer: Any) -> None:
    """Run baseline policy and log metrics identically to RL training."""
    n_envs = vec_env.num_envs
    obs = vec_env.reset()
    step = 0
    episode_rewards, episode_coi, episode_alpha_err, episode_revenues = [], [], [], []
    ep_rewards = np.zeros(n_envs)

    while step < total_timesteps:
        actions = np.array([model.predict(obs[i])[0] for i in range(n_envs)])
        obs, rewards, dones, infos = vec_env.step(actions)
        step += n_envs
        ep_rewards += rewards

        for i, info in enumerate(infos):
            if 'coi_erosion' in info:
                episode_coi.append(info['coi_erosion'])
            if 'alpha_true' in info and 'alpha_est' in info:
                episode_alpha_err.append(abs(info['alpha_true'] - info['alpha_est']))
            if 'revenue' in info:
                episode_revenues.append(info['revenue'])
            if dones[i]:
                episode_rewards.append(ep_rewards[i])
                ep_rewards[i] = 0.0

        if writer and len(episode_rewards) >= 5 and step % 1000 < n_envs:
            writer.add_scalar('rollout/ep_rew_mean', np.mean(episode_rewards[-10:]), step)
            if episode_coi:
                writer.add_scalar('coi/erosion_mean', np.mean(episode_coi[-100:]), step)
            if episode_alpha_err:
                writer.add_scalar('alpha/estimation_error', np.mean(episode_alpha_err[-100:]), step)
            if episode_revenues:
                writer.add_scalar('economics/revenue_mean', np.mean(episode_revenues[-100:]), step)

        if step % 10000 < n_envs:
            print(f"  step {step}/{total_timesteps}, avg_reward={np.mean(episode_rewards[-20:]) if episode_rewards else 0:.2f}")


def make_vec_env(cfg: TrainConfig, n_envs: int = 1) -> DummyVecEnv:
    """Create vectorized environment."""
    def _make():
        env_cfg = EnvConfig(
            n_products=cfg.n_products, max_steps=cfg.max_steps,
            alpha_true=cfg.alpha_true, reward_mode=cfg.reward_mode, seed=cfg.seed)
        env = make_env(env_cfg)
        return Monitor(env)
    return DummyVecEnv([_make for _ in range(n_envs)])


def run_baseline(
    policy_fn: Callable[[np.ndarray, int], np.ndarray],
    env: PricingEnv,
    n_episodes: int = 20,
    name: str = "baseline"
) -> Dict[str, float]:
    """Evaluate baseline policy and collect metrics."""
    episode_rewards, episode_coi, episode_alpha_err = [], [], []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done, ep_reward, ep_coi, ep_alpha_err = False, 0.0, [], []

        while not done:
            action = policy_fn(obs, env.n)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            if 'coi_erosion' in info:
                ep_coi.append(info['coi_erosion'])
            if 'alpha_true' in info and 'alpha_est' in info:
                ep_alpha_err.append(abs(info['alpha_true'] - info['alpha_est']))

        episode_rewards.append(ep_reward)
        if ep_coi:
            episode_coi.append(np.mean(ep_coi))
        if ep_alpha_err:
            episode_alpha_err.append(np.mean(ep_alpha_err))

    return {
        f'{name}/reward_mean': np.mean(episode_rewards),
        f'{name}/reward_std': np.std(episode_rewards),
        f'{name}/coi_erosion': np.mean(episode_coi) if episode_coi else 0.0,
        f'{name}/alpha_error': np.mean(episode_alpha_err) if episode_alpha_err else 0.0,
    }


def run_coi_demonstration(writer: Any, cfg: TrainConfig) -> Dict[str, Dict[str, float]]:
    """Demonstrate COI leakage across contamination levels."""
    results = {}

    for alpha in cfg.alpha_sweep:
        env_cfg = EnvConfig(
            n_products=cfg.n_products, max_steps=cfg.max_steps,
            alpha_true=alpha, reward_mode=cfg.reward_mode, seed=cfg.seed)
        env = make_env(env_cfg)

        # run fixed policy
        fixed_metrics = run_baseline(
            lambda obs, n: fixed_price_policy(np.ones(n), margin=0.15),
            env, n_episodes=10, name=f"fixed_alpha{alpha:.1f}")

        # run adaptive policy
        adaptive_metrics = run_baseline(
            lambda obs, n: adaptive_policy(obs, n, base_margin=0.15),
            env, n_episodes=10, name=f"adaptive_alpha{alpha:.1f}")

        # theoretical erosion
        n_agents = int(alpha * cfg.max_steps * 30)  # rough agent count
        theo_erosion = coi_erosion(max(1, n_agents), price_std=5.0)

        results[f'alpha_{alpha:.1f}'] = {
            'fixed_reward': fixed_metrics[f"fixed_alpha{alpha:.1f}/reward_mean"],
            'adaptive_reward': adaptive_metrics[f"adaptive_alpha{alpha:.1f}/reward_mean"],
            'fixed_coi': fixed_metrics[f"fixed_alpha{alpha:.1f}/coi_erosion"],
            'adaptive_coi': adaptive_metrics[f"adaptive_alpha{alpha:.1f}/coi_erosion"],
            'theoretical_erosion': theo_erosion,
        }

        if writer:
            writer.add_scalar(f'baseline/fixed_reward', fixed_metrics[f"fixed_alpha{alpha:.1f}/reward_mean"], int(alpha * 100))
            writer.add_scalar(f'baseline/adaptive_reward', adaptive_metrics[f"adaptive_alpha{alpha:.1f}/reward_mean"], int(alpha * 100))
            writer.add_scalar(f'baseline/coi_erosion_fixed', fixed_metrics[f"fixed_alpha{alpha:.1f}/coi_erosion"], int(alpha * 100))
            writer.add_scalar(f'baseline/coi_erosion_adaptive', adaptive_metrics[f"adaptive_alpha{alpha:.1f}/coi_erosion"], int(alpha * 100))
            writer.add_scalar(f'baseline/theoretical_erosion', theo_erosion, int(alpha * 100))

    return results


def train_rl(cfg: TrainConfig) -> Dict[str, Any]:
    """Train RL agent or baseline policy with TensorBoard logging."""
    is_baseline = cfg.algo.lower() in ["myopic", "random_myopic", "fixed", "adaptive"]
    if not HAS_SB3 and not is_baseline:
        raise ImportError("stable-baselines3 required: pip install stable-baselines3[extra]")

    log_path = Path(cfg.log_dir) / f"{cfg.algo}_alpha{cfg.alpha_true:.1f}_{cfg.reward_mode}"
    log_path.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_path) if HAS_TB else None

    # baseline demonstration
    if False and cfg.run_baselines:
        print("Running baseline demonstrations...")
        baseline_results = run_coi_demonstration(writer, cfg)
        for k, v in baseline_results.items():
            print(f"  {k}: reward_fixed={v['fixed_reward']:.2f}, reward_adapt={v['adaptive_reward']:.2f}, "
                  f"coi_fixed={v['fixed_coi']:.3f}, coi_adapt={v['adaptive_coi']:.3f}, theo={v['theoretical_erosion']:.3f}")

    # create envs
    train_env = make_vec_env(cfg, n_envs=cfg.n_envs)
    eval_env = make_vec_env(cfg, n_envs=1)

    # select algorithm
    algo_name = cfg.algo.lower()

    if is_baseline:
        # baseline policies wrapped for compatibility
        policy_map = {
            "myopic": lambda obs, n: myopic_policy(obs, n, greed=0.3),
            "random_myopic": random_myopic_policy,
            "fixed": lambda obs, n: fixed_price_policy(np.ones(n), margin=0.15),
            "adaptive": lambda obs, n: adaptive_policy(obs, n, base_margin=0.15),
        }
        model = BaselinePolicy(policy_map[algo_name], algo_name)
    else:
        if not HAS_SB3:
            raise ImportError("stable-baselines3 required for RL algos")

        algo_cls = {"ppo": PPO, "sac": SAC, "a2c": A2C}.get(algo_name)
        if algo_cls is None:
            raise ValueError(f"unknown algo: {cfg.algo}")

        common_kwargs = dict(
            verbose=1, seed=cfg.seed, tensorboard_log=str(log_path),
            device="auto"
        )
        if algo_name == "ppo":
            model = PPO(
                "MlpPolicy", train_env, learning_rate=3e-4, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, **common_kwargs)
        elif algo_name == "sac":
            model = SAC(
                "MlpPolicy", train_env, learning_rate=3e-4, buffer_size=100_000,
                batch_size=256, tau=0.005, gamma=0.99, train_freq=1,
                gradient_steps=1, ent_coef="auto", **common_kwargs)
        else:
            model = A2C(
                "MlpPolicy", train_env, learning_rate=7e-4, n_steps=5,
                gamma=0.99, gae_lambda=1.0, ent_coef=0.01, **common_kwargs)

    print(f"\nRunning {cfg.algo.upper()} for {cfg.total_timesteps} steps...")
    print(f"  alpha_true={cfg.alpha_true}, reward_mode={cfg.reward_mode}")
    print(f"  logs: {log_path}")

    if is_baseline:
        # run baseline through env manually with logging
        run_baseline_with_logging(model, train_env, cfg.total_timesteps, writer)
    else:
        coi_cb = COICallback(writer=writer, verbose=1)
        eval_cb = EvalCallback(
            eval_env, best_model_save_path=str(log_path / "best"),
            log_path=str(log_path), eval_freq=cfg.eval_freq,
            n_eval_episodes=cfg.n_eval_episodes, deterministic=True)
        model.learn(total_timesteps=cfg.total_timesteps, callback=[coi_cb, eval_cb], progress_bar=True)
        model.save(log_path / "final_model")

    # final evaluation
    final_metrics = evaluate_trained_model(model, cfg)
    if writer:
        for k, v in final_metrics.items():
            writer.add_scalar(f'final/{k}', v, cfg.total_timesteps)
        writer.close()

    train_env.close()
    eval_env.close()

    return {"model_path": str(log_path / "final_model"), "metrics": final_metrics}


def evaluate_trained_model(model: Any, cfg: TrainConfig, n_episodes: int = 20) -> Dict[str, float]:
    """Evaluate trained model."""
    env_cfg = EnvConfig(
        n_products=cfg.n_products, max_steps=cfg.max_steps,
        alpha_true=cfg.alpha_true, reward_mode=cfg.reward_mode, seed=cfg.seed + 1000)
    env = make_env(env_cfg)

    episode_rewards, episode_coi = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, ep_reward, ep_coi = False, 0.0, []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            if 'coi_erosion' in info:
                ep_coi.append(info['coi_erosion'])
        episode_rewards.append(ep_reward)
        if ep_coi:
            episode_coi.append(np.mean(ep_coi))

    return {
        'reward_mean': np.mean(episode_rewards),
        'reward_std': np.std(episode_rewards),
        'coi_erosion_mean': np.mean(episode_coi) if episode_coi else 0.0,
    }


def compare_policies(cfg: TrainConfig, model_paths: List[str] = None) -> None:
    """Compare trained models against baselines."""
    if model_paths and not HAS_SB3:
        raise ImportError("stable-baselines3 required for loading trained models")

    writer = SummaryWriter(Path(cfg.log_dir) / "comparison") if HAS_TB else None

    env_cfg = EnvConfig(
        n_products=cfg.n_products, max_steps=cfg.max_steps,
        alpha_true=cfg.alpha_true, reward_mode=cfg.reward_mode, seed=cfg.seed)
    env = make_env(env_cfg)

    results = {}

    # all baseline policies
    baselines = {
        'random': lambda obs, n: random_policy(n),
        'fixed': lambda obs, n: fixed_price_policy(np.ones(n), 0.15),
        'adaptive': lambda obs, n: adaptive_policy(obs, n, 0.15),
        'myopic': lambda obs, n: myopic_policy(obs, n, 0.3),
        'random_myopic': random_myopic_policy,
    }
    for name, policy_fn in baselines.items():
        results[name] = run_baseline(policy_fn, env, n_episodes=20, name=name)

    # trained models
    if model_paths:
        for path in model_paths:
            name = Path(path).parent.name
            model = PPO.load(path)  # assume PPO, could detect
            metrics = evaluate_trained_model(model, cfg)
            results[name] = {f'{name}/{k}': v for k, v in metrics.items()}

    print("\n=== Policy Comparison ===")
    for name, metrics in results.items():
        reward_key = [k for k in metrics if 'reward_mean' in k][0]
        coi_key = [k for k in metrics if 'coi' in k][0] if any('coi' in k for k in metrics) else None
        print(f"{name:20s}: reward={metrics[reward_key]:.2f}", end="")
        if coi_key:
            print(f", coi={metrics[coi_key]:.3f}")
        else:
            print()

        if writer:
            for k, v in metrics.items():
                writer.add_scalar(f'comparison/{k}', v, 0)

    if writer:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train RL pricing policies")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "sac", "a2c", "myopic", "random_myopic", "fixed", "adaptive"])
    parser.add_argument("--steps", type=int, default=100_000, help="total training steps")
    parser.add_argument("--alpha", type=float, default=0.2, help="true contamination level")
    parser.add_argument("--reward-mode", type=str, default="robust", choices=["revenue", "profit", "robust", "coi_aware"])
    parser.add_argument("--n-products", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="lab/case/thesis/runs")
    parser.add_argument("--no-baselines", action="store_true", help="skip baseline runs")
    parser.add_argument("--compare", nargs="*", help="compare model paths")
    args = parser.parse_args()

    cfg = TrainConfig(
        algo=args.algo, total_timesteps=args.steps, alpha_true=args.alpha,
        reward_mode=args.reward_mode, n_products=args.n_products,
        n_envs=args.n_envs, seed=args.seed, log_dir=args.log_dir,
        run_baselines=not args.no_baselines)

    if args.compare is not None:
        compare_policies(cfg, args.compare if args.compare else None)
    else:
        result = train_rl(cfg)
        print(f"\nTraining complete. Model saved to: {result['model_path']}")
        print(f"Final metrics: {result['metrics']}")


if __name__ == "__main__":
    main()
