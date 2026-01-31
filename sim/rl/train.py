import numpy as np
import logging
from pathlib import Path
from typing import Dict, Type, Optional
import pickle
from torch.utils.tensorboard import SummaryWriter
from sim.rl.environment import PHANTOMEnv, BusinessLogicConstraints

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    from sim.rl.engine import (BasePricingEngine, WildPricingEngine, StaticPricingEngine,
                       SimpleDemandEngine, RandomWalkEngine, ThompsonSamplingEngine)
except ImportError as e:
    BasePricingEngine = None  # engines not required for basic usage
    print(e)


"""
Target training loop:
have base prices p0 from env reset and run the env step, collect reward and metrics
pass this to the pricing engine which computes the price action to take based on previous reward by learning
the new action gets passed to the step
so we alternate, step -> reward -> engine (produces price delta) -> step with price delta -> reward
to make sure the reinforcement learning inside the engine can learn we need to have trajectory of prices
CURRENT SOLUTION BELOW does not implement correct learning or updates.
"""

class EngineTrainer:
    """wrapper to run pricing engines through episodes and collect metrics"""
    def __init__(self, engine, env: PHANTOMEnv, tb_writer: Optional[SummaryWriter] = None):
        self.engine = engine
        self.env = env
        self.episode_metrics = []
        self.tb_writer = tb_writer
        self.global_step = 0

    def train(self, n_episodes: int, seed: int = 42):
        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=seed + ep)
            self.engine.reset()
            done = False
            prev_prices = obs["elasticity"]["price"]
            episode_reward = 0.0
            last_info: Dict[str, float] = {}
            while not done:
                action_prices = self.engine.compute_prices(prev_prices, obs)
                obs, reward, done, _, info = self.env.step(action_prices)
                self.engine.update(obs, reward, done, info)
                episode_reward += reward
                prev_prices = obs["elasticity"]["price"]
                last_info = info
                if self.tb_writer:
                    self.tb_writer.add_scalar("reward/step", reward, self.global_step)
                    if "coi" in info:
                        self.tb_writer.add_scalar("diagnostics/coi", info["coi"], self.global_step)
                    if "alpha_hat" in info:
                        self.tb_writer.add_scalar("diagnostics/alpha_hat", info["alpha_hat"], self.global_step)
                self.global_step += 1
            last_info = dict(last_info)
            last_info.update({"episode_reward": episode_reward, "episode": ep})
            self.episode_metrics.append(last_info)
            if self.tb_writer:
                self.tb_writer.add_scalar("reward/episode", episode_reward, ep)
        return self

    def run_episode(self, seed: int = 42) -> Dict:
        """run single evaluation episode and return metrics"""
        obs, _ = self.env.reset(seed=seed)
        self.engine.reset()
        total_reward = 0.0
        prev_prices = obs["elasticity"]["price"]
        ep_metrics = {'total_reward': 0.0}
        done = False
        while not done:
            action_prices = self.engine.compute_prices(prev_prices, obs)
            obs, reward, done, _, info = self.env.step(action_prices)
            total_reward += reward
            for k, v in info.items():
                ep_metrics[k] = v
            prev_prices = obs["elasticity"]["price"]
        ep_metrics['total_reward'] = total_reward
        return ep_metrics

    def evaluate(self, n_episodes: int = 10, seed: int = 100) -> Dict:
        """evaluate trained engine"""
        results = {k: [] for k in ['total_reward', 'revenue_observed', 'revenue_oracle',
                                   'agent_loss', 'ux_volatility', 'look_to_book']}
        for ep in range(n_episodes):
            metrics = self.run_episode(seed=seed + ep)
            for k in results:
                results[k].append(metrics.get(k, 0.0))
        return {k: (np.mean(v), np.std(v)) for k, v in results.items()}


def make_env():
    return PHANTOMEnv(constraints=BusinessLogicConstraints())


def train_engine(engine_cls, env: PHANTOMEnv, n_episodes: int, seed: int = 42,
                tb_writer: Optional[SummaryWriter] = None) -> EngineTrainer:
    constraints = env.constraints
    engine = engine_cls(constraints=constraints, seed=seed)
    trainer = EngineTrainer(engine, env, tb_writer=tb_writer)
    trainer.train(n_episodes, seed=seed)
    return trainer


def save_trainer(trainer: EngineTrainer, path: Path):
    """save engine state and metrics"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'engine': trainer.engine, 'metrics': trainer.episode_metrics}, f)
    logger.info(f"Saved trainer to {path}")


def load_trainer(path: Path, env: PHANTOMEnv, tb_writer: Optional[SummaryWriter] = None) -> EngineTrainer:
    """load saved engine"""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    trainer = EngineTrainer(data['engine'], env, tb_writer=tb_writer)
    trainer.episode_metrics = data['metrics']
    return trainer


if __name__ == "__main__":
    if BasePricingEngine is None:
        logger.error("Engines not available, cannot run training")
        exit(1)

    base_dir = Path("./sim/rl/runs")
    base_dir.mkdir(exist_ok=True)

    engines = {
        "Wild": WildPricingEngine,
        "Static": StaticPricingEngine,
        "RandomWalk": RandomWalkEngine,
        "ThompsonSampling": ThompsonSamplingEngine,
    }
    n_train_episodes = 50
    n_eval_episodes = 10
    seed = 42

    logger.info(f"Training config: {n_train_episodes} episodes per engine")

    trained_trainers = {}

    for engine_name, engine_cls in engines.items():
        run_name = engine_name
        log_dir = base_dir / run_name
        log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training {engine_name}")
        logger.info(f"Log directory: {log_dir}")

        env = make_env()
        tb_writer = SummaryWriter(log_dir=str(log_dir))
        trainer = train_engine(engine_cls, env, n_train_episodes, seed, tb_writer=tb_writer)
        tb_writer.close()

        save_path = log_dir / "trainer.pkl"
        save_trainer(trainer, save_path)

        trained_trainers[run_name] = (trainer, env)

    logger.info("Starting evaluation")

    for run_name, (trainer, env) in trained_trainers.items():
        logger.info(f"Evaluating {run_name}")
        results = trainer.evaluate(n_episodes=n_eval_episodes, seed=seed + 1000)
        for metric, (mean, std) in results.items():
            logger.info(f"  {metric:20s}: {mean:10.2f} ± {std:6.2f}")

    logger.info(f"Results saved to: {base_dir}")
