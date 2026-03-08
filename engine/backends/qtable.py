from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .common import evaluate, make_env


def train_qtable(cfg: Mapping[str, Any]) -> tuple[object, dict[str, float | int]]:
    from ..lib.discrete import EventQTable

    np.random.seed(int(cfg["seed"]))
    env = make_env(cfg)
    eval_env = make_env(cfg)
    agent = EventQTable(
        env.action_space.n,
        int(cfg["n_products"]),
        (float(cfg["price_low"]), float(cfg["price_high"])),
        lr=float(cfg["q_lr"]),
        gamma=float(cfg["gamma"]),
        n_bins=int(cfg["q_bins"]),
    )

    total_reward = 0.0
    total_revenue = 0.0
    steps = 0
    epsilon = float(cfg["eps_start"])
    obs, _ = env.reset(seed=int(cfg["seed"]))

    for _ in range(int(cfg["total_timesteps"])):
        action, state = agent.act(obs, epsilon)
        nxt, reward, term, trunc, info = env.step(action)
        done = bool(term or trunc)
        agent.update(state, action, float(reward), agent.encode(nxt), done)

        total_reward += float(reward)
        total_revenue += float(info.get("economics", {}).get("revenue", 0.0))
        steps += 1
        epsilon = max(float(cfg["eps_end"]), epsilon * float(cfg["eps_decay"]))
        obs = env.reset()[0] if done else nxt

    metrics: dict[str, float | int] = {
        "train/reward_mean": total_reward / max(steps, 1),
        "train/revenue_mean": total_revenue / max(steps, 1),
        "train/epsilon": float(epsilon),
        "train/global_step": int(cfg["total_timesteps"]),
    }
    metrics.update(evaluate(agent, eval_env, int(cfg["eval_episodes"])))

    env.close()
    eval_env.close()
    return agent, metrics
