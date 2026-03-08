from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .common import evaluate, make_env


def train_qtable(
    cfg: Mapping[str, Any],
) -> tuple[object, dict[str, Any]]:
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
    log_freq = max(1, int(cfg.get("log_freq", 100)))
    obs, _ = env.reset(seed=int(cfg["seed"]))

    interval_sums = {
        "reward": 0.0,
        "revenue": 0.0,
        "agent_prob": 0.0,
        "alpha_adv": 0.0,
        "coi_leakage": 0.0,
    }
    interval_count = 0
    train_events: list[dict[str, float | int]] = []

    for _ in range(int(cfg["total_timesteps"])):
        action, state = agent.act(obs, epsilon)
        nxt, reward, term, trunc, info = env.step(action)
        done = bool(term or trunc)
        agent.update(state, action, float(reward), agent.encode(nxt), done)

        total_reward += float(reward)
        revenue = float(info.get("economics", {}).get("revenue", 0.0))
        total_revenue += revenue
        steps += 1
        interval_sums["reward"] += float(reward)
        interval_sums["revenue"] += revenue
        interval_sums["agent_prob"] += float(info.get("agent_prob", 0.0))
        interval_sums["alpha_adv"] += float(info.get("alpha_adv", 0.0))
        interval_sums["coi_leakage"] += float(info.get("coi_leakage", 0.0))
        interval_count += 1

        if steps % log_freq == 0 and interval_count > 0:
            denom = float(interval_count)
            train_events.append(
                {
                    "train/reward_mean": interval_sums["reward"] / denom,
                    "train/revenue_mean": interval_sums["revenue"] / denom,
                    "train/agent_prob": interval_sums["agent_prob"] / denom,
                    "train/alpha_adv": interval_sums["alpha_adv"] / denom,
                    "train/coi_leakage": interval_sums["coi_leakage"] / denom,
                    "train/epsilon": float(epsilon),
                    "train/global_step": int(steps),
                }
            )
            interval_sums = {key: 0.0 for key in interval_sums}
            interval_count = 0

        epsilon = max(float(cfg["eps_end"]), epsilon * float(cfg["eps_decay"]))
        obs = env.reset()[0] if done else nxt

    if interval_count > 0:
        denom = float(interval_count)
        train_events.append(
            {
                "train/reward_mean": interval_sums["reward"] / denom,
                "train/revenue_mean": interval_sums["revenue"] / denom,
                "train/agent_prob": interval_sums["agent_prob"] / denom,
                "train/alpha_adv": interval_sums["alpha_adv"] / denom,
                "train/coi_leakage": interval_sums["coi_leakage"] / denom,
                "train/epsilon": float(epsilon),
                "train/global_step": int(steps),
            }
        )

    metrics: dict[str, Any] = {
        "train/reward_mean": total_reward / max(steps, 1),
        "train/revenue_mean": total_revenue / max(steps, 1),
        "train/epsilon": float(epsilon),
        "train/global_step": int(cfg["total_timesteps"]),
    }
    metrics.update(evaluate(agent, eval_env, int(cfg["eval_episodes"])))
    metrics["_train_events"] = train_events

    env.close()
    eval_env.close()
    return agent, metrics
