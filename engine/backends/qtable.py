from __future__ import annotations

import logging
import time
from typing import Any, Mapping

import numpy as np

from .common import evaluate, make_env
from ..telemetry.wandb import get_wandb_module

logger = logging.getLogger(__name__)


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
    console_progress = bool(cfg.get("console_progress", False))
    obs, _ = env.reset(seed=int(cfg["seed"]))
    started_at = time.perf_counter()
    wandb = get_wandb_module()
    wandb_live = bool(wandb is not None and wandb.run is not None)
    step_offset = max(0, int(cfg.get("wandb_step_offset", 0)))

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
            event = {
                "train/reward_mean": interval_sums["reward"] / denom,
                "train/revenue_mean": interval_sums["revenue"] / denom,
                "train/agent_prob": interval_sums["agent_prob"] / denom,
                "train/alpha_adv": interval_sums["alpha_adv"] / denom,
                "train/coi_leakage": interval_sums["coi_leakage"] / denom,
                "train/epsilon": float(epsilon),
                "train/global_step": int(steps),
            }
            if wandb_live:
                try:
                    wandb.log(dict(event), step=step_offset + int(steps))
                except Exception:
                    wandb_live = False
                    train_events.append(event)
            else:
                train_events.append(event)
            if console_progress:
                elapsed = max(time.perf_counter() - started_at, 1e-6)
                speed = steps / elapsed
                logger.info(
                    "step=%d/%d reward=%.3f revenue=%.3f eps=%.4f speed=%.1f steps/s",
                    steps,
                    int(cfg["total_timesteps"]),
                    event["train/reward_mean"],
                    event["train/revenue_mean"],
                    event["train/epsilon"],
                    speed,
                )
            interval_sums = {key: 0.0 for key in interval_sums}
            interval_count = 0

        epsilon = max(float(cfg["eps_end"]), epsilon * float(cfg["eps_decay"]))
        obs = env.reset()[0] if done else nxt

    if interval_count > 0:
        denom = float(interval_count)
        tail_event = {
            "train/reward_mean": interval_sums["reward"] / denom,
            "train/revenue_mean": interval_sums["revenue"] / denom,
            "train/agent_prob": interval_sums["agent_prob"] / denom,
            "train/alpha_adv": interval_sums["alpha_adv"] / denom,
            "train/coi_leakage": interval_sums["coi_leakage"] / denom,
            "train/epsilon": float(epsilon),
            "train/global_step": int(steps),
        }
        if wandb_live:
            try:
                wandb.log(dict(tail_event), step=step_offset + int(steps))
            except Exception:
                wandb_live = False
                train_events.append(tail_event)
        else:
            train_events.append(tail_event)

    metrics: dict[str, Any] = {
        "train/reward_mean": total_reward / max(steps, 1),
        "train/revenue_mean": total_revenue / max(steps, 1),
        "train/epsilon": float(epsilon),
        "train/global_step": int(cfg["total_timesteps"]),
    }
    metrics.update(evaluate(agent, eval_env, int(cfg["eval_episodes"]), cfg=cfg))
    metrics["_train_events"] = train_events

    env.close()
    eval_env.close()
    return agent, metrics
