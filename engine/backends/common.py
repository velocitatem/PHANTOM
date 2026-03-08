from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def make_env(cfg: Mapping[str, Any]):
    from gymnasium.wrappers import FlattenObservation

    from ..lib.wrappers import EconomicMetricsWrapper
    from ..wrapper import PHANTOM

    env = PHANTOM(
        n_products=int(cfg["n_products"]),
        alpha=float(cfg["alpha"]),
        N=int(cfg["N"]),
        price_bounds=(float(cfg["price_low"]), float(cfg["price_high"])),
        lambda_coi=float(cfg["lambda_coi"]),
        robust_radius=float(cfg["robust_radius"]),
        robust_points=int(cfg["robust_points"]),
        info_value=float(cfg["info_value"]),
        action_levels=int(cfg["action_levels"]),
        action_scale_low=float(cfg["action_scale_low"]),
        action_scale_high=float(cfg["action_scale_high"]),
        max_steps=int(cfg.get("max_steps", 100)),
        margin_floor=float(cfg.get("margin_floor", 0.05)),
        margin_floor_patience=int(cfg.get("margin_floor_patience", 5)),
        render_mode=None,
    )
    env = EconomicMetricsWrapper(env)
    return FlattenObservation(env)


def _action(agent: Any, obs: Any, deterministic: bool = True):
    out = agent.predict(obs, deterministic=deterministic)
    action = out[0] if isinstance(out, tuple) else out
    if isinstance(action, np.ndarray) and action.size == 1:
        return int(action.reshape(-1)[0])
    return action


def evaluate(agent: Any, env: Any, episodes: int) -> dict[str, float]:
    rewards: list[float] = []
    revenues: list[float] = []
    margins: list[float] = []
    coi_levels: list[float] = []

    for _ in range(int(episodes)):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_revenue = 0.0
        ep_margin = 0.0
        ep_coi = 0.0
        steps = 0

        while not done:
            obs, reward, term, trunc, info = env.step(_action(agent, obs, True))
            done = bool(term or trunc)
            econ = info.get("economics", {})
            ep_reward += float(reward)
            ep_revenue += float(econ.get("revenue", info.get("revenue", 0.0)))
            ep_margin += float(econ.get("margin", 0.0))
            ep_coi += float(econ.get("coi_level", 0.0))
            steps += 1

        rewards.append(ep_reward)
        revenues.append(ep_revenue)
        denom = max(steps, 1)
        margins.append(ep_margin / denom)
        coi_levels.append(ep_coi / denom)

    return {
        "eval/reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "eval/reward_std": float(np.std(rewards)) if rewards else 0.0,
        "eval/revenue_mean": float(np.mean(revenues)) if revenues else 0.0,
        "eval/revenue_std": float(np.std(revenues)) if revenues else 0.0,
        "eval/margin_mean": float(np.mean(margins)) if margins else 0.0,
        "eval/coi_level_mean": float(np.mean(coi_levels)) if coi_levels else 0.0,
    }
