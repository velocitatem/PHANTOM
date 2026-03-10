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
        robust_rollouts=int(cfg.get("robust_rollouts", 1)),
        info_value=float(cfg["info_value"]),
        eta_ux=float(cfg.get("eta_ux", 0.5)),
        reward_profit_weight=float(cfg.get("reward_profit_weight", 1.0)),
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


def _evaluate_env(agent: Any, env: Any, episodes: int) -> dict[str, float]:
    rewards: list[float] = []
    revenues: list[float] = []
    margins: list[float] = []
    coi_levels: list[float] = []
    coi_leakages: list[float] = []
    volatilities: list[float] = []
    agent_probs: list[float] = []

    for _ in range(int(episodes)):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_revenue = 0.0
        ep_margin = 0.0
        ep_coi = 0.0
        ep_coi_leakage = 0.0
        ep_volatility = 0.0
        ep_agent_prob = 0.0
        steps = 0

        while not done:
            obs, reward, term, trunc, info = env.step(_action(agent, obs, True))
            done = bool(term or trunc)
            econ = info.get("economics", {})
            ep_reward += float(reward)
            ep_revenue += float(econ.get("revenue", info.get("revenue", 0.0)))
            ep_margin += float(econ.get("margin", 0.0))
            ep_coi += float(econ.get("coi_level", 0.0))
            ep_coi_leakage += float(econ.get("coi_leakage", 0.0))
            ep_volatility += float(econ.get("volatility", 0.0))
            ep_agent_prob += float(econ.get("agent_prob", info.get("agent_prob", 0.0)))
            steps += 1

        rewards.append(ep_reward)
        revenues.append(ep_revenue)
        denom = max(steps, 1)
        margins.append(ep_margin / denom)
        coi_levels.append(ep_coi / denom)
        coi_leakages.append(ep_coi_leakage / denom)
        volatilities.append(ep_volatility / denom)
        agent_probs.append(ep_agent_prob / denom)

    return {
        "eval/reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "eval/reward_std": float(np.std(rewards)) if rewards else 0.0,
        "eval/revenue_mean": float(np.mean(revenues)) if revenues else 0.0,
        "eval/revenue_std": float(np.std(revenues)) if revenues else 0.0,
        "eval/margin_mean": float(np.mean(margins)) if margins else 0.0,
        "eval/coi_level_mean": float(np.mean(coi_levels)) if coi_levels else 0.0,
        "eval/coi_leakage_mean": float(np.mean(coi_leakages)) if coi_leakages else 0.0,
        "eval/volatility_mean": float(np.mean(volatilities)) if volatilities else 0.0,
        "eval/agent_prob_mean": float(np.mean(agent_probs)) if agent_probs else 0.0,
    }


def evaluate(
    agent: Any,
    env: Any,
    episodes: int,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    metrics = _evaluate_env(agent, env, episodes)
    if cfg is None or not bool(cfg.get("robust_eval_enabled", True)):
        return metrics

    nominal_alpha = float(cfg.get("alpha", 0.0))
    eval_radius = max(float(cfg.get("robust_radius", 0.0)), 0.15)
    low_alpha = float(np.clip(nominal_alpha - eval_radius, 0.0, 1.0))
    high_alpha = float(np.clip(nominal_alpha + eval_radius, 0.0, 1.0))
    shifted_episodes = max(1, int(np.ceil(int(episodes) / 2)))

    shifted_rows = []
    for tag, alpha in (
        ("low", low_alpha),
        ("nominal", nominal_alpha),
        ("high", high_alpha),
    ):
        eval_cfg = dict(cfg)
        eval_cfg["alpha"] = float(alpha)
        shifted_env = make_env(eval_cfg)
        shifted_metrics = _evaluate_env(agent, shifted_env, shifted_episodes)
        shifted_env.close()
        shifted_rows.append((tag, alpha, shifted_metrics))

    metrics["eval/robust_alpha_low"] = low_alpha
    metrics["eval/robust_alpha_high"] = high_alpha
    metrics["eval/robust_reward_worst"] = float(
        min(row[2]["eval/reward_mean"] for row in shifted_rows)
    )
    metrics["eval/robust_revenue_worst"] = float(
        min(row[2]["eval/revenue_mean"] for row in shifted_rows)
    )
    metrics["eval/robust_coi_leakage_worst"] = float(
        max(row[2]["eval/coi_leakage_mean"] for row in shifted_rows)
    )
    for tag, alpha, shifted_metrics in shifted_rows:
        metrics[f"eval/{tag}_alpha"] = float(alpha)
        metrics[f"eval/{tag}_reward_mean"] = float(shifted_metrics["eval/reward_mean"])
        metrics[f"eval/{tag}_revenue_mean"] = float(
            shifted_metrics["eval/revenue_mean"]
        )
        metrics[f"eval/{tag}_coi_leakage_mean"] = float(
            shifted_metrics["eval/coi_leakage_mean"]
        )

    return metrics
