from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any


def _base_sweep(method: str, metric_name: str) -> dict[str, Any]:
    return {
        "method": str(method),
        "metric": {"name": str(metric_name), "goal": "maximize"},
    }


def _benchmark_sweep(method: str) -> dict[str, Any]:
    cfg = _base_sweep(method=method, metric_name="objective/score")
    cfg["name"] = "benchmark-all-algos-defense"
    cfg["parameters"] = {
        "tiers": {
            "values": [
                "static",
                "surge",
                "linear",
                "qtable",
                "ppo",
                "a2c",
                "dqn",
            ]
        },
        "alpha_values": {"values": ["0.0", "0.1", "0.25", "0.4", "0.6", "0.8"]},
        "baseline_mode": {"values": [False, True]},
        "seed": {"values": [42, 1337, 2026, 7777]},
        "episodes": {"values": [8, 12]},
        "total_timesteps": {"values": [15000, 30000, 50000]},
        "lambda_coi": {"values": [0.1, 0.2, 0.4]},
        "ambiguity_radius": {"values": [0.1, 0.2, 0.3]},
        "ambiguity_points": {"values": [5, 7]},
        "ambiguity_rollouts": {"values": [1, 2]},
        "eta_ux": {"values": [0.25, 0.5, 0.75]},
        "reward_profit_weight": {"values": [0.75, 1.0, 1.25]},
        "learning_rate": {"values": [1e-4, 3e-4, 1e-3]},
        "batch_size": {"values": [128, 256, 512]},
        "n_steps": {"values": [1024, 2048, 4096]},
        "device": {"value": "cpu"},
    }
    return cfg


def _train_sweep(method: str) -> dict[str, Any]:
    cfg = _base_sweep(method=method, metric_name="objective/score")
    cfg["name"] = "train-all-algos-defense"
    cfg["parameters"] = {
        "algo": {"values": ["qtable", "ppo", "a2c", "dqn"]},
        "alpha": {"values": [0.0, 0.1, 0.25, 0.4, 0.6]},
        "baseline_mode": {"values": [False, True]},
        "seed": {"values": [42, 1337, 2026, 7777]},
        "total_timesteps": {"values": [30000, 50000, 80000]},
        "learning_rate": {"values": [1e-4, 3e-4, 1e-3]},
        "batch_size": {"values": [128, 256, 512]},
        "n_steps": {"values": [1024, 2048, 4096]},
        "lambda_coi": {"values": [0.1, 0.2, 0.4]},
        "ambiguity_radius": {"values": [0.1, 0.2, 0.3]},
        "ambiguity_points": {"values": [3, 5, 7]},
        "ambiguity_rollouts": {"values": [1, 2]},
        "eta_ux": {"values": [0.25, 0.5, 0.75]},
        "reward_profit_weight": {"values": [0.75, 1.0, 1.25]},
        "N": {"values": [80, 100, 140]},
        "max_steps": {"values": [80, 100, 120]},
        "action_levels": {"values": [7, 9, 11]},
        "device": {"value": "cpu"},
    }
    return cfg


def _train_robust_revenue_sweep(method: str) -> dict[str, Any]:
    cfg = _base_sweep(method=method, metric_name="eval/stress_revenue_worst")
    cfg["name"] = "train-defense-revenue-search"
    cfg["parameters"] = {
        "algo": {"values": ["qtable", "ppo", "a2c", "dqn"]},
        "alpha": {"values": [0.4, 0.6, 0.8]},
        "baseline_mode": {"value": False},
        "seed": {"values": [42, 1337, 2026, 7777]},
        "total_timesteps": {"values": [60_000, 80_000, 120_000]},
        "learning_rate": {"values": [1e-4, 3e-4, 1e-3]},
        "batch_size": {"values": [128, 256, 512]},
        "n_steps": {"values": [1024, 2048, 4096]},
        "lambda_coi": {"values": [0.2, 0.4, 0.6]},
        "ambiguity_radius": {"values": [0.1, 0.2, 0.3]},
        "ambiguity_points": {"values": [5, 7, 9]},
        "ambiguity_rollouts": {"values": [1, 2]},
        "eta_ux": {"values": [0.25, 0.5, 0.75]},
        "reward_profit_weight": {"values": [1.0, 1.25]},
        "N": {"values": [80, 100, 140]},
        "max_steps": {"values": [80, 100, 120]},
        "action_levels": {"values": [7, 9, 11]},
        "margin_floor": {"value": 0.85},
        "device": {"value": "cpu"},
    }
    return cfg


def _ppo_calibration_sweep(method: str) -> dict[str, Any]:
    cfg = _base_sweep(method=method, metric_name="objective/score")
    cfg["name"] = "benchmark-ppo-calibration"
    cfg["parameters"] = {
        "tiers": {"value": "ppo"},
        "alpha_values": {"values": ["0.0", "0.1", "0.25", "0.4", "0.6", "0.8"]},
        "baseline_mode": {"values": [False, True]},
        "seed": {"values": [42, 1337, 2026, 7777]},
        "episodes": {"value": 12},
        "total_timesteps": {"value": 60000},
        "lambda_coi": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.6,
        },
        "ambiguity_radius": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.45,
        },
        "ambiguity_points": {"value": 7},
        "ambiguity_rollouts": {"value": 1},
        "eta_ux": {"value": 0.5},
        "reward_profit_weight": {"value": 1.0},
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-3,
        },
        "batch_size": {"values": [128, 256, 512]},
        "n_steps": {"values": [1024, 2048, 4096]},
        "device": {"value": "cpu"},
    }
    return cfg


def _ppo_block_a_sweep(method: str) -> dict[str, Any]:
    cfg = _base_sweep(method=method, metric_name="objective/score")
    cfg["name"] = "benchmark-ppo-block-a-calibration"
    cfg["parameters"] = {
        "tiers": {"value": "ppo"},
        "alpha_values": {"value": "0.25,0.6,0.8"},
        "seed": {"values": [42, 1337, 2026]},
        "episodes": {"value": 12},
        "total_timesteps": {"value": 80000},
        "lambda_coi": {"values": [0.05, 0.1, 0.2]},
        "ambiguity_radius": {"values": [0.05, 0.1, 0.2]},
        "ambiguity_points": {"value": 7},
        "ambiguity_rollouts": {"value": 1},
        "eta_ux": {"value": 0.5},
        "reward_profit_weight": {"value": 1.0},
        "learning_rate": {"value": 3e-4},
        "batch_size": {"value": 256},
        "n_steps": {"value": 2048},
        "device": {"value": "cpu"},
    }
    return cfg


def _ppo_shift_screen_sweep(method: str) -> dict[str, Any]:
    cfg = _base_sweep(method=method, metric_name="objective/score")
    cfg["name"] = "benchmark-ppo-shift-screen"
    cfg["parameters"] = {
        "tiers": {"value": "ppo"},
        "alpha_values": {"value": "0.25"},
        "eval_alpha_values": {"value": "0.6,0.8"},
        "seed": {"values": [42, 1337, 2026]},
        "episodes": {"value": 20},
        "total_timesteps": {"value": 80000},
        "lambda_coi": {"values": [0.0, 0.02, 0.05, 0.1]},
        "ambiguity_radius": {"values": [0.0, 0.02, 0.05, 0.1]},
        "ambiguity_points": {"value": 5},
        "ambiguity_rollouts": {"value": 1},
        "eta_ux": {"value": 0.0},
        "reward_profit_weight": {"value": 1.0},
        "learning_rate": {"value": 3e-4},
        "batch_size": {"value": 256},
        "n_steps": {"value": 2048},
        "device": {"value": "cpu"},
    }
    return cfg


def _ppo_rl_study_sweep(method: str) -> dict[str, Any]:
    cfg = _base_sweep(method=method, metric_name="eval/stress_revenue_worst")
    cfg["name"] = "train-ppo-standard-vs-defended-equilibrium"
    cfg["parameters"] = {
        "algo": {"value": "ppo"},
        "seed": {"values": [42, 1337, 7777]},
        "alpha": {"values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
        "n_products": {"values": [5, 25, 50, 100]},
        "N": {"value": 100},
        "no_robust": {"values": [False, True]},
        "lambda_coi": {"values": [0.05, 0.15, 0.3]},
        "ambiguity_radius": {"values": [0.1, 0.2, 0.3]},
        "ambiguity_points": {"value": 7},
        "ambiguity_rollouts": {"value": 1},
        "eta_ux": {"value": 0.0},
        "reward_profit_weight": {"value": 1.0},
        "total_timesteps": {"value": 100000},
        "eval_episodes": {"value": 10},
        "eval_freq": {"value": 1000},
        "log_freq": {"value": 100},
        "hist_freq": {"value": 500},
        "learning_rate": {"value": 3e-4},
        "batch_size": {"value": 256},
        "n_steps": {"value": 2048},
        "device": {"value": "cpu"},
    }
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Create W&B sweep for PHANTOM")
    parser.add_argument(
        "--kind",
        choices=[
            "benchmark",
            "train",
            "ppo_calibration",
            "ppo_block_a",
            "ppo_shift_screen",
            "ppo_rl_study",
        ],
        default="benchmark",
    )
    parser.add_argument(
        "--profile",
        choices=["default", "robust_revenue"],
        default="default",
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default="")
    parser.add_argument(
        "--method", choices=["random", "bayes", "grid"], default="random"
    )
    parser.add_argument("--run-cap", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--full-id", action="store_true")
    args = parser.parse_args()

    cwd = str(Path.cwd())
    sys.path = [p for p in sys.path if p not in {"", cwd}]

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is required to create sweeps") from exc

    if str(args.kind) == "benchmark":
        if str(args.profile) != "default":
            raise ValueError("benchmark sweeps only support --profile default")
        sweep_cfg = _benchmark_sweep(args.method)
    elif str(args.kind) == "train":
        if str(args.profile) == "robust_revenue":
            sweep_cfg = _train_robust_revenue_sweep(args.method)
        else:
            sweep_cfg = _train_sweep(args.method)
    elif str(args.kind) == "ppo_calibration":
        if str(args.profile) != "default":
            raise ValueError("ppo_calibration sweeps only support --profile default")
        sweep_cfg = _ppo_calibration_sweep(args.method)
    elif str(args.kind) == "ppo_block_a":
        if str(args.profile) != "default":
            raise ValueError("ppo_block_a sweeps only support --profile default")
        sweep_cfg = _ppo_block_a_sweep(args.method)
    elif str(args.kind) == "ppo_shift_screen":
        if str(args.profile) != "default":
            raise ValueError("ppo_shift_screen sweeps only support --profile default")
        sweep_cfg = _ppo_shift_screen_sweep(args.method)
    else:
        if str(args.profile) != "default":
            raise ValueError("ppo_rl_study sweeps only support --profile default")
        sweep_cfg = _ppo_rl_study_sweep(args.method)
    if int(args.run_cap) > 0:
        sweep_cfg["run_cap"] = int(args.run_cap)

    with contextlib.redirect_stdout(io.StringIO()):
        sweep_id = wandb.sweep(
            sweep=sweep_cfg,
            project=str(args.project),
            entity=str(args.entity) if str(args.entity).strip() else None,
        )
    full_id = (
        f"{args.entity}/{args.project}/{sweep_id}"
        if str(args.entity).strip()
        else f"{args.project}/{sweep_id}"
    )

    if bool(args.json):
        print(
            json.dumps(
                {
                    "kind": str(args.kind),
                    "profile": str(args.profile),
                    "project": str(args.project),
                    "entity": str(args.entity),
                    "sweep_id": str(sweep_id),
                    "full_id": str(full_id),
                }
            )
        )
        return
    print(full_id if bool(args.full_id) else sweep_id)


if __name__ == "__main__":
    main()
