from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, UTC
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lib.tiers import LinearElasticityPolicy, StaticPolicy, SurgePolicy
from .spec import TrainSpec
from .telemetry.wandb import get_wandb_module

wandb = get_wandb_module()
HAS_WANDB = wandb is not None


def _parse_list(raw: str) -> list[str]:
    return [x.strip().lower() for x in str(raw).split(",") if x.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _truthy(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _action(policy, obs: np.ndarray):
    out = policy.predict(obs, deterministic=True)
    action = out[0] if isinstance(out, tuple) else out
    if isinstance(action, np.ndarray) and action.size == 1:
        return int(action.reshape(-1)[0])
    return int(action)


def _run_eval_episode(env, policy) -> dict:
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    total_revenue = 0.0
    total_margin = 0.0
    total_coi = 0.0
    price_trace: list[float] = []
    step_count = 0

    while not done:
        action = _action(policy, obs)
        obs, reward, term, trunc, info = env.step(action)
        done = bool(term or trunc)
        econ = info.get("economics", {})
        total_reward += float(reward)
        total_revenue += float(econ.get("revenue", 0.0))
        total_margin += float(econ.get("margin", 0.0))
        total_coi += float(econ.get("coi_level", 0.0))
        prices = np.asarray(info.get("prices", []), dtype=np.float32)
        if prices.size > 0:
            price_trace.append(float(np.mean(prices)))
        step_count += 1

    denom = max(step_count, 1)
    return {
        "reward": total_reward,
        "revenue": total_revenue,
        "mean_margin": total_margin / denom,
        "mean_coi": total_coi / denom,
        "price_trace": price_trace,
    }


def _build_tier(name: str, cfg: dict, alpha: float):
    from .backends.common import make_env
    from .backends.qtable import train_qtable
    from .backends.sb3 import train_sb3

    tier = name.lower().strip()
    run_cfg = dict(cfg)
    run_cfg["alpha"] = float(alpha)

    if tier == "static":
        return StaticPolicy(int(run_cfg["action_levels"]))

    if tier == "surge":
        return SurgePolicy(
            n_actions=int(run_cfg["action_levels"]),
            n_products=int(run_cfg["n_products"]),
        )

    if tier == "linear":
        warmup_env = make_env(run_cfg)
        policy = LinearElasticityPolicy(
            n_actions=int(run_cfg["action_levels"]),
            n_products=int(run_cfg["n_products"]),
            price_low=float(run_cfg["price_low"]),
            price_high=float(run_cfg["price_high"]),
        )
        policy.fit(
            warmup_env,
            warmup_steps=int(run_cfg.get("linear_warmup_steps", 800)),
            seed=int(run_cfg["seed"]),
        )
        warmup_env.close()
        return policy

    if tier == "qtable":
        agent, _ = train_qtable(run_cfg)
        return agent

    if tier in {"ppo", "a2c", "dqn"}:
        run_cfg["algo"] = tier
        agent, _ = train_sb3(run_cfg)
        return agent

    raise ValueError(f"unsupported tier '{name}'")


def run_benchmark(
    cfg: dict, tiers: list[str], alpha_values: list[float], n_episodes: int
):
    from .backends.common import make_env

    rows: list[dict] = []
    traces: list[dict] = []

    for alpha in alpha_values:
        for tier_name in tiers:
            policy = _build_tier(tier_name, cfg, alpha)
            env = make_env({**cfg, "alpha": float(alpha)})
            eps = [_run_eval_episode(env, policy) for _ in range(int(n_episodes))]
            env.close()

            row = {
                "tier": tier_name,
                "alpha": float(alpha),
                "episodes": int(n_episodes),
                "mean_reward": float(np.mean([e["reward"] for e in eps])),
                "mean_revenue": float(np.mean([e["revenue"] for e in eps])),
                "mean_margin": float(np.mean([e["mean_margin"] for e in eps])),
                "mean_coi": float(np.mean([e["mean_coi"] for e in eps])),
                "std_revenue": float(np.std([e["revenue"] for e in eps])),
            }
            row["objective_score"] = (
                row["mean_reward"]
                + float(cfg.get("revenue_weight", 0.01)) * row["mean_revenue"]
            )
            rows.append(row)

            max_len = max((len(e["price_trace"]) for e in eps), default=0)
            step_means = []
            for step in range(max_len):
                vals = [
                    e["price_trace"][step] for e in eps if step < len(e["price_trace"])
                ]
                step_means.append(float(np.mean(vals)) if vals else np.nan)
            traces.append(
                {
                    "tier": tier_name,
                    "alpha": float(alpha),
                    "mean_price_trace": step_means,
                }
            )

            if HAS_WANDB and wandb.run is not None:
                wandb.log(
                    {
                        "study/alpha": float(alpha),
                        "eval/reward_mean": row["mean_reward"],
                        "eval/revenue_mean": row["mean_revenue"],
                        "eval/margin_mean": row["mean_margin"],
                        "objective/score": row["objective_score"],
                        "objective/coi_preserved": row["mean_coi"],
                    }
                )

    return pd.DataFrame(rows), traces


def _plot_outputs(df: pd.DataFrame, traces: list[dict], out_dir: Path, stamp: str):
    fig1 = plt.figure(figsize=(11, 4.5))
    if "mode" in df.columns:
        groups = sorted(df[["tier", "mode"]].drop_duplicates().values.tolist())
        for tier, mode in groups:
            sub = df[(df["tier"] == tier) & (df["mode"] == mode)].sort_values("alpha")
            plt.plot(
                sub["alpha"],
                sub["mean_revenue"],
                marker="o",
                label=f"{tier}:{mode}",
            )
    else:
        for tier in sorted(df["tier"].unique()):
            sub = df[df["tier"] == tier].sort_values("alpha")
            plt.plot(sub["alpha"], sub["mean_revenue"], marker="o", label=tier)
    plt.xlabel("contamination alpha")
    plt.ylabel("mean episode revenue")
    plt.title("Revenue under contamination")
    plt.grid(alpha=0.3)
    plt.legend()
    fig1.tight_layout()
    rev_path = out_dir / f"benchmark_revenue_{stamp}.png"
    fig1.savefig(rev_path, dpi=220)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(11, 4.5))
    if "mode" in df.columns:
        groups = sorted(df[["tier", "mode"]].drop_duplicates().values.tolist())
        for tier, mode in groups:
            sub = df[(df["tier"] == tier) & (df["mode"] == mode)].sort_values("alpha")
            plt.plot(
                sub["alpha"],
                sub["mean_coi"],
                marker="s",
                label=f"{tier}:{mode}",
            )
    else:
        for tier in sorted(df["tier"].unique()):
            sub = df[df["tier"] == tier].sort_values("alpha")
            plt.plot(sub["alpha"], sub["mean_coi"], marker="s", label=tier)
    plt.xlabel("contamination alpha")
    plt.ylabel("mean COI level")
    plt.title("COI preservation")
    plt.grid(alpha=0.3)
    plt.legend()
    fig2.tight_layout()
    coi_path = out_dir / f"benchmark_coi_{stamp}.png"
    fig2.savefig(coi_path, dpi=220)
    plt.close(fig2)

    focus_alpha = float(df["alpha"].min()) if not df.empty else 0.0
    alpha_traces = [t for t in traces if abs(float(t["alpha"]) - focus_alpha) < 1e-9]
    fig3 = plt.figure(figsize=(11, 4.5))
    for item in alpha_traces:
        xs = np.arange(len(item["mean_price_trace"]))
        ys = np.asarray(item["mean_price_trace"], dtype=np.float32)
        mode = item.get("mode")
        label = f"{item['tier']}:{mode}" if mode is not None else str(item["tier"])
        plt.plot(xs, ys, label=label)
    plt.xlabel("step")
    plt.ylabel("mean price")
    plt.title(f"Price evolution (alpha={focus_alpha:.2f})")
    plt.grid(alpha=0.3)
    plt.legend()
    fig3.tight_layout()
    price_path = out_dir / f"benchmark_price_trace_{stamp}.png"
    fig3.savefig(price_path, dpi=220)
    plt.close(fig3)

    return rev_path, coi_path, price_path


def _run_with_args(args):
    compare_robust = _truthy(os.environ.get("PHANTOM_BENCHMARK_COMPARE_ROBUST"))
    robust_modes = [False, True] if compare_robust else [bool(args.no_robust)]

    base_overrides = {
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "n_products": args.n_products,
        "N": args.N,
        "lambda_coi": args.lambda_coi,
        "robust_radius": args.robust_radius,
        "robust_points": args.robust_points,
        "price_low": args.price_low,
        "price_high": args.price_high,
        "action_levels": args.action_levels,
        "action_scale_low": args.action_scale_low,
        "action_scale_high": args.action_scale_high,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "linear_warmup_steps": args.linear_warmup_steps,
        "device": args.device,
    }
    tiers = _parse_list(args.tiers)
    alpha_values = _parse_float_list(args.alpha_values)

    all_frames: list[pd.DataFrame] = []
    all_traces: list[dict] = []
    for no_robust in robust_modes:
        overrides = dict(base_overrides)
        overrides["no_robust"] = bool(no_robust)
        cfg = TrainSpec.from_flat(
            {k: v for k, v in overrides.items() if v is not None}
        ).to_flat_dict()
        cfg["linear_warmup_steps"] = int(args.linear_warmup_steps)
        df_mode, traces_mode = run_benchmark(cfg, tiers, alpha_values, args.episodes)
        mode_label = "no_robust" if no_robust else "robust"
        df_mode["mode"] = mode_label
        for trace in traces_mode:
            trace["mode"] = mode_label
        all_frames.append(df_mode)
        all_traces.extend(traces_mode)

    df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    traces = all_traces

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"benchmark_{stamp}.csv"
    trace_path = out_dir / f"benchmark_traces_{stamp}.json"
    df.to_csv(csv_path, index=False)
    trace_path.write_text(json.dumps(traces, indent=2))
    rev_path, coi_path, price_path = _plot_outputs(df, traces, out_dir, stamp)

    if not df.empty:
        best_idx = int(df["mean_revenue"].idxmax())
        best = df.iloc[best_idx]
        print(
            "BEST_TIER="
            + json.dumps(
                {
                    "tier": best["tier"],
                    "mode": best.get("mode", "robust"),
                    "alpha": float(best["alpha"]),
                    "mean_revenue": float(best["mean_revenue"]),
                    "mean_coi": float(best["mean_coi"]),
                }
            )
        )
    print(f"BENCHMARK_CSV={csv_path}")
    print(f"BENCHMARK_TRACES={trace_path}")
    print(f"BENCHMARK_PLOT_REVENUE={rev_path}")
    print(f"BENCHMARK_PLOT_COI={coi_path}")
    print(f"BENCHMARK_PLOT_PRICE={price_path}")


def run_cli(raw_args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="PHANTOM benchmark orchestrator")
    parser.add_argument("--project", default="capstone")
    parser.add_argument("--tiers", default="static,surge,linear,qtable,ppo")
    parser.add_argument("--alpha-values", default="0.0,0.3,0.6")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output-dir", default="engine/studies/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=25_000)
    parser.add_argument("--n-products", type=int, default=10)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--lambda-coi", type=float, default=0.2)
    parser.add_argument("--robust-radius", type=float, default=0.15)
    parser.add_argument("--robust-points", type=int, default=5)
    parser.add_argument("--price-low", type=float, default=10.0)
    parser.add_argument("--price-high", type=float, default=150.0)
    parser.add_argument("--action-levels", type=int, default=9)
    parser.add_argument("--action-scale-low", type=float, default=0.8)
    parser.add_argument("--action-scale-high", type=float, default=1.2)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--linear-warmup-steps", type=int, default=800)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-robust", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--sweep-agent", action="store_true")
    parser.add_argument("--sweep-id", type=str)
    parser.add_argument("--count", type=int, default=0)
    args = parser.parse_args(raw_args)

    if args.sweep_agent:
        if args.no_wandb or not HAS_WANDB:
            raise ValueError("sweep agent requires wandb")
        if not args.sweep_id:
            raise ValueError("--sweep-id is required with --sweep-agent")

        def _sweep_run():
            run = wandb.init(mode="offline" if args.offline else "online")
            try:
                key_to_attr = {
                    "tiers": "tiers",
                    "alpha_values": "alpha_values",
                    "episodes": "episodes",
                    "total_timesteps": "total_timesteps",
                    "lambda_coi": "lambda_coi",
                    "robust_radius": "robust_radius",
                    "robust_points": "robust_points",
                    "learning_rate": "learning_rate",
                    "batch_size": "batch_size",
                    "n_steps": "n_steps",
                    "no_robust": "no_robust",
                    "device": "device",
                }
                for key in (
                    "tiers",
                    "alpha_values",
                    "episodes",
                    "total_timesteps",
                    "lambda_coi",
                    "robust_radius",
                    "robust_points",
                    "learning_rate",
                    "batch_size",
                    "n_steps",
                    "no_robust",
                    "device",
                ):
                    if key in wandb.config:
                        setattr(args, key_to_attr[key], wandb.config[key])
                _run_with_args(args)
            finally:
                if run is not None:
                    wandb.finish()

        wandb.agent(
            args.sweep_id,
            function=_sweep_run,
            count=args.count if args.count > 0 else None,
        )
        return

    if args.no_wandb or not HAS_WANDB:
        _run_with_args(args)
        return

    run = wandb.init(
        project=args.project,
        name=f"benchmark-{datetime.now(UTC).strftime('%m%d-%H%M%S')}",
        tags=[
            "benchmark",
            "robust-compare"
            if _truthy(os.environ.get("PHANTOM_BENCHMARK_COMPARE_ROBUST"))
            else "single-mode",
        ],
        config={
            "run.kind": "benchmark",
            "tiers": args.tiers,
            "alpha_values": args.alpha_values,
            "episodes": args.episodes,
            "total_timesteps": args.total_timesteps,
            "lambda_coi": args.lambda_coi,
            "robust_radius": args.robust_radius,
            "robust_points": args.robust_points,
            "learning_rate": args.learning_rate,
            "device": args.device,
        },
        mode="offline" if args.offline else "online",
    )
    try:
        _run_with_args(args)
    finally:
        if run is not None:
            wandb.finish()


if __name__ == "__main__":
    run_cli()
