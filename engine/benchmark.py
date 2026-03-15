from __future__ import annotations

import os
import subprocess
import sys

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# clear stale TPU locks on startup
if os.path.exists("/dev/accel0"):
    try:
        subprocess.run(
            ["rm", "-f", "/tmp/.libtpu_lockfile", "/tmp/libtpu_lockfile"],
            stderr=subprocess.DEVNULL,
        )
    except:
        pass

try:
    import jax

    jax.config.update("jax_threefry_partitionable", True)
except ImportError:
    pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lib.tiers import LinearElasticityPolicy, StaticPolicy, SurgePolicy
from .logging_utils import configure_logging
from .spec import TrainSpec
from .telemetry.wandb import get_wandb_module

wandb = get_wandb_module()
HAS_WANDB = wandb is not None
logger = logging.getLogger(__name__)


def _log(message: str) -> None:
    logger.info(message)


def _wandb_run_active() -> bool:
    return bool(HAS_WANDB and getattr(wandb, "run", None) is not None)


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


def _mode_label_from_baseline(is_baseline: bool) -> str:
    return "baseline" if bool(is_baseline) else "defended"


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


def _build_tier(name: str, cfg: dict, alpha: float, *, step_offset: int = 0):
    from .backends.common import make_env

    tier = name.lower().strip()
    run_cfg = dict(cfg)
    run_cfg["alpha"] = float(alpha)
    run_cfg["wandb_step_offset"] = int(step_offset)

    if tier == "static":
        return StaticPolicy(int(run_cfg["action_levels"])), []

    if tier == "surge":
        return (
            SurgePolicy(
                n_actions=int(run_cfg["action_levels"]),
                n_products=int(run_cfg["n_products"]),
            ),
            [],
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
        return policy, []

    if tier == "qtable":
        from .backends.qtable import train_qtable

        run_cfg["console_progress"] = True
        agent, metrics = train_qtable(run_cfg)
        events = metrics.get("_train_events", [])
        return agent, events if isinstance(events, list) else []

    if tier in {"ppo", "a2c", "dqn"}:
        from .backends.sb3 import train_sb3

        run_cfg["algo"] = tier
        agent, metrics = train_sb3(run_cfg)
        events = metrics.get("_train_events", [])
        return agent, events if isinstance(events, list) else []

    raise ValueError(f"unsupported tier '{name}'")


def _log_train_events(
    events: list[dict],
    *,
    tier_name: str,
    mode_label: str,
    alpha: float,
    step_offset: int,
) -> int:
    if not _wandb_run_active():
        return int(step_offset)
    if not events:
        return int(step_offset)

    ordered = sorted(
        [evt for evt in events if isinstance(evt, dict)],
        key=lambda evt: int(evt.get("train/global_step", 0)),
    )
    if not ordered:
        return int(step_offset)

    cursor = int(step_offset)
    for evt in ordered:
        rel_step = max(1, int(evt.get("train/global_step", 0)))
        payload = dict(evt)
        payload.update(
            {
                "run.kind": "benchmark",
                "runtime/backend": tier_name,
                "study/mode": mode_label,
                "study/baseline_mode": float(mode_label == "baseline"),
                "study/alpha": float(alpha),
            }
        )
        try:
            wandb.log(payload, step=cursor + rel_step)
        except Exception:
            return int(step_offset)
    max_rel = max(max(1, int(evt.get("train/global_step", 0))) for evt in ordered)
    return cursor + max_rel + 1


def run_benchmark(
    cfg: dict,
    tiers: list[str],
    alpha_values: list[float],
    n_episodes: int,
    mode_label: str,
    step_cursor_start: int = 0,
    eval_alpha_values: list[float] | None = None,
):
    from .backends.common import make_env

    rows: list[dict] = []
    traces: list[dict] = []
    total_runs = max(1, len(alpha_values) * len(tiers))
    run_index = 0
    wandb_step_cursor = int(step_cursor_start)

    for alpha in alpha_values:
        for tier_name in tiers:
            run_index += 1
            _log(
                f"[{run_index}/{total_runs}] alpha={float(alpha):.2f} tier={tier_name}: training"
            )
            policy, train_events = _build_tier(
                tier_name,
                cfg,
                alpha,
                step_offset=wandb_step_cursor,
            )
            prev_cursor = int(wandb_step_cursor)
            wandb_step_cursor = _log_train_events(
                train_events,
                tier_name=tier_name,
                mode_label=mode_label,
                alpha=float(alpha),
                step_offset=wandb_step_cursor,
            )
            if wandb_step_cursor == prev_cursor and tier_name in {
                "qtable",
                "ppo",
                "a2c",
                "dqn",
            }:
                wandb_step_cursor += max(1, int(cfg.get("total_timesteps", 1))) + 1
            eval_targets = (
                [float(value) for value in eval_alpha_values]
                if eval_alpha_values
                else [float(alpha)]
            )
            for eval_alpha in eval_targets:
                env = make_env({**cfg, "alpha": float(eval_alpha)})
                eps = [_run_eval_episode(env, policy) for _ in range(int(n_episodes))]
                env.close()

                row = {
                    "tier": tier_name,
                    "mode": mode_label,
                    "alpha": float(eval_alpha),
                    "train_alpha": float(alpha),
                    "eval_alpha": float(eval_alpha),
                    "episodes": int(n_episodes),
                    "mean_reward": float(np.mean([e["reward"] for e in eps])),
                    "mean_revenue": float(np.mean([e["revenue"] for e in eps])),
                    "mean_margin": float(np.mean([e["mean_margin"] for e in eps])),
                    "mean_coi": float(np.mean([e["mean_coi"] for e in eps])),
                    "std_revenue": float(np.std([e["revenue"] for e in eps])),
                }
                row["objective_score"] = row["mean_reward"]
                rows.append(row)
                _log(
                    f"[{run_index}/{total_runs}] train_alpha={float(alpha):.2f} "
                    f"eval_alpha={float(eval_alpha):.2f} tier={tier_name}: "
                    f"reward={row['mean_reward']:.3f} revenue={row['mean_revenue']:.3f} "
                    f"coi={row['mean_coi']:.4f} score={row['objective_score']:.3f}"
                )

                max_len = max((len(e["price_trace"]) for e in eps), default=0)
                step_means = []
                for step in range(max_len):
                    vals = [
                        e["price_trace"][step]
                        for e in eps
                        if step < len(e["price_trace"])
                    ]
                    step_means.append(float(np.mean(vals)) if vals else np.nan)
                traces.append(
                    {
                        "tier": tier_name,
                        "alpha": float(eval_alpha),
                        "train_alpha": float(alpha),
                        "eval_alpha": float(eval_alpha),
                        "mean_price_trace": step_means,
                    }
                )

                if _wandb_run_active():
                    try:
                        wandb.log(
                            {
                                "run.kind": "benchmark",
                                "runtime/backend": tier_name,
                                "study/mode": mode_label,
                                "study/baseline_mode": float(mode_label == "baseline"),
                                "study/alpha": float(eval_alpha),
                                "study/train_alpha": float(alpha),
                                "study/eval_alpha": float(eval_alpha),
                                "eval/reward_mean": row["mean_reward"],
                                "eval/revenue_mean": row["mean_revenue"],
                                "eval/margin_mean": row["mean_margin"],
                                "eval/coi_level_mean": row["mean_coi"],
                                "objective/score": row["objective_score"],
                                "objective/coi_preserved": row["mean_coi"],
                            },
                            step=wandb_step_cursor,
                        )
                    except Exception:
                        pass
                    wandb_step_cursor += 1

    return pd.DataFrame(rows), traces, int(wandb_step_cursor)


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


def _run_with_args(args, compare_robust_override: bool | None = None):
    compare_robust = (
        bool(compare_robust_override)
        if compare_robust_override is not None
        else _truthy(os.environ.get("PHANTOM_BENCHMARK_COMPARE_ROBUST"))
    )
    baseline_modes = [False, True] if compare_robust else [bool(args.no_robust)]

    base_overrides = {
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "n_products": args.n_products,
        "N": args.N,
        "lambda_coi": args.lambda_coi,
        "robust_radius": args.robust_radius,
        "robust_points": args.robust_points,
        "robust_rollouts": args.robust_rollouts,
        "margin_floor": args.margin_floor,
        "eta_ux": args.eta_ux,
        "reward_profit_weight": args.reward_profit_weight,
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
    eval_alpha_values = (
        _parse_float_list(args.eval_alpha_values)
        if str(getattr(args, "eval_alpha_values", "")).strip()
        else []
    )
    _log(
        "starting run "
        + json.dumps(
            {
                "tiers": tiers,
                "alpha_values": alpha_values,
                "eval_alpha_values": (
                    eval_alpha_values if eval_alpha_values else alpha_values
                ),
                "episodes": int(args.episodes),
                "total_timesteps": int(args.total_timesteps),
                "device": str(args.device),
            }
        )
    )

    all_frames: list[pd.DataFrame] = []
    all_traces: list[dict] = []
    wandb_step_cursor = 0
    for baseline_mode in baseline_modes:
        overrides = dict(base_overrides)
        overrides["baseline_mode"] = bool(baseline_mode)
        cfg = TrainSpec.from_flat(
            {k: v for k, v in overrides.items() if v is not None}
        ).to_flat_dict()
        cfg["linear_warmup_steps"] = int(args.linear_warmup_steps)
        mode_label = _mode_label_from_baseline(bool(baseline_mode))
        _log(f"mode={mode_label}: begin")
        df_mode, traces_mode, wandb_step_cursor = run_benchmark(
            cfg,
            tiers,
            alpha_values,
            args.episodes,
            mode_label=mode_label,
            step_cursor_start=wandb_step_cursor,
            eval_alpha_values=eval_alpha_values,
        )
        _log(f"mode={mode_label}: complete ({len(df_mode)} rows)")
        for trace in traces_mode:
            trace["mode"] = mode_label
        all_frames.append(df_mode)
        all_traces.extend(traces_mode)

    df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    traces = all_traces

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"benchmark_{stamp}.csv"
    trace_path = out_dir / f"benchmark_traces_{stamp}.json"
    df.to_csv(csv_path, index=False)
    trace_path.write_text(json.dumps(traces, indent=2))
    rev_path, coi_path, price_path = _plot_outputs(df, traces, out_dir, stamp)
    _log(f"artifacts written in {out_dir}")

    if not df.empty:
        best_idx = int(df["objective_score"].idxmax())
        best = df.iloc[best_idx]
        _log(
            "BEST_TIER="
            + json.dumps(
                {
                    "tier": best["tier"],
                    "mode": best.get("mode", "defended"),
                    "alpha": float(best["alpha"]),
                    "objective_score": float(best["objective_score"]),
                    "mean_revenue": float(best["mean_revenue"]),
                    "mean_coi": float(best["mean_coi"]),
                }
            )
        )
    _log(f"BENCHMARK_CSV={csv_path}")
    _log(f"BENCHMARK_TRACES={trace_path}")
    _log(f"BENCHMARK_PLOT_REVENUE={rev_path}")
    _log(f"BENCHMARK_PLOT_COI={coi_path}")
    _log(f"BENCHMARK_PLOT_PRICE={price_path}")


def run_cli(raw_args: list[str] | None = None):
    configure_logging()
    parser = argparse.ArgumentParser(description="PHANTOM benchmark orchestrator")
    parser.add_argument("--project", default="capstone")
    parser.add_argument("--tiers", default="static,surge,linear,qtable,ppo")
    parser.add_argument("--alpha-values", default="0.0,0.3,0.6")
    parser.add_argument("--eval-alpha-values", default="")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output-dir", default="engine/studies/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=25_000)
    parser.add_argument("--n-products", type=int, default=10)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--lambda-coi", type=float, default=0.2)
    parser.add_argument("--robust-radius", type=float, default=0.15)
    parser.add_argument("--robust-points", type=int, default=5)
    parser.add_argument("--robust-rollouts", type=int, default=1)
    parser.add_argument("--margin-floor", type=float, default=0.85)
    parser.add_argument("--eta-ux", type=float, default=0.5)
    parser.add_argument("--reward-profit-weight", type=float, default=1.0)
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
                    "eval_alpha_values": "eval_alpha_values",
                    "episodes": "episodes",
                    "total_timesteps": "total_timesteps",
                    "lambda_coi": "lambda_coi",
                    "robust_radius": "robust_radius",
                    "robust_points": "robust_points",
                    "robust_rollouts": "robust_rollouts",
                    "ambiguity_radius": "robust_radius",
                    "ambiguity_points": "robust_points",
                    "ambiguity_rollouts": "robust_rollouts",
                    "eta_ux": "eta_ux",
                    "reward_profit_weight": "reward_profit_weight",
                    "learning_rate": "learning_rate",
                    "batch_size": "batch_size",
                    "n_steps": "n_steps",
                    "baseline_mode": "no_robust",
                    "no_robust": "no_robust",
                    "margin_floor": "margin_floor",
                    "device": "device",
                }
                for key in (
                    "tiers",
                    "alpha_values",
                    "eval_alpha_values",
                    "episodes",
                    "total_timesteps",
                    "lambda_coi",
                    "robust_radius",
                    "robust_points",
                    "robust_rollouts",
                    "ambiguity_radius",
                    "ambiguity_points",
                    "ambiguity_rollouts",
                    "eta_ux",
                    "reward_profit_weight",
                    "learning_rate",
                    "batch_size",
                    "n_steps",
                    "baseline_mode",
                    "no_robust",
                    "margin_floor",
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

    tiers = _parse_list(args.tiers)
    alpha_values = _parse_float_list(args.alpha_values)
    run_stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
    compare_enabled = _truthy(os.environ.get("PHANTOM_BENCHMARK_COMPARE_ROBUST"))
    compare_tag = "defended-compare" if compare_enabled else "single-mode"
    modes = (
        [("baseline", True), ("defended", False)]
        if compare_enabled
        else [(_mode_label_from_baseline(bool(args.no_robust)), bool(args.no_robust))]
    )

    run_idx = 0
    for tier in tiers:
        for mode_label, baseline_mode in modes:
            for alpha in alpha_values:
                run_idx += 1
                alpha_token = (
                    f"{float(alpha):.2f}".rstrip("0").rstrip(".").replace(".", "p")
                )
                tier_args = argparse.Namespace(**vars(args))
                tier_args.tiers = tier
                tier_args.alpha_values = str(float(alpha))
                tier_args.no_robust = bool(baseline_mode)
                run = wandb.init(
                    project=args.project,
                    name=(
                        f"benchmark-{tier}-{mode_label}-a{alpha_token}-{run_stamp}-{run_idx}"
                    ),
                    tags=[
                        "benchmark",
                        compare_tag,
                        f"backend:{tier}",
                        f"mode:{mode_label}",
                        f"alpha:{alpha_token}",
                    ],
                    config={
                        "run.kind": "benchmark",
                        "runtime/backend": tier,
                        "study/mode": mode_label,
                        "study/baseline_mode": float(baseline_mode),
                        "study/alpha": float(alpha),
                        "tiers": tier,
                        "alpha_values": str(float(alpha)),
                        "eval_alpha_values": args.eval_alpha_values,
                        "episodes": args.episodes,
                        "total_timesteps": args.total_timesteps,
                        "lambda_coi": args.lambda_coi,
                        "ambiguity_radius": args.robust_radius,
                        "ambiguity_points": args.robust_points,
                        "ambiguity_rollouts": args.robust_rollouts,
                        "margin_floor": args.margin_floor,
                        "baseline_mode": float(baseline_mode),
                        "eta_ux": args.eta_ux,
                        "reward_profit_weight": args.reward_profit_weight,
                        "learning_rate": args.learning_rate,
                        "device": args.device,
                    },
                    mode="offline" if args.offline else "online",
                )
                try:
                    _run_with_args(tier_args, compare_robust_override=False)
                finally:
                    if run is not None:
                        wandb.finish()


if __name__ == "__main__":
    run_cli()
