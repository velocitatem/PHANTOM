from __future__ import annotations

import argparse
from typing import Any

from .logging_utils import configure_logging
from .orchestrators import run_benchmark_cli, run_sweep_agent, run_train_once
from .spec import TrainSpec


def _parse_tags(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [piece.strip() for piece in str(raw).split(",") if piece.strip()]


def _probe_run_kind(argv: list[str]) -> str:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--run-kind", choices=["train", "benchmark"])
    probe.add_argument("--run-mode", choices=["train", "benchmark"])
    args, _ = probe.parse_known_args(argv)
    return str(args.run_kind or args.run_mode or "train")


def _strip_run_kind(argv: list[str]) -> list[str]:
    stripped: list[str] = []
    skip_next = False
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item in {"--run-kind", "--run-mode"}:
            skip_next = True
            continue
        if item.startswith("--run-kind=") or item.startswith("--run-mode="):
            continue
        stripped.append(item)
    return stripped


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PHANTOM unified training entrypoint")
    parser.add_argument("--run-kind", choices=["train", "benchmark"], default="train")
    parser.add_argument("--run-mode", choices=["train", "benchmark"])

    parser.add_argument("--project", default="capstone")
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--group", type=str)
    parser.add_argument("--tags", type=str)

    parser.add_argument("--backend", choices=["auto", "sb3"], default="auto")
    parser.add_argument("--algo", choices=["ppo", "a2c", "dqn", "qtable", "sac"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--model-dir", type=str)
    parser.add_argument("--log-freq", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--device", type=str)

    parser.add_argument("--alpha", type=float)
    parser.add_argument("--N", type=int)
    parser.add_argument("--n-products", type=int)
    parser.add_argument("--lambda-coi", type=float)
    parser.add_argument("--info-value", type=float)
    parser.add_argument("--robust-radius", type=float)
    parser.add_argument("--robust-points", type=int)
    parser.add_argument("--no-robust", action="store_true")
    parser.add_argument("--revenue-weight", type=float)

    parser.add_argument("--price-low", type=float)
    parser.add_argument("--price-high", type=float)
    parser.add_argument("--action-levels", type=int)
    parser.add_argument("--action-scale-low", type=float)
    parser.add_argument("--action-scale-high", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--margin-floor", type=float)
    parser.add_argument("--margin-floor-patience", type=int)

    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--train-freq", type=int)
    parser.add_argument("--learning-starts", type=int)
    parser.add_argument("--target-update-interval", type=int)
    parser.add_argument("--exploration-fraction", type=float)
    parser.add_argument("--exploration-final-eps", type=float)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--clip-range", type=float)
    parser.add_argument("--ent-coef", type=float)
    parser.add_argument("--q-lr", type=float)
    parser.add_argument("--q-bins", type=int)
    parser.add_argument("--eps-start", type=float)
    parser.add_argument("--eps-end", type=float)
    parser.add_argument("--eps-decay", type=float)
    parser.add_argument("--arch", type=str)
    parser.add_argument("--activation", type=str)
    parser.add_argument("--vf-coef", type=float)
    parser.add_argument("--max-grad-norm", type=float)

    parser.add_argument("--eval-freq", type=int)
    parser.add_argument("--eval-episodes", type=int)

    parser.add_argument("--sweep-agent", action="store_true")
    parser.add_argument("--sweep-id", type=str)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def _overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    backend = None if args.backend == "auto" else args.backend

    overrides = {
        "project": args.project,
        "backend": backend,
        "algo": args.algo,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "model_dir": args.model_dir,
        "log_freq": args.log_freq,
        "checkpoint_interval": args.checkpoint_interval,
        "device": args.device,
        "alpha": args.alpha,
        "N": args.N,
        "n_products": args.n_products,
        "lambda_coi": args.lambda_coi,
        "info_value": args.info_value,
        "robust_radius": args.robust_radius,
        "robust_points": args.robust_points,
        "no_robust": args.no_robust,
        "revenue_weight": args.revenue_weight,
        "price_low": args.price_low,
        "price_high": args.price_high,
        "action_levels": args.action_levels,
        "action_scale_low": args.action_scale_low,
        "action_scale_high": args.action_scale_high,
        "max_steps": args.max_steps,
        "margin_floor": args.margin_floor,
        "margin_floor_patience": args.margin_floor_patience,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "tau": args.tau,
        "train_freq": args.train_freq,
        "learning_starts": args.learning_starts,
        "target_update_interval": args.target_update_interval,
        "exploration_fraction": args.exploration_fraction,
        "exploration_final_eps": args.exploration_final_eps,
        "n_steps": args.n_steps,
        "n_epochs": args.n_epochs,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "q_lr": args.q_lr,
        "q_bins": args.q_bins,
        "eps_start": args.eps_start,
        "eps_end": args.eps_end,
        "eps_decay": args.eps_decay,
        "arch": args.arch,
        "activation": args.activation,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
    }
    return {key: value for key, value in overrides.items() if value is not None}


def main(argv: list[str] | None = None) -> None:
    import sys

    configure_logging()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    run_kind = _probe_run_kind(raw_args)
    if run_kind == "benchmark":
        run_benchmark_cli(_strip_run_kind(raw_args))
        return

    parser = _build_parser()
    args, unknown = parser.parse_known_args(raw_args)
    if unknown:
        raise ValueError(f"Unknown arguments for training mode: {' '.join(unknown)}")

    overrides = _overrides_from_args(args)
    scenario = str(args.scenario)
    group = args.group
    extra_tags = tuple(_parse_tags(args.tags))

    if args.sweep_agent:
        run_sweep_agent(
            project=args.project,
            sweep_id=str(args.sweep_id or ""),
            count=int(args.count),
            offline=bool(args.offline),
            no_wandb=bool(args.no_wandb),
            base_overrides=overrides,
            kind="sweep",
            scenario=scenario,
            group=group,
            extra_tags=extra_tags,
        )
        return

    spec = TrainSpec.from_flat(overrides)
    run_train_once(
        spec,
        project=args.project,
        offline=bool(args.offline),
        no_wandb=bool(args.no_wandb),
        kind="train",
        scenario=scenario,
        group=group,
        extra_tags=extra_tags,
    )


if __name__ == "__main__":
    main()
