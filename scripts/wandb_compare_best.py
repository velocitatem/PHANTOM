from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _normalize_sweep_id(
    raw: str, entity: str, project: str
) -> tuple[str, str, str, str]:
    sweep_raw = str(raw).strip()
    if not sweep_raw:
        raise ValueError("--sweep-id is required")
    parts = [piece.strip() for piece in sweep_raw.split("/") if piece.strip()]
    if len(parts) == 3:
        return f"{parts[0]}/{parts[1]}/{parts[2]}", parts[0], parts[1], parts[2]
    if len(parts) == 2:
        if not entity.strip():
            raise ValueError("--entity is required when --sweep-id is '<project>/<id>'")
        return f"{entity}/{parts[0]}/{parts[1]}", entity, parts[0], parts[1]
    if len(parts) == 1:
        if not entity.strip() or not project.strip():
            raise ValueError(
                "--entity and --project are required when --sweep-id is '<id>'"
            )
        return f"{entity}/{project}/{parts[0]}", entity, project, parts[0]
    raise ValueError(f"invalid --sweep-id value: '{raw}'")


def _pick_best_defended_run(
    sweep: Any,
    metric: str,
    *,
    min_margin: float,
    min_coi: float,
) -> tuple[Any, float]:
    ranked: list[tuple[float, Any]] = []
    for run in list(sweep.runs):
        if str(getattr(run, "state", "")).lower() != "finished":
            continue
        cfg = dict(getattr(run, "config", {}) or {})
        is_baseline = (
            _truthy(cfg.get("baseline_mode"))
            if "baseline_mode" in cfg
            else _truthy(cfg.get("no_robust"))
        )
        if is_baseline:
            continue
        summary = dict(getattr(run, "summary", {}) or {})
        margin = _as_float(summary.get("eval/margin_mean"), -1.0)
        coi_level = _as_float(summary.get("eval/coi_level_mean"), -1.0)
        if margin < float(min_margin):
            continue
        if coi_level < float(min_coi):
            continue
        score = summary.get(metric)
        if score is None and str(metric) == "eval/stress_revenue_worst":
            score = summary.get("eval/robust_revenue_worst")
        if score is None:
            continue
        try:
            ranked.append((float(score), run))
        except (TypeError, ValueError):
            continue
    if not ranked:
        raise RuntimeError(
            f"no finished defended runs found with summary metric '{metric}' and constraints "
            f"margin>={min_margin}, coi>={min_coi}"
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1], ranked[0][0]


def _format_alpha_values(raw: str, fallback_alpha: float) -> str:
    cleaned = str(raw).strip()
    if cleaned:
        return cleaned
    return f"{float(fallback_alpha):.6g}"


def _benchmark_tokens(
    *,
    project: str,
    cfg: dict[str, Any],
    alpha_values: str,
    episodes: int,
) -> list[str]:
    algo = str(cfg.get("algo", "")).strip().lower()
    if algo not in {"qtable", "ppo", "a2c", "dqn"}:
        raise ValueError(f"unsupported algo in best run: '{algo}'")

    total_timesteps = _as_int(cfg.get("total_timesteps"), 80_000)
    max_steps = _as_int(cfg.get("max_steps"), 100)
    ambiguity_radius = _as_float(
        cfg.get("ambiguity_radius", cfg.get("robust_radius")), 0.2
    )
    ambiguity_points = _as_int(cfg.get("ambiguity_points", cfg.get("robust_points")), 7)
    ambiguity_rollouts = _as_int(
        cfg.get("ambiguity_rollouts", cfg.get("robust_rollouts")), 1
    )
    lambda_coi = _as_float(cfg.get("lambda_coi"), 0.2)
    eta_ux = _as_float(cfg.get("eta_ux"), 0.5)
    reward_profit_weight = _as_float(cfg.get("reward_profit_weight"), 1.0)
    learning_rate = _as_float(cfg.get("learning_rate"), 3e-4)
    batch_size = _as_int(cfg.get("batch_size"), 256)
    n_steps = _as_int(cfg.get("n_steps"), 2048)
    sessions = _as_int(cfg.get("N"), 100)
    action_levels = _as_int(cfg.get("action_levels"), 9)
    margin_floor = _as_float(cfg.get("margin_floor"), 0.85)
    seed = _as_int(cfg.get("seed"), 42)

    return [
        "--project",
        project,
        "--tiers",
        algo,
        "--alpha-values",
        alpha_values,
        "--episodes",
        str(int(episodes)),
        "--seed",
        str(seed),
        "--total-timesteps",
        str(total_timesteps),
        "--max-steps",
        str(max_steps),
        "--robust-radius",
        str(ambiguity_radius),
        "--robust-points",
        str(ambiguity_points),
        "--robust-rollouts",
        str(ambiguity_rollouts),
        "--lambda-coi",
        str(lambda_coi),
        "--eta-ux",
        str(eta_ux),
        "--reward-profit-weight",
        str(reward_profit_weight),
        "--learning-rate",
        str(learning_rate),
        "--batch-size",
        str(batch_size),
        "--n-steps",
        str(n_steps),
        "--N",
        str(sessions),
        "--action-levels",
        str(action_levels),
        "--margin-floor",
        str(margin_floor),
        "--device",
        "cpu",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find best defended sweep run and prepare defended-vs-baseline benchmark"
    )
    parser.add_argument("--sweep-id", required=True)
    parser.add_argument("--entity", default="")
    parser.add_argument("--project", default="")
    parser.add_argument("--metric", default="eval/stress_revenue_worst")
    parser.add_argument("--min-margin", type=float, default=0.90)
    parser.add_argument("--min-coi", type=float, default=120.0)
    parser.add_argument("--alpha-values", default="")
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--num-nodes", type=int, default=4)
    parser.add_argument("--tpu-per-task", type=float, default=0.0)
    parser.add_argument("--inner-workers", type=int, default=12)
    parser.add_argument("--inner-threads", type=int, default=1)
    parser.add_argument("--max-heavy-workers", type=int, default=3)
    parser.add_argument("--worker-cpus", type=int, default=24)
    parser.add_argument(
        "--output-root", default="engine/studies/results/overnight/best_compare"
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--ray-no-wait", action="store_true")
    parser.add_argument("--submission-id", default="")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cwd = str(Path.cwd())
    sys.path = [p for p in sys.path if p not in {"", cwd}]

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is required") from exc

    full_sweep_id, entity, project, _ = _normalize_sweep_id(
        raw=str(args.sweep_id),
        entity=str(args.entity).strip(),
        project=str(args.project).strip(),
    )
    api = wandb.Api(timeout=int(args.timeout))
    sweep = api.sweep(full_sweep_id)
    best_run, best_score = _pick_best_defended_run(
        sweep,
        str(args.metric),
        min_margin=float(args.min_margin),
        min_coi=float(args.min_coi),
    )

    best_cfg = dict(getattr(best_run, "config", {}) or {})
    best_alpha = _as_float(
        best_cfg.get(
            "alpha",
            getattr(best_run, "summary", {}).get("study/alpha", 0.6),
        ),
        0.6,
    )
    alpha_values = _format_alpha_values(
        str(args.alpha_values), fallback_alpha=best_alpha
    )
    benchmark_tokens = _benchmark_tokens(
        project=project,
        cfg=best_cfg,
        alpha_values=alpha_values,
        episodes=int(args.episodes),
    )
    benchmark_args = shlex.join(benchmark_tokens)

    submission_id = str(args.submission_id).strip()
    if not submission_id:
        stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M")
        submission_id = f"best-compare-{stamp}"

    env_overrides = {
        "RAY_MODE": "benchmark",
        "COMPARE_ROBUST": "1",
        "NUM_NODES": str(int(args.num_nodes)),
        "TPU_PER_TASK": str(float(args.tpu_per_task)),
        "PHANTOM_JAX_PLATFORM": "cpu",
        "WANDB_ENTITY": entity,
        "WANDB_PROJECT": project,
        "BENCHMARK_ARGS": benchmark_args,
        "INNER_WORKERS": str(int(args.inner_workers)),
        "INNER_THREADS": str(int(args.inner_threads)),
        "MAX_HEAVY_WORKERS": str(int(args.max_heavy_workers)),
        "WORKER_CPUS": str(int(args.worker_cpus)),
        "OUTPUT_ROOT": str(args.output_root),
        "SUBMISSION_ID": submission_id,
    }
    if bool(args.ray_no_wait):
        env_overrides["RAY_NO_WAIT"] = "1"

    command_str = (
        "cd "
        + shlex.quote(str(root))
        + " && "
        + " ".join(
            f"{key}={shlex.quote(str(value))}" for key, value in env_overrides.items()
        )
        + " bash ./submit_ray_job.sh"
    )

    payload = {
        "sweep_id": full_sweep_id,
        "selection_metric": str(args.metric),
        "constraints": {
            "min_margin": float(args.min_margin),
            "min_coi": float(args.min_coi),
        },
        "best_run": {
            "id": str(getattr(best_run, "id", "")),
            "name": str(getattr(best_run, "name", "")),
            "url": str(getattr(best_run, "url", "")),
            "score": float(best_score),
            "algo": str(best_cfg.get("algo", "")),
            "alpha": float(best_alpha),
            "eval_margin_mean": _as_float(
                getattr(best_run, "summary", {}).get("eval/margin_mean"), 0.0
            ),
            "eval_coi_level_mean": _as_float(
                getattr(best_run, "summary", {}).get("eval/coi_level_mean"), 0.0
            ),
        },
        "benchmark_compare_command": command_str,
    }
    print(json.dumps(payload, indent=2))

    output_json = str(args.output_json).strip()
    if output_json:
        out_path = Path(output_json)
        if not out_path.is_absolute():
            out_path = root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n")

    if bool(args.submit):
        run_env = dict(os.environ)
        run_env.update({key: str(value) for key, value in env_overrides.items()})
        subprocess.run(
            ["bash", "./submit_ray_job.sh"],
            cwd=str(root),
            env=run_env,
            check=True,
        )


if __name__ == "__main__":
    main()
