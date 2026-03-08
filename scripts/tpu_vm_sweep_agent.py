#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shlex
import shutil
import subprocess
import time
import resource
from pathlib import Path

import wandb


CLI_MAP: dict[str, str] = {
    "algo": "--algo",
    "total_timesteps": "--total-timesteps",
    "alpha": "--alpha",
    "N": "--N",
    "n_products": "--n-products",
    "lambda_coi": "--lambda-coi",
    "info_value": "--info-value",
    "robust_radius": "--robust-radius",
    "robust_points": "--robust-points",
    "no_robust": "--no-robust",
    "learning_rate": "--learning-rate",
    "gamma": "--gamma",
    "gae_lambda": "--gae-lambda",
    "clip_range": "--clip-range",
    "ent_coef": "--ent-coef",
    "revenue_weight": "--revenue-weight",
    "max_steps": "--max-steps",
    "margin_floor": "--margin-floor",
    "margin_floor_patience": "--margin-floor-patience",
    "arch": "--arch",
    "activation": "--activation",
    "jax_num_envs": "--jax-num-envs",
    "jax_num_steps": "--jax-num-steps",
    "jax_num_minibatches": "--jax-num-minibatches",
    "jax_update_epochs": "--jax-update-epochs",
    "jax_anneal_lr": "--jax-anneal-lr",
    "checkpoint_interval": "--checkpoint-interval",
    "action_levels": "--action-levels",
    "action_scale_low": "--action-scale-low",
    "action_scale_high": "--action-scale-high",
}


def _to_cli_args(cfg: dict) -> str:
    parts: list[str] = ["--jax", "--no-wandb"]
    for key, flag in CLI_MAP.items():
        if key not in cfg:
            continue
        value = cfg[key]
        if value is None:
            continue
        if isinstance(value, bool):
            if key == "jax_anneal_lr":
                parts.extend([flag, "true" if value else "false"])
            elif value:
                parts.append(flag)
            continue
        parts.extend([flag, str(value)])
    return " ".join(shlex.quote(p) for p in parts)


_SENTINEL = "PHANTOM_METRICS:"


def _raise_nofile_limit(min_soft: int = 8192) -> None:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(hard, max(soft, min_soft))
        if target > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception:
        return


def _extract_metrics(output: str) -> dict:
    # fast path: look for the dedicated sentinel line emitted by run_local
    for line in output.splitlines():
        if line.startswith(_SENTINEL):
            try:
                return json.loads(line[len(_SENTINEL) :])
            except Exception:
                break
    # fallback: scan for any JSON block containing eval/sweep keys;
    # use greedy match to capture the largest possible block first
    for block in re.findall(r"\{[^{}]*\}", output):
        try:
            obj = json.loads(block)
        except Exception:
            continue
        if isinstance(obj, dict) and (
            "objective/score" in obj
            or "eval/reward_mean" in obj
            or "sweep/score" in obj
        ):
            return obj
    return {}


def main() -> None:
    _raise_nofile_limit()
    p = argparse.ArgumentParser(
        description="Run W&B sweep where each trial uses full TPU pod"
    )
    p.add_argument("--sweep-id", required=True)
    p.add_argument("--tpu-name", required=True)
    p.add_argument("--tpu-zone", default="us-central2-b")
    p.add_argument("--tpu-project", default="phantom-trc")
    p.add_argument("--tpu-repo-dir", default="/tmp/PHANTOM")
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--workdir", default=str(Path(__file__).resolve().parents[1]))
    args = p.parse_args()

    workdir = Path(args.workdir).resolve()
    env = os.environ.copy()
    wandb_root = workdir / ".wandb-agent"
    wandb_root.mkdir(parents=True, exist_ok=True)

    prepare_cmd = [
        "make",
        "train.tpu.vm.prepare",
        f"TPU_NAME={args.tpu_name}",
        f"TPU_ZONE={args.tpu_zone}",
        f"TPU_PROJECT={args.tpu_project}",
        f"TPU_REPO_DIR={args.tpu_repo_dir}",
    ]
    prepare = subprocess.run(
        prepare_cmd,
        cwd=workdir,
        env=env,
        text=True,
        capture_output=False,
        check=False,
    )
    if prepare.returncode != 0:
        raise RuntimeError("Failed to prepare TPU workers for sweep")

    def run_trial() -> None:
        run = None
        trial_wandb_dir = wandb_root / f"trial-{time.time_ns()}"
        trial_wandb_dir.mkdir(parents=True, exist_ok=True)
        try:
            run = wandb.init(dir=str(trial_wandb_dir))
            cfg = dict(wandb.config)
            cli_args = _to_cli_args(cfg)
            env_trial = dict(env)
            env_trial["LOCAL_TRAIN_ARGS"] = cli_args
            env_trial["WANDB_DIR"] = str(trial_wandb_dir)
            env_trial["WANDB_CACHE_DIR"] = str(trial_wandb_dir / "cache")
            env_trial["WANDB_DATA_DIR"] = str(trial_wandb_dir / "data")

            cmd = [
                "make",
                "train.tpu.vm.run",
                f"TPU_NAME={args.tpu_name}",
                f"TPU_ZONE={args.tpu_zone}",
                f"TPU_PROJECT={args.tpu_project}",
                f"TPU_REPO_DIR={args.tpu_repo_dir}",
            ]

            proc = subprocess.run(
                cmd,
                cwd=workdir,
                env=env_trial,
                text=True,
                capture_output=True,
                check=False,
            )

            if proc.stdout:
                print(proc.stdout)
            if proc.stderr:
                print(proc.stderr)

            if proc.returncode != 0:
                if run is not None:
                    run.summary["runner/exit_code"] = proc.returncode
                raise RuntimeError(f"TPU trial failed with exit code {proc.returncode}")

            metrics = _extract_metrics(proc.stdout)
            if metrics:
                wandb.log(metrics)
                for k, v in metrics.items():
                    run.summary[k] = v
            run.summary["runner/exit_code"] = 0
        except Exception:
            time.sleep(2)
            raise
        finally:
            if run is not None and wandb.run is not None:
                wandb.finish()
            shutil.rmtree(trial_wandb_dir, ignore_errors=True)
            gc.collect()

    wandb.agent(
        args.sweep_id,
        function=run_trial,
        count=args.count if args.count > 0 else None,
    )


if __name__ == "__main__":
    main()
