#!/bin/bash
# Submits PHANTOM training to a Ray cluster with .env injection.
# Modes:
#   RAY_MODE=single       -> one run (default)
#   RAY_MODE=distributed  -> one run per TPU node (experimental)
#   RAY_MODE=benchmark    -> one benchmark run per TPU node (overnight)

set -euo pipefail

ROOT="/home/velocitatem/Documents/Projects/PHANTOM"
RAY_BIN="${RAY_BIN:-ray}"
if ! command -v "$RAY_BIN" >/dev/null 2>&1; then
  if [ -x "$ROOT/.venv-ray/bin/ray" ]; then
    RAY_BIN="$ROOT/.venv-ray/bin/ray"
  else
    echo "ray CLI not found. Activate .venv-ray or set RAY_BIN." >&2
    exit 1
  fi
fi

# 1. Parse .env and generate the JSON payload for Ray
export RUNTIME_ENV_JSON=$(python -c '
import json
import os
from dotenv import dotenv_values

env = dotenv_values(".env")
# Filter out empty/None values
env_vars = {k: v for k, v in env.items() if v}
env_vars.setdefault("CLOUD_TPU_TASK_ID", os.getenv("CLOUD_TPU_TASK_ID", "0"))
for k in ("WANDB_ENTITY", "WANDB_PROJECT", "PHANTOM_BENCHMARK_COMPARE_ROBUST"):
    if os.getenv(k):
        env_vars[k] = os.getenv(k)

print(json.dumps({
    "pip": [
        "stable-baselines3>=2.2.0", 
        "gymnasium>=0.29.0", 
        "wandb", 
        "tensorboard",
        "python-dotenv",
        "pandas",
        "pydantic",
        "graphviz",
        "huggingface_hub",
        "matplotlib"
    ],
    "env_vars": env_vars
}))
')

RAY_MODE="${RAY_MODE:-single}"
TRAIN_ARGS="${TRAIN_ARGS:---algo ppo --total-timesteps 1000000}"
BENCHMARK_ARGS="${BENCHMARK_ARGS:---project capstone_tpu --tiers static,surge,linear,qtable,ppo --alpha-values 0.0,0.1,0.25,0.4,0.6,0.8 --episodes 12 --total-timesteps 30000 --max-steps 100 --robust-radius 0.2 --robust-points 7 --robust-rollouts 1 --lambda-coi 0.2 --eta-ux 0.5 --reward-profit-weight 1.0 --device cpu}"

SUBMIT_ARGS=()
if [ "${RAY_NO_WAIT:-0}" = "1" ]; then
  SUBMIT_ARGS+=(--no-wait)
fi
if [ -n "${SUBMISSION_ID:-}" ]; then
  SUBMIT_ARGS+=(--submission-id "$SUBMISSION_ID")
fi

COMMON_ARGS=(
  job submit
  --address http://localhost:8265
  --working-dir "$ROOT"
  --runtime-env-json "$RUNTIME_ENV_JSON"
  "${SUBMIT_ARGS[@]}"
  --
)

if [ "$RAY_MODE" = "single" ]; then
  read -r -a TRAIN_TOKENS <<< "$TRAIN_ARGS"
  "$RAY_BIN" "${COMMON_ARGS[@]}" python -m engine.train "${TRAIN_TOKENS[@]}"
  exit 0
fi

if [ "$RAY_MODE" = "distributed" ]; then
  DIST_ARGS=(
    python
    scripts/ray_distributed_train.py
    --train-args "$TRAIN_ARGS"
    --num-nodes "${NUM_NODES:-4}"
    --tpu-per-task "${TPU_PER_TASK:-8}"
    --base-seed "${BASE_SEED:-42}"
  )
  if [ "${SYNC_JAX:-0}" = "1" ]; then
    DIST_ARGS+=(--sync-jax)
  fi
  "$RAY_BIN" "${COMMON_ARGS[@]}" "${DIST_ARGS[@]}"
  exit 0
fi

if [ "$RAY_MODE" = "benchmark" ]; then
  DIST_ARGS=(
    python
    scripts/ray_distributed_train.py
    --run-kind benchmark
    --entry-args "$BENCHMARK_ARGS"
    --num-nodes "${NUM_NODES:-4}"
    --tpu-per-task "${TPU_PER_TASK:-8}"
    --base-seed "${BASE_SEED:-42}"
    --output-root "${OUTPUT_ROOT:-engine/studies/results/overnight}"
    --wandb-entity "${WANDB_ENTITY:-lusiana}"
    --wandb-project "${WANDB_PROJECT:-capstone_tpu}"
  )
  if [ "${COMPARE_ROBUST:-1}" = "1" ]; then
    DIST_ARGS+=(--compare-robust)
  fi
  "$RAY_BIN" "${COMMON_ARGS[@]}" "${DIST_ARGS[@]}"
  exit 0
fi

echo "Unsupported RAY_MODE='$RAY_MODE' (expected 'single', 'distributed', or 'benchmark')." >&2
exit 1
