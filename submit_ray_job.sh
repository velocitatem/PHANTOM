#!/bin/bash
# Submits PHANTOM training to a Ray cluster with .env injection.
# Modes:
#   RAY_MODE=single       -> one run (default)
#   RAY_MODE=distributed  -> one run per TPU node (experimental)
#   RAY_MODE=benchmark    -> one benchmark run per TPU node (overnight)
#   RAY_MODE=sweep        -> distributed W&B sweep agents

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
for k in (
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "PHANTOM_BENCHMARK_COMPARE_ROBUST",
    "PHANTOM_JAX_PLATFORM",
    "PHANTOM_ALLOW_MULTI_NODE_TPU",
    "PHANTOM_TPU_AGENT_SLOTS",
):
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
INNER_WORKERS="${INNER_WORKERS:-16}"
INNER_THREADS="${INNER_THREADS:-1}"
MAX_HEAVY_WORKERS="${MAX_HEAVY_WORKERS:-3}"
WORKER_CPUS="${WORKER_CPUS:-$((INNER_WORKERS * INNER_THREADS))}"
SWEEP_KIND="${SWEEP_KIND:-benchmark}"
SWEEP_METHOD="${SWEEP_METHOD:-random}"
SWEEP_RUN_CAP="${SWEEP_RUN_CAP:-0}"
AGENTS_PER_NODE="${AGENTS_PER_NODE:-16}"
AGENT_COUNT="${AGENT_COUNT:-0}"

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
    --inner-workers "${INNER_WORKERS}"
    --inner-threads "${INNER_THREADS}"
    --max-heavy-workers "${MAX_HEAVY_WORKERS}"
    --worker-cpus "${WORKER_CPUS}"
  )
  if [ "${COMPARE_ROBUST:-1}" = "1" ]; then
    DIST_ARGS+=(--compare-robust)
  fi
  "$RAY_BIN" "${COMMON_ARGS[@]}" "${DIST_ARGS[@]}"
  exit 0
fi

if [ "$RAY_MODE" = "sweep" ]; then
  SWEEP_PROJECT="${WANDB_PROJECT:-capstone_tpu}"
  SWEEP_ENTITY="${WANDB_ENTITY:-lusiana}"
  SWEEP_ID_VALUE="${SWEEP_ID:-}"
  SWEEP_NUM_NODES="${NUM_NODES:-5}"
  PY_SWEEP_BIN="${PY_SWEEP_BIN:-}"
  if [ -z "$PY_SWEEP_BIN" ]; then
    for cand in "$ROOT/.venv/bin/python" "$ROOT/.venv-ray/bin/python" python3 python; do
      if [ "$cand" = "python3" ] || [ "$cand" = "python" ]; then
        command -v "$cand" >/dev/null 2>&1 || continue
      elif [ ! -x "$cand" ]; then
        continue
      fi
      if "$cand" - <<'PY' >/dev/null 2>&1
import sys
from pathlib import Path
cwd = str(Path.cwd())
sys.path = [p for p in sys.path if p not in {'', cwd}]
import wandb
print(wandb.__name__)
PY
      then
        PY_SWEEP_BIN="$cand"
        break
      fi
    done
  fi
  if [ -z "$PY_SWEEP_BIN" ]; then
    echo "No python interpreter with wandb is available for sweep creation." >&2
    exit 1
  fi

  if [ -z "$SWEEP_ID_VALUE" ]; then
    if [ -z "${WANDB_API_KEY:-}" ]; then
      export WANDB_API_KEY
      WANDB_API_KEY="$($PY_SWEEP_BIN - <<'PY'
from dotenv import dotenv_values
print(dotenv_values('.env').get('WANDB_API_KEY', '').strip())
PY
)"
    fi
    if [ -z "${WANDB_API_KEY:-}" ]; then
      echo "WANDB_API_KEY is required to create a sweep." >&2
      exit 1
    fi
    SWEEP_ID_VALUE="$($PY_SWEEP_BIN "$ROOT/scripts/wandb_create_sweep.py" \
      --kind "$SWEEP_KIND" \
      --project "$SWEEP_PROJECT" \
      --entity "$SWEEP_ENTITY" \
      --method "$SWEEP_METHOD" \
      --run-cap "$SWEEP_RUN_CAP")"
  fi

  SWEEP_ENTRY_ARGS="${SWEEP_ENTRY_ARGS:-}"
  if [ -z "$SWEEP_ENTRY_ARGS" ]; then
    SWEEP_ENTRY_ARGS="--sweep-agent --sweep-id $SWEEP_ID_VALUE --project $SWEEP_PROJECT --device cpu"
  fi

  if [ "$AGENT_COUNT" = "0" ] && [ "${SWEEP_RUN_CAP:-0}" -gt 0 ]; then
    TOTAL_AGENTS=$((SWEEP_NUM_NODES * AGENTS_PER_NODE))
    if [ "$TOTAL_AGENTS" -gt 0 ]; then
      AGENT_COUNT=$(((SWEEP_RUN_CAP + TOTAL_AGENTS - 1) / TOTAL_AGENTS))
      echo "Derived AGENT_COUNT=$AGENT_COUNT from SWEEP_RUN_CAP=$SWEEP_RUN_CAP across $TOTAL_AGENTS agents"
    fi
  fi

  DIST_ARGS=(
    python
    scripts/ray_distributed_train.py
    --run-kind "$SWEEP_KIND"
    --entry-args "$SWEEP_ENTRY_ARGS"
    --num-nodes "${SWEEP_NUM_NODES}"
    --tpu-per-task "${TPU_PER_TASK:-0}"
    --base-seed "${BASE_SEED:-42}"
    --wandb-entity "$SWEEP_ENTITY"
    --wandb-project "$SWEEP_PROJECT"
    --agents-per-node "$AGENTS_PER_NODE"
    --agent-count "$AGENT_COUNT"
    --inner-threads "$INNER_THREADS"
    --worker-cpus "${WORKER_CPUS:-$((AGENTS_PER_NODE * INNER_THREADS))}"
  )
  if [ "$SWEEP_KIND" = "benchmark" ]; then
    DIST_ARGS+=(--output-root "${OUTPUT_ROOT:-engine/studies/results/sweeps}")
  fi
  if [ "${COMPARE_ROBUST:-0}" = "1" ]; then
    DIST_ARGS+=(--compare-robust)
  fi
  echo "SWEEP_ID=$SWEEP_ID_VALUE"
  "$RAY_BIN" "${COMMON_ARGS[@]}" "${DIST_ARGS[@]}"
  exit 0
fi

echo "Unsupported RAY_MODE='$RAY_MODE' (expected 'single', 'distributed', 'benchmark', or 'sweep')." >&2
exit 1
