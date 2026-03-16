#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RAY_MODE="${RAY_MODE:-sweep}"
export SWEEP_KIND="${SWEEP_KIND:-ppo_block_a}"
export SWEEP_METHOD="${SWEEP_METHOD:-grid}"
export SWEEP_PROFILE="${SWEEP_PROFILE:-default}"
export SWEEP_RUN_CAP="${SWEEP_RUN_CAP:-27}"
export COMPARE_ROBUST="${COMPARE_ROBUST:-1}"
export NUM_NODES="${NUM_NODES:-3}"
export AGENTS_PER_NODE="${AGENTS_PER_NODE:-4}"
export AGENT_COUNT="${AGENT_COUNT:-0}"
export INNER_THREADS="${INNER_THREADS:-1}"
export PHANTOM_JAX_PLATFORM="${PHANTOM_JAX_PLATFORM:-cpu}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-engine/studies/results/block_a_sweep}"

if [ -z "${WORKER_CPUS:-}" ]; then
  export WORKER_CPUS="$((AGENTS_PER_NODE * INNER_THREADS))"
fi

printf '%s\n' "Launching Block A PPO calibration sweep"
printf '%s\n' "RAY_MODE=$RAY_MODE"
printf '%s\n' "SWEEP_KIND=$SWEEP_KIND"
printf '%s\n' "SWEEP_METHOD=$SWEEP_METHOD"
printf '%s\n' "SWEEP_RUN_CAP=$SWEEP_RUN_CAP"
printf '%s\n' "COMPARE_ROBUST=$COMPARE_ROBUST"
printf '%s\n' "NUM_NODES=$NUM_NODES"
printf '%s\n' "AGENTS_PER_NODE=$AGENTS_PER_NODE"
printf '%s\n' "AGENT_COUNT=$AGENT_COUNT"
printf '%s\n' "INNER_THREADS=$INNER_THREADS"
printf '%s\n' "WORKER_CPUS=$WORKER_CPUS"
printf '%s\n' "OUTPUT_ROOT=$OUTPUT_ROOT"

cd "$ROOT"
bash ./submit_ray_job.sh
