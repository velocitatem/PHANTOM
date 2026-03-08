#!/usr/bin/env bash

set -euo pipefail

cmd="${1:-}"
env_file="${SWEEP_ENV_FILE:-.env.sweep}"

load_sweep_env() {
  set -a
  [ -f "$env_file" ] && . "$env_file" || true
  set +a
}

require_var() {
  local name="$1"
  local msg="$2"
  if [ -z "${!name:-}" ]; then
    printf '%s\n' "$msg" >&2
    exit 1
  fi
}

case "$cmd" in
  install)
    [ -x .venv/bin/python ] || python3 -m venv .venv
    .venv/bin/python -m ensurepip --upgrade
    .venv/bin/python -m pip install -r requirements.txt
    ;;
  train)
    load_sweep_env
    require_var WANDB_API_KEY "WANDB_API_KEY required - set it in $env_file"
    WANDB_ENTITY="${WANDB_ENTITY:-}" \
    WANDB_PROJECT="${WANDB_PROJECT:-phantom-pricing}" \
    WANDB_API_KEY="$WANDB_API_KEY" \
      .venv/bin/python -m engine.train ${LOCAL_TRAIN_ARGS:---algo ppo --total-timesteps 50000}
    ;;
  train-agent)
    load_sweep_env
    require_var WANDB_API_KEY "WANDB_API_KEY required - set it in $env_file"
    require_var SWEEP_ID "SWEEP_ID required, e.g. SWEEP_ID=entity/project/id"
    args=(--sweep-agent --sweep-id "$SWEEP_ID")
    if [ -n "${AGENT_COUNT:-}" ] && [ "${AGENT_COUNT}" != "0" ]; then
      args+=(--count "$AGENT_COUNT")
    fi
    WANDB_ENTITY="${WANDB_ENTITY:-}" \
    WANDB_PROJECT="${WANDB_PROJECT:-phantom-pricing}" \
    WANDB_API_KEY="$WANDB_API_KEY" \
      .venv/bin/python -m engine.train "${args[@]}"
    ;;
  train-bootstrap)
    load_sweep_env
    require_var WANDB_API_KEY "WANDB_API_KEY required - set it in $env_file"
    require_var GITHUB_TOKEN "GITHUB_TOKEN required - set it in $env_file"
    require_var REPO_URL "REPO_URL required, e.g. REPO_URL=https://github.com/org/repo.git"
    require_var SWEEP_ID "SWEEP_ID required, e.g. SWEEP_ID=entity/project/id"
    WANDB_API_KEY="$WANDB_API_KEY" \
    WANDB_ENTITY="${WANDB_ENTITY:-}" \
    WANDB_PROJECT="${WANDB_PROJECT:-phantom-pricing}" \
    GITHUB_TOKEN="$GITHUB_TOKEN" \
    REPO_URL="$REPO_URL" \
    BRANCH="${BRANCH:-main}" \
    WORKDIR="${WORKDIR:-$HOME/PHANTOM-agent}" \
    SWEEP_ID="$SWEEP_ID" \
    AGENT_COUNT="${AGENT_COUNT:-0}" \
    AGENT_LOOP="${AGENT_LOOP:-1}" \
    RETRY_SECONDS="${RETRY_SECONDS:-20}" \
      bash scripts/wandb_agent_bootstrap.sh
    ;;
  stats)
    python3 - <<'PY'
from pathlib import Path

skip = {"node_modules", ".venv", "venv"}
exts = {".ts", ".py"}
total = 0
for path in Path(".").rglob("*"):
    if not path.is_file() or path.suffix not in exts or any(part in skip for part in path.parts):
        continue
    text = path.read_text(errors="ignore")
    total += text.count("\n") + (1 if text and not text.endswith("\n") else 0)
print(total)
PY
    ;;
  docker-train-publish)
    image_ref="${TRAIN_IMAGE_REF:-us-central1-docker.pkg.dev/phantom-trc/phantom/phantom-trainer}"
    docker build -f docker/Trainer.dockerfile --target gpu -t "$image_ref:gpu-latest" .
    docker push "$image_ref:gpu-latest"
    docker build -f docker/Trainer.dockerfile --target tpu -t "$image_ref:tpu-latest" .
    docker push "$image_ref:tpu-latest"
    ;;
  train-tpu-pod)
    load_sweep_env
    require_var TPU_NAME "TPU_NAME required, e.g. TPU_NAME=TPUlong"
    require_var SWEEP_ID "SWEEP_ID required, e.g. SWEEP_ID=entity/project/id"
    require_var WANDB_API_KEY "WANDB_API_KEY required - set it in $env_file"
    gcloud compute tpus tpu-vm scp scripts/tpu_pod_run.sh "$TPU_NAME":/tmp/tpu_pod_run.sh --zone="${TPU_ZONE:-us-central2-b}" --project="${TPU_PROJECT:-phantom-trc}" --worker=all
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="${TPU_ZONE:-us-central2-b}" --project="${TPU_PROJECT:-phantom-trc}" --worker=all --command="WANDB_API_KEY='$WANDB_API_KEY' SWEEP_ID='$SWEEP_ID' AGENT_COUNT='${AGENT_COUNT:-0}' sh /tmp/tpu_pod_run.sh"
    ;;
  train-tpu-vm-prepare)
    require_var TPU_NAME "TPU_NAME required, e.g. TPU_NAME=TPUlong"
    TPU_NAME="$TPU_NAME" \
    TPU_ZONE="${TPU_ZONE:-us-central2-b}" \
    TPU_PROJECT="${TPU_PROJECT:-phantom-trc}" \
    LOCAL_REPO_DIR="$PWD" \
    REMOTE_REPO_DIR="${TPU_REPO_DIR:-/tmp/PHANTOM}" \
      sh scripts/tpu_sync_repo.sh
    gcloud compute tpus tpu-vm scp scripts/tpu_vm_train.sh "$TPU_NAME":/tmp/tpu_vm_train.sh --zone="${TPU_ZONE:-us-central2-b}" --project="${TPU_PROJECT:-phantom-trc}" --worker=all
    ;;
  train-tpu-vm-run)
    load_sweep_env
    require_var TPU_NAME "TPU_NAME required, e.g. TPU_NAME=TPUlong"
    require_var LOCAL_TRAIN_ARGS "LOCAL_TRAIN_ARGS required, e.g. --algo ppo --jax --total-timesteps 200000"
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="${TPU_ZONE:-us-central2-b}" --project="${TPU_PROJECT:-phantom-trc}" --worker=all --command="REPO_DIR='${TPU_REPO_DIR:-/tmp/PHANTOM}' TRAIN_ARGS='${LOCAL_TRAIN_ARGS}' WANDB_API_KEY='${WANDB_API_KEY:-}' sh /tmp/tpu_vm_train.sh"
    ;;
  train-tpu-vm-sweep)
    load_sweep_env
    require_var TPU_NAME "TPU_NAME required, e.g. TPU_NAME=TPUlong"
    require_var SWEEP_ID "SWEEP_ID required, e.g. SWEEP_ID=lusiana/phantom-pricing/abc123"
    require_var WANDB_API_KEY "WANDB_API_KEY required - set it in $env_file"
    args=(
      --sweep-id "$SWEEP_ID"
      --tpu-name "$TPU_NAME"
      --tpu-zone "${TPU_ZONE:-us-central2-b}"
      --tpu-project "${TPU_PROJECT:-phantom-trc}"
      --tpu-repo-dir "${TPU_REPO_DIR:-/tmp/PHANTOM}"
    )
    if [ -n "${AGENT_COUNT:-}" ] && [ "${AGENT_COUNT}" != "0" ]; then
      args+=(--count "$AGENT_COUNT")
    fi
    WANDB_API_KEY="$WANDB_API_KEY" python3 scripts/tpu_vm_sweep_agent.py "${args[@]}"
    ;;
  *)
    printf '%s\n' "Unknown research command: $cmd" >&2
    exit 1
    ;;
esac
