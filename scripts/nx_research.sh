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
    WANDB_PROJECT="${WANDB_PROJECT:-capstone}" \
    WANDB_API_KEY="$WANDB_API_KEY" \
      .venv/bin/python -m engine.train ${LOCAL_TRAIN_ARGS:---algo ppo --total-timesteps 50000}
    ;;
  benchmark)
    load_sweep_env
    if [[ " ${LOCAL_BENCHMARK_ARGS:-} " != *" --no-wandb "* ]]; then
      require_var WANDB_API_KEY "WANDB_API_KEY required - set it in $env_file"
    fi
    WANDB_ENTITY="${WANDB_ENTITY:-}" \
    WANDB_PROJECT="${WANDB_PROJECT:-capstone}" \
    WANDB_API_KEY="${WANDB_API_KEY:-}" \
      .venv/bin/python -m engine.train --run-kind benchmark ${LOCAL_BENCHMARK_ARGS:---tiers static,surge,linear,qtable,ppo --alpha-values 0.0,0.3 --episodes 3 --total-timesteps 3000 --max-steps 40 --device cpu}
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
    WANDB_PROJECT="${WANDB_PROJECT:-capstone}" \
    WANDB_API_KEY="$WANDB_API_KEY" \
      .venv/bin/python -m engine.train "${args[@]}"
    ;;
  benchmark-agent)
    load_sweep_env
    require_var WANDB_API_KEY "WANDB_API_KEY required - set it in $env_file"
    require_var SWEEP_ID "SWEEP_ID required, e.g. SWEEP_ID=entity/project/id"
    args=(--sweep-agent --sweep-id "$SWEEP_ID")
    if [ -n "${AGENT_COUNT:-}" ] && [ "${AGENT_COUNT}" != "0" ]; then
      args+=(--count "$AGENT_COUNT")
    fi
    WANDB_ENTITY="${WANDB_ENTITY:-}" \
    WANDB_PROJECT="${WANDB_PROJECT:-capstone}" \
    WANDB_API_KEY="$WANDB_API_KEY" \
      .venv/bin/python -m engine.train --run-kind benchmark "${args[@]}" ${BENCHMARK_AGENT_ARGS:-}
    ;;
  train-bootstrap)
    load_sweep_env
    require_var WANDB_API_KEY "WANDB_API_KEY required - set it in $env_file"
    require_var GITHUB_TOKEN "GITHUB_TOKEN required - set it in $env_file"
    require_var REPO_URL "REPO_URL required, e.g. REPO_URL=https://github.com/org/repo.git"
    require_var SWEEP_ID "SWEEP_ID required, e.g. SWEEP_ID=entity/project/id"
    WANDB_API_KEY="$WANDB_API_KEY" \
    WANDB_ENTITY="${WANDB_ENTITY:-}" \
    WANDB_PROJECT="${WANDB_PROJECT:-capstone}" \
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
    ;;
  *)
    printf '%s\n' "Unknown research command: $cmd" >&2
    exit 1
    ;;
esac
