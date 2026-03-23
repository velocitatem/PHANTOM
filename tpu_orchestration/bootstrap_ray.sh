#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONF="${SCRIPT_DIR}/configs/v4_spot_us.conf"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_HOST="${RAY_DASHBOARD_HOST:-0.0.0.0}"
RAY_DASHBOARD_LOCAL_PORT="${RAY_DASHBOARD_LOCAL_PORT:-8265}"
RAY_CLIENT_LOCAL_PORT="${RAY_CLIENT_LOCAL_PORT:-10001}"
TPU_CHIPS_PER_HOST="${TPU_CHIPS_PER_HOST:-8}"
TPU_RESOURCE_PER_HOST="${TPU_RESOURCE_PER_HOST:-8}"

CONF_FILE="$DEFAULT_CONF"
DEPS_ONLY=0
VERIFY_ONLY=0
TEARDOWN=0

usage() {
  cat <<'EOF'
Usage: bootstrap_ray.sh [options]

Options:
  --conf <path>   Path to TPU config (default: tpu_orchestration/configs/v4_spot_us.conf)
  --deps-only     Install TPU dependencies on all workers and exit
  --verify-only   Run JAX distributed smoke test on all workers and exit
  --teardown      Stop Ray on all workers and head, then exit
  -h, --help      Show this help

Config file keys expected:
  ZONE, QR_NAME, ACCEL_TYPE

Optional env overrides:
  PROJECT_ID, TPU_CHIPS_PER_HOST, TPU_RESOURCE_PER_HOST,
  RAY_PORT, RAY_DASHBOARD_HOST, RAY_DASHBOARD_LOCAL_PORT, RAY_CLIENT_LOCAL_PORT
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "Missing required command: ${cmd}"
}

parse_args() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --conf)
        [ "$#" -ge 2 ] || die "--conf requires a path"
        CONF_FILE="$2"
        shift 2
        ;;
      --deps-only)
        DEPS_ONLY=1
        shift
        ;;
      --verify-only)
        VERIFY_ONLY=1
        shift
        ;;
      --teardown)
        TEARDOWN=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done
}

load_optional_sweep_env() {
  if [ -n "${SWEEP_ENV_FILE:-}" ] && [ -f "${SWEEP_ENV_FILE}" ]; then
    set -a
    . "${SWEEP_ENV_FILE}"
    set +a
    return
  fi

  local fallback_env="${SCRIPT_DIR}/../.env.sweep"
  if [ -f "$fallback_env" ]; then
    set -a
    . "$fallback_env"
    set +a
  fi
}

load_config() {
  [ -f "$CONF_FILE" ] || die "Config file not found: $CONF_FILE"
  # shellcheck disable=SC1090
  . "$CONF_FILE"

  [ -n "${ZONE:-}" ] || die "ZONE is required in config"
  [ -n "${QR_NAME:-}" ] || die "QR_NAME is required in config"
  [ -n "${ACCEL_TYPE:-}" ] || die "ACCEL_TYPE is required in config"
}

resolve_project() {
  if [ -n "${PROJECT_ID:-}" ]; then
    return
  fi

  local active_project
  active_project="$(gcloud config get-value project 2>/dev/null || true)"
  if [ -n "$active_project" ] && [ "$active_project" != "(unset)" ]; then
    PROJECT_ID="$active_project"
    return
  fi

  die "PROJECT_ID is not set and gcloud has no active project"
}

resolve_worker_count() {
  [ -n "$TPU_CHIPS_PER_HOST" ] || die "TPU_CHIPS_PER_HOST must be set"
  [[ "$TPU_CHIPS_PER_HOST" =~ ^[0-9]+$ ]] || die "TPU_CHIPS_PER_HOST must be numeric"
  [ "$TPU_CHIPS_PER_HOST" -gt 0 ] || die "TPU_CHIPS_PER_HOST must be > 0"

  local total_chips
  if [[ "$ACCEL_TYPE" =~ ([0-9]+)$ ]]; then
    total_chips="${BASH_REMATCH[1]}"
  else
    die "Unable to parse total chips from ACCEL_TYPE=$ACCEL_TYPE"
  fi

  if [ $((total_chips % TPU_CHIPS_PER_HOST)) -ne 0 ]; then
    die "ACCEL_TYPE=$ACCEL_TYPE is not divisible by TPU_CHIPS_PER_HOST=$TPU_CHIPS_PER_HOST"
  fi

  WORKER_COUNT=$((total_chips / TPU_CHIPS_PER_HOST))
  [ "$WORKER_COUNT" -gt 0 ] || die "Computed worker count must be > 0"
}

run_tpu_ssh() {
  local worker="$1"
  local remote_cmd="$2"
  local args=(compute tpus tpu-vm ssh "$QR_NAME" --zone "$ZONE" --project "$PROJECT_ID" --worker="$worker" --quiet --command "$remote_cmd")
  gcloud "${args[@]}"
}

install_deps() {
  local cmd='python3 -m pip install --user --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html stable-baselines3 gymnasium wandb tensorboard "ray[default]"'
  log "Installing JAX and Ray dependencies on all workers"
  run_tpu_ssh "all" "$cmd"
}

verify_jax() {
  local cmd='python3 -c "import jax; jax.distributed.initialize(); print(f\"process_index={jax.process_index()} local_devices={jax.local_device_count()} global_devices={jax.device_count()}\")"'
  log "Running JAX distributed smoke test on all workers"
  run_tpu_ssh "all" "$cmd"
}

start_ray_head() {
  local resources_json="{\"TPU\":${TPU_RESOURCE_PER_HOST}}"
  local cmd="export PATH=\$HOME/.local/bin:\$PATH; ray stop >/dev/null 2>&1 || true; ray start --head --port=${RAY_PORT} --dashboard-host=${RAY_DASHBOARD_HOST} --resources='${resources_json}' --disable-usage-stats"
  log "Starting Ray head on worker 0"
  run_tpu_ssh "0" "$cmd"
}

get_head_ip() {
  local cmd="hostname -I | awk '{print \$1}'"
  local head_ip
  head_ip="$(run_tpu_ssh "0" "$cmd" | awk 'NF { ip=$1 } END { print ip }')"
  [ -n "$head_ip" ] || die "Failed to resolve Ray head IP"
  printf '%s\n' "$head_ip"
}

start_ray_workers() {
  local head_ip="$1"
  local resources_json="{\"TPU\":${TPU_RESOURCE_PER_HOST}}"
  local cmd
  cmd="export PATH=\$HOME/.local/bin:\$PATH; ray stop >/dev/null 2>&1 || true; ray start --address=${head_ip}:${RAY_PORT} --resources='${resources_json}' --disable-usage-stats"

  if [ "$WORKER_COUNT" -le 1 ]; then
    log "Single-worker topology detected; skipping worker join step"
    return
  fi

  local worker
  for ((worker = 1; worker < WORKER_COUNT; worker++)); do
    log "Starting Ray worker on worker ${worker}"
    run_tpu_ssh "$worker" "$cmd"
  done
}

verify_ray_cluster() {
  local cmd='export PATH=$HOME/.local/bin:$PATH; ray status'
  log "Checking Ray cluster status from worker 0"
  run_tpu_ssh "0" "$cmd"
}

print_tunnel_hint() {
  cat <<EOF

Ray tunnel command:
gcloud compute tpus tpu-vm ssh ${QR_NAME} --zone ${ZONE} --project ${PROJECT_ID} --worker=0 -- -L ${RAY_DASHBOARD_LOCAL_PORT}:localhost:${RAY_DASHBOARD_LOCAL_PORT} -L ${RAY_CLIENT_LOCAL_PORT}:localhost:${RAY_CLIENT_LOCAL_PORT} -N
EOF
}

teardown_ray() {
  local cmd='export PATH=$HOME/.local/bin:$PATH; ray stop'
  local failures=0
  local worker

  if [ "$WORKER_COUNT" -gt 1 ]; then
    for ((worker = WORKER_COUNT - 1; worker >= 1; worker--)); do
      log "Stopping Ray on worker ${worker}"
      if ! run_tpu_ssh "$worker" "$cmd"; then
        failures=$((failures + 1))
      fi
    done
  fi

  log "Stopping Ray head on worker 0"
  if ! run_tpu_ssh "0" "$cmd"; then
    failures=$((failures + 1))
  fi

  [ "$failures" -eq 0 ] || die "Teardown completed with ${failures} failure(s)"
}

main() {
  parse_args "$@"
  require_cmd gcloud

  load_optional_sweep_env
  load_config
  resolve_project
  resolve_worker_count

  log "Target TPU: ${QR_NAME} (${ACCEL_TYPE}) in ${ZONE}"
  log "Computed workers: ${WORKER_COUNT} (chips per host: ${TPU_CHIPS_PER_HOST})"

  if [ "$TEARDOWN" -eq 1 ]; then
    teardown_ray
    return
  fi

  if [ "$DEPS_ONLY" -eq 1 ] && [ "$VERIFY_ONLY" -eq 1 ]; then
    install_deps
    verify_jax
    return
  fi

  if [ "$DEPS_ONLY" -eq 1 ]; then
    install_deps
    return
  fi

  if [ "$VERIFY_ONLY" -eq 1 ]; then
    verify_jax
    return
  fi

  install_deps
  verify_jax

  start_ray_head
  local head_ip
  head_ip="$(get_head_ip)"
  log "Ray head IP: ${head_ip}"

  start_ray_workers "$head_ip"
  verify_ray_cluster
  print_tunnel_hint
}

main "$@"
