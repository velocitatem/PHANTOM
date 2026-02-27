#!/usr/bin/env bash
set -euo pipefail

need_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "$name is required"
    exit 1
  fi
}

need_cmd() {
  local c="$1"
  command -v "$c" >/dev/null 2>&1 || {
    echo "Missing command: $c"
    exit 1
  }
}

need_cmd git
need_cmd python3

need_env WANDB_API_KEY
need_env GITHUB_TOKEN
need_env REPO_URL
need_env SWEEP_ID

BRANCH="${BRANCH:-main}"
WORKDIR="${WORKDIR:-$HOME/PHANTOM-agent}"
AGENT_COUNT="${AGENT_COUNT:-0}"
AGENT_LOOP="${AGENT_LOOP:-1}"
RETRY_SECONDS="${RETRY_SECONDS:-20}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$(dirname "$WORKDIR")"

ASKPASS_FILE="$(mktemp)"
cat >"$ASKPASS_FILE" <<'EOF'
#!/usr/bin/env sh
case "$1" in
  *Username*) echo "x-access-token" ;;
  *Password*) echo "$GITHUB_TOKEN" ;;
  *) echo "" ;;
esac
EOF
chmod 700 "$ASKPASS_FILE"

cleanup() {
  rm -f "$ASKPASS_FILE"
}
trap cleanup EXIT

git_auth() {
  GIT_TERMINAL_PROMPT=0 GIT_ASKPASS="$ASKPASS_FILE" git "$@"
}

sync_repo() {
  if [ ! -d "$WORKDIR/.git" ]; then
    rm -rf "$WORKDIR"
    git_auth clone --single-branch --branch "$BRANCH" "$REPO_URL" "$WORKDIR"
    return
  fi

  git -C "$WORKDIR" remote set-url origin "$REPO_URL"
  git_auth -C "$WORKDIR" fetch origin "$BRANCH" --prune
  git -C "$WORKDIR" checkout -B "$BRANCH" "origin/$BRANCH"
  git -C "$WORKDIR" reset --hard "origin/$BRANCH"
}

install_deps() {
  "$PYTHON_BIN" -m venv "$WORKDIR/.venv"
  "$WORKDIR/.venv/bin/pip" install --upgrade pip
  "$WORKDIR/.venv/bin/pip" install -r "$WORKDIR/requirements.txt"
}

run_agent() {
  local cmd=("$WORKDIR/.venv/bin/python" -m engine.train --sweep-agent --sweep-id "$SWEEP_ID")
  if [ "$AGENT_COUNT" != "0" ]; then
    cmd+=(--count "$AGENT_COUNT")
  fi

  (
    cd "$WORKDIR"
    WANDB_API_KEY="$WANDB_API_KEY" \
    WANDB_ENTITY="${WANDB_ENTITY:-}" \
    WANDB_PROJECT="${WANDB_PROJECT:-}" \
    "${cmd[@]}"
  )
}

while true; do
  sync_repo
  install_deps

  if run_agent; then
    if [ "$AGENT_LOOP" = "1" ] && [ "$AGENT_COUNT" = "0" ]; then
      sleep "$RETRY_SECONDS"
      continue
    fi
    exit 0
  fi

  if [ "$AGENT_LOOP" != "1" ]; then
    exit 1
  fi

  sleep "$RETRY_SECONDS"
done
