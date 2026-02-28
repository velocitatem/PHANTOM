#!/usr/bin/env sh
set -eu

TPU_NAME="${TPU_NAME:?TPU_NAME is required}"
TPU_ZONE="${TPU_ZONE:-us-central2-b}"
TPU_PROJECT="${TPU_PROJECT:-phantom-trc}"
LOCAL_REPO_DIR="${LOCAL_REPO_DIR:-$(pwd)}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/tmp/PHANTOM}"
ARCHIVE_PATH="${ARCHIVE_PATH:-/tmp/phantom-sync.tgz}"

FILE_LIST="$(mktemp /tmp/phantom-sync-files.XXXXXX)"
CLEANUP_LIST=true

cleanup() {
  if [ "$CLEANUP_LIST" = "true" ]; then
    rm -f "$FILE_LIST"
  fi
}
trap cleanup EXIT

if [ ! -d "$LOCAL_REPO_DIR" ]; then
  echo "local repo directory not found: $LOCAL_REPO_DIR"
  exit 1
fi

if git -C "$LOCAL_REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$LOCAL_REPO_DIR" ls-files -co --exclude-standard > "$FILE_LIST"
  python3 - "$FILE_LIST" <<'PY'
import sys
from pathlib import Path

file_list = Path(sys.argv[1])
skip_prefixes = (
    "wandb/",
    ".venv/",
    "venv/",
    "node_modules/",
    ".next/",
    ".turbo/",
    "__pycache__/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    "paper/build/",
    "tests/e2e/test-results/",
)

rows = file_list.read_text().splitlines()
kept = [
    row
    for row in rows
    if row and not any(row == p.rstrip("/") or row.startswith(p) for p in skip_prefixes)
]
file_list.write_text("\n".join(kept) + ("\n" if kept else ""))
PY
  tar -czf "$ARCHIVE_PATH" -C "$LOCAL_REPO_DIR" -T "$FILE_LIST"
else
  tar \
    --exclude-vcs \
    --exclude=".venv" --exclude="*/.venv" \
    --exclude="venv" --exclude="*/venv" \
    --exclude="node_modules" --exclude="*/node_modules" \
    --exclude=".next" --exclude="*/.next" \
    --exclude=".turbo" --exclude="*/.turbo" \
    --exclude="__pycache__" --exclude="*/__pycache__" \
    --exclude=".mypy_cache" --exclude="*/.mypy_cache" \
    --exclude=".pytest_cache" --exclude="*/.pytest_cache" \
    --exclude=".ruff_cache" --exclude="*/.ruff_cache" \
    --exclude="wandb" --exclude="*/wandb" \
    --exclude="paper/build" \
    --exclude="tests/e2e/test-results" \
    -czf "$ARCHIVE_PATH" \
    -C "$LOCAL_REPO_DIR" .
fi

gcloud compute tpus tpu-vm scp "$ARCHIVE_PATH" "$TPU_NAME:/tmp/phantom-sync.tgz" \
  --zone="$TPU_ZONE" --project="$TPU_PROJECT" --worker=all

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$TPU_ZONE" --project="$TPU_PROJECT" --worker=all \
  --command="rm -rf '$REMOTE_REPO_DIR' && mkdir -p '$REMOTE_REPO_DIR' && tar -xzf /tmp/phantom-sync.tgz -C '$REMOTE_REPO_DIR' && rm -f /tmp/phantom-sync.tgz"

rm -f "$ARCHIVE_PATH"
