#!/usr/bin/env sh
set -eu

REPO_DIR="${REPO_DIR:-$HOME/PHANTOM}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_ARGS="${TRAIN_ARGS:---algo ppo --jax --total-timesteps 200000 --jax-num-envs 32 --jax-num-steps 128 --jax-num-minibatches 4 --jax-update-epochs 4}"
EXTRA_PIP="${EXTRA_PIP:-flax optax distrax}"
INSTALL_FULL_REQUIREMENTS="${INSTALL_FULL_REQUIREMENTS:-0}"

if [ ! -d "$REPO_DIR" ]; then
  echo "repo directory not found: $REPO_DIR"
  exit 1
fi

cd "$REPO_DIR"

if [ -d "wandb" ]; then
  rm -rf wandb
fi

# keep install idempotent and avoid re-installing jax/libtpu each run
if [ "$INSTALL_FULL_REQUIREMENTS" = "1" ] && [ -f "requirements.txt" ]; then
  $PYTHON_BIN -m pip install -r requirements.txt
fi
if ! $PYTHON_BIN -c 'import flax, optax, distrax' >/dev/null 2>&1; then
  if [ -f "engine/jax/requirements.txt" ]; then
    $PYTHON_BIN -m pip install -r engine/jax/requirements.txt
  fi
  $PYTHON_BIN -m pip install -U $EXTRA_PIP
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  if ! $PYTHON_BIN -c 'import wandb; import inspect; assert hasattr(wandb, "init") and callable(wandb.init)' >/dev/null 2>&1; then
    $PYTHON_BIN -m pip install -U wandb
  fi
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  export WANDB_API_KEY
  exec $PYTHON_BIN -m engine.train $TRAIN_ARGS
fi

exec $PYTHON_BIN -m engine.train $TRAIN_ARGS --no-wandb
