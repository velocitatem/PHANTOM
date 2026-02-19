#!/usr/bin/env sh
set -eu

if [ -z "${SWEEP_ID:-}" ]; then
  echo "SWEEP_ID is required"
  exit 1
fi

set -- python -m engine.train --sweep-agent --sweep-id "${SWEEP_ID}"

if [ -n "${PHANTOM_DEFAULT_AGENT_ARGS:-}" ]; then
  set -- "$@" ${PHANTOM_DEFAULT_AGENT_ARGS}
fi

if [ -n "${TRAIN_ARGS:-}" ]; then
  set -- "$@" ${TRAIN_ARGS}
fi

if [ "${AGENT_COUNT:-0}" != "0" ]; then
  set -- "$@" --count "${AGENT_COUNT}"
fi

exec "$@"
