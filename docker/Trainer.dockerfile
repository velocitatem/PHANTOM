# syntax=docker/dockerfile:1.7

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS gpu

WORKDIR /app

COPY docker/trainer.requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Optional for JAX-on-GPU workflows.
ARG INSTALL_JAX_GPU=false
RUN if [ "${INSTALL_JAX_GPU}" = "true" ]; then \
      pip install --no-cache-dir "jax[cuda12]==0.4.30" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html; \
    fi

COPY --chmod=755 docker/trainer-agent-entrypoint.sh /usr/local/bin/trainer-agent-entrypoint
COPY engine /app/engine

ENV PYTHONPATH=/app \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

ENTRYPOINT ["/usr/local/bin/trainer-agent-entrypoint"]


FROM python:3.11-slim AS tpu

WORKDIR /app

COPY docker/trainer.requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN pip install --no-cache-dir "jax[tpu]==0.4.30" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

COPY --chmod=755 docker/trainer-agent-entrypoint.sh /usr/local/bin/trainer-agent-entrypoint
COPY engine /app/engine

ENV PYTHONPATH=/app \
    PHANTOM_USE_JAX=1 \
    PHANTOM_DEFAULT_AGENT_ARGS="--jax" \
    JAX_PLATFORMS=tpu,cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

ENTRYPOINT ["/usr/local/bin/trainer-agent-entrypoint"]
