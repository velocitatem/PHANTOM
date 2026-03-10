# syntax=docker/dockerfile:1.7

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS gpu

WORKDIR /app

COPY docker/trainer.requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY --chmod=755 docker/trainer-agent-entrypoint.sh /usr/local/bin/trainer-agent-entrypoint
COPY engine /app/engine

ENV PYTHONPATH=/app

ENTRYPOINT ["/usr/local/bin/trainer-agent-entrypoint"]
