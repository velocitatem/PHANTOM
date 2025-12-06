#!/bin/bash
set -e

# init db and create admin user on first run
airflow db migrate

# create admin user if not exists
airflow users create \
    --username "${AIRFLOW_ADMIN_USER:-admin}" \
    --password "${AIRFLOW_ADMIN_PASSWORD:-admin}" \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com || true

# start scheduler in background
airflow scheduler &

# start webserver in foreground (Railway needs one foreground process)
exec airflow webserver --port ${PORT:-8080}
