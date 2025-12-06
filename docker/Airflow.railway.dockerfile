FROM apache/airflow:2.7.3-python3.11

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    supervisor \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN pip install --no-cache-dir \
    psycopg2-binary \
    apache-airflow-providers-postgres

ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__EXECUTOR=SequentialExecutor
ENV AIRFLOW__CORE__LOAD_EXAMPLES=false
ENV AIRFLOW__CORE__ENABLE_XCOM_PICKLING=true
ENV AIRFLOW__WEBSERVER__EXPOSE_CONFIG=true

# copy all code into image (standalone - no volume mounts needed)
COPY --chown=airflow:root experiments/airflow/dags ${AIRFLOW_HOME}/dags
COPY --chown=airflow:root experiments/procesing ${AIRFLOW_HOME}/procesing
COPY --chown=airflow:root lib ${AIRFLOW_HOME}/lib

RUN mkdir -p ${AIRFLOW_HOME}/logs ${AIRFLOW_HOME}/plugins

# copy entrypoint script
COPY --chown=airflow:root docker/airflow-railway-entrypoint.sh /entrypoint.sh
USER root
RUN chmod +x /entrypoint.sh
USER airflow

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
