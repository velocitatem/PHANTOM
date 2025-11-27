FROM apache/airflow:2.7.3-python3.11

USER root

# install system deps if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# copy requirements for pipeline dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# install postgres driver and providers
RUN pip install --no-cache-dir \
    psycopg2-binary \
    apache-airflow-providers-postgres

# set airflow home
ENV AIRFLOW_HOME=/opt/airflow
