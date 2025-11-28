from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import pandas as pd
import logging
import sys
import os

# add procesing module to path (mounted at /opt/airflow/procesing in container)
sys.path.insert(0, '/opt/airflow/procesing')

from extract import KafkaDataFetcher, ExperimentJoiner, EventTitleAugmenter
from demand import DemandEstimator, ChunkInteractionsIntoSteps
from elasticity import TemporalElasticityEstimator, aggregate_price_logs

default_args = {
    'owner': 'phantom-research',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# callable functions for tasks (stateless, idempotent)
def fetch_interactions(**context):
    """Extract interaction data from Kafka and augment"""
    fetcher = KafkaDataFetcher(topic='user-interactions')
    data = fetcher.fit_transform(None)

    if data.empty:
        logging.warning("No interaction data fetched")
        return None

    data = ExperimentJoiner().fit_transform(data)
    data = EventTitleAugmenter().fit_transform(data)

    # push to XCom for downstream tasks
    context['task_instance'].xcom_push(key='interaction_data', value=data.to_json())
    logging.info(f"Fetched {len(data)} interaction records")
    return len(data)

def fetch_price_logs(**context):
    """Extract price logs from Kafka"""
    fetcher = KafkaDataFetcher(topic='price-logs')
    data = fetcher.fit_transform(None)

    if data.empty:
        logging.warning("No price data fetched")
        return None

    context['task_instance'].xcom_push(key='price_data', value=data.to_json())
    logging.info(f"Fetched {len(data)} price records")
    return len(data)

def compute_demand_chunks(**context):
    """Chunk interactions and compute demand per window"""
    import io
    ti = context['task_instance']
    window_size = context['dag_run'].conf.get('window_size', '30s')
    store_mode = context['dag_run'].conf.get('store_mode', 'hotel')

    # pull from XCom
    interaction_json = ti.xcom_pull(task_ids='fetch_interactions', key='interaction_data')
    if not interaction_json:
        logging.error("No interaction data available")
        return None

    interactions_df = pd.read_json(io.StringIO(interaction_json))

    # chunk into windows
    chunker = ChunkInteractionsIntoSteps(window_size=window_size, return_metadata=True)
    chunks = chunker.transform(interactions_df)

    if not chunks:
        logging.warning("No chunks generated")
        return None

    # compute demand per chunk
    estimator = DemandEstimator(store_mode=store_mode)
    demand_chunks = [
        {
            'window_start': c['window_start'].isoformat(),
            'window_end': c['window_end'].isoformat(),
            'demand_vector': estimator.transform(c['data']).to_json()
        }
        for c in chunks
    ]

    ti.xcom_push(key='demand_chunks', value=demand_chunks)
    logging.info(f"Generated {len(demand_chunks)} demand chunks @ {window_size}")
    return len(demand_chunks)

def aggregate_prices(**context):
    """Aggregate price logs into aligned windows"""
    import io
    ti = context['task_instance']
    window_size = context['dag_run'].conf.get('window_size', '30s')
    store_mode = context['dag_run'].conf.get('store_mode', 'hotel')

    price_json = ti.xcom_pull(task_ids='fetch_price_logs', key='price_data')
    if not price_json:
        logging.error("No price data available")
        return None

    price_df = pd.read_json(io.StringIO(price_json))
    price_chunks = aggregate_price_logs(price_df, window_size=window_size, store_mode=store_mode)

    # serialize for XCom
    serialized = [
        {
            'window_start': c['window_start'].isoformat(),
            'window_end': c['window_end'].isoformat(),
            'price_vector': c['price_vector'].to_json()
        }
        for c in price_chunks
    ]

    ti.xcom_push(key='price_chunks', value=serialized)
    logging.info(f"Aggregated {len(price_chunks)} price chunks")
    return len(price_chunks)

def compute_elasticity(**context):
    """Compute price elasticity from demand and price chunks"""
    import io
    ti = context['task_instance']
    store_mode = context['dag_run'].conf.get('store_mode', 'hotel')
    method = context['dag_run'].conf.get('elasticity_method', 'point')
    min_obs = context['dag_run'].conf.get('min_observations', 2)

    # pull chunks from XCom
    demand_chunks_raw = ti.xcom_pull(task_ids='compute_demand', key='demand_chunks')
    price_chunks_raw = ti.xcom_pull(task_ids='aggregate_prices', key='price_chunks')

    if not demand_chunks_raw or not price_chunks_raw:
        logging.error("Missing demand or price chunks")
        return None

    # deserialize
    demand_chunks = [
        {
            'window_start': pd.Timestamp(c['window_start']),
            'window_end': pd.Timestamp(c['window_end']),
            'demand_vector': pd.read_json(io.StringIO(c['demand_vector']))
        }
        for c in demand_chunks_raw
    ]

    price_chunks = [
        {
            'window_start': pd.Timestamp(c['window_start']),
            'window_end': pd.Timestamp(c['window_end']),
            'price_vector': pd.read_json(io.StringIO(c['price_vector']))
        }
        for c in price_chunks_raw
    ]

    # compute elasticity
    estimator = TemporalElasticityEstimator(method=method, min_observations=min_obs)
    elasticity_df = estimator.transform(demand_chunks, price_chunks, store_mode=store_mode)

    if elasticity_df is None or elasticity_df.empty:
        logging.warning("No elasticity results computed")
        return None

    # store results (could push to DB, S3, or XCom)
    ti.xcom_push(key='elasticity_results', value=elasticity_df.to_json())
    logging.info(f"Computed elasticity for {len(elasticity_df)} products")

    # summary stats
    return {
        'n_products': len(elasticity_df),
        'mean_elasticity': float(elasticity_df['elasticity'].mean()),
        'median_elasticity': float(elasticity_df['elasticity'].median())
    }

def publish_results(**context):
    """Publish elasticity results to model registry and train pricing model"""
    import io
    ti = context['task_instance']
    elasticity_json = ti.xcom_pull(task_ids='compute_elasticity', key='elasticity_results')

    if not elasticity_json:
        logging.error("No elasticity results to publish")
        return None

    elasticity_df = pd.read_json(io.StringIO(elasticity_json))

    # import registry and pricing modules
    import sys
    sys.path.insert(0, '/opt/airflow/procesing') # this is pretty janky
    sys.path.insert(0, '/opt/airflow')

    from lib.model_registry import ModelRegistry
    from procesing.pricing import ElasticityBasedPricingFunction

    # initialize registry
    registry = ModelRegistry()

    # publish elasticity data
    metadata = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'window_size': context['dag_run'].conf.get('window_size', '30s'),
        'store_mode': context['dag_run'].conf.get('store_mode', 'hotel'),
        'dag_run_id': context['dag_run'].run_id
    }

    registry.publish_elasticity(
        elasticity_df,
        model_name='latest',
        metadata=metadata
    )

    # train and publish pricing model
    pricing_model = ElasticityBasedPricingFunction(
        cost_floor=0.5,
        max_markup=2.5,
        min_markup=1.0,
        inelastic_markup=1.2
    )
    pricing_model.fit(elasticity_df)

    registry.publish_pricing_model(
        pricing_model,
        model_name='latest',
        metadata={
            **metadata,
            'model_type': 'ElasticityBasedPricingFunction'
        }
    )

    logging.info(f"Published elasticity + pricing model for {len(elasticity_df)} products to registry")

    return {
        'n_products': len(elasticity_df),
        'registry_status': 'success',
        'elasticity_mean': float(elasticity_df['elasticity'].mean())
    }


# DAG definition
with DAG(
    'elasticity_pricing_pipeline',
    default_args=default_args,
    description='E2E pipeline: interactions -> demand -> elasticity -> pricing',
    schedule_interval='*/15 * * * *',  # every 5 minutes for real-time pricing
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['pricing', 'elasticity', 'research'],
) as dag:

    # parallel data fetching
    fetch_interactions_task = PythonOperator(
        task_id='fetch_interactions',
        python_callable=fetch_interactions,
        provide_context=True,
    )

    fetch_price_logs_task = PythonOperator(
        task_id='fetch_price_logs',
        python_callable=fetch_price_logs,
        provide_context=True,
    )

    # demand computation (depends on interactions)
    compute_demand_task = PythonOperator(
        task_id='compute_demand',
        python_callable=compute_demand_chunks,
        provide_context=True,
    )

    # price aggregation (depends on price logs)
    aggregate_prices_task = PythonOperator(
        task_id='aggregate_prices',
        python_callable=aggregate_prices,
        provide_context=True,
    )

    # elasticity computation (depends on both demand and prices)
    compute_elasticity_task = PythonOperator(
        task_id='compute_elasticity',
        python_callable=compute_elasticity,
        provide_context=True,
    )

    # publish results (depends on elasticity)
    publish_results_task = PythonOperator(
        task_id='publish_results',
        python_callable=publish_results,
        provide_context=True,
    )

    # dependency graph
    fetch_interactions_task >> compute_demand_task
    fetch_price_logs_task >> aggregate_prices_task
    [compute_demand_task, aggregate_prices_task] >> compute_elasticity_task
    compute_elasticity_task >> publish_results_task
