from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import pandas as pd
import logging
import sys
import pickle
import io

# add parent dir to path so procesing package can be imported
sys.path.insert(0, '/opt/airflow')

from procesing.context import PipelineContext
from procesing.providers import SupabaseProvider, BackendAPIProvider
from procesing.steps import (
    FetchInteractionsStep,
    FetchPriceLogsStep,
    CreatePriceBucketsStep,
    AugmentEventNamesStep,
    ChunkByTimeWindowStep,
    ComputeDemandForChunksStep,
    AggregatePriceLogsStep,
    ComputeElasticityStep,
    BuildStateSpaceStep,
    FitPricingFunctionStep,
    PredictPricesStep,
)

default_args = {
    'owner': 'phantom-research',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def get_provider():
    """Factory to create composite provider"""
    class CompositeProvider(SupabaseProvider, BackendAPIProvider):
        def __init__(self):
            SupabaseProvider.__init__(self)
            BackendAPIProvider.__init__(self)
    return CompositeProvider()

def get_context(**kwargs):
    """Build pipeline context from Airflow config"""
    dag_conf = kwargs.get('dag_run').conf if kwargs.get('dag_run') else {}
    return PipelineContext(
        provider=get_provider(),
        store_mode=dag_conf.get('store_mode', 'hotel'),
        window_size=dag_conf.get('window_size', '30s'),
        n_price_buckets=dag_conf.get('n_price_buckets', 5),
        elasticity_method=dag_conf.get('elasticity_method', 'point'),
        min_observations=dag_conf.get('min_observations', 2),
    )

# atomic task functions (each wraps one sklearn step)
def fetch_interactions(**kwargs):
    """Task: Fetch interaction data from Kafka"""
    context = get_context(**kwargs)
    step = FetchInteractionsStep(context)
    df = step.transform(None)

    kwargs['ti'].xcom_push(key='interactions_raw', value=pickle.dumps(df))
    logging.info(f"Fetched {len(df)} interaction records")
    return len(df)

def fetch_price_logs(**kwargs):
    """Task: Fetch price logs from Kafka"""
    context = get_context(**kwargs)
    step = FetchPriceLogsStep(context)
    df = step.transform(None)

    kwargs['ti'].xcom_push(key='price_logs_raw', value=pickle.dumps(df))
    logging.info(f"Fetched {len(df)} price records")
    return len(df)

def create_price_buckets(**kwargs):
    """Task: Create price buckets for interactions"""
    ti = kwargs['ti']
    df = pickle.loads(ti.xcom_pull(key='interactions_raw'))

    context = get_context(**kwargs)
    step = CreatePriceBucketsStep(context)
    df = step.transform(df)

    ti.xcom_push(key='interactions_bucketed', value=pickle.dumps(df))
    logging.info(f"Created price buckets for {len(df)} interactions")
    return len(df)

def augment_event_names(**kwargs):
    """Task: Augment event names with product and price schema"""
    ti = kwargs['ti']
    df = pickle.loads(ti.xcom_pull(key='interactions_bucketed'))

    context = get_context(**kwargs)
    step = AugmentEventNamesStep(context)
    df = step.transform(df)

    ti.xcom_push(key='interactions_final', value=pickle.dumps(df))
    logging.info(f"Augmented event names for {len(df)} interactions")
    return len(df)

def chunk_interactions(**kwargs):
    """Task: Chunk interactions into time windows"""
    ti = kwargs['ti']
    df = pickle.loads(ti.xcom_pull(key='interactions_final'))

    context = get_context(**kwargs)
    step = ChunkByTimeWindowStep(context)
    chunks = step.transform(df)

    ti.xcom_push(key='interaction_chunks', value=pickle.dumps(chunks))
    logging.info(f"Generated {len(chunks)} interaction chunks")
    return len(chunks)

def compute_demand(**kwargs):
    """Task: Compute demand vectors for all chunks"""
    ti = kwargs['ti']
    chunks = pickle.loads(ti.xcom_pull(key='interaction_chunks'))

    context = get_context(**kwargs)
    step = ComputeDemandForChunksStep(context)
    demand_chunks = step.transform(chunks)

    ti.xcom_push(key='demand_chunks', value=pickle.dumps(demand_chunks))
    logging.info(f"Computed demand for {len(demand_chunks)} chunks")
    return len(demand_chunks)

def aggregate_price_logs(**kwargs):
    """Task: Aggregate price logs into time windows """
    ti = kwargs['ti']
    df = pickle.loads(ti.xcom_pull(key='price_logs_raw'))

    context = get_context(**kwargs)
    step = AggregatePriceLogsStep(context)
    price_chunks = step.transform(df)

    ti.xcom_push(key='price_chunks', value=pickle.dumps(price_chunks))
    logging.info(f"Aggregated {len(price_chunks)} price chunks")
    return len(price_chunks)

def compute_elasticity(**kwargs):
    """Task: Compute price elasticity from demand and price chunks"""
    ti = kwargs['ti']
    demand_chunks = pickle.loads(ti.xcom_pull(key='demand_chunks'))
    price_chunks = pickle.loads(ti.xcom_pull(key='price_chunks'))

    context = get_context(**kwargs)
    step = ComputeElasticityStep(context)
    elasticity_df = step.transform((demand_chunks, price_chunks))

    ti.xcom_push(key='elasticity_results', value=pickle.dumps(elasticity_df))
    logging.info(f"Computed elasticity for {len(elasticity_df)} products")

    return {
        'n_products': len(elasticity_df),
        'mean_elasticity': float(elasticity_df['elasticity'].mean()),
        'median_elasticity': float(elasticity_df['elasticity'].median())
    }

def build_state_space(**kwargs):
    """Task: Build state space from elasticity"""
    ti = kwargs['ti']
    elasticity_df = pickle.loads(ti.xcom_pull(key='elasticity_results'))

    context = get_context(**kwargs)
    step = BuildStateSpaceStep(context)
    state_space = step.transform(elasticity_df)

    ti.xcom_push(key='state_space', value=pickle.dumps(state_space))
    logging.info("Built state space for pricing")
    return True

def fit_pricing_function(**kwargs):
    """Task: Fit pricing function using elasticity"""
    ti = kwargs['ti']
    elasticity_df = pickle.loads(ti.xcom_pull(key='elasticity_results'))

    context = get_context(**kwargs)
    step = FitPricingFunctionStep(context)
    pricer = step.transform(elasticity_df)

    ti.xcom_push(key='pricer', value=pickle.dumps(pricer))
    logging.info("Fitted pricing function")
    return True

def predict_prices(**kwargs):
    """Task: Predict optimal prices"""
    ti = kwargs['ti']
    pricer = pickle.loads(ti.xcom_pull(key='pricer'))
    state_space = pickle.loads(ti.xcom_pull(key='state_space'))

    context = get_context(**kwargs)
    step = PredictPricesStep(context)
    prices_df = step.transform((pricer, state_space))

    ti.xcom_push(key='predicted_prices', value=pickle.dumps(prices_df))
    logging.info(f"Predicted prices for {len(prices_df)} products")
    return len(prices_df)

def publish_results(**kwargs):
    """Task: Publish elasticity, pricing model, and predicted prices to registry"""
    ti = kwargs['ti']
    elasticity_df = pickle.loads(ti.xcom_pull(key='elasticity_results'))
    prices_df = pickle.loads(ti.xcom_pull(key='predicted_prices'))

    sys.path.insert(0, '/opt/airflow')
    from lib.model_registry import ModelRegistry

    registry = ModelRegistry()
    dag_conf = kwargs.get('dag_run').conf if kwargs.get('dag_run') else {}

    metadata = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'window_size': dag_conf.get('window_size', '30s'),
        'store_mode': dag_conf.get('store_mode', 'hotel'),
        'dag_run_id': kwargs['dag_run'].run_id if kwargs.get('dag_run') else 'manual'
    }

    registry.publish_elasticity(elasticity_df, model_name='latest', metadata=metadata)

    pricer = pickle.loads(ti.xcom_pull(key='pricer'))
    registry.publish_pricing_model(
        pricer,
        model_name='latest',
        metadata={**metadata, 'model_type': type(pricer).__name__}
    )

    registry.publish_prices(prices_df, model_name='latest', metadata=metadata)

    logging.info(f"Published elasticity + pricing + prices for {len(elasticity_df)} products")

    return {
        'n_products': len(elasticity_df),
        'n_prices': len(prices_df),
        'registry_status': 'success',
        'elasticity_mean': float(elasticity_df['elasticity'].mean())
    }


# DAG definition
with DAG(
    'elasticity_pricing_pipeline',
    default_args=default_args,
    description='E2E refactored pipeline: atomic steps with proper separation',
    schedule_interval='*/15 * * * *',
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['pricing', 'elasticity', 'research', 'refactored'],
) as dag:

    # parallel data fetching
    t_fetch_interactions = PythonOperator(
        task_id='fetch_interactions',
        python_callable=fetch_interactions,
        provide_context=True,
    )

    t_fetch_price_logs = PythonOperator(
        task_id='fetch_price_logs',
        python_callable=fetch_price_logs,
        provide_context=True,
    )

    # interaction processing branch
    t_create_buckets = PythonOperator(
        task_id='create_price_buckets',
        python_callable=create_price_buckets,
        provide_context=True,
    )

    t_augment_events = PythonOperator(
        task_id='augment_event_names',
        python_callable=augment_event_names,
        provide_context=True,
    )

    t_chunk_interactions = PythonOperator(
        task_id='chunk_interactions',
        python_callable=chunk_interactions,
        provide_context=True,
    )

    t_compute_demand = PythonOperator(
        task_id='compute_demand',
        python_callable=compute_demand,
        provide_context=True,
    )

    # price processing branch (VECTORIZED)
    t_aggregate_prices = PythonOperator(
        task_id='aggregate_price_logs',
        python_callable=aggregate_price_logs,
        provide_context=True,
    )

    # convergence: compute elasticity
    t_compute_elasticity = PythonOperator(
        task_id='compute_elasticity',
        python_callable=compute_elasticity,
        provide_context=True,
    )

    # pricing tasks
    t_build_state = PythonOperator(
        task_id='build_state_space',
        python_callable=build_state_space,
        provide_context=True,
    )

    t_fit_pricer = PythonOperator(
        task_id='fit_pricing_function',
        python_callable=fit_pricing_function,
        provide_context=True,
    )

    t_predict_prices = PythonOperator(
        task_id='predict_prices',
        python_callable=predict_prices,
        provide_context=True,
    )

    # publish to registry
    t_publish = PythonOperator(
        task_id='publish_results',
        python_callable=publish_results,
        provide_context=True,
    )

    # dependency graph (clear atomic flow)
    # parallel fetches
    [t_fetch_interactions, t_fetch_price_logs]

    # interaction branch: fetch -> bucket -> augment -> chunk -> demand
    t_fetch_interactions >> t_create_buckets >> t_augment_events >> t_chunk_interactions >> t_compute_demand

    # price branch: fetch -> aggregate (vectorized)
    t_fetch_price_logs >> t_aggregate_prices

    # convergence: both branches -> elasticity
    [t_compute_demand, t_aggregate_prices] >> t_compute_elasticity

    # pricing: elasticity -> state + fit -> predict -> publish
    t_compute_elasticity >> [t_build_state, t_fit_pricer] >> t_predict_prices >> t_publish
