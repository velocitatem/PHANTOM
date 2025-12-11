from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import pandas as pd
import logging
import sys
import pickle

sys.path.insert(0, '/opt/airflow')

from procesing.context import PipelineContext
from procesing.providers import SupabaseProvider, BackendAPIProvider
from procesing.steps import (
    FetchInteractionsStep,
    FetchPriceLogsStep,
    ComputeDemandStep,
    AggregatePriceLogsStep,
    JoinProductFeaturesStep,
)
from procesing.pricers.simple import SimpleSurgePricer

DEFAULT_ARGS = {
    'owner': 'phantom-research',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

class CompositeProvider(SupabaseProvider, BackendAPIProvider):
    def __init__(self):
        SupabaseProvider.__init__(self)
        BackendAPIProvider.__init__(self)

def _get_provider():
    return CompositeProvider()

def _make_task_callables(store_mode: str):
    """Generate task callables bound to a specific store_mode."""

    def get_context(**kwargs):
        return PipelineContext(provider=_get_provider(), store_mode=store_mode)

    def fetch_interactions(**kwargs):
        ctx = get_context(**kwargs)
        df = FetchInteractionsStep(ctx).transform(None)
        kwargs['ti'].xcom_push(key='interactions_raw', value=pickle.dumps(df))
        logging.info(f"[{store_mode}] Fetched {len(df)} interaction records")
        return len(df)

    def fetch_price_logs(**kwargs):
        ctx = get_context(**kwargs)
        df = FetchPriceLogsStep(ctx).transform(None)
        kwargs['ti'].xcom_push(key='price_logs_raw', value=pickle.dumps(df))
        logging.info(f"[{store_mode}] Fetched {len(df)} price records")
        return len(df)

    def compute_demand(**kwargs):
        ti = kwargs['ti']
        df = pickle.loads(ti.xcom_pull(key='interactions_raw'))
        ctx = get_context(**kwargs)
        demand_df = ComputeDemandStep(ctx).transform(df)
        ti.xcom_push(key='demand_data', value=pickle.dumps(demand_df))
        logging.info(f"[{store_mode}] Computed demand for {len(demand_df)} products")
        return len(demand_df)

    def aggregate_price_logs(**kwargs):
        ti = kwargs['ti']
        df = pickle.loads(ti.xcom_pull(key='price_logs_raw'))
        ctx = get_context(**kwargs)
        price_df = AggregatePriceLogsStep(ctx).transform(df)
        ti.xcom_push(key='price_data', value=pickle.dumps(price_df))
        logging.info(f"[{store_mode}] Aggregated price logs for {len(price_df)} products")
        return len(price_df)

    def join_product_features(**kwargs):
        ti = kwargs['ti']
        demand_df = pickle.loads(ti.xcom_pull(key='demand_data'))
        price_df = pickle.loads(ti.xcom_pull(key='price_data'))
        ctx = get_context(**kwargs)
        joined_df = JoinProductFeaturesStep(ctx).transform((demand_df, price_df))
        ti.xcom_push(key='product_features', value=pickle.dumps(joined_df))
        logging.info(f"[{store_mode}] Joined features for {len(joined_df)} products")
        return len(joined_df)

    def apply_surge_pricing(**kwargs):
        ti = kwargs['ti']
        product_features = pickle.loads(ti.xcom_pull(key='product_features'))
        dag_conf = kwargs.get('dag_run').conf if kwargs.get('dag_run') else {}

        data = product_features.rename(columns={'demand_score': 'demand'})
        surge_pricer = SimpleSurgePricer(
            high_threshold=dag_conf.get('high_threshold', 10),
            low_threshold=dag_conf.get('low_threshold', 2),
            surge_multiplier=dag_conf.get('surge_multiplier', 1.2),
            discount_multiplier=dag_conf.get('discount_multiplier', 0.9)
        )
        surge_pricer.fit(data)
        data['optimal_price'] = surge_pricer.predict()

        prices_df = data[['productId', 'price', 'base_price', 'optimal_price', 'demand']].rename(columns={
            'price': 'current_price', 'demand': 'demand_score'
        })
        ti.xcom_push(key='predicted_prices', value=pickle.dumps(prices_df))
        logging.info(f"[{store_mode}] Applied surge pricing for {len(prices_df)} products")
        return len(prices_df)

    def publish_results(**kwargs):
        ti = kwargs['ti']
        prices_df = pickle.loads(ti.xcom_pull(key='predicted_prices'))
        from lib.model_registry import ModelRegistry

        registry = ModelRegistry()
        dag_conf = kwargs.get('dag_run').conf if kwargs.get('dag_run') else {}

        metadata = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'store_mode': store_mode,
            'dag_run_id': kwargs['dag_run'].run_id if kwargs.get('dag_run') else 'manual',
            'pricing_method': 'surge',
            'high_threshold': dag_conf.get('high_threshold', 10),
            'low_threshold': dag_conf.get('low_threshold', 2),
            'surge_multiplier': dag_conf.get('surge_multiplier', 1.2),
            'discount_multiplier': dag_conf.get('discount_multiplier', 0.9)
        }
        registry.publish_prices(prices_df, model_name=f'{store_mode}_latest', metadata=metadata)
        logging.info(f"[{store_mode}] Published surge pricing for {len(prices_df)} products")

        return {
            'n_products': len(prices_df),
            'registry_status': 'success',
            'store_mode': store_mode,
            'mean_demand': float(prices_df['demand_score'].mean()) if 'demand_score' in prices_df.columns else None
        }

    return {
        'fetch_interactions': fetch_interactions,
        'fetch_price_logs': fetch_price_logs,
        'compute_demand': compute_demand,
        'aggregate_price_logs': aggregate_price_logs,
        'join_product_features': join_product_features,
        'apply_surge_pricing': apply_surge_pricing,
        'publish_results': publish_results,
    }


def create_surge_pricing_dag(store_mode: str) -> DAG:
    """Factory: generates a surge pricing DAG for a given store_mode."""
    callables = _make_task_callables(store_mode)

    dag = DAG(
        f'surge_pricing_{store_mode}',
        default_args=DEFAULT_ARGS,
        description=f'Surge pricing pipeline for {store_mode} store mode',
        schedule_interval='*/15 * * * *',
        start_date=days_ago(1),
        catchup=False,
        max_active_runs=1,
        tags=['pricing', 'surge', 'research', store_mode],
    )

    with dag:
        t_fetch_interactions = PythonOperator(
            task_id='fetch_interactions',
            python_callable=callables['fetch_interactions'],
            provide_context=True,
        )
        t_fetch_price_logs = PythonOperator(
            task_id='fetch_price_logs',
            python_callable=callables['fetch_price_logs'],
            provide_context=True,
        )
        t_compute_demand = PythonOperator(
            task_id='compute_demand',
            python_callable=callables['compute_demand'],
            provide_context=True,
        )
        t_aggregate_prices = PythonOperator(
            task_id='aggregate_price_logs',
            python_callable=callables['aggregate_price_logs'],
            provide_context=True,
        )
        t_join_features = PythonOperator(
            task_id='join_product_features',
            python_callable=callables['join_product_features'],
            provide_context=True,
        )
        t_surge_pricing = PythonOperator(
            task_id='apply_surge_pricing',
            python_callable=callables['apply_surge_pricing'],
            provide_context=True,
        )
        t_publish = PythonOperator(
            task_id='publish_results',
            python_callable=callables['publish_results'],
            provide_context=True,
        )

        t_fetch_interactions >> t_compute_demand
        t_fetch_price_logs >> t_aggregate_prices
        [t_compute_demand, t_aggregate_prices] >> t_join_features >> t_surge_pricing >> t_publish

    return dag


# instantiate DAGs for Airflow to discover
dag_airline = create_surge_pricing_dag('airline')
dag_hotel = create_surge_pricing_dag('hotel')
