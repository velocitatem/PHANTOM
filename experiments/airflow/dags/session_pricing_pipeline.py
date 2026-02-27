"""
Session-Aware Pricing DAG
THIS implements the core pricing computation (policy layer).

Flow: τ → θ̂ → D → p*
  1. Fetch recent sessions from Kafka (last 10 active)
  2. Extract features per session (τ → θ̂)
  3. Map features to demand proxy (θ̂ → D)
  4. Compute optimal prices (D → p*)
  5. Write to Redis session:{sessionId}:prices

Scheduled: every 1 minute when enabled
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import pandas as pd
import numpy as np
import logging
import sys
import pickle

sys.path.insert(0, '/opt/airflow')

from procesing.context import PipelineContext
from procesing.providers import SupabaseProvider, BackendAPIProvider
from procesing.steps.session import ExtractSessionFeaturesStep
from procesing.pricers.simple import SimpleSurgePricer, session_features_to_demand
from procesing.pricing import StateSpace
from lib.model_registry import ModelRegistry

DEFAULT_ARGS = {
    'owner': 'phantom-research',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(seconds=30),
}


class CompositeProvider(SupabaseProvider, BackendAPIProvider):
    def __init__(self):
        SupabaseProvider.__init__(self)
        BackendAPIProvider.__init__(self)


def _get_context(store_mode: str = 'hotel') -> PipelineContext:
    return PipelineContext(provider=CompositeProvider(), store_mode=store_mode)


def fetch_recent_sessions(**kwargs):
    """
    Task: Fetch last N active sessions from Kafka.
    Returns: DataFrame of interaction events for recent sessions.
    """
    dag_conf = kwargs.get('dag_run').conf if kwargs.get('dag_run') else {}
    store_mode = dag_conf.get('store_mode', 'hotel')
    session_limit = dag_conf.get('session_limit', 10)

    ctx = _get_context(store_mode)
    provider = ctx.provider

    # fetch all recent interactions from Kafka
    try:
        interactions_df = provider.fetch_kafka_topic("user-interactions")
    except Exception as e:
        logging.error(f"Failed to fetch interactions: {e}")
        kwargs['ti'].xcom_push(key='sessions_data', value=pickle.dumps(pd.DataFrame()))
        return 0

    if interactions_df.empty or 'sessionId' not in interactions_df.columns:
        kwargs['ti'].xcom_push(key='sessions_data', value=pickle.dumps(pd.DataFrame()))
        return 0

    # identify last N active sessions (most recent by event count)
    recent_sessions = interactions_df['sessionId'].value_counts().head(session_limit).index.tolist()

    # filter to only those sessions
    filtered_df = interactions_df[interactions_df['sessionId'].isin(recent_sessions)].copy()

    kwargs['ti'].xcom_push(key='sessions_data', value=pickle.dumps(filtered_df))
    kwargs['ti'].xcom_push(key='session_ids', value=recent_sessions)

    logging.info(f"Fetched {len(filtered_df)} events for {len(recent_sessions)} sessions")
    return len(recent_sessions)


def extract_session_features(**kwargs):
    """
    Task: Extract behavioral features from session trajectories.
    THIS implements τ → θ̂ transformation.
    """
    ti = kwargs['ti']
    sessions_df = pickle.loads(ti.xcom_pull(key='sessions_data'))

    if sessions_df.empty:
        ti.xcom_push(key='session_features', value=pickle.dumps(pd.DataFrame()))
        return 0

    dag_conf = kwargs.get('dag_run').conf if kwargs.get('dag_run') else {}
    ctx = _get_context(dag_conf.get('store_mode', 'hotel'))

    # extract features using vectorized pipeline
    feature_extractor = ExtractSessionFeaturesStep(ctx)
    features_df = feature_extractor.transform(sessions_df)

    ti.xcom_push(key='session_features', value=pickle.dumps(features_df))

    logging.info(f"Extracted {len(features_df.columns)} features for {len(features_df)} sessions")
    logging.info(f"Feature columns: {list(features_df.columns)}")
    logging.info(f"Sample features (first session):\n{features_df.iloc[0].to_dict()}")

    return len(features_df)


def compute_session_prices(**kwargs):
    """
    Task: Compute optimal prices for each session.
    THIS implements θ̂ → D → p* transformation.
    """
    ti = kwargs['ti']
    features_df = pickle.loads(ti.xcom_pull(key='session_features'))

    if features_df.empty:
        ti.xcom_push(key='price_results', value=pickle.dumps({}))
        return 0

    dag_conf = kwargs.get('dag_run').conf if kwargs.get('dag_run') else {}
    store_mode = dag_conf.get('store_mode', 'hotel')
    ctx = _get_context(store_mode)

    # fetch product catalog for base prices
    products_df = ctx.provider.fetch_products(store_mode)
    if products_df.empty:
        logging.error("No products found in catalog")
        ti.xcom_push(key='price_results', value=pickle.dumps({}))
        return 0

    products_df['base_price'] = products_df['metadata'].apply(
        lambda m: m.get('base_price', 100.0) if isinstance(m, dict) else 100.0
    )

    # initialize pricing model
    pricer = SimpleSurgePricer(
        high_threshold=dag_conf.get('high_threshold', 10),
        low_threshold=dag_conf.get('low_threshold', 2),
        surge_multiplier=dag_conf.get('surge_multiplier', 1.15),
        discount_multiplier=dag_conf.get('discount_multiplier', 0.95)
    )
    pricer.fit(products_df)

    # compute prices per session
    price_results = {}
    n_products = len(products_df)

    logging.info(f"Starting price computation for {len(features_df)} sessions, {n_products} products")
    logging.info(f"Pricer config: high_thresh={pricer.high_threshold}, low_thresh={pricer.low_threshold}, surge_mult={pricer.surge_multiplier}")

    for idx, session_row in features_df.iterrows():
        session_id = session_row.get('sessionId')
        if not session_id:
            continue

        # map features to demand proxy (θ̂ → D)
        session_features_single = pd.DataFrame([session_row])
        demand_proxy = session_features_to_demand(session_features_single)

        logging.info(f"[Session {session_id}] Features → Demand: {demand_proxy:.2f}")
        logging.info(f"[Session {session_id}] Key features: velocity={session_row.get('interaction_velocity', 0):.2f}, cart_ratio={session_row.get('cart_to_view_ratio', 0):.2f}, item_views={session_row.get('item_views', 0)}")

        # build state space
        state_space = StateSpace(
            demand=np.full(n_products, demand_proxy),  # broadcast session demand to all products
            prices=products_df['base_price'].values,
            session_features=session_features_single
        )

        # compute optimal prices (D → p*)
        optimal_prices = pricer.predict(state_space)

        base_avg = products_df['base_price'].mean()
        optimal_avg = optimal_prices.mean()
        price_change_pct = ((optimal_avg - base_avg) / base_avg) * 100

        logging.info(f"[Session {session_id}] Price adjustment: base_avg={base_avg:.2f}, optimal_avg={optimal_avg:.2f}, change={price_change_pct:+.1f}%")

        # store as dict {productId: price}
        price_map = {
            str(products_df.iloc[i]['id']): float(optimal_prices[i])
            for i in range(n_products)
        }

        price_results[session_id] = price_map

    ti.xcom_push(key='price_results', value=pickle.dumps(price_results))

    logging.info(f"Computed prices for {len(price_results)} sessions, {n_products} products each")
    return len(price_results)


def publish_to_registry(**kwargs):
    """
    Task: Write session prices to Redis registry.
    THIS is the write path: prices → session:{sessionId}:prices
    """
    ti = kwargs['ti']
    price_results = pickle.loads(ti.xcom_pull(key='price_results'))

    if not price_results:
        logging.warning("No prices to publish")
        return 0

    registry = ModelRegistry()
    ttl = kwargs.get('dag_run').conf.get('ttl', 1800) if kwargs.get('dag_run') and kwargs.get('dag_run').conf else 1800

    published_count = 0
    for session_id, price_map in price_results.items():
        registry.set_session_prices(session_id, price_map, ttl=ttl)
        published_count += 1

    logging.info(f"Published prices for {published_count} sessions to registry (TTL={ttl}s)")

    return {
        'sessions_published': published_count,
        'products_per_session': len(next(iter(price_results.values()))) if price_results else 0,
        'status': 'success'
    }


# DAG definition
with DAG(
    'session_pricing_pipeline',
    default_args=DEFAULT_ARGS,
    description='Session-aware pricing: extract features → compute prices → publish to registry',
    schedule_interval='*/1 * * * *',  # every 1 minute
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['pricing', 'session-aware', 'research', 'real-time'],
) as dag:

    t_fetch_sessions = PythonOperator(
        task_id='fetch_recent_sessions',
        python_callable=fetch_recent_sessions,
        provide_context=True,
    )

    t_extract_features = PythonOperator(
        task_id='extract_session_features',
        python_callable=extract_session_features,
        provide_context=True,
    )

    t_compute_prices = PythonOperator(
        task_id='compute_session_prices',
        python_callable=compute_session_prices,
        provide_context=True,
    )

    t_publish = PythonOperator(
        task_id='publish_to_registry',
        python_callable=publish_to_registry,
        provide_context=True,
    )

    # linear dependency: fetch → extract → compute → publish
    t_fetch_sessions >> t_extract_features >> t_compute_prices >> t_publish
