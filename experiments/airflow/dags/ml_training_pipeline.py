from airflow import DAG, Dataset
from airflow.decorators import task
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
    ValidateDataStep,
    ExtractSessionFeaturesStep,
    JoinLabelsStep,
)

TRAINING_DATASET = Dataset('phantom://ml/training-data')

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


def _get_context(store_mode: str = 'hotel') -> PipelineContext:
    return PipelineContext(provider=CompositeProvider(), store_mode=store_mode)


with DAG(
    'ml_training_pipeline',
    default_args=DEFAULT_ARGS,
    description='ML training data pipeline: fetch -> validate -> extract features -> label -> publish',
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'training', 'features', 'research'],
) as dag:

    @task
    def fetch_interactions(**kwargs) -> bytes:
        dag_conf = kwargs.get('dag_run').conf if kwargs.get('dag_run') else {}
        ctx = _get_context(dag_conf.get('store_mode', 'hotel'))
        df = FetchInteractionsStep(ctx).transform(None)
        logging.info(f"Fetched {len(df)} interactions, {df['sessionId'].nunique()} sessions")
        return pickle.dumps(df)

    @task
    def validate_data(raw_data: bytes, **kwargs) -> bytes:
        df = pickle.loads(raw_data)
        dag_conf = kwargs.get('dag_run').conf if kwargs.get('dag_run') else {}
        ctx = _get_context(dag_conf.get('store_mode', 'hotel'))
        validated = ValidateDataStep(ctx).transform(df)
        report = ctx.get_cached('validation_report') or {}
        logging.info(f"Validation: {report.get('status')}, {report.get('sessions', 0)} sessions")
        return pickle.dumps(validated)

    @task
    def extract_session_features(validated_data: bytes, **kwargs) -> bytes:
        df = pickle.loads(validated_data)
        if df.empty:
            logging.warning("Empty input, skipping feature extraction")
            return pickle.dumps(pd.DataFrame())
        dag_conf = kwargs.get('dag_run').conf if kwargs.get('dag_run') else {}
        ctx = _get_context(dag_conf.get('store_mode', 'hotel'))
        features = ExtractSessionFeaturesStep(ctx).transform(df)
        logging.info(f"Extracted {len(features.columns)} features for {len(features)} sessions")
        return pickle.dumps(features)

    @task
    def join_labels(features_data: bytes, **kwargs) -> bytes:
        features_df = pickle.loads(features_data)
        if features_df.empty:
            logging.warning("Empty features, skipping label join")
            return pickle.dumps(pd.DataFrame())
        dag_conf = kwargs.get('dag_run').conf if kwargs.get('dag_run') else {}
        ctx = _get_context(dag_conf.get('store_mode', 'hotel'))
        labeled = JoinLabelsStep(ctx).transform(features_df)
        n_agents = labeled['is_agent'].sum() if 'is_agent' in labeled.columns else 0
        logging.info(f"Labeled {len(labeled)} sessions: {n_agents} agents")
        return pickle.dumps(labeled)

    @task(outlets=[TRAINING_DATASET])
    def publish_training_data(labeled_data: bytes, **kwargs) -> dict:
        labeled_df = pickle.loads(labeled_data)
        if labeled_df.empty:
            return {'status': 'skipped', 'reason': 'empty_data'}
        dag_conf = kwargs.get('dag_run').conf if kwargs.get('dag_run') else {}
        return {
            'status': 'success',
            'n_sessions': len(labeled_df),
            'n_features': len([c for c in labeled_df.columns if c not in ['sessionId', 'experimentId', 'is_agent']]),
            'store_mode': dag_conf.get('store_mode', 'hotel'),
            'timestamp': pd.Timestamp.now().isoformat(),
        }

    raw = fetch_interactions()
    validated = validate_data(raw)
    features = extract_session_features(validated)
    labeled = join_labels(features)
    publish_training_data(labeled)
