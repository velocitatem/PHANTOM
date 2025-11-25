import pandas as pd
import json
import numpy as np
import os
import requests
from dotenv import load_dotenv
from sklearn.base import BaseEstimator, TransformerMixin
from supabase import create_client, Client
from typing import Tuple, List, Dict
load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000")
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
N_PRICE_BUCKETS = 5

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


class KafkaDataFetcher(BaseEstimator, TransformerMixin):
    def __init__(self, topic: str = "user-interactions"):
        self.topic = topic # also can be price-logs
    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None):
        resp = requests.get(f"{BACKEND_URL}/api/kafka/dump?topic={self.topic}")
        resp.raise_for_status()
        data = resp.json()

        if not data.get('success') or not data.get('data'):
            return pd.DataFrame()

        df = pd.DataFrame(data['data'])
        if self.topic == 'user-interactions':
            if 'metadata' in df.columns: # explode metadata col json
                df = df.join(pd.json_normalize(df.pop("metadata"), sep=".").add_prefix("metadata_"))
            df = df.dropna(subset=['eventName'])
            # remape dateIndex
            df['dateIndex'] = df['metadata_dateIndex'].astype('Int64')
        return df


class ExperimentJoiner(BaseEstimator, TransformerMixin):
    def fit(self, X=None, y=None):
        return self

    def transform(self, df):
        if df.empty or 'experimentId' not in df.columns:
            return df

        unique_exp_ids = df['experimentId'].dropna().unique()
        if len(unique_exp_ids) == 0:
            return df

        resp = supabase.table('experiments').select(
            'id, subject_name, xp_human_only, xp_market_mode, xp_task_id, task:tasks(task_name, task_description, task_def_of_done)'
        ).in_('id', unique_exp_ids.tolist()).execute()

        if not resp.data:
            return df

        exp_df = pd.DataFrame(resp.data)

        # flatten task nested object if present
        if 'task' in exp_df.columns and exp_df['task'].notnull().any():
            task_normalized = pd.json_normalize(exp_df['task'].dropna())
            task_normalized.index = exp_df[exp_df['task'].notnull()].index
            exp_df = exp_df.drop(columns=['task']).join(task_normalized, rsuffix='_task')

        # rename experiment columns for clarity
        exp_df = exp_df.rename(columns={
            'id': 'experimentId',
            'subject_name': 'exp_subject',
            'xp_human_only': 'exp_human_only',
            'xp_market_mode': 'exp_market_mode',
            'xp_task_id': 'exp_task_id'
        })

        df = df.merge(exp_df, on='experimentId', how='left')
        return df


class EventTitleAugmenter(BaseEstimator, TransformerMixin):
    def fit(self, X=None, y=None):
        return self

    def transform(self, df):
        # from taking standard view_item_page in eventName to view_item_page_{metadata_schema}
        # we want metadata schema to create product specific event names

        # only create price buckets if we have enough unique prices
        if df["metadata_price"].notnull().sum() > 0:
            try:
                price_buckets = pd.qcut(
                    df["metadata_price"],
                    q=N_PRICE_BUCKETS,
                    labels=[f"PB_{i+1}" for i in range(N_PRICE_BUCKETS)],
                    duplicates='drop'  # handle duplicate bin edges
                )
            except ValueError:
                # fallback: if still not enough unique values, use cut with fixed ranges or just use raw price
                price_buckets = df["metadata_price"].apply(lambda x: f"P_{int(x)}" if pd.notnull(x) else "")
        else:
            price_buckets = pd.Series([""] * len(df), index=df.index)

        # metadata_schema: _product_id@price_bucket_{i} only if we have product metadata otherswise keep original event name
        # TODO: make this adaptive, if we have hover_over_title we append the title, if its view_page we say which page
        df["metadata_schema"] = np.where(
            df["productId"].notnull() & df["metadata_price"].notnull(),
            "_" + df["productId"].astype(str) + "@" + price_buckets.astype(str),
            ""
        )
        df["eventName"] = df["eventName"] + df["metadata_schema"].astype(str)
        return df


def chunk_shared_data(interactions_df: pd.DataFrame,
                      price_logs_df: pd.DataFrame,
                      window_size: str = '30s',
                      ts_col: str = 'ts') -> Tuple[List[Dict], List[Dict]]:
    """
    Chunk interaction and price data into aligned time windows.

    Args:
        interactions_df: interaction data with timestamp column
        price_logs_df: price log data with timestamp column
        window_size: pandas freq string ('30s', '1min', '1h', etc)
        ts_col: name of timestamp column

    Returns:
        tuple of (interaction_chunks, price_chunks) where each is list of dicts:
        {
            'window_start': timestamp,
            'window_end': timestamp,
            'data': dataframe for this window
        }
    """
    if interactions_df.empty and price_logs_df.empty:
        return [], []

    # convert timestamps to datetime
    interactions_df = interactions_df.copy()
    price_logs_df = price_logs_df.copy()

    if not interactions_df.empty:
        if not pd.api.types.is_datetime64_any_dtype(interactions_df[ts_col]):
            interactions_df[ts_col] = pd.to_datetime(interactions_df[ts_col])

    if not price_logs_df.empty:
        if not pd.api.types.is_datetime64_any_dtype(price_logs_df[ts_col]):
            price_logs_df[ts_col] = pd.to_datetime(price_logs_df[ts_col])

    # find global time bounds
    times = []
    if not interactions_df.empty:
        times.extend([interactions_df[ts_col].min(), interactions_df[ts_col].max()])
    if not price_logs_df.empty:
        times.extend([price_logs_df[ts_col].min(), price_logs_df[ts_col].max()])

    if not times:
        return [], []

    earliest = min(times)
    latest = max(times)

    # create shared time windows
    windows = pd.date_range(start=earliest, end=latest, freq=window_size)

    if len(windows) < 2:
        return [], []

    # chunk both datasets
    interaction_chunks = []
    price_chunks = []

    for i in range(len(windows) - 1):
        window_start = windows[i]
        window_end = windows[i + 1]

        # filter interactions in this window
        if not interactions_df.empty:
            mask = (interactions_df[ts_col] >= window_start) & (interactions_df[ts_col] < window_end)
            interaction_chunk = interactions_df[mask]
        else:
            interaction_chunk = pd.DataFrame()

        interaction_chunks.append({
            'window_start': window_start,
            'window_end': window_end,
            'data': interaction_chunk
        })

        # filter price logs in this window
        if not price_logs_df.empty:
            mask = (price_logs_df[ts_col] >= window_start) & (price_logs_df[ts_col] < window_end)
            price_chunk = price_logs_df[mask]
        else:
            price_chunk = pd.DataFrame()

        price_chunks.append({
            'window_start': window_start,
            'window_end': window_end,
            'data': price_chunk
        })

    return interaction_chunks, price_chunks
