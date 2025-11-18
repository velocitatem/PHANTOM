import pandas as pd
import json
import numpy as np
import os
import requests
from dotenv import load_dotenv
from sklearn.base import BaseEstimator, TransformerMixin
from supabase import create_client, Client
load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000")
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
N_PRICE_BUCKETS = 5

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_data_from_kafka() -> pd.DataFrame:
    """fetch all events from backend dump endpoint"""
    resp = requests.get(f"{BACKEND_URL}/api/kafka/dump")
    resp.raise_for_status()
    data = resp.json()

    if not data.get('success') or not data.get('data'):
        return pd.DataFrame()

    df = pd.DataFrame(data['data'])
    # explode metadata col json
    if 'metadata' in df.columns:
        df = df.join(pd.json_normalize(df.pop("metadata"), sep=".").add_prefix("metadata_"))
    df = df.dropna(subset=['eventName'])
    return df


def join_with_experiments(df: pd.DataFrame) -> pd.DataFrame:
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


def augment_event_titles(df: pd.DataFrame) -> pd.DataFrame:
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


def extract() -> pd.DataFrame:
    df = get_data_from_kafka()
    df = join_with_experiments(df)
    df = augment_event_titles(df)
    return df


class DataExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None):
        return extract()


if __name__ == "__main__":
    df = extract()
    print(df.head())
    print(df.tail())
    print(df.info())
