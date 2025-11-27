from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from supabase import create_client, Client
from typing import Optional, Literal
import os
import logging
log = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class ChunkInteractionsIntoSteps(BaseEstimator, TransformerMixin):
    """
    Split interaction data into time windows for temporal analysis.
    Returns a list of dataframes, one per time window.
    """
    def __init__(self,
                 window_size:str='1h',
                 ts_col:str='ts',
                 return_metadata:bool=True):
        """
        Args:
            window_size: pandas freq string ('1h', '30T', '1D', etc)
            ts_col: timestamp column name
            return_metadata: if True, return dict with metadata per chunk
        """
        self.window_size = window_size
        self.ts_col = ts_col
        self.return_metadata = return_metadata

    def fit(self, X):
        return self

    def transform(self, interactions: pd.DataFrame):
        """
        Returns:
            if return_metadata=False: list of dataframes, one per window
            if return_metadata=True: list of dicts with keys:
                - 'data': dataframe for this window
                - 'window_start': start timestamp
                - 'window_end': end timestamp
                - 'window_idx': integer index
        """
        if interactions.empty:
            return []

        df = interactions.copy()

        # ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.ts_col]):
            df[self.ts_col] = pd.to_datetime(df[self.ts_col])

        # sort by time
        df = df.sort_values(self.ts_col)

        # assign window
        df['_window'] = df[self.ts_col].dt.floor(self.window_size)

        # group by window
        chunks = []
        for idx, (window_start, group) in enumerate(df.groupby('_window')):
            chunk_data = group.drop(columns=['_window'])

            if self.return_metadata:
                chunks.append({
                    'data': chunk_data,
                    'window_start': window_start,
                    'window_end': window_start + pd.Timedelta(self.window_size),
                    'window_idx': idx
                })
            else:
                chunks.append(chunk_data)

        return chunks


class DemandEstimator(BaseEstimator, TransformerMixin):
    def __init__(self,
                 store_mode:str='hotel',
                 session_filter:str="",
                 experiment_filter:str=""):
        self.store=store_mode
        self.session_filter=session_filter if len(session_filter)>0 else None
        self.experiment_filter=experiment_filter if len(experiment_filter)>0 else None
    def fit(self, X):
        return self

    def transform(self, interactions : pd.DataFrame):
        if interactions.empty:
            return pd.DataFrame(columns=["productId", "demand_score"])
        if self.session_filter:
            interactions = interactions[interactions['sessionId'] == self.session_filter]
        if self.experiment_filter:
            interactions = interactions[interactions['experimentId'] == self.experiment_filter]

        products=supabase.table(f'{self.store}_products').select("id, room_type, date_index, metadata, availability").execute()
        products = pd.DataFrame(products.data)
        unique_products = products['id'].unique()
        log.info(f"Demand estimator found {len(unique_products)} in data")

        # filter out rows without productId
        interactions_with_products = interactions.dropna(subset=['productId'])

        if interactions_with_products.empty:
            # no interactions with products, return all zeros
            return pd.DataFrame({
                'productId': unique_products,
                'demand_score': 0
            })

        # TODO: improve demand score calculation rather than just counting interactions (use weights..)
        # while maintaining simplicity of a simple cross tab approach
        product_demand = pd.crosstab(interactions_with_products['productId'], "no_of_interactions")
        product_demand = product_demand.reindex(unique_products, fill_value=0).reset_index()
        product_demand.columns = ['productId', 'demand_score']
        return product_demand
