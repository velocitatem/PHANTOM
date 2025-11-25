from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from supabase import create_client, Client
import pandas as pd
import os

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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
        # TODO: improve demand score calculation rather than just counting interactions (use weights..)
        # while maintaining simplicity of a simple cross tab approach
        product_demand = pd.crosstab(interactions['productId'], "no_of_interactions")
        product_demand = product_demand.reindex(unique_products, fill_value=0).reset_index()
        product_demand.columns = ['productId', 'demand_score']
        return product_demand
