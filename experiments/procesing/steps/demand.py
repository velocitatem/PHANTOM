import pandas as pd
from procesing.steps.base import BaseContextStep

class ComputeDemandStep(BaseContextStep):
    """
    Compute demand vector for a single time window or dataframe.
    Input: single chunk dict OR raw dataframe
    Output: demand dataframe with [productId, demand_score]
    """

    def transform(self, chunk):
        # handle both chunk dict and raw dataframe
        if isinstance(chunk, dict):
            interactions = chunk['data']
            window_meta = {k: v for k, v in chunk.items() if k != 'data'}
        else:
            interactions = chunk
            window_meta = {}

        products = self.context.products
        unique_products = products['id'].unique()

        # apply filters if configured
        session_filter = self.context.config.get('session_filter')
        experiment_filter = self.context.config.get('experiment_filter')

        if session_filter and 'sessionId' in interactions.columns:
            interactions = interactions[interactions['sessionId'] == session_filter]
        if experiment_filter and 'experimentId' in interactions.columns:
            interactions = interactions[interactions['experimentId'] == experiment_filter]

        interactions_with_products = interactions.dropna(subset=['productId'])

        if interactions_with_products.empty:
            demand_df = pd.DataFrame({
                'productId': unique_products,
                'demand_score': 0
            })
        else:
            # crosstab for simple demand count
            demand_df = pd.crosstab(
                interactions_with_products['productId'],
                'count'
            ).reindex(unique_products, fill_value=0).reset_index()
            demand_df.columns = ['productId', 'demand_score']

        # attach window metadata if present
        if window_meta:
            return {**window_meta, 'demand_vector': demand_df}
        return demand_df


class ComputeDemandForChunksStep(BaseContextStep):
    """Apply ComputeDemandStep to list of chunks"""

    def transform(self, chunks: list):
        if not chunks:
            return []

        demand_step = ComputeDemandStep(self.context)
        return [demand_step.transform(chunk) for chunk in chunks]
