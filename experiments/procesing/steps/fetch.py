import pandas as pd
from procesing.steps.base import BaseContextStep

class FetchInteractionsStep(BaseContextStep):
    """Fetch raw interaction data from Kafka topic"""

    def transform(self, X=None):
        df = self.context.provider.fetch_kafka_topic('user-interactions')

        if df.empty:
            return df

        # Explode metadata JSON column
        if 'metadata' in df.columns:
            df = df.join(
                pd.json_normalize(df.pop('metadata'), sep='.').add_prefix('metadata_')
            )

        df = df.dropna(subset=['eventName'])
        # drop all where page has /admin/
        df = df[~df['page'].str.contains('/admin/', na=False)]

        # Remap dateIndex if present
        if 'metadata_dateIndex' in df.columns:
            df['dateIndex'] = df['metadata_dateIndex'].astype('Int64')

        return df


class FetchPriceLogsStep(BaseContextStep):
    """Fetch price log data from Kafka topic"""

    def transform(self, X=None):
        return self.context.provider.fetch_kafka_topic('price-logs')


class FetchExperimentsStep(BaseContextStep):
    """Fetch experiment metadata for given interaction data"""

    def transform(self, interactions_df: pd.DataFrame):
        if interactions_df.empty or 'experimentId' not in interactions_df.columns:
            return pd.DataFrame()

        exp_ids = interactions_df['experimentId'].dropna().unique().tolist()
        if not exp_ids:
            return pd.DataFrame()

        return self.context.provider.fetch_experiments(exp_ids)
