import pandas as pd
from procesing.steps.base import BaseContextStep

class FetchInteractionsStep(BaseContextStep):
    """Fetch raw interaction data from Kafka topic with optional time and store_mode filtering"""

    def __init__(self, context, lookback: str = None):
        super().__init__(context)
        self.lookback = lookback

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

        # filter by store_mode from context
        if 'storeMode' in df.columns:
            df = df[df['storeMode'] == self.context.store_mode]

        # Remap dateIndex if present
        if 'metadata_dateIndex' in df.columns:
            df['dateIndex'] = df['metadata_dateIndex'].astype('Int64')

        # Apply time filtering if lookback specified
        if self.lookback and 'ts' in df.columns:
            df['ts'] = pd.to_datetime(df['ts'])
            cutoff = pd.Timestamp.now() - pd.Timedelta(self.lookback)
            df = df[df['ts'] >= cutoff]

        return df


class FetchPriceLogsStep(BaseContextStep):
    """Fetch price log data from Kafka topic with optional time and store_mode filtering"""

    def __init__(self, context, lookback: str = None):
        super().__init__(context)
        self.lookback = lookback

    def transform(self, X=None):
        df = self.context.provider.fetch_kafka_topic('price-logs')

        if df.empty:
            return df

        # filter by store_mode from context
        if 'storeMode' in df.columns:
            df = df[df['storeMode'] == self.context.store_mode]

        # Apply time filtering if lookback specified
        if self.lookback and 'ts' in df.columns:
            df['ts'] = pd.to_datetime(df['ts'])
            cutoff = pd.Timestamp.now() - pd.Timedelta(self.lookback)
            df = df[df['ts'] >= cutoff]

        return df


class FetchExperimentsStep(BaseContextStep):
    """Fetch experiment metadata for given interaction data"""

    def transform(self, interactions_df: pd.DataFrame):
        if interactions_df.empty or 'experimentId' not in interactions_df.columns:
            return pd.DataFrame()

        exp_ids = interactions_df['experimentId'].dropna().unique().tolist()
        if not exp_ids:
            return pd.DataFrame()

        return self.context.provider.fetch_experiments(exp_ids)
