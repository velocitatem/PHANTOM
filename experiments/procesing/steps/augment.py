import numpy as np
import pandas as pd
from procesing.steps.base import BaseContextStep


class AugmentInteractionsStep(BaseContextStep):
    """
    Consolidated step: create price buckets, augment event names, join experiments.
    Input: (interactions_df, price_logs_df)
    Output: enriched interactions_df
    """

    def transform(self, data: tuple):
        interactions_df, price_logs_df = data

        if interactions_df.empty:
            return interactions_df

        # Step 1: Create price buckets
        interactions_df = self._create_price_buckets(interactions_df)

        # Step 2: Augment event names
        interactions_df = self._augment_event_names(interactions_df)

        # Step 3: Join experiments (optional)
        if 'experimentId' in interactions_df.columns:
            interactions_df = self._join_experiments(interactions_df)

        return interactions_df

    def _create_price_buckets(self, df: pd.DataFrame):
        """Create price bucket labels from price data"""
        if 'metadata_price' not in df.columns:
            df['price_bucket'] = ""
            return df

        n_buckets = self.context.config.get('n_price_buckets', 5)

        if df['metadata_price'].notnull().sum() > 0:
            try:
                price_buckets = pd.qcut(
                    df['metadata_price'],
                    q=n_buckets,
                    labels=[f"PB_{i+1}" for i in range(n_buckets)],
                    duplicates='drop'
                )
            except ValueError:
                # fallback for insufficient unique values
                price_buckets = df['metadata_price'].apply(
                    lambda x: f"P_{int(x)}" if pd.notnull(x) else ""
                )
        else:
            price_buckets = pd.Series([""] * len(df), index=df.index)

        df['price_bucket'] = price_buckets
        return df

    def _augment_event_names(self, df: pd.DataFrame):
        """Augment event names with product and price bucket schema"""
        # Create schema: _productId@price_bucket
        has_product = df.get('productId', pd.Series()).notnull()
        has_bucket = df.get('price_bucket', pd.Series()).notnull()

        df['metadata_schema'] = np.where(
            has_product & has_bucket,
            "_" + df['productId'].astype(str) + "@" + df['price_bucket'].astype(str),
            ""
        )

        df['eventName'] = df['eventName'] + df['metadata_schema']
        return df

    def _join_experiments(self, df: pd.DataFrame):
        """Join experiment metadata if experimentId present"""
        exp_ids = df['experimentId'].dropna().unique().tolist()
        if not exp_ids:
            return df

        experiments_df = self.context.provider.fetch_experiments(exp_ids)
        if experiments_df.empty:
            return df

        return df.merge(
            experiments_df,
            left_on='experimentId',
            right_on='id',
            how='left',
            suffixes=('', '_exp')
        )


class CreatePriceBucketsStep(BaseContextStep):
    """Create price bucket labels from price data"""

    def transform(self, df: pd.DataFrame):
        if df.empty or 'metadata_price' not in df.columns:
            df['price_bucket'] = ""
            return df

        n_buckets = self.context.config.get('n_price_buckets', 5)

        if df['metadata_price'].notnull().sum() > 0:
            try:
                price_buckets = pd.qcut(
                    df['metadata_price'],
                    q=n_buckets,
                    labels=[f"PB_{i+1}" for i in range(n_buckets)],
                    duplicates='drop'
                )
            except ValueError:
                # fallback for insufficient unique values
                price_buckets = df['metadata_price'].apply(
                    lambda x: f"P_{int(x)}" if pd.notnull(x) else ""
                )
        else:
            price_buckets = pd.Series([""] * len(df), index=df.index)

        df['price_bucket'] = price_buckets
        return df


class AugmentEventNamesStep(BaseContextStep):
    """Augment event names with product and price bucket schema"""

    def transform(self, df: pd.DataFrame):
        if df.empty:
            return df

        # Create schema: _productId@price_bucket
        has_product = df.get('productId', pd.Series()).notnull()
        has_bucket = df.get('price_bucket', pd.Series()).notnull()

        df['metadata_schema'] = np.where(
            has_product & has_bucket,
            "_" + df['productId'].astype(str) + "@" + df['price_bucket'].astype(str),
            ""
        )

        df['eventName'] = df['eventName'] + df['metadata_schema']
        return df
