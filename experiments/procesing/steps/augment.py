import numpy as np
import pandas as pd
from .base import BaseContextStep

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
