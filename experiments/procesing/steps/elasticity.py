import numpy as np
import pandas as pd
from typing import Dict, List
from procesing.steps.base import BaseContextStep

class AggregatePriceLogsStep(BaseContextStep):
    """
    Aggregate price logs into time windows using VECTORIZED operations.
    Input: price_logs_df
    Output: list of price chunks with [productId, price]
    """

    def transform(self, price_logs_df: pd.DataFrame):
        if price_logs_df.empty:
            return []

        df = price_logs_df.copy()
        ts_col = self.context.config.get('ts_col', 'ts')
        #window_size = self.context.window_size WE ARE NOT USING CHUNKS ANYMORE

        # ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
            df[ts_col] = pd.to_datetime(df[ts_col])

        df = df.sort_values([ts_col, 'productId'])
        products = self.context.products
        # get base price from metadata if available 1) read the metadata col as json and get the base_price
        products['base_price'] = products.apply(
            lambda row: row['metadata'].get('base_price', 0) if isinstance(row['metadata'], dict) else 0,
            axis=1
        )

        unique_products = products['id'].unique()

        df_indexed = df.set_index(ts_col)
        # we return a df of average price per product over the entire period
        # TODO: maybe consider different opration to handle price aggregation over time
        avg_prices = df_indexed.groupby('productId')['price'].mean().reindex(unique_products, fill_value=0).reset_index()
        avg_prices.columns = ['productId', 'price']
        # fill 0s with base_price from products
        base_price_map = products.set_index('id')['base_price'].to_dict()
        return avg_prices
