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



class ComputeElasticityStep(BaseContextStep):
    """
    Compute price elasticity from demand and price chunks.
    Input: (demand_chunks, price_chunks)
    Output: elasticity_df [productId, elasticity, std_error, n_obs]
    """

    def transform(self, chunk_tuple: tuple):
        demand_chunks, price_chunks = chunk_tuple

        method = self.context.config.get('elasticity_method', 'point')
        min_obs = self.context.config.get('min_observations', 2)

        products = self.context.products
        all_product_ids = products['id'].unique()

        # align chunks by window_start
        # aligned = self._align_chunks(demand_chunks, price_chunks)

        if None:
            return pd.DataFrame({
                'productId': all_product_ids,
                'elasticity': 0.0,
                'std_error': 0.0,
                'n_obs': 0
            })

        # build time series per product
        product_series = self._build_timeseries(aligned)

        # compute elasticity per product
        elasticities = []
        for pid, series in product_series.items():
            if len(series) < min_obs:
                elasticities.append({
                    'productId': pid,
                    'elasticity': 0.0,
                    'std_error': 0.0,
                    'n_obs': len(series)
                })
                continue

            elast = self._compute_elasticity(series, method)
            elasticities.append({
                'productId': pid,
                'elasticity': elast['value'],
                'std_error': elast.get('std_error', 0.0),
                'n_obs': len(series)
            })

        result_df = pd.DataFrame(elasticities)

        # fill missing products with zero elasticity
        observed_pids = set(result_df['productId'])
        missing_pids = [p for p in all_product_ids if p not in observed_pids]

        if missing_pids:
            missing_df = pd.DataFrame({
                'productId': missing_pids,
                'elasticity': 0.0,
                'std_error': 0.0,
                'n_obs': 0
            })
            result_df = pd.concat([result_df, missing_df], ignore_index=True)

        return result_df

    def _align_chunks(self, demand_chunks: List[Dict], price_chunks: List[Dict]):
        """Align demand and price chunks by window_start"""
        price_lookup = {c['window_start']: c for c in price_chunks}
        aligned = []

        for dc in demand_chunks:
            ws = dc['window_start']
            if ws in price_lookup:
                aligned.append({
                    'window_start': ws,
                    'window_end': dc['window_end'],
                    'demand': dc['demand_vector'],
                    'prices': price_lookup[ws]['price_vector']
                })

        return aligned

    def _build_timeseries(self, aligned: List[Dict]):
        """Build time series [timestamp, price, quantity] per product"""
        series_by_product = {}

        for chunk in aligned:
            merged = chunk['demand'].merge(chunk['prices'], on='productId', how='inner')

            for _, row in merged.iterrows():
                pid = row['productId']
                if pid not in series_by_product:
                    series_by_product[pid] = []

                series_by_product[pid].append({
                    'timestamp': chunk['window_start'],
                    'price': row['price'],
                    'quantity': row['demand_score']
                })

        return series_by_product

    def _compute_elasticity(self, series: List[Dict], method: str):
        """Compute point or arc elasticity"""
        prices = np.array([s['price'] for s in series])
        quantities = np.array([s['quantity'] for s in series])

        # filter out zero/negative values
        valid = (prices > 0) & (quantities > 0)
        if valid.sum() < 2:
            return {'value': 0.0, 'std_error': 0.0}

        prices = prices[valid]
        quantities = quantities[valid]

        if method == 'point':
            return self._point_elasticity(prices, quantities)
        elif method == 'arc':
            return self._arc_elasticity(prices, quantities)
        else:
            raise ValueError(f"Unknown elasticity method: {method}")

    def _point_elasticity(self, prices: np.ndarray, quantities: np.ndarray):
        """Point elasticity via log-log regression: log(Q) = a + b*log(P), elasticity = b"""
        if len(prices) < 2:
            return {'value': 0.0, 'std_error': 0.0}

        log_p = np.log(prices)
        log_q = np.log(quantities)

        if log_p.std() == 0:
            return {'value': 0.0, 'std_error': 0.0}

        cov = np.cov(log_p, log_q)[0, 1]
        var = np.var(log_p)
        b = cov / var

        # std error estimate
        if len(prices) > 2:
            residuals = log_q - (log_q.mean() + b * (log_p - log_p.mean()))
            mse = (residuals ** 2).sum() / (len(prices) - 2)
            se_b = np.sqrt(mse / (len(prices) * var))
        else:
            se_b = 0.0

        return {'value': b, 'std_error': se_b}

    def _arc_elasticity(self, prices: np.ndarray, quantities: np.ndarray):
        """Arc elasticity: average period-over-period elasticity"""
        elasticities = []

        for i in range(1, len(prices)):
            p1, p2 = prices[i-1], prices[i]
            q1, q2 = quantities[i-1], quantities[i]

            p_avg = (p1 + p2) / 2
            q_avg = (q1 + q2) / 2

            if p_avg == 0 or q_avg == 0:
                continue

            delta_p = p2 - p1
            delta_q = q2 - q1

            if delta_p == 0:
                continue

            e = (delta_q / q_avg) / (delta_p / p_avg)
            elasticities.append(e)

        if not elasticities:
            return {'value': 0.0, 'std_error': 0.0}

        return {
            'value': np.mean(elasticities),
            'std_error': np.std(elasticities) / np.sqrt(len(elasticities))
        }
