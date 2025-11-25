import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.base import BaseEstimator, TransformerMixin

class TemporalElasticityEstimator(BaseEstimator, TransformerMixin):
    """
    Compute price elasticity from time-series demand and price data.

    Elasticity = (% change in quantity) / (% change in price)

    Works with chunked time-window data from ChunkInteractionsIntoSteps.
    """

    def __init__(self,
                 method:str='point',
                 min_observations:int=2,
                 smooth_window:Optional[int]=None):
        """
        Args:
            method: 'point' (point elasticity) or 'arc' (arc elasticity)
            min_observations: min data points needed per product
            smooth_window: if set, apply rolling avg smoothing to time series
        """
        self.method = method
        self.min_observations = min_observations
        self.smooth_window = smooth_window

    def fit(self, X):
        return self

    def transform(self,
                  demand_chunks: List[Dict],
                  price_chunks: List[Dict]) -> pd.DataFrame:
        """
        Args:
            demand_chunks: list from ChunkInteractionsIntoSteps + DemandEstimator
                           each item: {'window_start', 'window_end', 'demand_vector'}
            price_chunks: list of dicts with {'window_start', 'window_end', 'price_vector'}

        Returns:
            df with [productId, elasticity, std_error, n_observations]
        """
        aligned = self._align_chunks(demand_chunks, price_chunks)
        if not aligned:
            return pd.DataFrame(columns=['productId', 'elasticity', 'std_error', 'n_obs'])

        # build time series per product
        product_series = self._build_product_timeseries(aligned)

        # compute elasticity per product
        elasticities = []
        for pid, series in product_series.items():
            if len(series) < self.min_observations:
                continue

            # apply smoothing if requested
            if self.smooth_window and len(series) >= self.smooth_window:
                series = self._smooth_series(series, self.smooth_window)

            elast = self._compute_elasticity(series)
            if elast is not None:
                elasticities.append({
                    'productId': pid,
                    'elasticity': elast['value'],
                    'std_error': elast.get('std_error', np.nan),
                    'n_obs': len(series)
                })

        return pd.DataFrame(elasticities)

    def _align_chunks(self, demand_chunks, price_chunks):
        """Align demand and price data by matching time windows."""
        aligned = []

        # create lookup for price chunks by window_start
        price_lookup = {chunk['window_start']: chunk for chunk in price_chunks}

        for demand_chunk in demand_chunks:
            window_start = demand_chunk['window_start']
            if window_start in price_lookup:
                aligned.append({
                    'window_start': window_start,
                    'window_end': demand_chunk['window_end'],
                    'demand': demand_chunk['demand_vector'],
                    'prices': price_lookup[window_start]['price_vector']
                })

        return aligned

    def _build_product_timeseries(self, aligned_chunks):
        """Build time series [price, quantity] per product."""
        series_by_product = {}

        for chunk in aligned_chunks:
            demand_df = chunk['demand']
            price_df = chunk['prices']

            # merge on productId
            merged = demand_df.merge(price_df, on='productId', how='inner')

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

    def _smooth_series(self, series, window):
        """Apply rolling average smoothing."""
        df = pd.DataFrame(series)
        df['price_smooth'] = df['price'].rolling(window=window, center=True).mean()
        df['quantity_smooth'] = df['quantity'].rolling(window=window, center=True).mean()
        df = df.dropna()

        return [{'timestamp': row['timestamp'],
                 'price': row['price_smooth'],
                 'quantity': row['quantity_smooth']}
                for _, row in df.iterrows()]

    def _compute_elasticity(self, series):
        """Compute elasticity from time series."""
        if len(series) < 2:
            return None

        prices = np.array([s['price'] for s in series])
        quantities = np.array([s['quantity'] for s in series])

        # filter out zero/negative values
        valid = (prices > 0) & (quantities > 0)
        if valid.sum() < 2:
            return None

        prices = prices[valid]
        quantities = quantities[valid]

        if self.method == 'point':
            return self._point_elasticity(prices, quantities)
        elif self.method == 'arc':
            return self._arc_elasticity(prices, quantities)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _point_elasticity(self, prices, quantities):
        """
        Point elasticity using log-log regression.
        log(Q) = a + b*log(P), elasticity = b
        """
        if len(prices) < 2:
            return None

        log_p = np.log(prices)
        log_q = np.log(quantities)

        # simple linear regression
        if log_p.std() == 0:
            return None

        cov = np.cov(log_p, log_q)[0, 1]
        var = np.var(log_p)
        b = cov / var

        # std error estimate
        residuals = log_q - (log_q.mean() + b * (log_p - log_p.mean()))
        mse = (residuals ** 2).sum() / (len(prices) - 2)
        se_b = np.sqrt(mse / (len(prices) * var))

        return {'value': b, 'std_error': se_b}

    def _arc_elasticity(self, prices, quantities):
        """
        Arc elasticity: average of period-over-period elasticities.
        E_t = (ΔQ/Q_avg) / (ΔP/P_avg)
        """
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
            return None

        return {
            'value': np.mean(elasticities),
            'std_error': np.std(elasticities) / np.sqrt(len(elasticities))
        }


def aggregate_price_logs(price_logs: pd.DataFrame,
                         window_size: str = '1H',
                         ts_col: str = 'ts') -> List[Dict]:
    """
    Recover price vectors treating prices as persistent state changes.

    Prices are set-operations that persist until next change. For each window:
    - If price logs exist: average all changes within window
    - If no logs: carry forward last price before window end

    Args:
        price_logs: df with [productId, price, ts, ...]
        window_size: time window size matching ChunkInteractionsIntoSteps
        ts_col: timestamp column name

    Returns:
        list of dicts with {'window_start', 'window_end', 'price_vector'}
        where price_vector is df with [productId, price]
    """
    if price_logs.empty:
        return []

    df = price_logs.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col])

    df = df.sort_values([ts_col, 'productId'])

    # generate windows across data range
    min_time, max_time = df[ts_col].min(), df[ts_col].max()
    windows = pd.date_range(
        start=min_time.floor(window_size),
        end=max_time,
        freq=window_size
    )

    chunks = []

    for window_start in windows:
        window_end = window_start + pd.Timedelta(window_size)
        price_vector = []

        # all products with price history by window_end
        historical_products = df[df[ts_col] < window_end]['productId'].unique()

        for pid in historical_products:
            product_data = df[df['productId'] == pid]

            # logs within window
            in_window = product_data[
                (product_data[ts_col] >= window_start) &
                (product_data[ts_col] < window_end)
            ]

            if not in_window.empty:
                # average changes within window
                price = in_window['price'].mean()
            else:
                # carry forward: last price before window end
                before_window = product_data[product_data[ts_col] < window_end]
                if before_window.empty:
                    continue
                price = before_window['price'].iloc[-1]

            price_vector.append({'productId': pid, 'price': price})

        if price_vector:
            chunks.append({
                'window_start': window_start,
                'window_end': window_end,
                'price_vector': pd.DataFrame(price_vector)
            })

    return chunks
