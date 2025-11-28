import pandas as pd
from .base import BaseContextStep

class ChunkByTimeWindowStep(BaseContextStep):
    """
    Chunk dataframe into time windows.
    Returns list of dicts with window metadata.
    """

    def transform(self, df: pd.DataFrame):
        if df.empty:
            return []

        df = df.copy()
        ts_col = self.context.config.get('ts_col', 'ts')
        window_size = self.context.window_size

        # ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
            df[ts_col] = pd.to_datetime(df[ts_col])

        df = df.sort_values(ts_col)
        df['_window'] = df[ts_col].dt.floor(window_size)

        chunks = []
        for idx, (window_start, group) in enumerate(df.groupby('_window')):
            chunks.append({
                'window_start': window_start,
                'window_end': window_start + pd.Timedelta(window_size),
                'window_idx': idx,
                'data': group.drop(columns=['_window'])
            })

        return chunks
