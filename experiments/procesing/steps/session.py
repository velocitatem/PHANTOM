"""
Session feature extraction for S_t component of state space.
Computes behavioral signals from interaction data already in pipeline.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from collections import Counter
from procesing.steps.base import BaseContextStep

def _extract_features_for_session(session_df: pd.DataFrame, session_timeout_sec: float = 900) -> Dict[str, Any]:
    """Compute features for single session.

    Args:
        session_df: interaction events for this session
        session_timeout_sec: max gap between events before resetting duration (default 900s = 15min)
    """
    features = {}

    # basic counts
    features['total_interactions'] = len(session_df)

    event_counts = session_df['eventName'].value_counts().to_dict()
    features['page_views'] = event_counts.get('page_view', 0) + event_counts.get('view_item_page', 0)
    features['item_views'] = event_counts.get('view_item_page', 0)
    features['searches'] = event_counts.get('search', 0)
    features['cart_adds'] = event_counts.get('add_item_to_cart', 0)

    # hover events
    hover_events = ['hover_over_title', 'hover_over_paragraph', 'hover_over_link', 'hover_over_button']
    features['hovers'] = sum(event_counts.get(ev, 0) for ev in hover_events)

    # product-level signals
    product_ids = session_df['productId'].dropna()
    features['unique_products_viewed'] = product_ids.nunique()

    if len(product_ids) > 0:
        product_view_counts = Counter(product_ids)
        features['product_view_depth'] = max(product_view_counts.values())
    else:
        features['product_view_depth'] = 0

    # temporal features with session timeout logic
    if 'ts' in session_df.columns:
        timestamps = session_df['ts'].sort_values()

        # compute active duration considering timeout gaps
        if len(timestamps) > 1:
            time_diffs = timestamps.diff().dropna().dt.total_seconds()
            # only count gaps shorter than timeout towards active session duration
            active_diffs = time_diffs[time_diffs <= session_timeout_sec]
            features['session_duration_sec'] = active_diffs.sum() if len(active_diffs) > 0 else 0.0

            features['avg_time_between_events'] = time_diffs.mean()
            features['std_time_between_events'] = time_diffs.std()
        else:
            features['session_duration_sec'] = 0.0
            features['avg_time_between_events'] = 0.0
            features['std_time_between_events'] = 0.0

        if features['session_duration_sec'] > 0:
            features['interaction_velocity'] = (features['total_interactions'] / features['session_duration_sec']) * 60
        else:
            features['interaction_velocity'] = 0.0
    else:
        features['session_duration_sec'] = 0.0
        features['interaction_velocity'] = 0.0
        features['avg_time_between_events'] = 0.0
        features['std_time_between_events'] = 0.0

    # cart/conversion signals
    features['cart_to_view_ratio'] = features['cart_adds'] / features['item_views'] if features['item_views'] > 0 else 0.0

    return features


def _apply_to_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature extraction to sliding window of interactions."""
    # add columns of all features at each step
    new_cols = ["total_interactions", "page_views", "item_views", "searches",
                "cart_adds", "hovers", "unique_products_viewed", "product_view_depth",
                "session_duration_sec", "interaction_velocity",
                "avg_time_between_events", "std_time_between_events",
                "cart_to_view_ratio"]
    for col in new_cols: df[col] = np.nan
    for idx in range(1, len(df) + 1):
        features = _extract_features_for_session(df.iloc[:idx])
        # fillna kinda meh
        features = { k: (v if not pd.isna(v) else 0.0) for k, v in features.items() }
        for col in new_cols:
            df.at[df.index[idx - 1], col] = features[col]
        #print(f"Processed {idx}/{len(df)} events for session {df['sessionId'].iloc[0]}")
    return df

class BuildStateSpaceStep(BaseContextStep):
    """
    Build state space representation S_t from session features.

    Input: session_features DataFrame
    Output: state_space_df DataFrame with S_t vectors
    """

    def transform(self, rich_dataset: pd.DataFrame) -> pd.DataFrame:
        # check if features are present
        required_cols = ["total_interactions", "page_views", "item_views", "searches",
                         "cart_adds", "hovers", "unique_products_viewed", "product_view_depth",
                         "session_duration_sec", "interaction_velocity",
                         "avg_time_between_events", "std_time_between_events",
                         "cart_to_view_ratio"]
        if not all(col in rich_dataset.columns for col in required_cols):
            raise ValueError("Missing required columns for feature extraction.")
        if rich_dataset.empty:
            return pd.DataFrame()


        # For simplicity, we return as is
        return rich_dataset.copy()




class ExtractSessionFeaturesStep(BaseContextStep):
    """
    Extract session-level behavioral features from interaction logs.

    Input: interactions_df (user-interactions from earlier pipeline step)
    Output: interactions_df with added session feature columns
    """

    def transform(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        if interactions_df.empty:
            return pd.DataFrame()

        # ensure timestamp column
        if 'ts' in interactions_df.columns:
            interactions_df = interactions_df.copy()
            interactions_df['ts'] = pd.to_datetime(interactions_df['ts'])

        # group by session and compute features
        session_features = []
        for session_id, session_df in interactions_df.groupby('sessionId'):
            new_slice = _apply_to_slice(session_df.sort_values('ts'))
            session_features.append(new_slice)

        return pd.concat(session_features, ignore_index=True)



class FilterSessionInteractionsStep(BaseContextStep):
    """
    Filter interactions DataFrame to specific session.

    Input: (interactions_df, session_id)
    Output: interactions_df filtered to session_id
    """

    def transform(self, data: tuple) -> pd.DataFrame:
        interactions_df, session_id = data
        return interactions_df[interactions_df['sessionId'] == session_id].copy()
