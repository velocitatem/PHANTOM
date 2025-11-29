"""
Session feature extraction for S_t component of state space.
Computes behavioral signals from interaction data already in pipeline.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from collections import Counter
from procesing.steps.base import BaseContextStep


class ExtractSessionFeaturesStep(BaseContextStep):
    """
    Extract session-level behavioral features from interaction logs.

    Input: interactions_df (user-interactions from earlier pipeline step)
    Output: session_features DataFrame [sessionId, feature_1, feature_2, ...]

    Features computed:
        - total_interactions: count of all events
        - page_views, item_views, searches, cart_adds: event type counts
        - hovers: hover event counts
        - unique_products_viewed: distinct product IDs
        - interaction_velocity: events per minute
        - session_duration_sec: time span of session
        - avg_time_between_events: mean inter-event time
        - product_view_depth: max views for single product (attention signal)
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
            features = self._extract_features_for_session(session_id, session_df)
            session_features.append(features)

        return pd.DataFrame(session_features)

    def _extract_features_for_session(self, session_id: str, session_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute features for single session."""
        features = {'sessionId': session_id}

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

        # temporal features
        if 'ts' in session_df.columns:
            timestamps = session_df['ts'].sort_values()
            features['session_duration_sec'] = (timestamps.max() - timestamps.min()).total_seconds()

            if features['session_duration_sec'] > 0:
                features['interaction_velocity'] = (features['total_interactions'] / features['session_duration_sec']) * 60
            else:
                features['interaction_velocity'] = 0.0

            # inter-event timing
            if len(timestamps) > 1:
                time_diffs = timestamps.diff().dropna().dt.total_seconds()
                features['avg_time_between_events'] = time_diffs.mean()
                features['std_time_between_events'] = time_diffs.std()
            else:
                features['avg_time_between_events'] = 0.0
                features['std_time_between_events'] = 0.0
        else:
            features['session_duration_sec'] = 0.0
            features['interaction_velocity'] = 0.0
            features['avg_time_between_events'] = 0.0
            features['std_time_between_events'] = 0.0

        # cart/conversion signals
        features['cart_to_view_ratio'] = features['cart_adds'] / features['item_views'] if features['item_views'] > 0 else 0.0

        return features


class FilterSessionInteractionsStep(BaseContextStep):
    """
    Filter interactions DataFrame to specific session.

    Input: (interactions_df, session_id)
    Output: interactions_df filtered to session_id
    """

    def transform(self, data: tuple) -> pd.DataFrame:
        interactions_df, session_id = data
        return interactions_df[interactions_df['sessionId'] == session_id].copy()
