"""
Session feature extraction for ML training pipeline.
"""
import pandas as pd
import numpy as np
import re
from typing import Dict, Any
from procesing.steps.base import BaseContextStep

EVENT_CATS = {
    'page_view': ['page_view'],
    'item_view': ['view_item_page', 'learn_more_about_item'],
    'cart_add': ['add_item_to_cart'],
    'purchase': ['purchase', 'checkout_complete'],
    'hover': ['hover_over_title', 'hover_over_paragraph', 'hover_over_link', 'hover_over_button'],
    # 'filter': ['filter', 'search', 'apply_filter'],
}
HEADLESS_RE = re.compile(r'HeadlessChrome|Headless|PhantomJS', re.I)
AUTOMATION_RE = re.compile(r'Selenium|Playwright|Puppeteer|WebDriver|chromedriver|geckodriver', re.I)
BROWSER_PATTERNS = [('Chrome', r'Chrome/[\d.]+'), ('Firefox', r'Firefox/[\d.]+'),
                    ('Safari', r'Safari/[\d.]+'), ('Edge', r'Edg/[\d.]+')]


class TemporalFeatureStep(BaseContextStep):
    """Vectorized time-based features: durations, velocities, gaps."""

    def __init__(self, context, timeout_sec: float = 900, velocity_window: str = '5min'):
        super().__init__(context)
        self.timeout_sec = timeout_sec
        self.velocity_window = velocity_window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or 'ts' not in df.columns:
            return pd.DataFrame(columns=['sessionId'])

        df = df.copy()
        df['ts_dt'] = pd.to_datetime(df['ts'])
        df = df.sort_values(['sessionId', 'ts_dt'])
        df['time_diff'] = df.groupby('sessionId')['ts_dt'].diff().dt.total_seconds()
        df['active_diff'] = df['time_diff'].where(df['time_diff'] <= self.timeout_sec, 0)

        agg = df.groupby('sessionId').agg(
            session_duration_sec=('active_diff', 'sum'),
            total_interactions=('sessionId', 'count'),
            avg_time_between_events=('time_diff', 'mean'),
            std_time_between_events=('time_diff', 'std'),
            min_time_between_events=('time_diff', 'min'),
            session_start_hour=('ts_dt', lambda x: x.min().hour),
        ).reset_index()
        agg['std_time_between_events'] = agg['std_time_between_events'].fillna(0)
        agg['interaction_velocity'] = np.where(
            agg['session_duration_sec'] > 0,
            (agg['total_interactions'] / agg['session_duration_sec']) * 60, 0)

        vel = df.set_index('ts_dt').groupby('sessionId').resample(self.velocity_window).size()
        agg = agg.merge(vel.groupby('sessionId').max().rename('max_velocity_5min'),
                        on='sessionId', how='left').fillna({'max_velocity_5min': 0})
        return agg


class BehavioralFeatureStep(BaseContextStep):
    """Vectorized event counts and ratios per session."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or 'eventName' not in df.columns:
            return pd.DataFrame(columns=['sessionId'])

        df = df.copy()
        for cat, events in EVENT_CATS.items():
            df[f'is_{cat}'] = df['eventName'].isin(events)
        df['is_hover'] = df['is_hover'] | df['eventName'].str.startswith('hover_over_')

        agg = df.groupby('sessionId').agg(
            total_events=('eventName', 'count'), unique_pages=('page', 'nunique'),
            page_views=('is_page_view', 'sum'), item_views=('is_item_view', 'sum'),
            cart_adds=('is_cart_add', 'sum'), purchases=('is_purchase', 'sum'),
            hover_events=('is_hover', 'sum'),
            # filter_events=('is_filter', 'sum'),
        ).reset_index()
        agg['cart_to_view_ratio'] = np.where(agg['item_views'] > 0, agg['cart_adds'] / agg['item_views'], 0)
        agg['conversion_rate'] = np.where(agg['item_views'] > 0, agg['purchases'] / agg['item_views'], 0)
        agg['hover_intensity'] = np.where(agg['total_events'] > 0, agg['hover_events'] / agg['total_events'], 0)
        return agg


class ProductFeatureStep(BaseContextStep):
    """Vectorized product interaction features: diversity, depth, price sensitivity."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=['sessionId'])

        df = df.copy()
        price_col = next((c for c in ['metadata_base_price', 'metadata_price', 'base_price'] if c in df.columns), None)
        df['price_seen'] = pd.to_numeric(df[price_col], errors='coerce') if price_col else np.nan

        prod_df = df[df['productId'].notna()]
        if prod_df.empty:
            return pd.DataFrame(columns=pd.Series(['sessionId', 'unique_products_viewed', 'product_view_depth', 'avg_price_seen', 'min_price_seen', 'max_price_seen', 'price_range']))

        agg = prod_df.groupby('sessionId').agg(
            unique_products_viewed=('productId', 'nunique'),
            product_view_depth=('productId', lambda x: x.value_counts().iloc[0] if len(x) > 0 else 0),
            avg_price_seen=('price_seen', 'mean'), min_price_seen=('price_seen', 'min'),
            max_price_seen=('price_seen', 'max'),
        ).reset_index()
        agg['price_range'] = (agg['max_price_seen'] - agg['min_price_seen']).fillna(0)
        return agg


class UserAgentFeatureStep(BaseContextStep):
    """Parse userAgent into bot-detection signals."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or 'userAgent' not in df.columns:
            return pd.DataFrame(columns=['sessionId'])

        ua = df.groupby('sessionId')['userAgent'].first().reset_index()
        ua['is_headless'] = ua['userAgent'].str.contains(HEADLESS_RE, na=False)
        ua['is_automation'] = ua['userAgent'].str.contains(AUTOMATION_RE, na=False)

        def get_browser(s):
            if pd.isna(s): return 'Unknown'
            for name, pat in BROWSER_PATTERNS:
                if re.search(pat, s): return name
            return 'Other'
        ua['browser_family'] = ua['userAgent'].apply(get_browser)
        return ua[['sessionId', 'is_headless', 'is_automation', 'browser_family']]


class ExtractSessionFeaturesStep(BaseContextStep):
    """
    Vectorized session feature extraction - replaces O(n^2) per-row loop.
    Input: interactions_df
    Output: session-level feature matrix
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        # run all feature steps and merge on sessionId
        temporal = TemporalFeatureStep(self.context).transform(df)
        behavioral = BehavioralFeatureStep(self.context).transform(df)
        product = ProductFeatureStep(self.context).transform(df)
        ua = UserAgentFeatureStep(self.context).transform(df)

        result = temporal
        for other in [behavioral, product, ua]:
            if not other.empty and 'sessionId' in other.columns:
                result = result.merge(other, on='sessionId', how='left')

        # carry forward experimentId for label joining
        if 'experimentId' in df.columns:
            exp_map = df.groupby('sessionId')['experimentId'].first()
            result = result.merge(exp_map, on='sessionId', how='left')

        return result


class JoinLabelsStep(BaseContextStep):
    """
    Join experiment labels to session features.
    Input: (features_df, experiments_df) or features_df (fetches experiments)
    Output: labeled feature matrix with is_agent column
    """

    def transform(self, data) -> pd.DataFrame:
        if isinstance(data, tuple):
            features_df, experiments_df = data
        else:
            features_df = data
            if 'experimentId' not in features_df.columns:
                return features_df
            exp_ids = features_df['experimentId'].dropna().unique().tolist()
            experiments_df = self.context.provider.fetch_experiments(exp_ids) if exp_ids else pd.DataFrame()

        if features_df.empty:
            return features_df
        if experiments_df.empty:
            features_df['is_agent'] = np.nan
            return features_df

        exp = experiments_df.copy()
        if 'id' in exp.columns:
            exp = exp.rename(columns={'id': 'experimentId'})
        if 'xp_human_only' in exp.columns:
            exp['is_agent'] = ~exp['xp_human_only']

        cols = ['experimentId'] + [c for c in ['is_agent', 'xp_human_only', 'xp_market_mode'] if c in exp.columns]
        return features_df.merge(exp[cols].drop_duplicates(), on='experimentId', how='left')


class ValidateDataStep(BaseContextStep):
    """
    Data quality checks before training.
    Input: df
    Output: df (unchanged, but logs validation report to context)
    """
    REQUIRED = ['sessionId', 'eventName', 'ts']

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        report = {'status': 'valid', 'rows': len(df), 'sessions': 0}
        if df.empty:
            report['status'] = 'empty'
            self.context.cache('validation_report', report)
            return df

        missing = [c for c in self.REQUIRED if c not in df.columns]
        if missing:
            report['status'] = 'invalid'
            report['missing_cols'] = missing

        report['sessions'] = df['sessionId'].nunique() if 'sessionId' in df.columns else 0
        report['null_sessions'] = int(df['sessionId'].isna().sum()) if 'sessionId' in df.columns else 0
        if 'experimentId' in df.columns:
            report['null_experiments'] = int(df['experimentId'].isna().sum())

        self.context.cache('validation_report', report)
        return df


# legacy compat - kept for backwards compatibility with existing code
def _extract_features_for_session(session_df: pd.DataFrame, session_timeout_sec: float = 900) -> Dict[str, Any]:
    """Single-session feature extraction (legacy interface)."""
    defaults = {k: 0 for k in ['total_interactions', 'page_views', 'item_views', 'searches',
                               'cart_adds', 'hovers', 'unique_products_viewed', 'product_view_depth',
                               'session_duration_sec', 'interaction_velocity',
                               'avg_time_between_events', 'std_time_between_events', 'cart_to_view_ratio']}
    if session_df.empty:
        return defaults

    session_df = session_df.copy()
    if 'sessionId' not in session_df.columns:
        session_df['sessionId'] = 'tmp'

    # use a dummy context for the steps
    class DummyCtx: config = {}
    ctx = DummyCtx()

    t = TemporalFeatureStep(ctx, timeout_sec=session_timeout_sec).transform(session_df)
    b = BehavioralFeatureStep(ctx).transform(session_df)
    p = ProductFeatureStep(ctx).transform(session_df)

    result = {}
    for df in [t, b, p]:
        if not df.empty:
            for col in df.columns:
                if col != 'sessionId':
                    result[col] = df[col].iloc[0] if len(df) > 0 else 0

    remap = {'hover_events': 'hovers', 'filter_events': 'searches', 'unique_pages': 'unique_pages_visited'}
    for old, new in remap.items():
        if old in result:
            result[new] = result.pop(old)
    return result
