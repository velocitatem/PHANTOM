"""
Revenue and KPI benchmark framework for pricing strategies.

Computes session-level and aggregate metrics to compare pricing functions:
    - Revenue: R_T = Σ P_t^T · Q_t
    - Conversion rate
    - Average order value (AOV)
    - Agent exploitation loss: L_agent = R_oracle - R_observed
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np


@dataclass
class SessionMetrics:
    """KPIs for single session."""
    session_id: str
    experiment_id: Optional[str] = None

    # interaction metrics
    total_interactions: int = 0
    page_views: int = 0
    item_views: int = 0
    searches: int = 0
    cart_adds: int = 0

    # revenue metrics
    items_purchased: int = 0
    total_revenue: float = 0.0
    avg_item_price: float = 0.0
    conversion_rate: float = 0.0

    # pricing signals
    total_price_shown: float = 0.0  # sum of all prices displayed
    avg_markup: float = 0.0  # avg (price / base_price)

    # behavioral features (for agent detection)
    interaction_velocity: float = 0.0  # interactions per minute
    session_duration_sec: float = 0.0
    unique_products_viewed: int = 0

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateMetrics:
    """Aggregate KPIs across sessions/experiments."""
    experiment_id: Optional[str] = None
    n_sessions: int = 0

    # revenue aggregates
    total_revenue: float = 0.0
    avg_revenue_per_session: float = 0.0
    median_revenue_per_session: float = 0.0

    # conversion aggregates
    total_conversions: int = 0
    conversion_rate: float = 0.0  # purchases / sessions

    # pricing aggregates
    avg_markup: float = 0.0
    median_markup: float = 0.0

    # agent exploitation metrics
    estimated_agent_sessions: int = 0  # sessions flagged as agent-driven
    agent_revenue: float = 0.0
    human_revenue: float = 0.0
    agent_loss: float = 0.0  # L_agent = R_oracle - R_observed (if available)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsComputer:
    """Compute session and aggregate metrics from interaction/price logs."""

    @staticmethod
    def compute_session_metrics(
        session_id: str,
        interactions: pd.DataFrame,
        price_logs: pd.DataFrame,
        purchases: Optional[pd.DataFrame] = None,
        experiment_id: Optional[str] = None
    ) -> SessionMetrics:
        """
        Compute metrics for single session.

        Args:
            session_id: session identifier
            interactions: user-interactions events for this session
            price_logs: price-logs for this session
            purchases: purchase events (if available)
            experiment_id: experiment identifier
        """
        metrics = SessionMetrics(session_id=session_id, experiment_id=experiment_id)

        if interactions.empty:
            return metrics

        # interaction counts
        event_counts = interactions['eventName'].value_counts().to_dict()
        metrics.total_interactions = len(interactions)
        metrics.page_views = event_counts.get('page_view', 0) + event_counts.get('view_item_page', 0)
        metrics.item_views = event_counts.get('view_item_page', 0)
        metrics.searches = event_counts.get('search', 0)
        metrics.cart_adds = event_counts.get('add_item_to_cart', 0)

        # unique products viewed
        metrics.unique_products_viewed = interactions['productId'].dropna().nunique()

        # session duration
        if 'ts' in interactions.columns:
            timestamps = pd.to_datetime(interactions['ts'])
            metrics.session_duration_sec = (timestamps.max() - timestamps.min()).total_seconds()
            if metrics.session_duration_sec > 0:
                metrics.interaction_velocity = (metrics.total_interactions / metrics.session_duration_sec) * 60

        # revenue from purchases
        if purchases is not None and not purchases.empty:
            metrics.items_purchased = len(purchases)
            metrics.total_revenue = purchases['price'].sum() if 'price' in purchases.columns else 0.0
            metrics.avg_item_price = metrics.total_revenue / metrics.items_purchased if metrics.items_purchased > 0 else 0.0
            metrics.conversion_rate = 1.0 if metrics.items_purchased > 0 else 0.0

        # pricing metrics
        if not price_logs.empty:
            metrics.total_price_shown = price_logs['price'].sum()
            # compute markup if base_price available in price logs or join with product catalog
            if 'base_price' in price_logs.columns:
                valid_markup = price_logs[price_logs['base_price'] > 0]
                if not valid_markup.empty:
                    metrics.avg_markup = (valid_markup['price'] / valid_markup['base_price']).mean()

        return metrics

    @staticmethod
    def compute_aggregate_metrics(
        session_metrics_list: List[SessionMetrics],
        experiment_id: Optional[str] = None,
        agent_detector_fn: Optional[callable] = None
    ) -> AggregateMetrics:
        """
        Aggregate metrics across sessions.

        Args:
            session_metrics_list: list of SessionMetrics
            experiment_id: experiment identifier
            agent_detector_fn: optional function to classify session as agent (returns bool)
        """
        agg = AggregateMetrics(experiment_id=experiment_id)
        agg.n_sessions = len(session_metrics_list)

        if agg.n_sessions == 0:
            return agg

        df = pd.DataFrame([m.to_dict() for m in session_metrics_list])

        # revenue aggregates
        agg.total_revenue = df['total_revenue'].sum()
        agg.avg_revenue_per_session = df['total_revenue'].mean()
        agg.median_revenue_per_session = df['total_revenue'].median()

        # conversion aggregates
        agg.total_conversions = (df['items_purchased'] > 0).sum()
        agg.conversion_rate = agg.total_conversions / agg.n_sessions

        # pricing aggregates
        valid_markups = df[df['avg_markup'] > 0]
        if not valid_markups.empty:
            agg.avg_markup = valid_markups['avg_markup'].mean()
            agg.median_markup = valid_markups['avg_markup'].median()

        # agent detection (if detector provided)
        if agent_detector_fn is not None:
            agent_flags = [agent_detector_fn(m) for m in session_metrics_list]
            agg.estimated_agent_sessions = sum(agent_flags)

            agent_revenue = sum(m.total_revenue for m, is_agent in zip(session_metrics_list, agent_flags) if is_agent)
            human_revenue = sum(m.total_revenue for m, is_agent in zip(session_metrics_list, agent_flags) if not is_agent)

            agg.agent_revenue = agent_revenue
            agg.human_revenue = human_revenue

        return agg

    @staticmethod
    def compare_pricing_strategies(
        experiments: Dict[str, List[SessionMetrics]],
        baseline_experiment_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple pricing strategies/experiments.

        Args:
            experiments: dict mapping experiment_id -> list of SessionMetrics
            baseline_experiment_id: experiment to use as baseline for comparison

        Returns:
            DataFrame with comparative metrics
        """
        results = []
        baseline_agg = None

        for exp_id, session_metrics in experiments.items():
            agg = MetricsComputer.compute_aggregate_metrics(session_metrics, experiment_id=exp_id)
            result = agg.to_dict()

            if exp_id == baseline_experiment_id:
                baseline_agg = agg

            results.append(result)

        df = pd.DataFrame(results)

        # add relative metrics if baseline exists
        if baseline_agg is not None:
            df['revenue_lift_pct'] = ((df['total_revenue'] - baseline_agg.total_revenue) / baseline_agg.total_revenue * 100)
            df['conversion_lift_pct'] = ((df['conversion_rate'] - baseline_agg.conversion_rate) / baseline_agg.conversion_rate * 100)

        return df


def simple_agent_detector(session_metrics: SessionMetrics, velocity_threshold: float = 5.0) -> bool:
    """
    Simple heuristic agent detector based on interaction velocity.

    Args:
        session_metrics: SessionMetrics instance
        velocity_threshold: interactions per minute threshold (default: 5.0)

    Returns:
        True if session likely agent-driven
    """
    # agents tend to have higher interaction velocity and lower session duration
    if session_metrics.interaction_velocity > velocity_threshold:
        return True
    # agents often view many products quickly without converting
    if session_metrics.unique_products_viewed > 10 and session_metrics.conversion_rate == 0:
        return True
    return False
