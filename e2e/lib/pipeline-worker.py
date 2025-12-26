#!/usr/bin/env python3
"""
E2E Test Pipeline Worker

A lightweight worker that runs the surge pricing pipeline for E2E tests.
This bypasses Airflow for faster, more reliable test execution.

Usage:
    python pipeline-worker.py --store-mode hotel --high-threshold 3 --surge-multiplier 1.5
"""

import argparse
import json
import logging
import os
import sys
from typing import Optional
from datetime import datetime

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'experiments'))
sys.path.insert(0, os.path.join(project_root, 'lib'))

from procesing.context import PipelineContext
from procesing.providers import BackendAPIProvider
from procesing.steps import (
    FetchInteractionsStep,
    FetchPriceLogsStep,
    ComputeDemandStep,
    AggregatePriceLogsStep,
    JoinProductFeaturesStep,
)
from procesing.pricers.simple import SimpleSurgePricer
from lib.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)


class E2ETestProvider(BackendAPIProvider):
    """Provider configured for E2E test environment"""

    def __init__(self, backend_url: str = None):
        self.backend_url = backend_url or os.getenv('BACKEND_URL', 'http://localhost:5000')
        super().__init__()


def run_pricing_pipeline(
    store_mode: str = 'hotel',
    high_threshold: int = 3,
    low_threshold: int = 1,
    surge_multiplier: float = 1.5,
    discount_multiplier: float = 0.9,
    dry_run: bool = False
) -> dict:
    """
    Execute the surge pricing pipeline and publish results to Redis.

    Args:
        store_mode: 'hotel' or 'airline'
        high_threshold: Demand threshold for surge pricing
        low_threshold: Demand threshold for discount pricing
        surge_multiplier: Price multiplier for high demand
        discount_multiplier: Price multiplier for low demand
        dry_run: If True, don't publish to Redis

    Returns:
        dict with pipeline results and statistics
    """
    log.info(f"Starting E2E pricing pipeline: mode={store_mode}, "
             f"high_threshold={high_threshold}, surge_multiplier={surge_multiplier}")

    # Initialize provider and context
    provider = E2ETestProvider()
    context = PipelineContext(provider=provider, store_mode=store_mode)

    # Step 1: Fetch interactions from Kafka
    log.info("Fetching interactions from Kafka...")
    fetch_interactions = FetchInteractionsStep(context)
    interactions_df = fetch_interactions.transform(None)
    log.info(f"Fetched {len(interactions_df)} interaction records")

    if interactions_df.empty:
        log.warning("No interactions found. Pipeline will produce no price updates.")
        return {
            'success': True,
            'interactions_count': 0,
            'products_count': 0,
            'prices_published': False,
            'message': 'No interactions to process'
        }

    # Step 2: Fetch price logs from Kafka
    log.info("Fetching price logs from Kafka...")
    fetch_prices = FetchPriceLogsStep(context)
    price_logs_df = fetch_prices.transform(None)
    log.info(f"Fetched {len(price_logs_df)} price log records")

    # Step 3: Compute demand scores
    log.info("Computing demand scores...")
    compute_demand = ComputeDemandStep(context)
    demand_df = compute_demand.transform(interactions_df)
    log.info(f"Computed demand for {len(demand_df)} products")

    if demand_df.empty:
        log.warning("No demand data computed.")
        return {
            'success': True,
            'interactions_count': len(interactions_df),
            'products_count': 0,
            'prices_published': False,
            'message': 'No demand data to process'
        }

    # Step 4: Aggregate price logs
    log.info("Aggregating price logs...")
    aggregate_prices = AggregatePriceLogsStep(context)
    price_agg_df = aggregate_prices.transform(price_logs_df)
    log.info(f"Aggregated prices for {len(price_agg_df)} products")

    # Step 5: Join product features
    log.info("Joining product features...")
    join_features = JoinProductFeaturesStep(context)
    features_df = join_features.transform((demand_df, price_agg_df))
    log.info(f"Joined features for {len(features_df)} products")

    if features_df.empty:
        log.warning("No product features after join.")
        return {
            'success': True,
            'interactions_count': len(interactions_df),
            'products_count': 0,
            'prices_published': False,
            'message': 'No product features to price'
        }

    # Step 6: Apply surge pricing
    log.info(f"Applying surge pricing (high={high_threshold}, surge={surge_multiplier}x)...")

    # Rename columns for pricer compatibility
    data = features_df.rename(columns={'demand_score': 'demand'})

    surge_pricer = SimpleSurgePricer(
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        surge_multiplier=surge_multiplier,
        discount_multiplier=discount_multiplier
    )
    surge_pricer.fit(data)
    data['optimal_price'] = surge_pricer.predict()

    # Prepare output DataFrame
    prices_df = data[['productId', 'price', 'base_price', 'optimal_price', 'demand']].rename(columns={
        'price': 'current_price',
        'demand': 'demand_score'
    })

    log.info(f"Generated optimal prices for {len(prices_df)} products")

    # Log pricing decisions
    for _, row in prices_df.iterrows():
        markup = row['optimal_price'] / row['base_price'] if row['base_price'] > 0 else 1.0
        log.info(f"  {row['productId'][:8]}...: base=${row['base_price']:.2f} "
                 f"-> optimal=${row['optimal_price']:.2f} (demand={row['demand_score']:.0f}, markup={markup:.2f}x)")

    # Step 7: Publish to Redis
    if not dry_run:
        log.info("Publishing prices to Redis registry...")
        registry = ModelRegistry()

        metadata = {
            'timestamp': datetime.utcnow().isoformat(),
            'store_mode': store_mode,
            'pipeline': 'e2e_test_worker',
            'high_threshold': high_threshold,
            'low_threshold': low_threshold,
            'surge_multiplier': surge_multiplier,
            'discount_multiplier': discount_multiplier,
        }

        registry.publish_prices(prices_df, model_name='latest', metadata=metadata)
        log.info(f"✅ Published {len(prices_df)} prices to Redis")
    else:
        log.info("Dry run - skipping Redis publish")

    return {
        'success': True,
        'interactions_count': len(interactions_df),
        'products_count': len(prices_df),
        'prices_published': not dry_run,
        'prices': prices_df.to_dict(orient='records'),
        'timestamp': datetime.utcnow().isoformat()
    }


def main():
    parser = argparse.ArgumentParser(description='E2E Test Pipeline Worker')
    parser.add_argument('--store-mode', choices=['hotel', 'airline'], default='hotel',
                        help='Store mode (hotel or airline)')
    parser.add_argument('--high-threshold', type=int, default=3,
                        help='Demand threshold for surge pricing')
    parser.add_argument('--low-threshold', type=int, default=1,
                        help='Demand threshold for discount pricing')
    parser.add_argument('--surge-multiplier', type=float, default=1.5,
                        help='Price multiplier for high demand')
    parser.add_argument('--discount-multiplier', type=float, default=0.9,
                        help='Price multiplier for low demand')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run without publishing to Redis')
    parser.add_argument('--json-output', action='store_true',
                        help='Output results as JSON')

    args = parser.parse_args()

    try:
        result = run_pricing_pipeline(
            store_mode=args.store_mode,
            high_threshold=args.high_threshold,
            low_threshold=args.low_threshold,
            surge_multiplier=args.surge_multiplier,
            discount_multiplier=args.discount_multiplier,
            dry_run=args.dry_run
        )

        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            log.info(f"Pipeline completed: {result['products_count']} products priced")

        sys.exit(0 if result['success'] else 1)

    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        if args.json_output:
            print(json.dumps({'success': False, 'error': str(e)}))
        sys.exit(1)


if __name__ == '__main__':
    main()
