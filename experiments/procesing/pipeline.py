from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging
log = logging.getLogger(__name__)

from extract import KafkaDataFetcher, ExperimentJoiner, EventTitleAugmenter, chunk_shared_data
from mapping import SessionTransitionProbMatrixTransformer, render_graph
from demand import DemandEstimator, ChunkInteractionsIntoSteps
from elasticity import TemporalElasticityEstimator, aggregate_price_logs



# elasticity pipeline components (not sklearn compatible, manual orchestration)
def elasticity_pipeline(interactions_df, price_logs_df, window_size='30s', store_mode='hotel'):
    """
    Compute price elasticity from interaction and price data.

    Args:
        interactions_df: raw interaction data from demand_data_pipeline
        price_logs_df: price log data from price_data_pipeline
        window_size: time window for chunking
        store_mode: 'hotel' or 'airline'

    Returns:
        df with [productId, elasticity, std_error, n_obs]
    """
    # step 1: chunk interactions into time windows
    chunker = ChunkInteractionsIntoSteps(window_size=window_size, return_metadata=True)
    interaction_chunks = chunker.transform(interactions_df)
    log.info(f"Chunked interactions into {len(interaction_chunks)} windows of size {window_size}")

    if not interaction_chunks:
        return None

    # step 2: compute demand per window
    demand_estimator = DemandEstimator(store_mode=store_mode)
    demand_chunks = []
    for chunk in interaction_chunks:
        demand_vector = demand_estimator.transform(chunk['data'])
        demand_chunks.append({
            'window_start': chunk['window_start'],
            'window_end': chunk['window_end'],
            'demand_vector': demand_vector # each has a full list of all products, even if demand is 0
        })
    # [q_chunk1, q_chunk2, ...]

    # step 3: aggregate price logs into windows
    price_chunks = aggregate_price_logs(price_logs_df, window_size=window_size)

    # step 4: compute elasticity
    elasticity_estimator = TemporalElasticityEstimator(method='point', min_observations=2)
    elasticity_df = elasticity_estimator.transform(demand_chunks, price_chunks, store_mode=store_mode)

    return elasticity_df


# exposable pipelines
interaction_pipeline = Pipeline([
    ('kafka_fetch', KafkaDataFetcher(topic='user-interactions')),
    ('experiment_join', ExperimentJoiner()),
    ('event_augment', EventTitleAugmenter()),
])

price_data_pipeline = Pipeline([
    ('kafka_fetch', KafkaDataFetcher(topic='price-logs')),
])

# interaction_data + price_data -> elasticity (demand)
# elasticity -> pricing

pricing_pipeline = Pipeline([
    ('demand_estimation', DemandEstimator()),
])
if __name__ == "__main__":
    # fetch both datasets
    interaction_data = interaction_pipeline.fit_transform(None)
    pricing_data = price_data_pipeline.fit_transform(None)
    if interaction_data.empty or pricing_data.empty:
        print("Insufficient data for elasticity computation"); exit(0)
    # compute elasticity via unified pipeline
    window_size = "30s"
    elasticity_results = elasticity_pipeline(interaction_data, pricing_data, window_size=window_size)
    elasticity_value_array = elasticity_results['elasticity'].values if elasticity_results is not None else np.array([])
    print(elasticity_value_array)

    if elasticity_results is not None and not elasticity_results.empty:
        print(elasticity_results.to_string(index=False))
    else:
        print("\nInsufficient data for elasticity computation")
