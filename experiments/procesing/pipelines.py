from sklearn.pipeline import Pipeline
import pandas as pd
from procesing.context import PipelineContext
from procesing.providers import SupabaseProvider, BackendAPIProvider
from typing import Union
from procesing.steps import (
    FetchInteractionsStep,
    FetchPriceLogsStep,
    FetchExperimentsStep,
    JoinExperimentsStep,
    CreatePriceBucketsStep,
    AugmentEventNamesStep,
    ChunkByTimeWindowStep,
    ComputeDemandForChunksStep,
    AggregatePriceLogsStep,
    ComputeElasticityStep,
    # BuildStateSpaceStep,
    FitPricingFunctionStep,
    PredictPricesStep,
)

def interaction_extraction_pipeline(context: PipelineContext):
    """Pipeline for extracting and augmenting interaction data"""
    return Pipeline([
        ('fetch', FetchInteractionsStep(context)),
        ('create_buckets', CreatePriceBucketsStep(context)),
        ('augment_events', AugmentEventNamesStep(context)),
    ])


def price_extraction_pipeline(context: PipelineContext):
    """Pipeline for extracting price logs"""
    return Pipeline([
        ('fetch', FetchPriceLogsStep(context)),
    ])


def elasticity_computation_pipeline(context: PipelineContext,
                                    interactions_df: pd.DataFrame,
                                    price_logs_df: pd.DataFrame):
    """
    Compute elasticity from interactions and price logs.
    Manual orchestration needed for branching logic.
    """
    # branch 1: chunk interactions and compute demand
    chunk_step = ChunkByTimeWindowStep(context)
    interaction_chunks = chunk_step.transform(interactions_df)

    demand_step = ComputeDemandForChunksStep(context)
    demand_chunks = demand_step.transform(interaction_chunks)

    # branch 2: aggregate price logs
    price_step = AggregatePriceLogsStep(context)
    price_chunks = price_step.transform(price_logs_df)

    # convergence: compute elasticity
    elasticity_step = ComputeElasticityStep(context)
    elasticity_df = elasticity_step.transform((demand_chunks, price_chunks))

    return elasticity_df


def pricing_pipeline(context: PipelineContext, elasticity_df: pd.DataFrame):
    """
    Generate optimal prices from elasticity estimates.
    """
    # build state space
    state_step = BuildStateSpaceStep(context)
    state_space = state_step.transform(elasticity_df)

    # fit pricing function
    fit_step = FitPricingFunctionStep(context)
    pricer = fit_step.transform(elasticity_df)

    # predict prices
    predict_step = PredictPricesStep(context)
    prices_df = predict_step.transform((pricer, state_space))

    return prices_df


def full_pipeline(context: PipelineContext):
    """
    Complete end-to-end pipeline: data extraction -> elasticity -> pricing
    Returns: (elasticity_df, prices_df)
    """
    # extract interactions
    interaction_pipe = interaction_extraction_pipeline(context)
    interactions_df = interaction_pipe.fit_transform(None)

    # extract price logs
    price_pipe = price_extraction_pipeline(context)
    price_logs_df = price_pipe.fit_transform(None)

    if interactions_df.empty or price_logs_df.empty:
        return None, None

    # compute elasticity
    elasticity_df = elasticity_computation_pipeline(
        context,
        interactions_df,
        price_logs_df
    )

    if elasticity_df is None or elasticity_df.empty:
        return elasticity_df, None

    # generate prices
    prices_df = pricing_pipeline(context, elasticity_df)

    return elasticity_df, prices_df


if __name__ == '__main__':

    class Provider(SupabaseProvider, BackendAPIProvider):
        def __init__(self, backend_url: str):
            SupabaseProvider.__init__(self)
            BackendAPIProvider.__init__(self, backend_url=backend_url)
    # example run
    context = PipelineContext(
        provider=Provider(backend_url="http://localhost:5000"),
        store_mode='hotel',
        # 15 min not month
        window_size='15min',
    )

    elasticity_df, prices_df = full_pipeline(context)

    if elasticity_df is not None and not elasticity_df.empty:
        print("Elasticity Estimates:")
        print(elasticity_df.to_string(index=False))
    else:
        print("No elasticity estimates computed.")

    if prices_df is not None and not prices_df.empty:
        print("\nPredicted Prices:")
        print(prices_df.to_string(index=False))
    else:
        print("No prices predicted.")
