from sklearn.pipeline import Pipeline
import pandas as pd
from procesing.context import PipelineContext
from procesing.providers import SupabaseProvider, BackendAPIProvider
import os
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
    FitPricingFunctionStep,
    PredictPricesStep,
    ComputeDemandStep,
    JoinProductFeaturesStep,
    ExtractSessionFeaturesStep,
    JoinLabelsStep,
    ValidateDataStep,
)
from procesing.pricers import SimpleSurgePricer

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


def product_features_pipeline(context: PipelineContext,
                                    interactions_df: pd.DataFrame,
                                    price_logs_df: pd.DataFrame):
    demand_step = ComputeDemandStep(context)
    price_step = AggregatePriceLogsStep(context)
    join_step = JoinProductFeaturesStep(context)


    demand_data = demand_step.transform(interactions_df)
    price_data= price_step.transform(price_logs_df)
    joined_data = join_step.transform((demand_data, price_data))

    return joined_data



def pricing_pipeline(context: "PipelineContext",
                     data: pd.DataFrame,
                     high_threshold: int = 10,
                     low_threshold: int = 2,
                     surge_multiplier: float = 1.2,
                     discount_multiplier: float = 0.9) -> pd.DataFrame:

    if data.empty or 'productId' not in data.columns:
        return pd.DataFrame()

    surge_pricer = SimpleSurgePricer()
    surge_pricer.fit(data)
    data['optimal_price'] = surge_pricer.predict()
    return data


def full_pipeline(context: PipelineContext,
                  high_threshold: int = 10,
                  low_threshold: int = 2,
                  surge_multiplier: float = 1.2,
                  discount_multiplier: float = 0.9):
    """
    Complete end-to-end pipeline: data extraction -> demand/price aggregation -> surge pricing

    Args:
        context: Pipeline context
        high_threshold: Demand threshold for surge pricing
        low_threshold: Demand threshold for discounts
        surge_multiplier: Price multiplier for high demand
        discount_multiplier: Price multiplier for low demand

    Returns:
        tuple: (product_features_df, optimal_prices_df)
            - product_features_df: [productId, demand_score, price]
            - optimal_prices_df: [productId, current_price, optimal_price, demand_score]
    """
    interaction_pipe = interaction_extraction_pipeline(context)
    price_pipe = price_extraction_pipeline(context)

    interactions_df = interaction_pipe.fit_transform(None)
    price_logs_df = price_pipe.fit_transform(None)
    product_features_df = product_features_pipeline(context, interactions_df, price_logs_df)
    print(product_features_df.to_string())

    # generate optimal prices using surge rules
    optimal_prices_df = pricing_pipeline(context, product_features_df,
                                          high_threshold=high_threshold,
                                          low_threshold=low_threshold,
                                          surge_multiplier=surge_multiplier,
                                          discount_multiplier=discount_multiplier)

    return product_features_df, optimal_prices_df


def ml_training_pipeline(context: PipelineContext) -> pd.DataFrame:
    """
    Build labeled session-level feature matrix for ML model training.
    Pipeline: fetch -> validate -> extract features -> join labels

    Returns:
        DataFrame with ~25 features per session + is_agent label
        Columns: sessionId, experimentId, temporal/behavioral/product/ua features, is_agent
    """
    # fetch raw interactions
    interactions_df = FetchInteractionsStep(context).transform(None)

    # validate data quality (report cached in context)
    interactions_df = ValidateDataStep(context).transform(interactions_df)
    if interactions_df.empty:
        return pd.DataFrame()

    # extract vectorized session features
    features_df = ExtractSessionFeaturesStep(context).transform(interactions_df)
    if features_df.empty:
        return pd.DataFrame()

    # join experiment labels (is_agent = ~xp_human_only)
    labeled_df = JoinLabelsStep(context).transform(features_df)

    return labeled_df




if __name__ == '__main__':

    class ExperimentsProvider(SupabaseProvider, BackendAPIProvider):
        def fetch_kafka_topic(self, topic: str) -> pd.DataFrame:
            path = "/home/velocitatem/Documents/Projects/PHANTOM/experiments/collected_data/"
            subdirs = os.listdir(path)
            full_df = pd.DataFrame()
            files = {"user-interactions": "int.json", "price-logs": "price.json"}
            for d in subdirs:
                path += d + "/"
                data = pd.read_json(path + files.get(topic, files["user-interactions"]))
                data = pd.DataFrame([r['payload'] for r in data['value'].to_list()])
                full_df = pd.concat([full_df, data], ignore_index=True)
            return full_df

    # demo: run ML training pipeline
    context = PipelineContext(provider=ExperimentsProvider(), store_mode='hotel')
    features = ml_training_pipeline(context)
    print(f"Feature matrix: {features.shape}")
    print(features.head())
    print(features.info())
