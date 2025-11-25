from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from extract import KafkaDataFetcher, ExperimentJoiner, EventTitleAugmenter
from mapping import SessionTransitionProbMatrixTransformer, render_graph
from demand import DemandEstimator


# exposable pipelines
etl_pipeline = Pipeline([
    ('kafka_fetch', KafkaDataFetcher()),
    ('experiment_join', ExperimentJoiner()),
    ('event_augment', EventTitleAugmenter()),
])
pricing_pipeline = Pipeline([
    ('demand_estimation', DemandEstimator()),
])

if __name__ == "__main__":
    processed_data = etl_pipeline.fit_transform(None)
    pricing = pricing_pipeline.fit_transform(processed_data)
    print(pricing)
