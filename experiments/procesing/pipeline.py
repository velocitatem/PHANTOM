from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from extract import DataExtractor
from mapping import SessionTransitionProbMatrixTransformer, render_graph
from demand import DemandEstimator


# exposable pipelines
etl_pipeline = Pipeline([
    ('data_extraction', DataExtractor()),
])
pricing_pipeline = Pipeline([
    ('demand_estimation', DemandEstimator()),
    ('scaling', StandardScaler()),
])

if __name__ == "__main__":
    processed_data = etl_pipeline.fit_transform(None)
    pricing = pricing_pipeline.fit_transform(processed_data)
