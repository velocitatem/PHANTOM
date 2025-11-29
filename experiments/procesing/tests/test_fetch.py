import pytest
import pandas as pd
from procesing.steps import (
    FetchInteractionsStep,
    FetchPriceLogsStep,
    FetchExperimentsStep,
)


def test_fetch_interactions_data(pipeline_context):
    step = FetchInteractionsStep(pipeline_context)
    data = step.transform(None)
    assert data is not None
    assert isinstance(data, pd.DataFrame)
    expected_cols = [
        "eventName",
        "dateIndex",
        "experimentId",
        "storeMode",
        "metadata_elementText"
    ]
    for expected in expected_cols:
        assert expected in data.columns

def test_fetch_price_logs(pipeline_context):
    step = FetchPriceLogsStep(pipeline_context)
    data = step.transform(None)
    assert data is not None
    assert isinstance(data, pd.DataFrame)
    expected_cols = [
        "price",
        "productId"
    ]
    for expected in expected_cols:
        assert expected in data.columns
    prices = data['price'].to_list()
    assert min(prices) >= 0
    assert max(prices) <= 9999


def test_experiments_fetching(pipeline_context):
    interactions = FetchInteractionsStep(pipeline_context).transform(None)
    assert interactions is not None
    experiments = FetchExperimentsStep(pipeline_context)
    experiment_data = experiments.transform(interactions)
    assert experiment_data is not None
    assert isinstance(experiment_data, pd.DataFrame)
    assert not experiment_data.empty
    assert 'id' in experiment_data.columns
    assert len(experiment_data) == 2
    assert '53aefd07-f66a-4d7f-ba8b-7ea1fc562d35' in experiment_data['id'].values
