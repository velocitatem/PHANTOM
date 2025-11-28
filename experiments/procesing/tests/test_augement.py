import pytest
import random
import pandas as pd
from procesing.steps import (
    CreatePriceBucketsStep,
    AugmentEventNamesStep
)

def test_bucketing(pipeline_context):
    step = CreatePriceBucketsStep(context=pipeline_context)

    # Test with normal price data
    df = pd.DataFrame({
        'metadata_price': random.sample(range(10, 1000), 100)
    })
    result = step.transform(df)
    assert 'price_bucket' in result.columns
    # test if is categorical
    assert isinstance(result['price_bucket'].dtype, pd.CategoricalDtype)
    assert result['price_bucket'].nunique() == 3 # as per context config
    # distribution check
    counts = result['price_bucket'].value_counts()
    assert all(counts > 0)
    assert counts.max() - counts.min() <= 10  # roughly equal distribution for 100 samples
    # Test with empty DataFrame
    df = pd.DataFrame()
    result = step.transform(df)
    assert 'price_bucket' in result.columns
    assert result.empty


def test_augment_names(pipeline_context):
    df = pd.DataFrame({
        'eventName': ['click', 'view', 'purchase'],
        'productId': ['prod_1', 'prod_2', None],
        'price_bucket': ['PB_1', None, 'PB_3']
    })
    step = AugmentEventNamesStep(context=pipeline_context)
    result = step.transform(df)
    expected_event_names = [
        'click_prod_1@PB_1',
        'view',
        'purchase'
    ]
    assert result['eventName'].tolist() == expected_event_names
