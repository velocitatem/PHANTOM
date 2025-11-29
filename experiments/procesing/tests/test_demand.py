import pytest
import random
import pandas as pd
from procesing.steps import (
    ComputeDemandStep
)

def test_compute_demand(pipeline_context):
    step = ComputeDemandStep(context=pipeline_context)

    # Test with normal interaction data
    df = pd.DataFrame({
        'ts': pd.date_range(start='2023-01-01', periods=100, freq='h'),
        'productId': random.choices([
            'd018efc1-25e9-4284-b276-80386e048b25',
            '51266ddb-5b07-47b7-89ee-5b5cae94bb11',
            '2cd7f756-fc65-4ba0-ab01-74521c1fff43'
        ], k=100),
        'eventName': random.choices(['view', 'click', 'purchase'], k=100)
    })
    result = step.transform(df)
    assert type(result) == pd.DataFrame
    assert not result.empty
    assert set(result['productId']) == set(pipeline_context.products['id'])
    assert all(result['demand_score'] > 100/3 -10)


def test_compute_demand_skewed(pipeline_context):
    step = ComputeDemandStep(context=pipeline_context)

    # Test with normal interaction data
    df = pd.DataFrame({
        'ts': pd.date_range(start='2023-01-01', periods=100, freq='h'),
        'productId': random.choices([
            'd018efc1-25e9-4284-b276-80386e048b25',
            '51266ddb-5b07-47b7-89ee-5b5cae94bb11',
            '2cd7f756-fc65-4ba0-ab01-74521c1fff43'
        ], weights=[0.7, 0.2, 0.1], k=100),
        'eventName': random.choices(['view', 'click', 'purchase'], k=100)
    })
    result = step.transform(df)
    assert type(result) == pd.DataFrame
    assert not result.empty
    assert set(result['productId']) == set(pipeline_context.products['id'])
    # test for skewness
    scores = result.set_index('productId')['demand_score'].to_dict()
    assert scores['d018efc1-25e9-4284-b276-80386e048b25'] > \
           scores['51266ddb-5b07-47b7-89ee-5b5cae94bb11'] > \
           scores['2cd7f756-fc65-4ba0-ab01-74521c1fff43']
