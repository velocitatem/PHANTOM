import pytest
import pandas as pd
import numpy as np
from procesing.steps import (
    AggregatePriceLogsStep,
    ComputeElasticityStep
)


def test_aggregate_price_logs_basic(pipeline_context):
    """Test basic price aggregation into time windows"""
    step = AggregatePriceLogsStep(pipeline_context)

    # Create price logs with known window structure
    df = pd.DataFrame({
        'ts': pd.date_range(start='2023-01-01 10:00:00', periods=100, freq='10s'),
        'productId': np.tile([
            'd018efc1-25e9-4284-b276-80386e048b25',
            '51266ddb-5b07-47b7-89ee-5b5cae94bb11',
            '2cd7f756-fc65-4ba0-ab01-74521c1fff43'
        ], 34)[:100],
        'price': np.random.uniform(100, 200, 100)
    })

    result = step.transform(df)
    assert isinstance(result, list)
    assert len(result) > 0
    # each chunk should have window metadata and price vector
    for chunk in result:
        assert 'window_start' in chunk
        assert 'window_end' in chunk
        assert 'price_vector' in chunk
        assert isinstance(chunk['price_vector'], pd.DataFrame)
        assert 'productId' in chunk['price_vector'].columns
        assert 'price' in chunk['price_vector'].columns


def test_aggregate_price_logs_handles_gaps(pipeline_context):
    """Test that price aggregation forward-fills missing windows"""
    step = AggregatePriceLogsStep(pipeline_context)

    # create sparse data with gaps
    df = pd.DataFrame({
        'ts': pd.to_datetime([
            '2023-01-01 10:00:00',
            '2023-01-01 10:00:05',
            '2023-01-01 10:02:00',  # gap of ~2 mins
            '2023-01-01 10:02:30'
        ]),
        'productId': [
            'd018efc1-25e9-4284-b276-80386e048b25',
            'd018efc1-25e9-4284-b276-80386e048b25',
            '51266ddb-5b07-47b7-89ee-5b5cae94bb11',
            '51266ddb-5b07-47b7-89ee-5b5cae94bb11'
        ],
        'price': [100, 102, 150, 153]
    })

    result = step.transform(df)
    assert isinstance(result, list)
    # should have multiple windows despite gaps
    assert len(result) >= 2


def test_compute_elasticity_with_known_relationship(pipeline_context):
    """Test elasticity computation with known price-demand relationship"""
    step = ComputeElasticityStep(pipeline_context)

    # simulate elastic demand: when price ↑10%, demand ↓15% (elasticity ~ -1.5)
    base_price = 100
    base_demand = 50

    demand_chunks = [
        {
            'window_start': pd.Timestamp('2023-01-01 10:00:00'),
            'window_end': pd.Timestamp('2023-01-01 10:00:30'),
            'demand_vector': pd.DataFrame({
                'productId': ['d018efc1-25e9-4284-b276-80386e048b25'],
                'demand_score': [base_demand]
            })
        },
        {
            'window_start': pd.Timestamp('2023-01-01 10:00:30'),
            'window_end': pd.Timestamp('2023-01-01 10:01:00'),
            'demand_vector': pd.DataFrame({
                'productId': ['d018efc1-25e9-4284-b276-80386e048b25'],
                'demand_score': [base_demand * 0.85]  # 15% decrease
            })
        },
        {
            'window_start': pd.Timestamp('2023-01-01 10:01:00'),
            'window_end': pd.Timestamp('2023-01-01 10:01:30'),
            'demand_vector': pd.DataFrame({
                'productId': ['d018efc1-25e9-4284-b276-80386e048b25'],
                'demand_score': [base_demand * 0.70]  # further decrease
            })
        }
    ]

    price_chunks = [
        {
            'window_start': pd.Timestamp('2023-01-01 10:00:00'),
            'window_end': pd.Timestamp('2023-01-01 10:00:30'),
            'price_vector': pd.DataFrame({
                'productId': ['d018efc1-25e9-4284-b276-80386e048b25'],
                'price': [base_price]
            })
        },
        {
            'window_start': pd.Timestamp('2023-01-01 10:00:30'),
            'window_end': pd.Timestamp('2023-01-01 10:01:00'),
            'price_vector': pd.DataFrame({
                'productId': ['d018efc1-25e9-4284-b276-80386e048b25'],
                'price': [base_price * 1.10]  # 10% increase
            })
        },
        {
            'window_start': pd.Timestamp('2023-01-01 10:01:00'),
            'window_end': pd.Timestamp('2023-01-01 10:01:30'),
            'price_vector': pd.DataFrame({
                'productId': ['d018efc1-25e9-4284-b276-80386e048b25'],
                'price': [base_price * 1.20]  # 20% increase
            })
        }
    ]

    result = step.transform((demand_chunks, price_chunks))
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'productId' in result.columns
    assert 'elasticity' in result.columns
    assert 'n_obs' in result.columns

    # check elasticity is negative (normal good)
    product_elast = result[result['productId'] == 'd018efc1-25e9-4284-b276-80386e048b25']
    assert len(product_elast) == 1
    assert product_elast.iloc[0]['elasticity'] < 0
    # should be roughly elastic (< -1)
    assert product_elast.iloc[0]['n_obs'] == 3


def test_compute_elasticity_inelastic_product(pipeline_context):
    """Test with inelastic demand: price changes, demand barely moves"""
    step = ComputeElasticityStep(pipeline_context)

    base_price = 150
    base_demand = 40

    demand_chunks = [
        {
            'window_start': pd.Timestamp('2023-01-01 10:00:00'),
            'window_end': pd.Timestamp('2023-01-01 10:00:30'),
            'demand_vector': pd.DataFrame({
                'productId': ['51266ddb-5b07-47b7-89ee-5b5cae94bb11'],
                'demand_score': [base_demand]
            })
        },
        {
            'window_start': pd.Timestamp('2023-01-01 10:00:30'),
            'window_end': pd.Timestamp('2023-01-01 10:01:00'),
            'demand_vector': pd.DataFrame({
                'productId': ['51266ddb-5b07-47b7-89ee-5b5cae94bb11'],
                'demand_score': [base_demand * 0.98]  # tiny 2% decrease
            })
        }
    ]

    price_chunks = [
        {
            'window_start': pd.Timestamp('2023-01-01 10:00:00'),
            'window_end': pd.Timestamp('2023-01-01 10:00:30'),
            'price_vector': pd.DataFrame({
                'productId': ['51266ddb-5b07-47b7-89ee-5b5cae94bb11'],
                'price': [base_price]
            })
        },
        {
            'window_start': pd.Timestamp('2023-01-01 10:00:30'),
            'window_end': pd.Timestamp('2023-01-01 10:01:00'),
            'price_vector': pd.DataFrame({
                'productId': ['51266ddb-5b07-47b7-89ee-5b5cae94bb11'],
                'price': [base_price * 1.20]  # 20% increase
            })
        }
    ]

    result = step.transform((demand_chunks, price_chunks))
    product_elast = result[result['productId'] == '51266ddb-5b07-47b7-89ee-5b5cae94bb11']
    assert len(product_elast) == 1
    # inelastic: elasticity between 0 and -1
    assert -1 < product_elast.iloc[0]['elasticity'] < 0


def test_compute_elasticity_multiple_products(pipeline_context):
    """Test elasticity computation across multiple products simultaneously"""
    step = ComputeElasticityStep(pipeline_context)

    products = [
        'd018efc1-25e9-4284-b276-80386e048b25',
        '51266ddb-5b07-47b7-89ee-5b5cae94bb11',
        '2cd7f756-fc65-4ba0-ab01-74521c1fff43'
    ]

    # create 5 time windows with all 3 products
    demand_chunks = []
    price_chunks = []

    for i in range(5):
        ts = pd.Timestamp('2023-01-01 10:00:00') + pd.Timedelta(f'{i*30}s')

        demand_chunks.append({
            'window_start': ts,
            'window_end': ts + pd.Timedelta('30s'),
            'demand_vector': pd.DataFrame({
                'productId': products,
                'demand_score': [
                    50 * (0.9 ** i),  # elastic: decreases as price rises
                    40 * (0.98 ** i), # inelastic: barely changes
                    30 * (0.85 ** i)  # very elastic
                ]
            })
        })

        price_chunks.append({
            'window_start': ts,
            'window_end': ts + pd.Timedelta('30s'),
            'price_vector': pd.DataFrame({
                'productId': products,
                'price': [
                    100 * (1.05 ** i),
                    150 * (1.10 ** i),
                    120 * (1.08 ** i)
                ]
            })
        })

    result = step.transform((demand_chunks, price_chunks))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # all products should have elasticity
    assert set(result['productId']) == set(products)
    assert all(result['n_obs'] == 5)
    assert all(result['elasticity'] < 0)  # all normal goods


def test_compute_elasticity_insufficient_data(pipeline_context):
    """Test behavior with insufficient observations"""
    step = ComputeElasticityStep(pipeline_context)

    # only 1 observation
    demand_chunks = [{
        'window_start': pd.Timestamp('2023-01-01 10:00:00'),
        'window_end': pd.Timestamp('2023-01-01 10:00:30'),
        'demand_vector': pd.DataFrame({
            'productId': ['d018efc1-25e9-4284-b276-80386e048b25'],
            'demand_score': [50]
        })
    }]

    price_chunks = [{
        'window_start': pd.Timestamp('2023-01-01 10:00:00'),
        'window_end': pd.Timestamp('2023-01-01 10:00:30'),
        'price_vector': pd.DataFrame({
            'productId': ['d018efc1-25e9-4284-b276-80386e048b25'],
            'price': [100]
        })
    }]

    result = step.transform((demand_chunks, price_chunks))
    # should still return result but with low n_obs
    product_elast = result[result['productId'] == 'd018efc1-25e9-4284-b276-80386e048b25']
    assert len(product_elast) == 1
    assert product_elast.iloc[0]['n_obs'] == 1
    assert product_elast.iloc[0]['elasticity'] == 0.0  # not enough data


def test_compute_elasticity_misaligned_chunks(pipeline_context):
    """Test with non-overlapping demand and price windows"""
    step = ComputeElasticityStep(pipeline_context)

    demand_chunks = [{
        'window_start': pd.Timestamp('2023-01-01 10:00:00'),
        'window_end': pd.Timestamp('2023-01-01 10:00:30'),
        'demand_vector': pd.DataFrame({
            'productId': ['d018efc1-25e9-4284-b276-80386e048b25'],
            'demand_score': [50]
        })
    }]

    price_chunks = [{
        'window_start': pd.Timestamp('2023-01-01 11:00:00'),  # different time
        'window_end': pd.Timestamp('2023-01-01 11:00:30'),
        'price_vector': pd.DataFrame({
            'productId': ['d018efc1-25e9-4284-b276-80386e048b25'],
            'price': [100]
        })
    }]

    result = step.transform((demand_chunks, price_chunks))
    # should handle gracefully with no aligned data
    assert isinstance(result, pd.DataFrame)
    assert all(result['n_obs'] == 0)


def test_elasticity_arc_method(pipeline_context):
    """Test arc elasticity computation method"""
    # configure context for arc method
    pipeline_context.config['elasticity_method'] = 'arc'
    step = ComputeElasticityStep(pipeline_context)

    demand_chunks = [
        {
            'window_start': pd.Timestamp('2023-01-01 10:00:00'),
            'window_end': pd.Timestamp('2023-01-01 10:00:30'),
            'demand_vector': pd.DataFrame({
                'productId': ['d018efc1-25e9-4284-b276-80386e048b25'],
                'demand_score': [100]
            })
        },
        {
            'window_start': pd.Timestamp('2023-01-01 10:00:30'),
            'window_end': pd.Timestamp('2023-01-01 10:01:00'),
            'demand_vector': pd.DataFrame({
                'productId': ['d018efc1-25e9-4284-b276-80386e048b25'],
                'demand_score': [80]
            })
        }
    ]

    price_chunks = [
        {
            'window_start': pd.Timestamp('2023-01-01 10:00:00'),
            'window_end': pd.Timestamp('2023-01-01 10:00:30'),
            'price_vector': pd.DataFrame({
                'productId': ['d018efc1-25e9-4284-b276-80386e048b25'],
                'price': [100]
            })
        },
        {
            'window_start': pd.Timestamp('2023-01-01 10:00:30'),
            'window_end': pd.Timestamp('2023-01-01 10:01:00'),
            'price_vector': pd.DataFrame({
                'productId': ['d018efc1-25e9-4284-b276-80386e048b25'],
                'price': [110]
            })
        }
    ]

    result = step.transform((demand_chunks, price_chunks))
    product_elast = result[result['productId'] == 'd018efc1-25e9-4284-b276-80386e048b25']
    assert len(product_elast) == 1
    assert product_elast.iloc[0]['elasticity'] < 0
    # reset config
    pipeline_context.config['elasticity_method'] = 'point'
