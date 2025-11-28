import pytest
import pandas as pd

from procesing.pricers import (
    StaticPricer,
    RandomPricer,
    ElasticityBasedPricer
)


def test_static_pricer_fit_and_predict():
    # Sample historical data
    historical_data = pd.DataFrame({
        'product_id': [1, 2, 3],
        'base_price': [100.0, 150.0, 200.0]
    })

    # Initialize and fit StaticPricer
    pricer = StaticPricer()
    pricer.fit(historical_data)

    # Predict prices
    predicted_prices = pricer.predict(None)

    # Assert that predicted prices match base prices
    expected_prices = historical_data['base_price'].values
    assert all(predicted_prices == expected_prices), "Predicted prices do not match base prices"


def test_random_pricer_fit_and_predict():
    # Sample historical data
    historical_data = pd.DataFrame({
        'product_id': [1, 2, 3],
        'base_price': [100.0, 150.0, 200.0]
    })

    # Initialize and fit RandomPricer
    pricer = RandomPricer(price_min=50.0, price_max=250.0, seed=42)
    pricer.fit(historical_data)

    # Predict prices
    predicted_prices = pricer.predict(None)

    # Assert that predicted prices are within bounds
    assert predicted_prices.min() >= 50.0, "Predicted prices are below minimum bound"
    assert predicted_prices.max() <= 250.0, "Predicted prices are above maximum bound"
    # distribution check (not so strict)
    assert len(set(predicted_prices)) > 1, "Predicted prices are not varied enough"
    assert len(predicted_prices) == len(historical_data), "Number of predicted prices does not match number of products"

def test_elasticity_based_pricer_fit_and_predict():
    # Sample historical data
    historical_data = pd.DataFrame({
        'productId': [1, 2, 3],
        'elasticity': [-1.5, -0.5, -2.0],
        'base_price': [100.0, 150.0, 200.0],
        'mean_demand': [10, 20, 15]
    })

    # Initialize and fit ElasticityBasedPricer
    pricer = ElasticityBasedPricer(alpha=0.1, price_floor=50.0, price_ceil=300.0)
    pricer.fit(historical_data)

    # Create a mock state space with demand deviations
    class MockStateSpace:
        def __init__(self, demand):
            self.demand = demand

    # Simulate demand higher than mean for all products
    state_space = MockStateSpace(demand=[15, 25, 20])

    # Predict prices
    predicted_prices = pricer.predict(state_space)

    # Assert that predicted prices are within bounds
    assert predicted_prices.min() >= 50.0, "Predicted prices are below minimum bound"
    assert predicted_prices.max() <= 300.0, "Predicted prices are above maximum bound"
    assert len(predicted_prices) == len(historical_data), "Number of predicted prices does not match number of products"

    # now we gotta check semantic validity
    # since demand is higher than mean, prices should generally increase
    for i, row in historical_data.iterrows():
        base_price = row['base_price']
        elasticity = row['elasticity']
        expected_increase = base_price * (1 + 0.1 * abs(elasticity) * ((state_space.demand[i] - row['mean_demand']) / row['mean_demand']))
        assert predicted_prices[i] >= base_price, f"Predicted price for product {row['productId']} did not increase as expected"
        assert abs(predicted_prices[i] - expected_increase) < 1e-5, f"Predicted price for product {row['productId']} does not match expected calculation within 1e-5 tolerance"
