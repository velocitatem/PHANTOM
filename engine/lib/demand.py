import logging
import numpy as np
from logging import getLogger
logger = getLogger(__name__)

def generate_demand(prices, distribution_method = np.random.normal, distribution_params = (50.0, 10.0)):
    # assumption 1: each product has an intrinsic valuation drawn from a normal distribution centered at 50
    product_valuations = distribution_method(*distribution_params, size=len(prices))
    # assumption 2: demand decreases as price increases, following a simple linear model
    demand = np.maximum(0, product_valuations - prices) # demand cannot be negative
    total = np.sum(demand)
    demand = demand / total * 100 if total > 0 else demand  # normalize to percentage, avoid div by zero
    logger.info(f"Generated demand for prices {prices}: {demand} with valuations from distribution {distribution_params}")
    return demand

def estimate_demand(trajectories):
    demand_estimate = {}
    for traj in trajectories:
        for event in traj:
            if 'view_product' in event:
                product_id = int(event.split('_')[-1].replace('product', ''))
                demand_estimate[product_id] = demand_estimate.get(product_id, 0) + 1
    total_views = sum(demand_estimate.values())
    for product_id in demand_estimate:
        demand_estimate[product_id] = (demand_estimate[product_id] / total_views) * 100  # normalize to percentage
    return demand_estimate

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    prices = np.array([20.0, 35.0, 50.0, 65.0])
    demand = generate_demand(prices)
    print("Generated Demand:", demand)
    from .behavior import sample_behavior
    N, alphat =200, 0.1
    trajectories = []
    for _ in range(int(N*(1 - alphat))):
        trajectories.append(sample_behavior(demand, human=True))
    for _ in range(int(N*alphat)):
        trajectories.append(sample_behavior(demand, human=False))
    demand_estimate = estimate_demand(trajectories)
    print("Estimated Demand from Behavior:", demand_estimate)
    delta = {k: demand_estimate.get(k, 0) - demand[i] for i, k in enumerate(range(len(prices)))}
    delta = np.mean([np.abs(v) for v in delta.values()])
    print("Demand Delta:", delta)
