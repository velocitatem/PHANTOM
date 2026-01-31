import logging
import numpy as np
from logging import getLogger
logger = getLogger(__name__)

def generate_demand_for_actor(prices: np.ndarray, params: tuple, noise_std: float = 1.0, distribution_method=np.random.normal) -> np.ndarray:
    """d(p;0) = max(0, valuation - price) + epsi for single actor type
    params: (mean, std) for valuation distribution D_H or D_A"""
    val = distribution_method(*params, size=len(prices))
    noise = distribution_method(0, noise_std, len(prices))
    demand = np.maximum(0, val - prices + noise)
    total = np.sum(demand)
    return demand / total * 100 if total > 0 else demand


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
    # demo actor-specific demands
    human_params, agent_params = (50, 10), (45, 15)
    demand_h = generate_demand_for_actor(prices, human_params)
    demand_a = generate_demand_for_actor(prices, agent_params)
    print("Human Demand:", demand_h)
    print("Agent Demand:", demand_a)
    from .behavior import sample_behavior
    N, alpha = 200, 0.3
    n_h, n_a = int(N * (1 - alpha)), int(N * alpha)
    human_t = [sample_behavior(demand_h, human=True) for _ in range(n_h)]
    agent_t = [sample_behavior(demand_a, human=False) for _ in range(n_a)]
    demand_estimate = estimate_demand(human_t + agent_t)
    print("Estimated Demand from Behavior:", demand_estimate)
