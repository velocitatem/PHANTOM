import numpy as np

CATEGORY_WEIGHTS = {"cart": 4.0, "dwell": 2.0, "nav": 1.0, "filter": 0.5}
ACTION_CATEGORIES = {
    "cart": {"add_item", "add_to_cart", "remove", "checkout", "purchase"},
    "dwell": {"hover_title", "hover_paragraph", "hover_link"},
    "nav": {"page_view", "view_item", "view", "learn_more"},
    "filter": {"search", "filter_date", "filter_price", "sort"},
}
DEFAULT_ACTION_WEIGHTS = {
    a: CATEGORY_WEIGHTS[c] for c, actions in ACTION_CATEGORIES.items() for a in actions
}


def generate_demand_for_actor(
    prices: np.ndarray,
    params: tuple,
    noise_std: float = 1.0,
    distribution_method=np.random.normal,
) -> np.ndarray:
    """d(p;0) = max(0, valuation - price) + epsi for single actor type
    params: (mean, std) for valuation distribution D_H or D_A"""
    val = distribution_method(*params, size=len(prices))
    noise = distribution_method(0, noise_std, len(prices))
    demand = np.maximum(0, val - prices + noise)
    total = np.sum(demand)
    return demand / total * 100 if total > 0 else demand


def estimate_demand(trajectories, action_weights=None):
    return estimate_weighted_demand(trajectories, action_weights)


def _parse_event_state(state: str):
    if "_product" not in state:
        return state, None
    action, raw_pid = state.rsplit("_product", 1)
    return action, int(raw_pid) if raw_pid.isdigit() else None


def _weight_for_action(action: str, action_weights: dict) -> float:
    if action in action_weights:
        return action_weights[action]
    if action.startswith("hover"):
        return CATEGORY_WEIGHTS["dwell"]
    if action.startswith("filter") or action in {"search", "sort"}:
        return CATEGORY_WEIGHTS["filter"]
    if action.startswith("add") or action in {"checkout", "purchase", "remove"}:
        return CATEGORY_WEIGHTS["cart"]
    return CATEGORY_WEIGHTS["nav"]


def estimate_weighted_demand(trajectories, action_weights=None):
    action_weights = (
        DEFAULT_ACTION_WEIGHTS if action_weights is None else action_weights
    )
    scores = {}
    for traj in trajectories:
        for state in traj:
            action, product_id = _parse_event_state(state)
            if product_id is None:
                continue
            w = _weight_for_action(action, action_weights)
            if w <= 0:
                continue
            scores[product_id] = scores.get(product_id, 0.0) + w
    total = sum(scores.values())
    return (
        {pid: (score / total) * 100 for pid, score in scores.items()}
        if total > 0
        else {}
    )


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
