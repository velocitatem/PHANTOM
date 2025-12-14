import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
import pandas as pd

# here when we say "learner" we mean the agent that is learning to optimize the pricing and "agent" is part of the envrionment where the agent is creating demand that that "learner" is processing"

@dataclass
class BusinessLogicConstraints():
    max_price_adjustment : float = 0.3 # maximum adjustment of price
    system_max_price : float = 500.0 # maximum price allowed in the system
    system_min_price : float = 1.0 # minimum price allowed in the system
    product_catelogue_size : int = 100 # number of products in the catalogue


class CommercePlatform:
    def __init__(self, product_catelogue_size: int, max_price: float, min_price: float):
        self.product_catelogue_size = product_catelogue_size
        self.max_price = max_price
        self.min_price = min_price
        self.simulation_history = []


    def setup_true_demand(self,prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        human_price_elasticity = -1.5  # Example elasticity value
        base_demand = 100  # Base demand for products
        demand = base_demand * (prices / self.max_price) ** human_price_elasticity

        agent_price_elasticity = -2.0  # Example elasticity value for agents
        agent_base_demand = 150  # Base demand for agents
        agent_demand = agent_base_demand * (prices / self.max_price) ** agent_price_elasticity

        return demand + agent_demand, agent_demand


    def compute_interaction_features(self, interaction_data: np.ndarray) -> dict:
        df = pd.DataFrame(interaction_data)
        return {
            'mean_sale_price': df[df['action'] == 'purchase']['price'].mean(),
        }

    def run_pricing_simulation(self, prices: np.ndarray) -> np.ndarray:
        # Simulate demand based on prices

        observed_demand, demand_from_agents = self.setup_true_demand(prices)
        true_demand = observed_demand - demand_from_agents

        interaction_data = self.get_interaction_data()
        interaction_features = self.compute_interaction_features(interaction_data)
        demand_estimates = self.demand_estimate(interaction_data)
        internal_error = np.abs(true_demand - demand_estimates) / (true_demand + 1e-6)

        self.simulation_history.append(
            {
                'prices': prices,
                'true_demand': true_demand,
                'demand_estimates': demand_estimates,
                'internal_error': internal_error,
                'interaction_data': interaction_data,
                'interaction_features': interaction_features
            })
        return np.array(interaction_data)

    def get_interaction_data(self) -> np.ndarray:
        # Simulate interaction data
        interaction_data = []
        return np.array(interaction_data)


    def demand_estimate(self, interactions : np.ndarray) -> np.ndarray:
        demand_estimates = np.random.rand(self.product_catelogue_size) * 100  # Dummy demand estimates
        return demand_estimates









class PHANTOMEnv(gym.Env):
    def __init__(self):
        super(PHANTOMEnv, self).__init__()
        self.constraints = BusinessLogicConstraints()
        self.action_space = spaces.Box(
            low=-self.constraints.max_price_adjustment, high=self.constraints.max_price_adjustment,
            shape=(self.constraints.product_catelogue_size,), dtype=np.float32) #  we allow teh learner to adjust price by some BusinessLogicConstraints factor
        # Example for using image as input:
        self.commerce_platform = CommercePlatform(
            product_catelogue_size=self.constraints.product_catelogue_size,
            max_price=self.constraints.system_max_price,
            min_price=self.constraints.system_min_price
        )
        self.observation_space = spaces.Dict({
            'elasticity': spaces.Dict({
                'price': spaces.Box(low=0, high=self.constraints.system_max_price,
                                    shape=(self.constraints.product_catelogue_size,), dtype=np.float32),
                'demand': spaces.Box(low=0, high=np.inf,
                                     shape=(self.constraints.product_catelogue_size,), dtype=np.float32)
            })
        })

    def reset(self, seed :int, options) -> tuple[dict, dict]:
        super().reset(seed=seed)
        # Initialize state
        self.state = {
            'elasticity': {
                'price': np.full((self.constraints.product_catelogue_size,), 100.0, dtype=np.float32),
                'demand': np.full((self.constraints.product_catelogue_size,), 50.0, dtype=np.float32)
            }
        }
        return self.state, {}

    def step(self, action):
        self.state['price'] = np.clip(self.state['price'] * (1 + action),
                            self.constraints.system_min_price,
                            self.constraints.system_max_price)



        # Calculate reward (e.g., revenue)
        reward = new_price * demand

        # Check if episode is done
        done = self.state['price'] <= 0.0 or self.state['demand'] <= 0.0


        return self.state, reward, done, False, {}
    def simulate_demand(self, price):
        # Simple linear demand model: demand decreases as price increases
        base_demand = 200
        price_sensitivity = 0.5
        demand = max(0, base_demand - price_sensitivity * price)
        return demand

if __name__ == "__main__":
    env = PHANTOMEnv()
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        print(f"Price: {obs['price']:.2f}, Demand: {obs['demand']:.2f}, Reward: {reward:.2f}")
        if done:
            break

    print(f"Total Reward: {total_reward:.2f}")
