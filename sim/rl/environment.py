import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass

# here when we say "learner" we mean the agent that is learning to optimize the pricing and "agent" is part of the envrionment where the agent is creating demand that that "learner" is processing"

@dataclass
class BusinessLogicConstraints():
    max_price_adjustment : float = 0.3 # maximum adjustment of price
    system_max_price : float = 500.0 # maximum price allowed in the system
    product_catelogue_size : int = 100 # number of products in the catalogue


class PHANTOMEnv(gym.Env):
    def __init__(self):
        super(PHANTOMEnv, self).__init__()
        self.constraints = BusinessLogicConstraints()
        self.action_space = spaces.Box(
            low=-self.constraints.max_price_adjustment, high=self.constraints.max_price_adjustment,
            shape=(1,), dtype=np.float32) #  we allow teh learner to adjust price by some BusinessLogicConstraints factor
        # Example for using image as input:
        self.observation_space = spaces.Dict({
            'elasticity': spaces.Dict({
                'price': spaces.Box(low=0, high=self.constraints.system_max_price,
                                    shape=(self.constraints.product_catelogue_size,), dtype=np.float32),
                'demand': spaces.Box(low=0, high=np.inf,
                                     shape=(self.constraints.product_catelogue_size,), dtype=np.float32)
            })
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize state
        self.state = {
            'price': 100.0,  # base price
            'demand': 0.0
        }
        return self.state, {}

    def step(self, action):
        # Apply action
        price_adjustment = action[0]
        new_price = self.state['price'] * (1 + price_adjustment)
        self.state['price'] = new_price

        # Simulate demand based on new price
        demand = self.simulate_demand(new_price)
        self.state['demand'] = demand

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
