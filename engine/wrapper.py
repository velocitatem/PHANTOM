import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from .engine import Limbo, MarketEngine, PricingEngine


class PHANTOM(gym.Env):
    """Gymnasium wrapper for the Limbo pricing-market simulation. Platform sets prices, market responds with demand."""
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self,
                 n_products: int = 10,
                 alpha: float = 0.3,
                 N: int = 100,
                 price_bounds: tuple = (10.0, 150.0),
                 lambda_coi: float = 0.1,  # coi leakage penalty weight
                 render_mode: str = None):
        super().__init__()
        self.n_products = n_products
        self.price_bounds = price_bounds
        self.lambda_coi = lambda_coi
        self.render_mode = render_mode

        self.market = MarketEngine(alpha=alpha, N=N)
        self._platform_stub = PricingEngine()
        self._limbo = Limbo(self._platform_stub, self.market)

        # action: continuous prices for each product
        self.action_space = spaces.Box(
            low=price_bounds[0], high=price_bounds[1],
            shape=(n_products,), dtype=np.float32
        )
        # observation: demand estimate + previous prices
        self.observation_space = spaces.Dict({
            "demand": spaces.Box(low=0.0, high=100.0, shape=(n_products,), dtype=np.float32),
            "prices": spaces.Box(low=price_bounds[0], high=price_bounds[1], shape=(n_products,), dtype=np.float32),
        })

        self._prices = None
        self._demand = None
        self._step_count = 0
        self._demand_history = []  # list of demand arrays over time
        self._price_history = []   # list of price arrays over time
        self._fig, self._axes = None, None

    def _get_obs(self) -> dict:
        demand_arr = np.array([self._demand.get(i, 0.0) for i in range(self.n_products)], dtype=np.float32)
        return {"demand": demand_arr, "prices": self._prices.astype(np.float32)}

    def _compute_reward(self, prices: np.ndarray, demand: dict) -> float:
        demand_arr = np.array([demand.get(i, 0.0) for i in range(self.n_products)])
        revenue = np.sum(prices * demand_arr)  # revenue = price * quantity proxy
        base_price = self.price_bounds[0]
        return float(revenue)# - self.lambda_coi * coi_leak)

    def _record_history(self):
        demand_arr = np.array([self._demand.get(i, 0.0) for i in range(self.n_products)])
        self._demand_history.append(demand_arr)
        self._price_history.append(self._prices.copy())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._prices = np.random.uniform(*self.price_bounds, size=self.n_products)
        self._demand = self.market.act(self._prices)
        self._step_count = 0
        self._demand_history, self._price_history = [], []
        self._record_history()
        if self._fig: plt.close(self._fig)
        self._fig, self._axes = None, None
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._prices = np.clip(action, *self.price_bounds)
        self._demand = self.market.act(self._prices)
        self._step_count += 1
        self._record_history()

        reward = self._compute_reward(self._prices, self._demand)
        terminated = self._step_count >= 100
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {"step": self._step_count}

    def render(self):
        if self.render_mode == "human":
            if self._fig is None:
                plt.ion()
                self._fig, self._axes = plt.subplots(2, 2, figsize=(12, 8))
                self._fig.suptitle("PHANTOM Environment")

            demand_mat = np.array(self._demand_history).T  # shape: (n_products, timesteps)
            price_mat = np.array(self._price_history).T
            revenue_per_step = np.sum(demand_mat * price_mat, axis=0)  # revenue = demand * price
            demand_variance = np.var(demand_mat, axis=0)  # how spread demand is across products

            for row in self._axes:
                for ax in row: ax.clear()

            self._axes[0, 0].imshow(demand_mat, aspect='auto', cmap='viridis', origin='lower')
            self._axes[0, 0].set_xlabel("Time Step")
            self._axes[0, 0].set_ylabel("Product")
            self._axes[0, 0].set_title("Demand Over Time")

            self._axes[0, 1].imshow(price_mat, aspect='auto', cmap='plasma', origin='lower')
            self._axes[0, 1].set_xlabel("Time Step")
            self._axes[0, 1].set_ylabel("Product")
            self._axes[0, 1].set_title("Price Over Time")

            self._axes[1, 0].plot(revenue_per_step, color='teal', linewidth=1.5)
            self._axes[1, 0].set_xlabel("Time Step")
            self._axes[1, 0].set_ylabel("Revenue")
            self._axes[1, 0].set_title("Revenue per Step")

            self._axes[1, 1].plot(demand_variance, color='orangered', linewidth=1.5)
            self._axes[1, 1].set_xlabel("Time Step")
            self._axes[1, 1].set_ylabel("Variance")
            self._axes[1, 1].set_title("Demand Variance")

            self._fig.tight_layout()
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(0.01)

        elif self.render_mode == "ansi":
            return f"step={self._step_count}, prices={self._prices}, demand={self._demand}"
        return None

    def close(self):
        if self._fig: plt.close(self._fig)
        self._fig, self._axes = None, None


if __name__ == "__main__":
    env = PHANTOM(n_products=100, render_mode="human")
    obs, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        env.render()
        print(f"Reward: {reward:.2f}")
        if term: break
