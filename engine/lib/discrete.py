from collections import defaultdict
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DiscretePriceActionWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env: gym.Env,
        n_levels: int = 9,
        min_scale: float = 0.8,
        max_scale: float = 1.2,
    ):
        super().__init__(env)
        self.scales = np.linspace(min_scale, max_scale, n_levels, dtype=np.float32)
        self.action_space = spaces.Discrete(n_levels)

    def action(self, action: int):
        scale = float(self.scales[int(action)])
        cur = np.asarray(self.env.unwrapped._prices, dtype=np.float32)
        lo, hi = self.env.unwrapped.price_bounds
        return np.clip(cur * scale, lo, hi).astype(np.float32)


class EventQTable:
    def __init__(
        self,
        n_actions: int,
        n_products: int,
        price_bounds: tuple,
        lr: float = 0.1,
        gamma: float = 0.99,
        n_bins: int = 6,
    ):
        self.n_actions = int(n_actions)
        self.n_products = int(n_products)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.q = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))
        lo, hi = price_bounds
        self.demand_bins = np.linspace(0.0, 100.0, n_bins + 1)[1:-1]
        self.price_bins = np.linspace(lo, hi, n_bins + 1)[1:-1]

    def encode(self, obs: np.ndarray) -> tuple:
        obs = np.asarray(obs, dtype=np.float32)
        d = obs[: self.n_products]
        p = obs[self.n_products : 2 * self.n_products]
        d_mean = float(np.mean(d)) if d.size else 0.0
        d_std = float(np.std(d)) if d.size else 0.0
        p_mean = float(np.mean(p)) if p.size else 0.0
        return (
            int(np.digitize(d_mean, self.demand_bins)),
            int(np.digitize(d_std, self.demand_bins)),
            int(np.digitize(p_mean, self.price_bins)),
        )

    def act(self, obs: np.ndarray, eps: float = 0.0) -> tuple[int, tuple]:
        s = self.encode(obs)
        if np.random.random() < eps:
            return int(np.random.randint(self.n_actions)), s
        return int(np.argmax(self.q[s])), s

    def update(self, s: tuple, a: int, r: float, s2: tuple, done: bool):
        target = r + (0.0 if done else self.gamma * float(np.max(self.q[s2])))
        self.q[s][a] += self.lr * (target - self.q[s][a])

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        a, _ = self.act(obs, 0.0 if deterministic else 0.05)
        return a, None
