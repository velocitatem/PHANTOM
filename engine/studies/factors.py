"""shared factor definitions for experimental designs"""
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Any

@dataclass
class Factor:
    name: str
    levels: list
    primary: bool = True  # full cross vs sampled

# demand functions with compatible signatures
def demand_linear(mu, sigma, size): return np.maximum(0, np.random.normal(mu, sigma, size))
def demand_uniform(mu, sigma, size): return np.random.uniform(mu - sigma, mu + sigma, size)
def demand_exponential(mu, sigma, size): return np.random.exponential(mu, size)
def demand_logistic(mu, sigma, size): return np.random.logistic(mu, sigma, size)

DEMAND_FUNCTIONS = {
    "linear": demand_linear,
    "uniform": demand_uniform,
    "exponential": demand_exponential,
    "logistic": demand_logistic,
}

FACTORS = [
    Factor("demand_fn", list(DEMAND_FUNCTIONS.keys()), primary=True),
    Factor("alpha", [0.1, 0.3, 0.5, 0.7], primary=True),
    Factor("n_products", [5, 15, 30, 50], primary=True),
    Factor("demand_mu", [30.0, 50.0, 70.0], primary=False),
    Factor("demand_sigma", [5.0, 10.0, 20.0], primary=False),
    Factor("N", [100, 500, 1000], primary=False),
]

SEEDS_PER_CONFIG = 5
