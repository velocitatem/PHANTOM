"""
Numerical utilities for stable computation.

This module provides numerically stable implementations of common operations:
- safe_exp, safe_log: Avoid overflow/underflow
- softmax: Numerically stable softmax
- sigmoid, clamp: Standard transformations
- intensity_decay: Avellaneda-Stoikov fill intensity
- inventory_penalty: Quadratic inventory risk
- poisson_arrivals, hawkes_intensity: Arrival process helpers

All functions accept both scalars and numpy arrays.
"""
import numpy as np

EPS = 1e-8  # small constant to avoid division by zero
MAX_EXP = 700.0  # maximum safe exponent to avoid overflow

def safe_exp(x: np.ndarray | float) -> np.ndarray | float:
    return np.exp(np.clip(x, -MAX_EXP, MAX_EXP))

def safe_log(x: np.ndarray | float) -> np.ndarray | float:
    return np.log(np.maximum(x, EPS))

def clamp(x: np.ndarray | float, lo: float, hi: float) -> np.ndarray | float:
    return np.clip(x, lo, hi)

def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + safe_exp(-x))

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = safe_exp(x - x_max)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + EPS)

def geometric_series(base: float, ratio: float, n: int) -> np.ndarray:
    return base * (ratio ** np.arange(n))

def ema(old: float, new: float, alpha: float = 0.1) -> float:
    return alpha * new + (1 - alpha) * old

def intensity_decay(distance: float, kappa: float = 1.0) -> float:
    """Avellaneda-Stoikov style fill intensity decay with quote distance"""
    return safe_exp(-kappa * distance)

def inventory_penalty(q: float, gamma: float = 0.1, sigma: float = 1.0) -> float:
    """Quadratic inventory risk penalty"""
    return gamma * sigma**2 * q**2 / 2

def poisson_arrivals(rate: float, dt: float, rng: np.random.Generator) -> int:
    return rng.poisson(rate * dt)

def hawkes_intensity(base: float, history: np.ndarray, alpha: float, beta: float, t: float) -> float:
    """Self-exciting Hawkes process intensity"""
    if len(history) == 0: return base
    decays = safe_exp(-beta * (t - history[history < t]))
    return base + alpha * np.sum(decays)
