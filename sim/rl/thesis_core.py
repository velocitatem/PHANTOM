from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from sim.case.thesis_simplified.simplified import Session


@dataclass(frozen=True)
class PricingStep:
    sessions: list[Session]
    demand_by_session: Dict[str, float]
    demand_by_product: np.ndarray
    purchases_by_product: np.ndarray
    revenue: float
    cost: float
    n_agents: int


def clip_prices(prices: np.ndarray, min_price: float, max_price: float) -> np.ndarray:
    return np.clip(prices, min_price, max_price).astype(np.float32)


def constrain_prices(
    prev_prices: Optional[np.ndarray],
    proposed: np.ndarray,
    *,
    costs: np.ndarray,
    min_price: float,
    max_price: float,
    max_adjustment: float,
    min_margin_pct: float,
) -> np.ndarray:
    prices = clip_prices(proposed, min_price, max_price)
    floor = (costs * (1.0 + float(min_margin_pct))).astype(np.float32)
    prices = np.maximum(prices, floor)
    if prev_prices is None:
        return prices
    prev_prices = prev_prices.astype(np.float32)
    ratio = np.clip(prices / (prev_prices + 1e-6), 1.0 - max_adjustment, 1.0 + max_adjustment)
    return (prev_prices * ratio).astype(np.float32)


def aggregate_demand_by_product(
    sessions: list[Session],
    demand_by_session: Dict[str, float],
    n_products: int,
) -> np.ndarray:
    demand = np.zeros(n_products, dtype=np.float32)
    sessions_by_id = {s.sid: s for s in sessions}
    for sid, q in demand_by_session.items():
        sess = sessions_by_id.get(sid)
        if not sess or not sess.events:
            continue
        pidx = int(sess.events[0].product_idx)
        if 0 <= pidx < n_products:
            demand[pidx] += float(q)
    return demand


def aggregate_purchases(
    sessions: list[Session],
    costs: np.ndarray,
    n_products: int,
) -> tuple[np.ndarray, float, float, int]:
    purchases = np.zeros(n_products, dtype=np.float32)
    revenue = 0.0
    cost = 0.0
    n_agents = 0

    for sess in sessions:
        if sess.actor == "A":
            n_agents += 1
        for e in sess.events:
            if e.action != "purchase":
                continue
            pidx = int(e.product_idx)
            if 0 <= pidx < n_products:
                purchases[pidx] += 1.0
                revenue += float(e.price_seen)
                cost += float(costs[pidx])

    return purchases, revenue, cost, n_agents

