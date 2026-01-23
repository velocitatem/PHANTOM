"""
Inventory/position management and instrument factories.

This module provides:
- PositionConfig: Configuration for position constraints and costs
- PositionModel: Manages inventory (retail) or position (finance)
- make_instruments: Factory for creating instrument sets

The PositionModel handles demand censorship by limiting executions
to available inventory, computing holding costs, and managing replenishment.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .types import Instrument, InstrumentSet, Execution
from .constants import Side, InstrumentType

@dataclass
class PositionConfig:
    """Configuration for position/inventory management.

    Attributes:
        initial_position: Starting inventory (None = unlimited, float = same for all)
        max_position: Maximum long position per instrument
        min_position: Maximum short position (negative, for finance)
        holding_cost_rate: Cost per unit per step for holding inventory
        shortage_cost_rate: Opportunity cost rate for stockouts
        lead_time: Steps until replenishment orders arrive
    """
    initial_position: np.ndarray | float | None = None
    max_position: float = 1000.0
    min_position: float = -1000.0
    holding_cost_rate: float = 0.001
    shortage_cost_rate: float = 0.05
    lead_time: int = 0

@dataclass
class PositionModel:
    """Manages inventory (retail) or position (finance) with censorship.

    Key responsibilities:
    - Track current position per instrument
    - Censor executions when position is insufficient
    - Compute holding costs per step
    - Track shortage/stockout costs
    - Handle replenishment orders with lead time

    For retail: position is inventory (positive), selling reduces it
    For finance: position can be positive (long) or negative (short)
    """
    cfg: PositionConfig
    n: int = 0
    _position: np.ndarray = field(default_factory=lambda: np.array([]))
    _pending_orders: list[tuple[int, np.ndarray]] = field(default_factory=list)
    _step_holding_cost: float = 0.0
    _step_shortage_cost: float = 0.0

    def reset(self, instruments: InstrumentSet, rng: np.random.Generator) -> None:
        self.n = instruments.n
        if self.cfg.initial_position is None:
            self._position = np.full(self.n, np.inf)  # unlimited
        elif isinstance(self.cfg.initial_position, (int, float)):
            self._position = np.full(self.n, float(self.cfg.initial_position))
        else:
            self._position = self.cfg.initial_position.copy().astype(np.float64)
        self._pending_orders = []
        self._step_holding_cost = 0.0
        self._step_shortage_cost = 0.0

    def available(self, instrument_id: int, side: Side) -> float:
        pos = self._position[instrument_id]
        if np.isinf(pos): return np.inf
        if side == Side.BUY:
            return max(0, pos)  # can sell up to current inventory
        else:
            return max(0, self.cfg.max_position - pos)  # can buy up to max

    def apply_execution(self, exe: Execution) -> Execution:
        idx = int(exe.instrument_id)
        avail = self.available(idx, exe.side)
        filled = min(exe.size_requested, avail)
        shortage = exe.size_requested - filled

        if exe.side == Side.BUY:
            self._position[idx] -= filled  # sold from inventory
        else:
            self._position[idx] += filled  # bought into inventory

        if shortage > 0:
            self._step_shortage_cost += shortage * exe.price * self.cfg.shortage_cost_rate

        return Execution(
            opportunity_id=exe.opportunity_id, instrument_id=exe.instrument_id,
            side=exe.side, size_requested=exe.size_requested,
            size_filled=filled, price=exe.price, propensity=exe.propensity, t=exe.t
        )

    def order(self, quantity: np.ndarray) -> None:
        if self.cfg.lead_time > 0:
            self._pending_orders.append((self.cfg.lead_time, quantity.copy()))
        else:
            self._position += quantity

    def step(self, t: float) -> None:
        # compute holding cost
        pos = np.where(np.isinf(self._position), 0, self._position)
        self._step_holding_cost = float(np.sum(np.abs(pos)) * self.cfg.holding_cost_rate)

        # receive pending orders
        new_pending = []
        for (remaining, qty) in self._pending_orders:
            if remaining <= 1:
                self._position += qty
            else:
                new_pending.append((remaining - 1, qty))
        self._pending_orders = new_pending

    @property
    def position(self) -> np.ndarray:
        return np.where(np.isinf(self._position), -1, self._position)

    @property
    def holding_cost(self) -> float:
        return self._step_holding_cost

    @property
    def shortage_cost(self) -> float:
        return self._step_shortage_cost

def make_instruments(n: int, cost_range: tuple[float, float] = (1.0, 10.0),
                     margin_range: tuple[float, float] = (0.2, 0.5),
                     inst_type: InstrumentType = InstrumentType.SKU,
                     rng: np.random.Generator | None = None) -> InstrumentSet:
    """Factory function to create a random instrument set.

    Args:
        n: Number of instruments to create
        cost_range: (min, max) for uniform cost sampling
        margin_range: (min, max) for uniform margin sampling
        inst_type: Type of instruments (SKU, ASSET, etc.)
        rng: Random generator (uses default if None)

    Returns:
        InstrumentSet with n instruments having random costs and margins
    """
    rng = rng or np.random.default_rng()
    costs = rng.uniform(*cost_range, n)
    margins = rng.uniform(*margin_range, n)
    items = [Instrument(id=i, type=inst_type, cost_basis=c, reference_price=c*(1+m))
             for i, (c, m) in enumerate(zip(costs, margins))]
    return InstrumentSet(instruments=items)
