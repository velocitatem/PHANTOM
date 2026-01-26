"""Minimal implementation of thesis pricing system.

Implements the core loop: prices -> sessions -> demand -> prices
with behavioral separability and robust pricing objective.

Objects:
- Session trajectories tau_s from mixture of H/A behavioral profiles
- Demand proxy q_hat via weighted action aggregation
- COI leakage penalty for agent reconnaissance
- Limbo: alternating price/demand history for trajectory analysis

COI Correction (Jan 2026):
The fundamental COI formulation is:
    COI = E[p_start] - p_transaction

This measures price erosion over time, not instantaneous margin × alpha.
Agents use multiple sessions to gather information and find minimum prices.
The price path from episode start to transaction captures information leakage.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

from .coi import COIWindow, compute_coi_window
from .separability import TRANS_H, TRANS_A, kl_div, build_kernel, compute_divergence, estimate_alpha

ACTION_WEIGHTS = {"add_to_cart": 0.8, "checkout": 0.9, "purchase": 1.0, "view": 0.15, "detail": 0.25, "hover": 0.3, "start": 0.05, "end": 0.0}


@dataclass
class Event:
    action: str
    product_idx: int
    price_seen: float
    ts: float


@dataclass
class Session:
    sid: str
    events: List[Event]
    actor: str  # H or A (ground truth label)
    theta: Dict[str, float] = field(default_factory=dict)


def compute_demand(session: Session) -> float:
    """Compute demand proxy q_hat = sum_k omega(a_k) for session."""
    return sum(ACTION_WEIGHTS.get(e.action, 0.1) for e in session.events)


def sample_trajectory(rng: np.random.Generator, trans: Dict, prices: np.ndarray, costs: np.ndarray, theta: Dict[str, float],
                      is_agent: bool, session_noise: float = 0.02, surge: float = 0.08, max_mult: float = 1.8) -> Tuple[List[Event], int]:
    """Sample session trajectory from behavioral kernel."""
    pidx = int(rng.integers(0, len(prices)))
    cost, base = float(costs[pidx]), float(prices[pidx]) * (1.0 + rng.normal(0.0, session_noise))
    base = float(np.clip(base, cost * 1.01, float(prices[pidx]) * 2.0))
    price, signal, state, t = base, 0.0, "start", 0.0
    events = []

    while state != "end" and len(events) < 30:
        probs = trans.get(state, {"end": 1.0})
        nxt = rng.choice(list(probs.keys()), p=list(probs.values()))
        if nxt == "purchase":  # purchase conversion check
            rel = max((price - cost) / (cost + 1e-6), 0.0)
            p_buy = float(np.clip(theta.get("base_conv", 0.2) * np.exp(-theta.get("price_sens", 2.0) * rel), 0.0, 1.0))
            if rng.random() > p_buy:
                nxt = "end"
        state = nxt
        if state not in {"start", "end"}:
            events.append(Event(action=state, product_idx=pidx, price_seen=float(price), ts=t))
            signal += float(ACTION_WEIGHTS.get(state, 0.1))
            price = float(np.clip(base * (1.0 + surge * signal), cost * 1.01, base * max_mult))
        t += max(0.2, rng.gamma(1.5, 0.8) if is_agent else rng.gamma(2.0, 1.2))
    return events, pidx


def put_prices_to_market(prices: np.ndarray, costs: np.ndarray, alpha: float = 0.2, n_sessions: int = 50,
                         seed: int | None = None) -> Tuple[List[Session], Dict[str, float]]:
    """Generate sessions from mixture model. Returns sessions and demand mapping sid -> q_hat."""
    rng = np.random.default_rng(seed)
    sessions, demand = [], {}
    for i in range(n_sessions):
        sid = f"s{i:04d}"
        is_agent = rng.random() < alpha
        trans = TRANS_A if is_agent else TRANS_H
        theta = {"price_sens": rng.uniform(0.05, 0.2), "base_conv": 0.01} if is_agent else \
                {"price_sens": rng.uniform(1.5, 4.0), "base_conv": rng.uniform(0.2, 0.5)}
        events, _ = sample_trajectory(rng, trans, prices, costs=costs, theta=theta, is_agent=is_agent)
        session = Session(sid=sid, events=events, actor="A" if is_agent else "H", theta=theta)
        sessions.append(session)
        demand[sid] = compute_demand(session)
    return sessions, demand


@dataclass
class LimboUpdate:
    utype: str  # "prices" or "demand"
    data: np.ndarray | Dict[str, float]
    t: int


class Limbo:
    """Historical trajectory of alternating price/demand observations."""

    def __init__(self):
        self.history: List[LimboUpdate] = []
        self._t = 0

    def add_update(self, utype: str, data: np.ndarray | Dict[str, float]) -> Dict:
        self.history.append(LimboUpdate(utype=utype, data=data, t=self._t))
        self._t += 1
        return {"action": "observe_demand" if utype == "prices" else "set_prices"}

    def get_prices_history(self) -> List[np.ndarray]:
        return [u.data for u in self.history if u.utype == "prices"]

    def get_demand_history(self) -> List[Dict[str, float]]:
        return [u.data for u in self.history if u.utype == "demand"]


class System:
    """Main pricing system implementing robust Stackelberg objective.

    Manages the alternating loop: set prices p_t -> observe demand Q_hat(p_t) ->
    estimate contamination alpha from behavioral signals -> compute next prices.
    """

    def __init__(self, n_products: int = 10, costs: np.ndarray | None = None, lambda_coi: float = 0.5, seed: int | None = 42):
        self.n = n_products
        self.rng = np.random.default_rng(seed)
        self.costs = costs if costs is not None else self.rng.uniform(10, 50, n_products)
        self.refs = self.costs * (1 + self.rng.uniform(0.2, 0.5, n_products))
        self.lambda_coi = lambda_coi
        self.limbo = Limbo()
        self._alpha_est = 0.2
        self._sessions: List[Session] = []
        self._last_sessions: List[Session] = []
        self._last_coi: COIWindow | None = None

    @property
    def alpha(self) -> float:
        return self._alpha_est

    def _estimate_alpha_from_sessions(self) -> float:
        if not self._sessions:
            return self._alpha_est
        return float(np.mean([estimate_alpha(s) for s in self._sessions[-50:]]))

    def _revenue_under_demand(self, prices: np.ndarray, demand: Dict[str, float]) -> float:
        agg = np.zeros(self.n)
        for sid, q in demand.items():
            sess = next((s for s in self._sessions if s.sid == sid), None)
            if sess and sess.events:
                agg[sess.events[0].product_idx] += q
        return float(np.dot(prices, agg))

    def _compute_coi_window(self, demand: Dict[str, float]) -> COIWindow:
        if not self._last_sessions:
            zeros = np.zeros(self.n, dtype=float)
            return COIWindow(policy=0.0, agent=0.0, leak=0.0, survival_ratio=0.0,
                             policy_by_product=zeros, agent_by_product=zeros, demand_weights=zeros)
        return compute_coi_window(self._last_sessions, self.costs, demand_mapping=demand)

    def _objective(self, prices: np.ndarray, demand: Dict[str, float]) -> float:
        """Robust objective: R(p,d) - lambda * COI_leak."""
        profit = self._revenue_under_demand(prices, demand) - float(np.sum(self.costs))
        self._last_coi = self._compute_coi_window(demand)
        return profit - self.lambda_coi * self._last_coi.leak

    def compute_prices(self, demand: Dict[str, float] | None = None) -> np.ndarray:
        """Compute next prices via heuristic margin adjustment based on alpha estimate."""
        self._alpha_est = self._estimate_alpha_from_sessions()
        margin_scale = 1.0 - 0.5 * self._alpha_est  # defensive pricing under high contamination
        margins = (self.refs - self.costs) * margin_scale
        noise = self.rng.normal(0, 0.02, self.n) * self.costs
        prices = np.clip(self.costs + margins + noise, self.costs * 1.02, self.refs * 1.3)
        self.limbo.add_update("prices", prices)
        return prices

    def observe_demand(self, prices: np.ndarray, alpha_true: float = 0.2, n_sessions: int = 50) -> Dict[str, float]:
        sessions, demand_map = put_prices_to_market(prices, costs=self.costs, alpha=alpha_true,
                                                    n_sessions=n_sessions, seed=int(self.rng.integers(0, 10000)))
        self._last_sessions = sessions
        self._sessions.extend(sessions)
        self.limbo.add_update("demand", demand_map)
        return demand_map

    def step(self, alpha_true: float = 0.2, n_sessions: int = 50) -> Tuple[np.ndarray, Dict[str, float], float, COIWindow]:
        demand_hist = self.limbo.get_demand_history()
        prices = self.compute_prices(demand_hist[-1] if demand_hist else None)
        demand = self.observe_demand(prices, alpha_true, n_sessions)
        reward = self._objective(prices, demand)
        return prices, demand, reward, self._last_coi or self._compute_coi_window(demand)

    def run(self, n_steps: int = 100, alpha_true: float = 0.2) -> Dict:
        traj = {"prices": [], "demand": [], "rewards": [], "alpha_est": [], "alpha_true": alpha_true,
                "coi_policy": [], "coi_agent": [], "coi_leak": [], "coi_survival": []}
        for _ in range(n_steps):
            p, d, r, coi = self.step(alpha_true)
            traj["prices"].append(p); traj["demand"].append(d); traj["rewards"].append(r)
            traj["alpha_est"].append(self._alpha_est)
            traj["coi_policy"].append(coi.policy); traj["coi_agent"].append(coi.agent)
            traj["coi_leak"].append(coi.leak); traj["coi_survival"].append(coi.survival_ratio)
        return traj


if __name__ == "__main__":
    sys = System(n_products=5, seed=42)
    traj = sys.run(n_steps=20, alpha_true=0.25)
    print(f"avg reward: {np.mean(traj['rewards']):.2f}, final alpha_hat: {traj['alpha_est'][-1]:.3f}, "
          f"COI_policy: {np.mean(traj['coi_policy']):.3f}, COI_agent: {np.mean(traj['coi_agent']):.3f}, leak: {np.mean(traj['coi_leak']):.3f}")

    prices = np.array([20.0, 35.0, 50.0, 25.0, 40.0])
    costs = np.array([15.0, 28.0, 40.0, 18.0, 30.0])
    sessions, demand = put_prices_to_market(prices, costs=costs, alpha=0.3, n_sessions=20, seed=123)
    print(f'sessions: {len(sessions)}, agents: {sum(1 for s in sessions if s.actor=="A")}')

    for n in [1, 5, 10, 50, 100]:
        # theoretical: erosion = 1 - 2/(N+1) for uniform order statistic
        print(f'N={n:3d} agents -> COI erosion: {1.0 - 2.0/(n+1):.3f}')

    events = [Event('view', 0, 20.0, 0.1), Event('detail', 0, 20.0, 0.5), Event('cart', 0, 20.0, 1.0), Event('purchase', 0, 20.0, 2.0)]
    print(f'human-like session alpha_hat: {estimate_alpha(Session(sid="test", events=events, actor="H")):.3f}')

    events_a = [Event('view', 0, 20.0, 0.1), Event('detail', 0, 20.0, 0.2), Event('view', 0, 20.0, 0.3), Event('detail', 0, 20.0, 0.4)]
    print(f'agent-like session alpha_hat: {estimate_alpha(Session(sid="test2", events=events_a, actor="A")):.3f}')
