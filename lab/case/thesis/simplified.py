"""Minimal implementation of thesis pricing system.

Implements the core loop: prices -> sessions -> demand -> prices
with behavioral separability and robust pricing objective (Eq 23).

Objects:
- Session trajectories τ_s from mixture of H/A behavioral profiles
- Demand proxy q̂ via weighted action aggregation (Eq 2)
- COI leakage penalty for agent reconnaissance
- Limbo: alternating price/demand history for trajectory analysis
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

ACTION_WEIGHTS = {"add_to_cart": 0.8, "checkout": 0.9, "purchase": 1.0, "view": 0.15, "detail": 0.25, "hover": 0.3, "start": 0.05, "end": 0.0}
TRANS_H = {"start": {"view": 0.85, "end": 0.15}, "view": {"detail": 0.4, "cart": 0.3, "view": 0.2, "end": 0.1},
           "detail": {"cart": 0.5, "view": 0.3, "end": 0.2}, "cart": {"purchase": 0.6, "view": 0.25, "end": 0.15}, "purchase": {"end": 1.0}}
TRANS_A = {"start": {"view": 0.95, "end": 0.05}, "view": {"detail": 0.6, "view": 0.25, "cart": 0.1, "end": 0.05},
           "detail": {"view": 0.5, "cart": 0.15, "detail": 0.3, "end": 0.05}, "cart": {"view": 0.4, "purchase": 0.2, "end": 0.4}, "purchase": {"end": 1.0}}


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
    """Compute demand proxy q̂ = Σ_k ω(a_k) for session (Eq 2)."""
    return sum(ACTION_WEIGHTS.get(e.action, 0.1) for e in session.events)


def kl_div(p: Dict[str, float], q: Dict[str, float]) -> float:
    """KL divergence D_KL(p || q) for transition kernels."""
    eps = 1e-10
    keys = set(p.keys()) | set(q.keys())
    return sum(p.get(k, eps) * np.log((p.get(k, eps) + eps) / (q.get(k, eps) + eps)) for k in keys)


def build_kernel(events: List[Event]) -> Dict[str, Dict[str, float]]:
    """Build empirical transition kernel from trajectory."""
    trans: Dict[str, Dict[str, int]] = {}
    prev = "start"
    for e in events:
        curr = e.action
        trans.setdefault(prev, {})
        trans[prev][curr] = trans[prev].get(curr, 0) + 1
        prev = curr
    kernel = {}
    for s, dsts in trans.items():
        total = sum(dsts.values())
        kernel[s] = {d: c / total for d, c in dsts.items()} if total > 0 else {}
    return kernel


def compute_divergence(session: Session) -> Tuple[float, float]:
    """Compute Δ_H, Δ_A divergence signals (Eq 20-21)."""
    kernel = build_kernel(session.events)
    delta_h = sum(kl_div(kernel.get(s, {}), TRANS_H.get(s, {})) for s in kernel) / max(len(kernel), 1)
    delta_a = sum(kl_div(kernel.get(s, {}), TRANS_A.get(s, {})) for s in kernel) / max(len(kernel), 1)
    return delta_h, delta_a


def estimate_alpha(session: Session, beta: float = 2.0) -> float:
    """Per-session contamination estimate α̂(τ') = σ(β(Δ_H - Δ_A))."""
    dh, da = compute_divergence(session)
    return 1.0 / (1.0 + np.exp(-beta * (dh - da))) if (dh + da) > 0 else 0.5


def sample_trajectory(rng: np.random.Generator, trans: Dict, prices: np.ndarray, is_agent: bool) -> Tuple[List[Event], int]:
    """Sample session trajectory from behavioral kernel."""
    state, t, pidx = "start", 0.0, int(rng.integers(0, len(prices)))
    events = []
    while state != "end" and len(events) < 30:
        if state != "start":
            events.append(Event(action=state, product_idx=pidx, price_seen=float(prices[pidx]), ts=t))
        probs = trans.get(state, {"end": 1.0})
        state = rng.choice(list(probs.keys()), p=list(probs.values()))
        t += max(0.2, rng.gamma(1.5, 0.8) if is_agent else rng.gamma(2.0, 1.2))
    return events, pidx


def put_prices_to_market(prices: np.ndarray, alpha: float = 0.2, n_sessions: int = 50,
                         seed: int | None = None) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Generate sessions from mixture model Q(p) = (1-α)E[d_H] + αE[d_A] (Eq 3).

    Returns:
        demand_mapping: session_id -> demand proxy q̂
        hidden_labels: session_id -> actor class (H or A)
    """
    rng = np.random.default_rng(seed)
    demand_mapping, hidden_labels = {}, {}

    for i in range(n_sessions):
        sid = f"s{i:04d}"
        is_agent = rng.random() < alpha
        trans = TRANS_A if is_agent else TRANS_H
        theta = {"price_sens": rng.uniform(0.05, 0.2), "base_conv": 0.01} if is_agent else {"price_sens": rng.uniform(1.5, 4.0), "base_conv": rng.uniform(0.2, 0.5)}
        events, _ = sample_trajectory(rng, trans, prices, is_agent)
        session = Session(sid=sid, events=events, actor="A" if is_agent else "H", theta=theta)
        demand_mapping[sid] = compute_demand(session)
        hidden_labels[sid] = session.actor

    return demand_mapping, hidden_labels


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
        return self.on_update(utype)

    def on_update(self, utype: str) -> Dict:
        """React to update: after prices -> return observed demand; after demand -> signal price update needed."""
        if utype == "prices":
            return {"action": "observe_demand", "msg": "awaiting market response"}
        return {"action": "set_prices", "msg": "demand observed, update prices"}

    def get_prices_history(self) -> List[np.ndarray]:
        return [u.data for u in self.history if u.utype == "prices"]

    def get_demand_history(self) -> List[Dict[str, float]]:
        return [u.data for u in self.history if u.utype == "demand"]


class System:
    """Main pricing system implementing robust Stackelberg objective.

    Manages the alternating loop:
    1. Set prices p_t
    2. Observe demand response Q̂(p_t)
    3. Estimate contamination α from behavioral signals
    4. Compute next prices via robust objective (Eq 23)
    """

    def __init__(self, n_products: int = 10, costs: np.ndarray | None = None, lambda_coi: float = 0.5, seed: int | None = 42):
        self.n = n_products
        self.rng = np.random.default_rng(seed)
        self.costs = costs if costs is not None else self.rng.uniform(10, 50, n_products)
        self.refs = self.costs * (1 + self.rng.uniform(0.2, 0.5, n_products))  # base prices with margin
        self.lambda_coi = lambda_coi
        self.limbo = Limbo()
        self._alpha_est = 0.2  # current contamination estimate
        self._sessions: List[Session] = []

    @property
    def alpha(self) -> float:
        return self._alpha_est

    def _estimate_alpha_from_sessions(self) -> float:
        """Aggregate per-session α̂ estimates."""
        if not self._sessions:
            return self._alpha_est
        alphas = [estimate_alpha(s) for s in self._sessions[-50:]]  # use recent sessions
        return float(np.mean(alphas))

    def _revenue_under_demand(self, prices: np.ndarray, demand: Dict[str, float]) -> float:
        """Compute expected revenue R(p, d) from demand proxy."""
        agg_demand = np.zeros(self.n)
        for sid, q in demand.items():
            if self._sessions:
                sess = next((s for s in self._sessions if s.sid == sid), None)
                if sess and sess.events:
                    pidx = sess.events[0].product_idx
                    agg_demand[pidx] += q
        return float(np.dot(prices, agg_demand))

    def _coi_leakage(self, prices: np.ndarray) -> float:
        """COI_leak = α · InfoValue (query-tax surrogate)."""
        return self._alpha_est * 1.0

    def _objective(self, prices: np.ndarray, demand: Dict[str, float]) -> float:
        """Robust objective: R(p,d) - λ·COI_leak (Eq 23 simplified)."""
        revenue = self._revenue_under_demand(prices, demand)
        cost = float(np.sum(self.costs))  # fixed cost approximation
        profit = revenue - cost
        coi_penalty = self.lambda_coi * self._coi_leakage(prices) * float(np.mean(prices - self.costs))
        return profit - coi_penalty

    def compute_prices(self, demand: Dict[str, float] | None = None) -> np.ndarray:
        """Compute next prices via simple gradient-like update on robust objective.

        In a full implementation this would be replaced by DR-RL policy output.
        Here we use a heuristic: adjust margins based on α estimate.
        """
        self._alpha_est = self._estimate_alpha_from_sessions()

        # base margin adjustment: higher α -> lower margins (defensive pricing)
        margin_scale = 1.0 - 0.5 * self._alpha_est  # reduce margins under high contamination
        margins = (self.refs - self.costs) * margin_scale

        # add small noise for exploration
        noise = self.rng.normal(0, 0.02, self.n) * self.costs
        prices = np.clip(self.costs + margins + noise, self.costs * 1.02, self.refs * 1.3)

        self.limbo.add_update("prices", prices)
        return prices

    def observe_demand(self, prices: np.ndarray, alpha_true: float = 0.2, n_sessions: int = 50) -> Dict[str, float]:
        """Observe market response to prices."""
        demand_map, labels = put_prices_to_market(prices, alpha=alpha_true, n_sessions=n_sessions, seed=int(self.rng.integers(0, 10000)))

        # reconstruct sessions for α estimation
        for sid, actor in labels.items():
            events, _ = sample_trajectory(self.rng, TRANS_A if actor == "A" else TRANS_H, prices, actor == "A")
            self._sessions.append(Session(sid=sid, events=events, actor=actor))

        self.limbo.add_update("demand", demand_map)
        return demand_map

    def step(self, alpha_true: float = 0.2, n_sessions: int = 50) -> Tuple[np.ndarray, Dict[str, float], float]:
        """Single simulation step: prices -> demand -> reward."""
        demand_hist = self.limbo.get_demand_history()
        prices = self.compute_prices(demand_hist[-1] if demand_hist else None)
        demand = self.observe_demand(prices, alpha_true, n_sessions)
        reward = self._objective(prices, demand)
        return prices, demand, reward

    def run(self, n_steps: int = 100, alpha_true: float = 0.2) -> Dict:
        """Run simulation for n_steps, return trajectory."""
        trajectory = {"prices": [], "demand": [], "rewards": [], "alpha_est": [], "alpha_true": alpha_true}
        for _ in range(n_steps):
            p, d, r = self.step(alpha_true)
            trajectory["prices"].append(p)
            trajectory["demand"].append(d)
            trajectory["rewards"].append(r)
            trajectory["alpha_est"].append(self._alpha_est)
        return trajectory


def coi_erosion(n_agents: int, price_std: float) -> float:
    """COI erosion from Theorem 1: as N->inf, min(p_1..p_N)->p_min."""
    if n_agents <= 1:
        return 0.0
    log_n = np.log(n_agents)
    shift = price_std * (np.sqrt(2 * log_n) - (np.log(log_n) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * log_n) + 1e-6))
    return float(min(shift / (price_std * 2 + 1e-6), 1.0))


if __name__ == "__main__":
    # quick demo
    sys = System(n_products=5, seed=42)
    traj = sys.run(n_steps=20, alpha_true=0.25)
    print(f"avg reward: {np.mean(traj['rewards']):.2f}, final α̂: {traj['alpha_est'][-1]:.3f}")

    prices = np.array([20.0, 35.0, 50.0, 25.0, 40.0])
    demand, labels = put_prices_to_market(prices, alpha=0.3, n_sessions=20, seed=123)
    print(f'sessions: {len(demand)}, agents: {sum(1 for l in labels.values() if l=="A")}')

    for n in [1, 5, 10, 50, 100]:
        ero = coi_erosion(n, price_std=5.0)
        print(f'N={n:3d} agents -> COI erosion: {ero:.3f}')

    # test separability
    events = [Event('view', 0, 20.0, 0.1), Event('detail', 0, 20.0, 0.5), Event('cart', 0, 20.0, 1.0),
    Event('purchase', 0, 20.0, 2.0)]
    sess_h = Session(sid='test', events=events, actor='H')
    print(f'human-like session α̂: {estimate_alpha(sess_h):.3f}')

    events_a = [Event('view', 0, 20.0, 0.1), Event('detail', 0, 20.0, 0.2), Event('view', 0, 20.0, 0.3),
    Event('detail', 0, 20.0, 0.4)]
    sess_a = Session(sid='test2', events=events_a, actor='A')
    print(f'agent-like session α̂: {estimate_alpha(sess_a):.3f}')
