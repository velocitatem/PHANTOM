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


@dataclass(frozen=True)
class COIWindow:
    """Windowed COI metrics computed from realized price exposures.

    COI_policy is the definition-level KPI: E[p_shown] - p_min.
    COI_agent is the theorem-level object: E[p^(1)] - p_min, where p^(1) is the minimum price realized under agent querying.
    In this simplified simulator, p^(1) is approximated as the minimum price exposed to any agent in the window (per product).
    Leak is the observable gap between them.
    """

    policy: float
    agent: float
    leak: float
    survival_ratio: float
    policy_by_product: np.ndarray
    agent_by_product: np.ndarray
    demand_weights: np.ndarray


def _prices_by_product(sessions: List[Session]) -> Dict[int, List[float]]:
    prices: Dict[int, List[float]] = {}
    for s in sessions:
        for e in s.events:
            prices.setdefault(e.product_idx, []).append(float(e.price_seen))
    return prices


def _min_session_prices_by_product(sessions: List[Session]) -> Dict[int, List[float]]:
    mins: Dict[int, List[float]] = {}
    for s in sessions:
        by_p: Dict[int, float] = {}
        for e in s.events:
            pidx = int(e.product_idx)
            price = float(e.price_seen)
            by_p[pidx] = price if pidx not in by_p else min(by_p[pidx], price)
        for pidx, pmin in by_p.items():
            mins.setdefault(pidx, []).append(pmin)
    return mins


def _min_price_across_sessions_by_product(sessions: List[Session]) -> Dict[int, float]:
    mins: Dict[int, float] = {}
    for s in sessions:
        for e in s.events:
            pidx = int(e.product_idx)
            price = float(e.price_seen)
            mins[pidx] = price if pidx not in mins else min(mins[pidx], price)
    return mins


def _demand_weights_by_product(
    sessions: List[Session],
    demand_mapping: Dict[str, float],
    n_products: int,
) -> np.ndarray:
    w = np.zeros(n_products, dtype=float)
    sessions_by_id = {s.sid: s for s in sessions}
    for sid, q in demand_mapping.items():
        sess = sessions_by_id.get(sid)
        if not sess or not sess.events:
            continue
        pidx = int(sess.events[0].product_idx)
        w[pidx] += float(q)
    s = float(np.sum(w))
    return (w / s) if s > 0 else w


def compute_coi_window(
    sessions: List[Session],
    costs: np.ndarray,
    demand_mapping: Dict[str, float] | None = None,
) -> COIWindow:
    n_products = int(len(costs))
    prices = _prices_by_product(sessions)
    agent_min_across = _min_price_across_sessions_by_product([s for s in sessions if s.actor == "A"])

    policy_by_product = np.zeros(n_products, dtype=float)
    agent_by_product = np.zeros(n_products, dtype=float)
    seen = np.array([(i in prices) for i in range(n_products)], dtype=bool)
    agent_seen = np.array([(i in agent_min_across) for i in range(n_products)], dtype=bool)

    for pidx, ps in prices.items():
        if 0 <= pidx < n_products and ps:
            policy_by_product[pidx] = float(np.mean(ps) - float(costs[pidx]))

    for pidx, pmin in agent_min_across.items():
        if 0 <= pidx < n_products:
            agent_by_product[pidx] = float(pmin - float(costs[pidx]))

    # If no agent exposure exists for a product in the window, there is no realized erosion for that product.
    agent_by_product[seen & ~agent_seen] = policy_by_product[seen & ~agent_seen]

    demand_weights = (
        _demand_weights_by_product(sessions, demand_mapping, n_products)
        if demand_mapping is not None
        else np.zeros(n_products, dtype=float)
    )

    has_weights = float(np.sum(demand_weights)) > 0
    if has_weights:
        policy = float(np.dot(demand_weights, policy_by_product))
        agent = float(np.dot(demand_weights, agent_by_product))
    else:
        if not bool(np.any(seen)):
            policy = 0.0
            agent = 0.0
        else:
            policy = float(np.mean(policy_by_product[seen]))
            agent = float(np.mean(agent_by_product[seen]))

    leak = float(max(policy - agent, 0.0))
    survival_ratio = float(np.clip(agent / policy, 0.0, 1.0)) if policy > 0 else 0.0

    return COIWindow(
        policy=policy,
        agent=agent,
        leak=leak,
        survival_ratio=survival_ratio,
        policy_by_product=policy_by_product,
        agent_by_product=agent_by_product,
        demand_weights=demand_weights,
    )


def sample_trajectory(
    rng: np.random.Generator,
    trans: Dict,
    prices: np.ndarray,
    costs: np.ndarray,
    theta: Dict[str, float],
    is_agent: bool,
    session_price_noise: float = 0.02,
    surge: float = 0.08,
    max_markup_mult: float = 1.8,
) -> Tuple[List[Event], int]:
    """Sample session trajectory from behavioral kernel."""
    state, t, pidx = "start", 0.0, int(rng.integers(0, len(prices)))
    cost = float(costs[pidx])
    base_price = float(prices[pidx]) * float(1.0 + rng.normal(0.0, session_price_noise))
    base_price = float(np.clip(base_price, cost * 1.01, float(prices[pidx]) * 2.0))
    current_price = base_price
    signal = 0.0
    events = []
    # TODO: instead of this very controlled setup implement same session samplin as in models.py
    while state != "end" and len(events) < 30:
        probs = trans.get(state, {"end": 1.0})
        nxt = rng.choice(list(probs.keys()), p=list(probs.values()))

        if nxt == "purchase":
            price_sens = float(theta.get("price_sens", 2.0))
            base_conv = float(theta.get("base_conv", 0.2))
            rel = max((current_price - cost) / (cost + 1e-6), 0.0)
            p_buy = float(np.clip(base_conv * np.exp(-price_sens * rel), 0.0, 1.0))
            if rng.random() > p_buy:
                nxt = "end"

        state = nxt
        if state not in {"start", "end"}:
            events.append(Event(action=state, product_idx=pidx, price_seen=float(current_price), ts=t))
            signal += float(ACTION_WEIGHTS.get(state, 0.1))
            current_price = float(np.clip(base_price * (1.0 + surge * signal), cost * 1.01, base_price * max_markup_mult))

        t += max(0.2, rng.gamma(1.5, 0.8) if is_agent else rng.gamma(2.0, 1.2))
    return events, pidx


def put_prices_to_market(prices: np.ndarray, costs: np.ndarray, alpha: float = 0.2, n_sessions: int = 50,
                         seed: int | None = None) -> Tuple[List[Session], Dict[str, float]]:
    """Generate sessions from mixture model

    Returns:
        sessions: list of Session objects with events and product attribution
        demand_mapping: session_id -> demand proxy q̂
    """
    rng = np.random.default_rng(seed)
    sessions, demand_mapping = [], {}

    for i in range(n_sessions):
        sid = f"s{i:04d}"
        is_agent = rng.random() < alpha
        trans = TRANS_A if is_agent else TRANS_H
        theta = {"price_sens": rng.uniform(0.05, 0.2), "base_conv": 0.01} if is_agent else {"price_sens": rng.uniform(1.5, 4.0), "base_conv": rng.uniform(0.2, 0.5)}
        events, _ = sample_trajectory(rng, trans, prices, costs=costs, theta=theta, is_agent=is_agent)
        session = Session(sid=sid, events=events, actor="A" if is_agent else "H", theta=theta)
        sessions.append(session)
        demand_mapping[sid] = compute_demand(session)

    return sessions, demand_mapping


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
        self._last_sessions: List[Session] = []
        self._last_coi: COIWindow | None = None

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

    def _compute_coi_window(self, demand: Dict[str, float]) -> COIWindow:
        if not self._last_sessions:
            zeros = np.zeros(self.n, dtype=float)
            return COIWindow(
                policy=0.0,
                agent=0.0,
                leak=0.0,
                survival_ratio=0.0,
                policy_by_product=zeros,
                agent_by_product=zeros,
                demand_weights=zeros,
            )
        return compute_coi_window(self._last_sessions, self.costs, demand_mapping=demand)

    def _objective(self, prices: np.ndarray, demand: Dict[str, float]) -> float:
        """Robust objective: R(p,d) - λ·COI_leak (Eq 23 simplified)."""
        revenue = self._revenue_under_demand(prices, demand)
        cost = float(np.sum(self.costs))  # fixed cost approximation
        profit = revenue - cost
        self._last_coi = self._compute_coi_window(demand)
        return profit - self.lambda_coi * self._last_coi.leak

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
        sessions, demand_map = put_prices_to_market(prices, costs=self.costs, alpha=alpha_true, n_sessions=n_sessions, seed=int(self.rng.integers(0, 10000)))
        self._last_sessions = sessions
        self._sessions.extend(sessions)  # store actual sessions for correct product attribution
        self.limbo.add_update("demand", demand_map)
        return demand_map

    def step(self, alpha_true: float = 0.2, n_sessions: int = 50) -> Tuple[np.ndarray, Dict[str, float], float, COIWindow]:
        """Single simulation step: prices -> demand -> reward."""
        demand_hist = self.limbo.get_demand_history()
        prices = self.compute_prices(demand_hist[-1] if demand_hist else None)
        demand = self.observe_demand(prices, alpha_true, n_sessions)
        reward = self._objective(prices, demand)
        coi = self._last_coi or self._compute_coi_window(demand)
        return prices, demand, reward, coi

    def run(self, n_steps: int = 100, alpha_true: float = 0.2) -> Dict:
        """Run simulation for n_steps, return trajectory."""
        trajectory = {
            "prices": [],
            "demand": [],
            "rewards": [],
            "alpha_est": [],
            "alpha_true": alpha_true,
            "coi_policy": [],
            "coi_agent": [],
            "coi_leak": [],
            "coi_survival": [],
        }
        for _ in range(n_steps):
            p, d, r, coi = self.step(alpha_true)
            trajectory["prices"].append(p)
            trajectory["demand"].append(d)
            trajectory["rewards"].append(r)
            trajectory["alpha_est"].append(self._alpha_est)
            trajectory["coi_policy"].append(coi.policy)
            trajectory["coi_agent"].append(coi.agent)
            trajectory["coi_leak"].append(coi.leak)
            trajectory["coi_survival"].append(coi.survival_ratio)
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
    print(
        f"avg reward: {np.mean(traj['rewards']):.2f}, "
        f"final α̂: {traj['alpha_est'][-1]:.3f}, "
        f"COI_policy: {np.mean(traj['coi_policy']):.3f}, "
        f"COI_agent: {np.mean(traj['coi_agent']):.3f}, "
        f"leak: {np.mean(traj['coi_leak']):.3f}"
    )

    prices = np.array([20.0, 35.0, 50.0, 25.0, 40.0])
    costs = np.array([15.0, 28.0, 40.0, 18.0, 30.0])
    sessions, demand = put_prices_to_market(prices, costs=costs, alpha=0.3, n_sessions=20, seed=123)
    print(f'sessions: {len(sessions)}, agents: {sum(1 for s in sessions if s.actor=="A")}')

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
