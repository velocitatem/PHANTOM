"""Contaminated arrivals using learned MDP kernels from behavior_loader.

Implements thesis demand model (Section 3.1):
- Aggregate demand Q(p) = (1-α)E[d(p;θ_H)] + αE[d(p;θ_A)] + ε_t  (Eq 3)
- Demand proxy q̂_{t,i} = Σ_s Σ_k ω(a_{s,k}) · 1[i_{s,k} = i]     (Eq 2)
- Per-session separability via KL divergence Δ_H, Δ_A              (Eq 20-21)

The arrival model samples sessions from a mixture of human/agent behavioral profiles,
each session produces a trajectory τ_s and associated demand computation q(τ').
"""
from __future__ import annotations
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional
import numpy as np
from ...outlet.types import Opportunity, InstrumentSet, MarketState, HiddenState
from ...outlet.constants import Side, OpportunityType
from ...outlet.math_util import poisson_arrivals

try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from sim.rl.behavior_loader.models import (
        BehaviorModel, AgentBehaviorModel, aggregate_event_transitions, kl_divergence
    )
    REAL_MDP = True
except ImportError:
    REAL_MDP = False
    kl_divergence = None

EVENT_PAGE = {"session_start": "/", "view_item_page": "/products", "learn_more_about_item": "/products/details",
              "add_item_to_cart": "/cart", "purchase_complete": "/checkout", "session_end": "/checkout/success"}
EVENT_CANON = {"page_view": "session_start", "hover_over_paragraph": "view_item_page", "hover_over_title": "view_item_page",
               "view_item_page": "view_item_page", "learn_more_about_item": "learn_more_about_item",
               "add_item_to_cart": "add_item_to_cart", "checkout_start": "purchase_complete", "remove_item": "view_item_page"}

# action space partition A = A_nav ∪ A_cart ∪ A_filter ∪ A_dwell with signal weights ω (Table 1)
ACTION_WEIGHTS: Dict[str, float] = {
    "add_item_to_cart": 0.8, "remove_item": 0.6, "checkout_start": 0.9, "purchase_complete": 1.0,  # A_cart
    "hover_over_title": 0.3, "hover_over_paragraph": 0.35, "hover_over_link": 0.25,               # A_dwell
    "page_view": 0.1, "session_start": 0.05, "view_item_page": 0.15, "learn_more_about_item": 0.2, # A_nav
    "search": 0.05, "filter_date": 0.05, "filter_price": 0.08, "sort": 0.03, "session_end": 0.0,   # A_filter
}


@dataclass
class SessionDemand:
    """Per-session demand computation per thesis formulation (Section 3.1).

    Each session s ∈ S produces trajectory τ_s and demand proxy q̂. The platform uses
    divergence signals Δ_H, Δ_A to estimate per-session contamination α̂(τ').
    """
    session_id: str
    q: Dict[int, float]               # q̂_i demand proxy per product (Eq 2)
    trajectory: List[Dict]            # τ_s = (e_{s,1}, ..., e_{s,L_s})
    delta_h: float = 0.0              # D_KL(T̂' || T̄_H) (Eq 20)
    delta_a: float = 0.0              # D_KL(T̂' || T̄_A) (Eq 21)
    alpha_hat: float = 0.0            # per-session contamination estimate
    actor_class: str = "H"            # ground truth Y_s ∈ {H, A}
    theta: Dict[str, float] = field(default_factory=dict)


def compute_demand_proxy(events: List[Dict], n_products: int) -> Dict[int, float]:
    """Compute q̂_{t,i} = Σ_k ω(a_{s,k}) · 1[i_{s,k} = i] per Eq 2."""
    q = {i: 0.0 for i in range(n_products)}
    for e in events:
        action, pidx = e.get("eventName", ""), e.get("product_idx")
        if pidx is not None and 0 <= pidx < n_products:
            q[pidx] += ACTION_WEIGHTS.get(action, 0.1)
    return q


def compute_session_divergence(events: List[Dict], ref_h: Dict, ref_a: Dict) -> Tuple[float, float]:
    """Compute Δ_H, Δ_A divergence signals from trajectory (Eq 20-21)."""
    if not events or kl_divergence is None:
        return 0.0, 0.0
    # build empirical transition kernel from trajectory
    trans: Dict[str, Dict[str, int]] = {}
    prev = "session_start"
    for e in events:
        curr = e.get("eventName", "session_end")
        trans.setdefault(prev, {})
        trans[prev][curr] = trans[prev].get(curr, 0) + 1
        prev = curr
    # normalize to probabilities
    kernel = {}
    for s, dests in trans.items():
        total = sum(dests.values())
        kernel[s] = {d: c / total for d, c in dests.items()} if total > 0 else {}
    # aggregate to event-level and compute KL divergence against reference kernels
    delta_h = sum(kl_divergence(kernel.get(s, {}), ref_h.get(s, {})) for s in kernel) / max(len(kernel), 1)
    delta_a = sum(kl_divergence(kernel.get(s, {}), ref_a.get(s, {})) for s in kernel) / max(len(kernel), 1)
    return delta_h, delta_a

def _canonicalize(raw: Dict) -> Dict:
    out = {}
    for src, dsts in raw.items():
        sc = EVENT_CANON.get(src, src)
        out.setdefault(sc, {})
        for dst, p in dsts.items():
            dc = EVENT_CANON.get(dst, dst)
            out[sc][dc] = out[sc].get(dc, 0.0) + p
    return {s: {k: v/sum(d.values()) for k, v in d.items()} for s, d in out.items() if sum(d.values()) > 0}


class BehavioralProfile:
    """Markov profile from learned MDP kernels (Section 3.5.2).

    Transition kernel T̂_Y estimated via MLE: P̂(s'|s) = N(s,s') / Σ_k N(s,k) (Eq 19)
    """
    STATES = ["session_start", "view_item_page", "learn_more_about_item", "add_item_to_cart", "purchase_complete", "session_end"]
    # fallback kernels T̄_H, T̄_A when real data unavailable
    FALLBACK_H = {"session_start": {"view_item_page": 0.85, "session_end": 0.15},
                  "view_item_page": {"learn_more_about_item": 0.4, "add_item_to_cart": 0.3, "view_item_page": 0.2, "session_end": 0.1},
                  "learn_more_about_item": {"add_item_to_cart": 0.5, "view_item_page": 0.3, "session_end": 0.2},
                  "add_item_to_cart": {"purchase_complete": 0.6, "view_item_page": 0.25, "session_end": 0.15},
                  "purchase_complete": {"session_end": 1.0}}
    FALLBACK_A = {"session_start": {"view_item_page": 0.95, "session_end": 0.05},
                  "view_item_page": {"learn_more_about_item": 0.6, "view_item_page": 0.25, "add_item_to_cart": 0.1, "session_end": 0.05},
                  "learn_more_about_item": {"view_item_page": 0.5, "add_item_to_cart": 0.15, "learn_more_about_item": 0.3, "session_end": 0.05},
                  "add_item_to_cart": {"view_item_page": 0.4, "purchase_complete": 0.2, "session_end": 0.4},
                  "purchase_complete": {"session_end": 1.0}}

    def __init__(self, actor: str, pprobs: np.ndarray, data_dir: str = ""):
        self.actor, self.pprobs = actor, np.clip(pprobs, 0.0, 0.95)
        self.trans = self._load(data_dir)  # T̂_Y transition kernel
        self._ensure_terminal()
        self.dwell = {s: (1.2, 0.5) if actor == "agents" else (2.0, 1.2) for s in self.STATES}

    def _load(self, data_dir: str) -> Dict:
        if not REAL_MDP or not data_dir:
            print("using fallback")
            return dict(self.FALLBACK_A if self.actor == "agents" else self.FALLBACK_H)
        try:
            mdp = (AgentBehaviorModel if self.actor == "agents" else BehaviorModel)(data_dir).build_MDP()
            raw = aggregate_event_transitions(mdp) if mdp.get("transitions") else {}
            return _canonicalize(raw) if raw else dict(self.FALLBACK_A if self.actor == "agents" else self.FALLBACK_H)
        except Exception:
            print("using fallback")
            return dict(self.FALLBACK_A if self.actor == "agents" else self.FALLBACK_H)

    def _ensure_terminal(self):
        self.trans.setdefault("purchase_complete", {})["session_end"] = self.trans.get("purchase_complete", {}).get("session_end", 1.0)
        self.trans.setdefault("session_start", {"view_item_page": 0.7, "learn_more_about_item": 0.2, "session_end": 0.1})

    def _tprobs(self, state: str, pidx: int) -> Dict[str, float]:
        probs = dict(self.trans.get(state, {"session_end": 1.0}))
        if state == "add_item_to_cart":
            base = probs.get("purchase_complete", 0.0)
            df = float(self.pprobs[pidx]) * (0.3 if self.actor == "agents" else 1.0)
            adj = np.clip(base * 0.5 + df * 0.5, 0.0, 0.95)
            rem = max(1e-6, 1.0 - adj)
            other = sum(v for k, v in probs.items() if k != "purchase_complete")
            probs = {k: (adj if k == "purchase_complete" else v * rem / max(other, 1e-6)) for k, v in probs.items()}
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()} if total > 0 else {"session_end": 1.0}

    def sample(self, rng: np.random.Generator, sid: str, prices: np.ndarray, costs: np.ndarray) -> Tuple[List[Dict], List[SimpleNamespace]]:
        events, fevts = [], []
        state, t, pidx = "session_start", 0.0, int(rng.integers(0, len(prices)))
        cost, cprice = float(costs[pidx]), max(float(prices[pidx]), float(costs[pidx]) * 1.05)

        while state != "session_end" and len(events) < 40:
            if state != "session_start":
                row = {"session_id": sid, "actor": "agent" if self.actor == "agents" else "human",
                       "eventName": state, "product_idx": pidx, "productId": f"product-{pidx:04d}",
                       "price_offered": cprice, "price_paid": 0.0, "page": EVENT_PAGE.get(state, "/"),
                       "ts": t, "unit_cost": cost, "base_price": float(prices[pidx])}
                if state == "purchase_complete":
                    row["price_paid"] = max(cprice * (1.0 + rng.normal(0.0, 0.015)), cost)
                events.append(row)
                fevts.append(SimpleNamespace(eventName=state, page=row["page"], productId=row["productId"], ts=t))

            probs = self._tprobs(state, pidx)
            state = rng.choice(list(probs.keys()), p=list(probs.values()))
            sh, sc = self.dwell.get(state, (2.0, 1.0))
            t += max(0.3, rng.gamma(shape=sh, scale=sc))
        return events, fevts


@dataclass
class ContaminatedArrivalConfig:
    base_rate: float = 20.0
    alpha_contamination: float = 0.2
    alpha_drift: float = 0.0
    alpha_bounds: tuple[float, float] = (0.0, 0.5)
    human_views_range: tuple[int, int] = (1, 4)
    agent_views_range: tuple[int, int] = (3, 10)
    agent_systematic: bool = True
    use_real_behavior: bool = True
    human_data_dir: str = ""
    agent_data_dir: str = ""


class ContaminatedArrivalModel:
    """Mixture model Q(p) = (1-α)E[d(p;θ_H)] + αE[d(p;θ_A)] + ε_t (Eq 3).

    Samples sessions from human/agent behavioral profiles, computes per-session
    demand proxy q̂ and divergence signals Δ_H, Δ_A for separability.
    """

    def __init__(self, cfg: ContaminatedArrivalConfig | None = None):
        self.cfg = cfg or ContaminatedArrivalConfig()
        self._alpha = self.cfg.alpha_contamination
        self._scount = 0
        self._profiles: Dict[str, BehavioralProfile] = {}
        self._ref_kernels: Dict[str, Dict] = {}  # T̄_H, T̄_A reference kernels
        self._session_demands: List[SessionDemand] = []  # collected session demands

    @property
    def alpha(self) -> float:
        return self._alpha

    def _profile(self, actor: str, pprobs: np.ndarray) -> BehavioralProfile:
        key = actor
        if key not in self._profiles:
            ddir = self.cfg.agent_data_dir if actor == "agents" else self.cfg.human_data_dir
            if not ddir and self.cfg.use_real_behavior:
                base = Path(__file__).parent.parent.parent.parent / "experiments"
                ddir = str(base / ("agents/collected_data" if actor == "agents" else "collected_data"))
            profile = BehavioralProfile(actor, pprobs, ddir if self.cfg.use_real_behavior else "")
            self._profiles[key] = profile
            self._ref_kernels[key] = profile.trans  # cache T̄_Y for divergence
        return self._profiles[key]

    def get_ref_kernels(self) -> Tuple[Dict, Dict]:
        """Return reference transition kernels T̄_H, T̄_A for divergence computation."""
        return (self._ref_kernels.get("humans", BehavioralProfile.FALLBACK_H),
                self._ref_kernels.get("agents", BehavioralProfile.FALLBACK_A))

    def get_session_demands(self) -> List[SessionDemand]:
        """Return collected session demands for downstream analysis."""
        return self._session_demands

    def sample(self, t: float, dt: float, instruments: InstrumentSet,
               market: MarketState | None, hidden: HiddenState, rng: np.random.Generator) -> list[Opportunity]:
        """Sample arrivals as per Eq 3: mixture of human/agent demand distributions.

        For each session s, computes:
        - Trajectory τ_s from behavioral profile sampling
        - Demand proxy q̂ via weighted action aggregation (Eq 2)
        - Divergence signals Δ_H, Δ_A for separability (Eq 20-21)
        - Per-session contamination estimate α̂(τ')
        """
        cfg = self.cfg
        if cfg.alpha_drift != 0:
            self._alpha = np.clip(self._alpha + cfg.alpha_drift * rng.normal(), *cfg.alpha_bounds)
        hidden.contamination = self._alpha

        n_sess = poisson_arrivals(cfg.base_rate * hidden.true_demand_intensity, dt, rng)
        prices, costs = instruments.refs, instruments.costs
        margin = np.clip((prices - costs) / np.maximum(costs, 1e-3), -0.9, 2.0)
        hprob, aprob = 0.08 * np.exp(-1.2 * margin), 0.05 * np.exp(-0.6 * margin)
        ref_h, ref_a = self.get_ref_kernels()

        opps = []
        for _ in range(n_sess):
            self._scount += 1
            sid = f"s{self._scount:06d}"
            is_agent = rng.random() < self._alpha
            actor, probs = ("agents", aprob) if is_agent else ("humans", hprob)
            profile = self._profile(actor, probs)
            events, fevts = profile.sample(rng, sid, prices, costs)

            # compute demand proxy q̂ per Eq 2
            q = compute_demand_proxy(events, instruments.n)

            # compute divergence signals Δ_H, Δ_A per Eq 20-21
            delta_h, delta_a = compute_session_divergence(events, ref_h, ref_a)
            # per-session contamination estimate α̂(τ') = σ(β(Δ_H - Δ_A))
            alpha_hat = 1.0 / (1.0 + np.exp(-2.0 * (delta_h - delta_a))) if (delta_h + delta_a) > 0 else 0.5

            theta = ({'price_sensitivity': rng.uniform(0.05, 0.2), 'base_conversion': 0.01, 'info_value': 1.0} if is_agent
                     else {'price_sensitivity': rng.uniform(1.5, 4.0), 'base_conversion': rng.uniform(0.2, 0.5), 'info_value': 0.0})

            # store session demand for downstream analysis
            self._session_demands.append(SessionDemand(
                session_id=sid, q=q, trajectory=events, delta_h=delta_h, delta_a=delta_a,
                alpha_hat=alpha_hat, actor_class="A" if is_agent else "H", theta=theta))

            viewed = list({e["product_idx"] for e in events if "product_idx" in e})
            if not viewed:
                vr = cfg.agent_views_range if is_agent else cfg.human_views_range
                viewed = list(rng.choice(instruments.n, size=min(rng.integers(*vr), instruments.n), replace=False))

            for vi, iid in enumerate(viewed):
                opps.append(Opportunity(
                    id=f"{sid}-{iid}", type=OpportunityType.SESSION, side=Side.BUY,
                    instrument_id=int(iid), size=1.0, t=t + rng.uniform(0, dt),
                    context={'session_id': sid, 'actor_class': 'AGENT' if is_agent else 'HUMAN', 'is_agent': is_agent,
                             'reconnaissance_intent': is_agent, 'view_index': vi, 'total_views': len(viewed),
                             'theta': theta, 'trajectory_events': fevts, 'mdp_trajectory': events,
                             'demand_proxy': q, 'alpha_hat': alpha_hat, 'delta_h': delta_h, 'delta_a': delta_a}))
        return opps


@dataclass
class AdversarialArrivalConfig:
    base_rate: float = 5.0
    n_parallel_agents: int = 3
    query_all_products: bool = True


class AdversarialArrivalModel:
    """Adversarial coordination (Theorem 1): as N->inf, COI->0."""

    def __init__(self, cfg: AdversarialArrivalConfig | None = None):
        self.cfg = cfg or AdversarialArrivalConfig()
        self._qcount = 0

    def sample(self, t: float, dt: float, instruments: InstrumentSet,
               market: MarketState | None, hidden: HiddenState, rng: np.random.Generator) -> list[Opportunity]:
        cfg, opps = self.cfg, []
        for _ in range(poisson_arrivals(cfg.base_rate, dt, rng)):
            self._qcount += 1
            for ai in range(cfg.n_parallel_agents):
                sid = f"adv{self._qcount:06d}-{ai}"
                prods = np.arange(instruments.n) if cfg.query_all_products else rng.choice(instruments.n, size=1)
                for iid in prods:
                    opps.append(Opportunity(
                        id=f"{sid}-{iid}", type=OpportunityType.SESSION, side=Side.BUY,
                        instrument_id=int(iid), size=1.0, t=t,
                        context={'session_id': sid, 'actor_class': 'AGENT', 'is_agent': True, 'adversarial': True,
                                 'agent_index': ai, 'query_group': self._qcount,
                                 'theta': {'price_sensitivity': 0.0, 'base_conversion': 0.0, 'info_value': 1.0}}))
        return opps
