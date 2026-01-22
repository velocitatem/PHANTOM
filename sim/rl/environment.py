import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
import pandas as pd
from types import SimpleNamespace
from typing import Optional, Dict, Any, List, Tuple

from lib.separability import load_artifacts, score_session, estimate_alpha
from sim.rl.behavior_loader.models import AgentBehaviorModel, BehaviorModel, aggregate_event_transitions

try:
    import jax
    from sim.rl.jax_core import JAX_AVAILABLE, compile_transitions, fallback_transitions, sample_sessions, compute_metrics
    from sim.rl.jax_core import session_features, compute_session_transitions, compute_divergences, estimate_alpha_batch
except ImportError:
    JAX_AVAILABLE = False

# "learner" agent learning to optimize pricing
# "agent" part of environment creating demand signals that learner processes

base_dir = "/home/velocitatem/Documents/Projects/PHANTOM/experiments"
human_dir, agent_dir = f"{base_dir}/collected_data/", f"{base_dir}/agents/collected_data/"
@dataclass
class BusinessLogicConstraints():
    max_price_adjustment: float = 0.30
    system_max_price: float = 500.0
    system_min_price: float = 1.0
    product_catalogue_size: int = 100
    episode_length: int = 2000
    sessions_per_step: int = 250
    agent_share: float = 0.2
    agent_recon_multiplier: float = 6.0
    agent_purchase_probability: float = 0.20
    coi_strength: float = 0.25
    coi_threshold: float = 4.0
    coi_sigmoid_temp: float = 1.25
    base_human_demand: float = 0.08
    base_agent_demand: float = 0.05
    human_price_elasticity: float = -1.2 # assumptions here
    agent_price_elasticity: float = -0.6
    w_agent_loss: float = 1.0
    w_volatility: float = 5.0
    w_estimation_error: float = 0.25
    seed: int = 7


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

EVENT_PAGE_MAP = {
    "session_start": "/",
    "page_view": "/",
    "view_item_page": "/products",
    "learn_more_about_item": "/products/details",
    "add_item_to_cart": "/cart",
    "checkout_start": "/checkout",
    "purchase_complete": "/checkout",
    "session_end": "/checkout/success",
}

# map real collected event names to canonical simulation states
EVENT_CANONICAL_MAP = {
    "page_view": "session_start",
    "hover_over_paragraph": "view_item_page",
    "hover_over_title": "view_item_page",
    "view_item_page": "view_item_page",
    "learn_more_about_item": "learn_more_about_item",
    "add_item_to_cart": "add_item_to_cart",
    "checkout_start": "purchase_complete",
    "remove_item": "view_item_page",
}


def _canonicalize_transitions(raw_trans: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Map real event transition names to canonical simulation states."""
    canonical: Dict[str, Dict[str, float]] = {}
    for src, dsts in raw_trans.items():
        src_canon = EVENT_CANONICAL_MAP.get(src, src)
        if src_canon not in canonical:
            canonical[src_canon] = {}
        for dst, prob in dsts.items():
            dst_canon = EVENT_CANONICAL_MAP.get(dst, dst)
            canonical[src_canon][dst_canon] = canonical[src_canon].get(dst_canon, 0.0) + prob
    # re-normalize after aggregation
    for src in canonical:
        total = sum(canonical[src].values())
        if total > 0:
            canonical[src] = {k: v / total for k, v in canonical[src].items()}
    return canonical


class BehavioralProfile:
    """Synthetic Markov profile used to generate interaction sessions.
    Uses aggregate_event_transitions from models.py to build transition kernels from real data."""

    def __init__(self, actor: str, purchase_probs: np.ndarray):
        self.actor = actor
        self.purchase_probs = np.clip(purchase_probs, 0.0, 0.95)
        self.states = [
            "session_start",
            "view_item_page",
            "learn_more_about_item",
            "add_item_to_cart",
            "purchase_complete",
            "session_end",
        ]
        model = AgentBehaviorModel(agent_dir) if actor == "agents" else BehaviorModel(human_dir)
        mdp = model.build_MDP()
        raw_trans = aggregate_event_transitions(mdp) if mdp.get("transitions") else {}
        self.transitions = _canonicalize_transitions(raw_trans) if raw_trans else self._fallback_transitions()
        self._ensure_terminal_states()
        self.dwell_params = self._extract_dwell_params(mdp)

    def _ensure_terminal_states(self):
        # guarantee purchase_complete leads to session_end and session_start exists
        if "purchase_complete" not in self.transitions:
            self.transitions["purchase_complete"] = {"session_end": 1.0}
        elif "session_end" not in self.transitions.get("purchase_complete", {}):
            self.transitions["purchase_complete"]["session_end"] = 1.0
            total = sum(self.transitions["purchase_complete"].values())
            self.transitions["purchase_complete"] = {k: v/total for k, v in self.transitions["purchase_complete"].items()}
        if "session_start" not in self.transitions:
            self.transitions["session_start"] = {"view_item_page": 0.7, "learn_more_about_item": 0.2, "session_end": 0.1}

    def _fallback_transitions(self) -> Dict[str, Dict[str, float]]:
        return {
            "session_start": {"view_item_page": 0.85, "session_end": 0.15},
            "view_item_page": {"learn_more_about_item": 0.4, "add_item_to_cart": 0.3, "view_item_page": 0.2, "session_end": 0.1},
            "learn_more_about_item": {"add_item_to_cart": 0.5, "view_item_page": 0.3, "session_end": 0.2},
            "add_item_to_cart": {"purchase_complete": 0.6, "view_item_page": 0.25, "session_end": 0.15},
            "purchase_complete": {"session_end": 1.0},
        }

    def _extract_dwell_params(self, mdp: Dict) -> Dict[str, Tuple[float, float]]:
        state_vals = mdp.get("state_values", {})
        params = {}
        for state in self.states:
            # try canonical and raw state names
            val = state_vals.get(state, 0.5)
            for raw, canon in EVENT_CANONICAL_MAP.items():
                if canon == state and raw in state_vals:
                    val = state_vals[raw]
                    break
            shape = 1.5 + val * 2.0
            scale = 0.8 + (1.0 - val) * 1.2
            params[state] = (shape, scale)
        return params

    def _transition_probs(self, state: str, product_idx: int) -> Dict[str, float]:
        probs = dict(self.transitions.get(state, {"session_end": 1.0}))
        if state == "add_item_to_cart":
            base = probs.get("purchase_complete", 0.0)
            demand_factor = float(self.purchase_probs[int(product_idx)])
            if self.actor == "agents":
                demand_factor *= 0.7
            adjusted = np.clip(base * 0.5 + demand_factor * 0.5, 0.0, 0.95)
            remainder = max(1e-6, 1.0 - adjusted)
            other_total = sum(v for k, v in probs.items() if k != "purchase_complete")
            scale = remainder / max(other_total, 1e-6)
            for key in probs:
                if key == "purchase_complete":
                    probs[key] = adjusted
                else:
                    probs[key] = probs[key] * scale
        total = sum(probs.values())
        if total <= 0:
            return {"session_end": 1.0}
        return {state: val / total for state, val in probs.items()}

    def sample_session(
        self,
        rng: np.random.Generator,
        session_id: str,
        prices: np.ndarray,
        unit_cost: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], List[SimpleNamespace]]:
        """Generate a single session trajectory respecting business constraints."""
        events: List[Dict[str, Any]] = []
        feature_events: List[SimpleNamespace] = []
        state = "session_start"
        t = 0.0
        product_idx = int(rng.integers(0, len(prices)))
        product_id = f"product-{product_idx:04d}"


        # enforce price >= cost constraint (lipschitz bound on pricing)
        # This is a sort of last resort to not let an pricing learner go rogue
        cost = float(unit_cost[product_idx])
        constrained_price = max(float(prices[product_idx]), cost * 1.05)  # 5% min margin

        while state != "session_end" and len(events) < 40:
            if state != "session_start":
                row = {
                    "session_id": session_id,
                    "actor": "agent" if self.actor == "agents" else "human",
                    "eventName": state,
                    "product_idx": product_idx,
                    "productId": product_id,
                    "price_offered": constrained_price,
                    "price_paid": 0.0,
                    "page": EVENT_PAGE_MAP.get(state, "/"),
                    "ts": t,
                    "unit_cost": cost,
                    "base_price": float(prices[product_idx]),
                }
                if state == "purchase_complete":
                    noise = float(rng.normal(0.0, 0.015))
                    row["price_paid"] = max(constrained_price * (1.0 + noise), cost)
                events.append(row)
                feature_events.append(
                    SimpleNamespace(
                        eventName=row["eventName"],
                        page=row["page"],
                        productId=row["productId"],
                        ts=row["ts"],
                    )
                )

            transitions = self._transition_probs(state, product_idx)
            next_state = rng.choice(list(transitions.keys()), p=list(transitions.values()))
            shape, scale = self.dwell_params.get(state, (2.0, 1.0))
            dwell = max(0.3, rng.gamma(shape=shape, scale=scale))
            t += dwell
            state = next_state

        return events, feature_events


def _load_behavioral_profile(actor: str, demand_forcing: np.ndarray) -> BehavioralProfile:
    """returns a behavioral profile for generating synthetic sessions
    actor: 'humans' or 'agents'
    demand_forcing: per-product purchase probabilities used to weight interactions
    """
    return BehavioralProfile(actor, demand_forcing)


class CommercePlatform:
    """state management for the environment, simulates demand"""
    def __init__(self, product_catalogue_size: int, max_price: float, min_price: float, constraints: BusinessLogicConstraints):
        self.product_catalogue_size = product_catalogue_size
        self.max_price = max_price
        self.min_price = min_price
        self.constraints = constraints
        self.simulation_history: List[Dict[str, Any]] = []
        self._rng = np.random.default_rng(constraints.seed)
        self._last_interaction_df: pd.DataFrame = pd.DataFrame()
        self.unit_cost = np.random.uniform(low=15.0, high=60.0, size=(self.product_catalogue_size,)).astype(np.float32)
        self.base_price = np.random.uniform(low=60.0, high=140.0, size=(self.product_catalogue_size,)).astype(np.float32)
        self.alpha_hat = constraints.agent_share
        try:
            self.separability_artifacts = load_artifacts()
        except FileNotFoundError:
            self.separability_artifacts = None

    def setup_true_demand(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        p = np.clip(prices, self.min_price, self.max_price)
        cost = np.clip(self.unit_cost, self.min_price * 0.2, self.max_price)
        margin = np.clip((p - cost) / np.maximum(cost, 1e-3), -0.9, 2.0)
        # isoelastic demand approximation
        human_prob = self.constraints.base_human_demand * np.exp(self.constraints.human_price_elasticity * margin)
        agent_prob = self.constraints.base_agent_demand * np.exp(self.constraints.agent_price_elasticity * margin)
        return {
            "human_purchase_prob": np.clip(human_prob, 0.0, 0.95),
            "agent_purchase_prob": np.clip(agent_prob, 0.0, 0.95),
        }

    def _simulate_sessions(self, prices: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        demand = self.setup_true_demand(prices)
        T = self.constraints.sessions_per_step
        effective_share = float(np.clip(self.alpha_hat, 0.0, 0.95))
        n_agent_sessions = max(1, int(round(T * effective_share)))
        n_human_sessions = max(1, T - n_agent_sessions)

        session_map = {
            "humans": n_human_sessions,
            "agents": n_agent_sessions,
        }
        pprob_map = {
            "humans": demand["human_purchase_prob"],
            "agents": demand["agent_purchase_prob"],
        }

        rows: List[Dict[str, Any]] = []
        session_scores: List[Dict[str, float]] = []
        demand_human = np.zeros_like(prices, dtype=np.float32)
        demand_agent = np.zeros_like(prices, dtype=np.float32)

        for actor, n_sessions in session_map.items():
            profile = _load_behavioral_profile(actor, pprob_map[actor])
            for idx in range(n_sessions):
                session_id = f"{actor}_{idx:06d}"
                session_rows, feature_events = profile.sample_session(
                    self._rng, session_id, prices, self.unit_cost
                )
                rows.extend(session_rows)
                if session_rows:
                    df_session = pd.DataFrame(session_rows)
                    purchases = df_session[df_session["eventName"] == "purchase_complete"]
                    if not purchases.empty:
                        counts = purchases.groupby("product_idx").size()
                        if actor == "agents":
                            demand_agent[counts.index.to_numpy(dtype=int)] += counts.to_numpy(dtype=np.float32)
                        else:
                            demand_human[counts.index.to_numpy(dtype=int)] += counts.to_numpy(dtype=np.float32)
                if self.separability_artifacts and feature_events:
                    score = score_session(feature_events, self.separability_artifacts)
                    session_scores.append(score)

        interactions_df = pd.DataFrame(rows)
        diagnostics = {
            "alpha_hat": float(self.alpha_hat),
            "session_scores": session_scores,
            "demand_human": demand_human,
            "demand_agent": demand_agent,
        }

        if session_scores:
            alphas = [
                estimate_alpha(s["prob_agent"], s["delta_h"], s["delta_a"], temperature=2.0)
                for s in session_scores
            ]
            mean_alpha = float(np.mean(alphas))
            # exponential moving average for stability
            self.alpha_hat = 0.7 * self.alpha_hat + 0.3 * mean_alpha
            diagnostics.update(
                {
                    "alpha_hat": float(self.alpha_hat),
                    "delta_h_mean": float(np.mean([s["delta_h"] for s in session_scores])),
                    "delta_a_mean": float(np.mean([s["delta_a"] for s in session_scores])),
                    "prob_agent_mean": float(np.mean([s["prob_agent"] for s in session_scores])),
                }
            )

        self._last_interaction_df = interactions_df
        return interactions_df, diagnostics

    def compute_interaction_features(self, interaction_df: pd.DataFrame) -> Dict[str, float]:
        if interaction_df.empty:
            return {
                "revenue_observed": 0.0,
                "revenue_oracle": 0.0,
                "agent_loss": 0.0,
                "true_human_purchases": 0.0,
                "true_agent_purchases": 0.0,
                "mean_sale_price": 0.0,
                "look_to_book": 0.0,
                "coi": 0.0,
                "expected_premium": 0.0,
            }

        purchases = interaction_df[interaction_df["eventName"] == "purchase_complete"]
        human_purchases = purchases[purchases["actor"] == "human"]
        agent_purchases = purchases[purchases["actor"] == "agent"]

        revenue_observed = float(purchases["price_paid"].sum())
        revenue_oracle = float(purchases["base_price"].sum())
        agent_loss = float((agent_purchases["base_price"] - agent_purchases["price_paid"]).sum())

        mean_sale_price = float(purchases["price_paid"].mean()) if not purchases.empty else 0.0
        views = float((interaction_df["eventName"] == "view_item_page").sum())
        look_to_book = float(views / (len(purchases) + 1e-6))
        true_human = float(len(human_purchases))
        true_agent = float(len(agent_purchases))

        human_prices = human_purchases["price_offered"] if not human_purchases.empty else pd.Series(dtype=float)
        human_costs = human_purchases["unit_cost"] if not human_purchases.empty else pd.Series(dtype=float)
        human_base = human_purchases["base_price"] if not human_purchases.empty else pd.Series(dtype=float)
        coi = 0.0
        if not human_prices.empty and not human_costs.empty:
            # COI = E[P] - p_min where p_min is cost, accounting for expected premium (base - realized)
            margin = human_prices.mean() - human_costs.mean()
            expected_premium = human_base.mean() - human_prices.mean() if not human_base.empty else 0.0
            coi = float(np.maximum(0.0, margin - expected_premium * 0.5))

        return {
            "revenue_observed": revenue_observed,
            "revenue_oracle": revenue_oracle,
            "agent_loss": agent_loss,
            "true_human_purchases": true_human,
            "true_agent_purchases": true_agent,
            "mean_sale_price": mean_sale_price,
            "look_to_book": look_to_book,
            "coi": coi,
            "expected_premium": float(expected_premium) if not human_base.empty else 0.0,
        }

    def _session_feature_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract per-session behavioral features for separability analysis."""
        if df.empty:
            return pd.DataFrame()
        g = df.groupby("session_id", sort=False)
        session_duration = g["ts"].max() - g["ts"].min()
        total_interactions = g.size()
        avg_time_between = g["ts"].apply(lambda x: float(np.diff(np.sort(x.to_numpy())).mean()) if len(x) > 1 else 0.0)
        interaction_velocity = total_interactions / (session_duration + 1e-6)
        views = g.apply(lambda x: int((x["eventName"] == "view_item_page").sum()), include_groups=False)
        cart_adds = g.apply(lambda x: int((x["eventName"] == "add_item_to_cart").sum()), include_groups=False)
        purchases = g.apply(lambda x: int((x["eventName"] == "purchase_complete").sum()), include_groups=False)
        learn_more = g.apply(lambda x: int((x["eventName"] == "learn_more_about_item").sum()), include_groups=False)
        conversion_rate = purchases / (views + 1e-6)
        is_agent = g["actor"].apply(lambda s: bool((s == "agent").any()), include_groups=False)
        # price sensitivity features
        price_variance = g["price_offered"].var().fillna(0.0)
        avg_price_seen = g["price_offered"].mean().fillna(0.0)
        products_viewed = g["product_idx"].nunique()

        return pd.DataFrame({
            "session_duration_sec": session_duration.astype(float),
            "avg_time_between_events": avg_time_between.astype(float),
            "total_interactions": total_interactions.astype(int),
            "interaction_velocity": interaction_velocity.astype(float),
            "item_views": views.astype(int),
            "cart_adds": cart_adds.astype(int),
            "purchases": purchases.astype(int),
            "learn_more_clicks": learn_more.astype(int),
            "conversion_rate": conversion_rate.astype(float),
            "price_variance": price_variance.astype(float),
            "avg_price_seen": avg_price_seen.astype(float),
            "products_viewed": products_viewed.astype(int),
            "is_agent": is_agent.astype(bool),
        }).reset_index()

    def get_interaction_data(self) -> np.ndarray:
        if self._last_interaction_df.empty:
            return np.array([], dtype=object)
        return self._last_interaction_df.to_dict(orient="records")


class PHANTOMEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, constraints: Optional[BusinessLogicConstraints] = None, use_jax: bool = True):
        super().__init__()
        self.constraints = constraints if isinstance(constraints, BusinessLogicConstraints) else BusinessLogicConstraints()
        self.use_jax = use_jax and JAX_AVAILABLE
        self.action_space = spaces.Box(low=-self.constraints.max_price_adjustment,
                                       high=self.constraints.max_price_adjustment,
                                       shape=(self.constraints.product_catalogue_size,), dtype=np.float32)
        n_products = self.constraints.product_catalogue_size
        self.observation_space = spaces.Dict({
            "elasticity": spaces.Dict({
                "price": spaces.Box(
                    low=np.full((n_products,), self.constraints.system_min_price, dtype=np.float32),
                    high=np.full((n_products,), self.constraints.system_max_price, dtype=np.float32),
                    dtype=np.float32),
                "demand": spaces.Box(
                    low=np.zeros((n_products,), dtype=np.float32),
                    high=np.full((n_products,), 1e6, dtype=np.float32),
                    dtype=np.float32),
            }),
            "market": spaces.Dict({
                "alpha_hat": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "revenue_rate": spaces.Box(low=0.0, high=1e6, shape=(1,), dtype=np.float32),
                "conversion_rate": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "price_volatility": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }),
            "cost": spaces.Box(low=0.0, high=self.constraints.system_max_price, shape=(n_products,), dtype=np.float32),
        })
        self.commerce_platform = CommercePlatform(
            product_catalogue_size=self.constraints.product_catalogue_size,
            max_price=self.constraints.system_max_price,
            min_price=self.constraints.system_min_price,
            constraints=self.constraints)
        self._rng = np.random.default_rng(self.constraints.seed)
        self.t = 0
        self._prev_prices: Optional[np.ndarray] = None
        self.state: Dict[str, Any] = {}
        self._jax_key = None
        self._jax_trans = None
        if self.use_jax:
            self._jax_key = jax.random.PRNGKey(self.constraints.seed)
            self._init_jax_transitions()

    def _init_jax_transitions(self):
        try:
            human_profile = _load_behavioral_profile("humans", np.ones(self.constraints.product_catalogue_size) * 0.1)
            agent_profile = _load_behavioral_profile("agents", np.ones(self.constraints.product_catalogue_size) * 0.1)
            self._jax_trans = compile_transitions(human_profile, agent_profile).to_jax()
        except Exception:
            self._jax_trans = fallback_transitions().to_jax()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.commerce_platform._rng = np.random.default_rng(seed)
            if self.use_jax:
                self._jax_key = jax.random.PRNGKey(seed)
        self.commerce_platform.alpha_hat = self.constraints.agent_share
        self.t = 0
        init_prices = self._rng.uniform(
            low=60.0,
            high=140.0,
            size=(self.constraints.product_catalogue_size,),
        ).astype(np.float32)
        self.commerce_platform.unit_cost = self._rng.uniform(
            low=15.0,
            high=60.0,
            size=(self.constraints.product_catalogue_size,),
        ).astype(np.float32)
        self.commerce_platform.base_price = init_prices.copy()
        self._prev_prices = init_prices.copy()
        self.state = {
            "elasticity": {
                "price": init_prices,
                "demand": np.zeros((self.constraints.product_catalogue_size,), dtype=np.float32),
            },
            "market": {
                "alpha_hat": np.array([self.constraints.agent_share], dtype=np.float32),
                "revenue_rate": np.array([0.0], dtype=np.float32),
                "conversion_rate": np.array([0.0], dtype=np.float32),
                "price_volatility": np.array([0.0], dtype=np.float32),
            },
            "cost": self.commerce_platform.unit_cost.astype(np.float32),
        }
        return self.state, {}

    def _step_jax(self, new_prices: np.ndarray) -> Tuple[Dict, Dict]:
        self._jax_key, subkey = jax.random.split(self._jax_key)
        alpha = float(np.clip(self.commerce_platform.alpha_hat, 0.0, 0.95))
        n_agent = max(1, int(self.constraints.sessions_per_step * alpha))
        n_human = max(1, self.constraints.sessions_per_step - n_agent)
        batch = sample_sessions(subkey, self._jax_trans, n_human, n_agent, len(new_prices))
        sim = compute_metrics(batch, new_prices, self.commerce_platform.unit_cost, self.commerce_platform.base_price)
        result = {"revenue_observed": sim.revenue, "revenue_oracle": sim.revenue_oracle,
                  "agent_loss": sim.agent_loss, "coi": sim.coi, "look_to_book": sim.look_to_book,
                  "mean_sale_price": sim.mean_sale_price, "true_human_purchases": sim.n_human_purchases,
                  "true_agent_purchases": sim.n_agent_purchases}
        diagnostics = {"demand_human": sim.demand_human, "demand_agent": sim.demand_agent, "alpha_hat": alpha}
        return result, diagnostics

    def step(self, action: np.ndarray):
        self.t += 1
        base_prices = self.state["elasticity"]["price"].astype(np.float32)
        new_prices = np.clip(base_prices * (1.0 + action.astype(np.float32)),
                           self.constraints.system_min_price,
                           self.constraints.system_max_price).astype(np.float32)

        self.state["elasticity"]["price"] = new_prices
        if self.use_jax:
            result, diagnostics = self._step_jax(new_prices)
        else:
            interactions_df, diagnostics = self.commerce_platform._simulate_sessions(new_prices)
            result = self.commerce_platform.compute_interaction_features(interactions_df)
        COI = float(result.get("coi", 0.0))

        demand_vector = diagnostics.get("demand_human", np.zeros_like(new_prices)) + diagnostics.get(
            "demand_agent", np.zeros_like(new_prices)
        )
        self.state["elasticity"]["demand"] = demand_vector.astype(np.float32)

        volatility = 0.0 if self._prev_prices is None else \
            float(np.mean(np.abs((new_prices - self._prev_prices) / (self._prev_prices + 1e-6))))
        self._prev_prices = new_prices.copy()

        # update market observation features
        total_demand = float(np.sum(demand_vector))
        total_purchases = float(result.get("true_human_purchases", 0.0) + result.get("true_agent_purchases", 0.0))
        conv_rate = total_purchases / max(total_demand, 1.0)
        self.state["market"] = {
            "alpha_hat": np.array([float(diagnostics.get("alpha_hat", self.commerce_platform.alpha_hat))], dtype=np.float32),
            "revenue_rate": np.array([float(result.get("revenue_observed", 0.0))], dtype=np.float32),
            "conversion_rate": np.array([float(np.clip(conv_rate, 0.0, 1.0))], dtype=np.float32),
            "price_volatility": np.array([float(volatility)], dtype=np.float32),
        }
        self.state["cost"] = self.commerce_platform.unit_cost.astype(np.float32)

        # extract metrics with safe defaults for incomplete simulation
        revenue_observed = float(result.get("revenue_observed", 0.0))
        agent_loss = float(result.get("agent_loss", 0.0))

        reward = (revenue_observed
                  - COI
                  - self.constraints.w_agent_loss * agent_loss
                  - self.constraints.w_volatility * volatility
                  - self.constraints.w_estimation_error)

        terminated = self.t >= self.constraints.episode_length
        info = {
            "t": self.t,
            "revenue_observed": revenue_observed,
            "revenue_oracle": float(result.get("revenue_oracle", revenue_observed)),
            "agent_loss": agent_loss,
            "ux_volatility": volatility,
            "look_to_book": float(result.get("look_to_book", 0.0)),
            "mean_sale_price": float(result.get("mean_sale_price", 0.0)),
            "true_human_purchases_total": float(result.get("true_human_purchases", 0.0)),
            "true_agent_purchases_total": float(result.get("true_agent_purchases", 0.0)),
            "coi": COI,
            "alpha_hat": diagnostics.get("alpha_hat", self.commerce_platform.alpha_hat),
            "mean_human_demand": float(np.mean(diagnostics.get("demand_human", np.zeros_like(new_prices)))),
            "mean_agent_demand": float(np.mean(diagnostics.get("demand_agent", np.zeros_like(new_prices)))),
        }
        if "delta_h_mean" in diagnostics:
            info.update(
                {
                    "delta_h_mean": diagnostics["delta_h_mean"],
                    "delta_a_mean": diagnostics["delta_a_mean"],
                    "prob_agent_mean": diagnostics["prob_agent_mean"],
                }
            )
        return self.state, float(reward), terminated, False, info


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from collections import defaultdict

    env = PHANTOMEnv(constraints=BusinessLogicConstraints())
    obs, _ = env.reset(seed=42)
    metrics = defaultdict(list)
    total_reward = 0.0
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        p_mean = float(np.mean(obs["elasticity"]["price"]))
        q_mean = float(np.mean(obs["elasticity"]["demand"]))
        p_std = float(np.std(obs["elasticity"]["price"]))

        metrics['t'].append(info['t'])
        metrics['price_mean'].append(p_mean)
        metrics['price_std'].append(p_std)
        metrics['demand_mean'].append(q_mean)
        metrics['revenue_observed'].append(info['revenue_observed'])
        metrics['revenue_oracle'].append(info['revenue_oracle'])
        metrics['agent_loss'].append(info['agent_loss'])
        metrics['ux_volatility'].append(info['ux_volatility'])
        metrics['look_to_book'].append(info['look_to_book'])
        metrics['reward'].append(reward)
        metrics['human_purchases'].append(info['true_human_purchases_total'])
        metrics['agent_purchases'].append(info['true_agent_purchases_total'])
        metrics['coi'].append(info.get('coi', 0.0))
        metrics['alpha_hat'].append(info.get('alpha_hat', env.commerce_platform.alpha_hat))
        metrics['mean_human_demand'].append(info.get('mean_human_demand', 0.0))
        metrics['mean_agent_demand'].append(info.get('mean_agent_demand', 0.0))
        metrics['delta_h_mean'].append(info.get('delta_h_mean', 0.0))
        metrics['delta_a_mean'].append(info.get('delta_a_mean', 0.0))
        metrics['prob_agent_mean'].append(info.get('prob_agent_mean', 0.0))

        if info['t'] % 20 == 0 or done:
            print(f"t={info['t']:03d} p={p_mean:6.2f}±{p_std:4.2f} q={q_mean:6.2f} "
                  f"rev={info['revenue_observed']:7.2f} oracle={info['revenue_oracle']:7.2f} "
                  f"loss={info['agent_loss']:6.2f} ux={info['ux_volatility']:.3f} "
                  f"coi={info.get('coi', 0.0):6.2f} alpha={info.get('alpha_hat', 0.0):4.2f} "
                  f"ltb={info['look_to_book']:5.2f} r={reward:7.2f}")

    print(f"total_reward={total_reward:.2f}")

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle('PHANTOM Environment Run', fontsize=14, fontweight='bold')

    plot_configs = [
        ('price_mean', 'Mean Price', 'Price'),
        ('demand_mean', 'Mean Demand (All)', 'Demand'),
        ('mean_human_demand', 'Mean Human Demand', 'Count'),
        ('mean_agent_demand', 'Mean Agent Demand', 'Count'),
        ('revenue_observed', 'Revenue (Observed)', 'Revenue'),
        ('agent_loss', 'Agent Loss (Oracle - Observed)', 'Loss'),
        ('coi', 'Cost of Information', 'COI'),
        ('alpha_hat', 'Estimated α̂', 'alpha'),
        ('ux_volatility', 'UX Volatility (Price Change)', 'Volatility'),
        ('look_to_book', 'Look-to-Book Ratio', 'Ratio'),
        ('reward', 'Step Reward', 'Reward'),
        ('prob_agent_mean', 'Avg Agent Probability', 'Probability'),
    ]

    for idx, (key, title, ylabel) in enumerate(plot_configs):
        ax = axes[idx // 4, idx % 4]
        ax.plot(metrics['t'], metrics[key], color='blue', alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('phantom_env_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved to phantom_env_comparison.png")
    plt.show()
