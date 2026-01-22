import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
import pandas as pd
from types import SimpleNamespace
from typing import Optional, Dict, Any, List, Tuple

from lib.separability import load_artifacts, score_session, estimate_alpha
from sim.rl.behavior_loader.models import AgentBehaviorModel, BehaviorModel

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
    episode_length: int = 200
    sessions_per_step: int = 250
    agent_share: float = 0.25
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
    "view_item_page": "/products",
    "learn_more_about_item": "/products/details",
    "add_item_to_cart": "/cart",
    "purchase_complete": "/checkout",
    "session_end": "/checkout/success",
}


class BehavioralProfile:
    """Synthetic Markov profile used to generate interaction sessions."""
    # TODO: a lot of this is duplicated from models.py - refactor to share code better

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
        # base transition structure (human default)
        self.transitions : Dict[str, Dict[str, float]];

        model = AgentBehaviorModel(agent_dir) if actor == "agents" else BehaviorModel(human_dir)
        self.transitions = # TODO similarly to model.build_MDP_event_transitions() in models.py buidl the dict

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
        """Generate a single session trajectory."""
        # TODO: this is similar to the sample trajectory method in models.
        # we also have to respect business constraints which constrain the lipshitz continuity of the transitions and prices
        # we must apply constraints on purcahses not to let the platform offer prices under the cost of a productid

        events: List[Dict[str, Any]] = []
        feature_events: List[SimpleNamespace] = []
        state = "session_start"
        t = 0.0
        product_idx = int(rng.integers(0, len(prices)))
        product_id = f"product-{product_idx:04d}"

        while state != "session_end" and len(events) < 40:
            if state != "session_start":
                price = float(prices[product_idx])
                row = {
                    "session_id": session_id,
                    "actor": "agent" if self.actor == "agents" else "human",
                    "eventName": state,
                    "product_idx": product_idx,
                    "productId": product_id,
                    "price_offered": price,
                    "price_paid": 0.0,
                    "page": EVENT_PAGE_MAP.get(state, "/"),
                    "ts": t,
                    "unit_cost": float(unit_cost[product_idx]),
                    "base_price": float(prices[product_idx]),
                }
                if state == "purchase_complete":
                    noise = float(rng.normal(0.0, 0.015))
                    row["price_paid"] = max(price * (1.0 + noise), row["unit_cost"])
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
            dwell = max(0.5, rng.gamma(shape=2.0, scale=1.0)) # TODO: should use params from the profile data
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
        coi = 0.0
        if not human_prices.empty and not human_costs.empty:
            # of the purchased items, what is the margin between the price and cost
            # TODO: this should take into account the expected price we could have charged also
            coi = float(np.maximum(0.0, human_prices.mean() - human_costs.mean()))

        return {
            "revenue_observed": revenue_observed,
            "revenue_oracle": revenue_oracle,
            "agent_loss": agent_loss,
            "true_human_purchases": true_human,
            "true_agent_purchases": true_agent,
            "mean_sale_price": mean_sale_price,
            "look_to_book": look_to_book,
            "coi": coi,
        }

    def _session_feature_table(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: adapt this
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
        conversion_rate = purchases / (views + 1e-6)
        is_agent = g["actor"].apply(lambda s: bool((s == "agent").any()), include_groups=False)

        return pd.DataFrame({
            "session_duration_sec": session_duration.astype(float),
            "avg_time_between_events": avg_time_between.astype(float),
            "total_interactions": total_interactions.astype(int),
            "interaction_velocity": interaction_velocity.astype(float),
            "item_views": views.astype(int),
            "cart_adds": cart_adds.astype(int),
            "purchases": purchases.astype(int),
            "conversion_rate": conversion_rate.astype(float),
            "is_agent": is_agent.astype(bool),
        }).reset_index()

    def get_interaction_data(self) -> np.ndarray:
        if self._last_interaction_df.empty:
            return np.array([], dtype=object)
        return self._last_interaction_df.to_dict(orient="records")


class PHANTOMEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, constraints: Optional[BusinessLogicConstraints] = None):
        super().__init__()
        self.constraints = constraints if isinstance(constraints, BusinessLogicConstraints) else BusinessLogicConstraints()
        self.action_space = spaces.Box(low=-self.constraints.max_price_adjustment,
                                       high=self.constraints.max_price_adjustment,
                                       shape=(self.constraints.product_catalogue_size,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "elasticity": spaces.Dict({
                "price": spaces.Box(
                    low=np.full((self.constraints.product_catalogue_size,), self.constraints.system_min_price, dtype=np.float32),
                    high=np.full((self.constraints.product_catalogue_size,), self.constraints.system_max_price, dtype=np.float32),
                    dtype=np.float32),
                "demand": spaces.Box(
                    low=np.zeros((self.constraints.product_catalogue_size,), dtype=np.float32),
                    high=np.full((self.constraints.product_catalogue_size,), 1e6, dtype=np.float32),
                    dtype=np.float32),
            })
            # TODO: define more features that we compute from the interaction data
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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.commerce_platform._rng = np.random.default_rng(seed)
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
            }
        }
        return self.state, {}

    def step(self, action: np.ndarray):
        self.t += 1
        base_prices = self.state["elasticity"]["price"].astype(np.float32)
        new_prices = np.clip(base_prices * (1.0 + action.astype(np.float32)),
                           self.constraints.system_min_price,
                           self.constraints.system_max_price).astype(np.float32)

        self.state["elasticity"]["price"] = new_prices
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
