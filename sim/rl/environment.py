import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
import pandas as pd
from typing import Callable, Optional, Dict, Any, List

# "learner"  agent learning to optimize pricing
# "agent"  part of environment creating demand signals that learner processes

@dataclass
class BusinessLogicConstraints():
    max_price_adjustment: float = 0.30
    system_max_price: float = 500.0
    system_min_price: float = 1.0
    product_catelogue_size: int = 100
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
    human_price_elasticity: float = -1.2
    agent_price_elasticity: float = -0.6
    w_agent_loss: float = 1.0
    w_volatility: float = 5.0
    w_estimation_error: float = 0.25
    seed: int = 7


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simple_agent_detector(session_df: pd.DataFrame) -> pd.Series:
    # baseline heuristic: high velocity + low conversion
    v = session_df.get("interaction_velocity", pd.Series(0.0, index=session_df.index))
    cr = session_df.get("conversion_rate", pd.Series(0.0, index=session_df.index))
    total = session_df.get("total_interactions", pd.Series(0, index=session_df.index))
    return (total >= 12) & (v >= 0.20) & (cr <= 0.01)


class CommercePlatform:
    def __init__(self, product_catelogue_size: int, max_price: float, min_price: float,
                 constraints: BusinessLogicConstraints, agent_detector: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
                 use_defense: bool = False):
        self.product_catelogue_size = product_catelogue_size
        self.max_price = max_price
        self.min_price = min_price
        self.constraints = constraints
        self.use_defense = use_defense
        self.agent_detector = agent_detector
        self.simulation_history: List[Dict[str, Any]] = []
        self._rng = np.random.default_rng(constraints.seed)
        self._popularity = self._rng.lognormal(mean=0.0, sigma=0.6, size=self.product_catelogue_size)
        self._popularity = self._popularity / (self._popularity.mean() + 1e-12)
        self._last_interaction_df: pd.DataFrame = pd.DataFrame()

    def setup_true_demand(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        # ground truth purchase propensities
        p = np.clip(prices, self.min_price, self.max_price)
        pn = p / self.max_price
        human_prob = self.constraints.base_human_demand * (pn ** self.constraints.human_price_elasticity)
        agent_prob = self.constraints.base_agent_demand * (pn ** self.constraints.agent_price_elasticity)
        return {
            "human_purchase_prob": np.clip(human_prob * self._popularity, 0.0, 0.95),
            "agent_purchase_prob": np.clip(agent_prob * self._popularity, 0.0, 0.95)
        }

    def _session_markup_multiplier(self, signal_score: float) -> float:
        # session-based COI markup based on demand signal expression
        x = (signal_score - self.constraints.coi_threshold) / max(self.constraints.coi_sigmoid_temp, 1e-6)
        return 1.0 + self.constraints.coi_strength * float(_sigmoid(np.array([x]))[0])

    def _simulate_sessions(self, base_prices: np.ndarray) -> pd.DataFrame:
        demand = self.setup_true_demand(base_prices)
        human_pprob = demand["human_purchase_prob"]
        agent_pprob = demand["agent_purchase_prob"]
        events: List[Dict[str, Any]] = []
        T = self.constraints.sessions_per_step
        n_agent_sessions = int(round(T * self.constraints.agent_share))
        n_human_sessions = T - n_agent_sessions

        # human sessions: normal browse with possible purchase
        for s in range(n_human_sessions):
            session_id = f"h_{len(events)}_{s}"
            k = int(self._rng.integers(1, 4))
            prod_ids = self._rng.choice(self.product_catelogue_size, size=k, replace=False)
            t = 0.0
            inter_times = self._rng.gamma(shape=2.0, scale=3.0, size=3 * k)
            signal_score = 0.0
            purchased_any = False

            for i, pid in enumerate(prod_ids):
                t += float(inter_times[i])
                price_shown = float(base_prices[pid])
                events.append({
                    "session_id": session_id, "actor": "human", "agent_id": None, "product_id": int(pid),
                    "action": "view", "t": t, "price_shown": price_shown, "is_purchase": 0,
                    "price_paid": 0.0, "oracle_price_paid": 0.0, "signal_score": 0.0,
                })
                signal_score += 1.0

                if self._rng.random() < 0.35:
                    t += float(inter_times[i + k])
                    events.append({
                        "session_id": session_id, "actor": "human", "agent_id": None, "product_id": int(pid),
                        "action": "cart", "t": t, "price_shown": price_shown, "is_purchase": 0,
                        "price_paid": 0.0, "oracle_price_paid": 0.0, "signal_score": 0.0,
                    })
                    signal_score += 2.0

                if (not purchased_any) and (self._rng.random() < float(human_pprob[pid])):
                    t += float(inter_times[i + 2 * k])
                    mult = self._session_markup_multiplier(signal_score)
                    price_paid = float(np.clip(base_prices[pid] * mult, self.min_price, self.max_price))
                    events.append({
                        "session_id": session_id, "actor": "human", "agent_id": None, "product_id": int(pid),
                        "action": "purchase", "t": t, "price_shown": float(base_prices[pid]), "is_purchase": 1,
                        "price_paid": price_paid, "oracle_price_paid": price_paid, "signal_score": signal_score,
                    })
                    purchased_any = True

        # agent sessions: split recon/purchase to circumvent COI
        n_agent_ids = max(1, n_agent_sessions // 2)
        for a in range(n_agent_ids):
            agent_id = f"a_{a}"
            recon_session_id = f"{agent_id}_recon"
            t = 0.0
            n_views = int(self._rng.poisson(lam=8) * self.constraints.agent_recon_multiplier) + 5
            inter_times = self._rng.gamma(shape=2.0, scale=0.6, size=max(n_views, 1))
            prod_ids = self._rng.integers(0, self.product_catelogue_size, size=n_views)
            recon_signal = 0.0

            for i, pid in enumerate(prod_ids):
                t += float(inter_times[i])
                events.append({
                    "session_id": recon_session_id, "actor": "agent", "agent_id": agent_id, "product_id": int(pid),
                    "action": "view", "t": t, "price_shown": float(base_prices[pid]), "is_purchase": 0,
                    "price_paid": 0.0, "oracle_price_paid": 0.0, "signal_score": 0.0,
                })
                recon_signal += 1.0

            # clean purchase session with minimal interactions
            if self._rng.random() < self.constraints.agent_purchase_probability:
                purchase_session_id = f"{agent_id}_clean"
                pid = int(self._rng.integers(0, self.product_catelogue_size))
                t2 = 0.0
                clean_signal = 0.0
                t2 += float(self._rng.gamma(shape=2.0, scale=0.7))
                events.append({
                    "session_id": purchase_session_id, "actor": "agent", "agent_id": agent_id, "product_id": pid,
                    "action": "view", "t": t2, "price_shown": float(base_prices[pid]), "is_purchase": 0,
                    "price_paid": 0.0, "oracle_price_paid": 0.0, "signal_score": 0.0,
                })
                clean_signal += 1.0

                if self._rng.random() < float(agent_pprob[pid]):
                    t2 += float(self._rng.gamma(shape=2.0, scale=0.7))
                    obs_mult = self._session_markup_multiplier(clean_signal)
                    obs_paid = float(np.clip(base_prices[pid] * obs_mult, self.min_price, self.max_price))
                    oracle_mult = self._session_markup_multiplier(recon_signal)  # oracle links recon->purchase
                    oracle_paid = float(np.clip(base_prices[pid] * oracle_mult, self.min_price, self.max_price))
                    events.append({
                        "session_id": purchase_session_id, "actor": "agent", "agent_id": agent_id, "product_id": pid,
                        "action": "purchase", "t": t2, "price_shown": float(base_prices[pid]), "is_purchase": 1,
                        "price_paid": obs_paid, "oracle_price_paid": oracle_paid, "signal_score": clean_signal,
                    })

        return pd.DataFrame(events)

    def compute_interaction_features(self, interaction_df: pd.DataFrame) -> Dict[str, float]:
        if interaction_df.empty:
            return {"mean_sale_price": 0.0, "look_to_book": 0.0}
        purchases = interaction_df[interaction_df["action"] == "purchase"]
        mean_sale_price = float(purchases["price_paid"].mean()) if not purchases.empty else 0.0
        views = float((interaction_df["action"] == "view").sum())
        buys = float((interaction_df["action"] == "purchase").sum())
        return {"mean_sale_price": mean_sale_price, "look_to_book": float(views / (buys + 1e-6))}

    def _session_feature_table(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        g = df.groupby("session_id", sort=False)
        session_duration = g["t"].max() - g["t"].min()
        total_interactions = g.size()
        avg_time_between = g["t"].apply(lambda x: float(np.diff(np.sort(x.to_numpy())).mean()) if len(x) > 1 else 0.0)
        interaction_velocity = total_interactions / (session_duration + 1e-6)
        views = g.apply(lambda x: int((x["action"] == "view").sum()), include_groups=False)
        cart_adds = g.apply(lambda x: int((x["action"] == "cart").sum()), include_groups=False)
        purchases = g.apply(lambda x: int((x["action"] == "purchase").sum()), include_groups=False)
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

    def demand_estimate(self, interaction_df: pd.DataFrame, exclude_sessions: Optional[pd.Series] = None) -> np.ndarray:
        # proxy demand from weighted interaction events
        if interaction_df.empty:
            return np.zeros(self.product_catelogue_size, dtype=np.float32)
        df = interaction_df
        if exclude_sessions is not None:
            bad_sessions = set(exclude_sessions.loc[exclude_sessions].index)
            df = df[~df["session_id"].isin(bad_sessions)]
        weights = {"view": 0.15, "cart": 0.75, "purchase": 2.5}
        w = df["action"].map(weights).fillna(0.0).to_numpy(dtype=float)
        prod = df["product_id"].to_numpy(dtype=int)
        q_hat = np.zeros(self.product_catelogue_size, dtype=float)
        np.add.at(q_hat, prod, w)
        return q_hat.astype(np.float32)

    def run_pricing_simulation(self, prices: np.ndarray) -> Dict[str, Any]:
        interaction_df = self._simulate_sessions(prices)
        self._last_interaction_df = interaction_df
        session_df = self._session_feature_table(interaction_df)

        predicted_agent_sessions = None
        if (self.use_defense and self.agent_detector is not None and not session_df.empty):
            predicted_agent_sessions = self.agent_detector(session_df.set_index("session_id"))

        q_hat_naive = self.demand_estimate(interaction_df, exclude_sessions=None)
        q_hat_defended = self.demand_estimate(interaction_df, exclude_sessions=predicted_agent_sessions) \
            if predicted_agent_sessions is not None else q_hat_naive.copy()

        true_human = np.zeros(self.product_catelogue_size, dtype=float)
        true_agent = np.zeros(self.product_catelogue_size, dtype=float)
        if not interaction_df.empty:
            purchases = interaction_df[interaction_df["action"] == "purchase"]
            if not purchases.empty:
                for _, r in purchases.iterrows():
                    if r["actor"] == "human":
                        true_human[int(r["product_id"])] += 1.0
                    else:
                        true_agent[int(r["product_id"])] += 1.0

        revenue_observed = float(interaction_df["price_paid"].sum()) if not interaction_df.empty else 0.0
        revenue_oracle = float(interaction_df["oracle_price_paid"].sum()) if not interaction_df.empty else 0.0
        agent_loss = max(0.0, revenue_oracle - revenue_observed)

        eps = 1e-6
        internal_error_naive = np.abs(true_human - q_hat_naive) / (true_human + eps)
        internal_error_def = np.abs(true_human - q_hat_defended) / (true_human + eps)
        interaction_features = self.compute_interaction_features(interaction_df)

        summary = {
            "prices": prices.copy(),
            "interaction_df": interaction_df,
            "session_df": session_df,
            "q_hat_naive": q_hat_naive,
            "q_hat_defended": q_hat_defended,
            "true_human_demand": true_human.astype(np.float32),
            "true_agent_purchases": true_agent.astype(np.float32),
            "internal_error_naive": internal_error_naive.astype(np.float32),
            "internal_error_defended": internal_error_def.astype(np.float32),
            "interaction_features": interaction_features,
            "revenue_observed": revenue_observed,
            "revenue_oracle": revenue_oracle,
            "agent_loss": agent_loss,
            "predicted_agent_sessions": predicted_agent_sessions,
        }
        self.simulation_history.append(summary)
        return summary

    def get_interaction_data(self) -> np.ndarray:
        if self._last_interaction_df.empty:
            return np.array([], dtype=object)
        return self._last_interaction_df.to_dict(orient="records")


class PHANTOMEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, use_defense: bool = False):
        super().__init__()
        self.constraints = BusinessLogicConstraints()
        self.action_space = spaces.Box(low=-self.constraints.max_price_adjustment,
                                       high=self.constraints.max_price_adjustment,
                                       shape=(self.constraints.product_catelogue_size,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "elasticity": spaces.Dict({
                "price": spaces.Box(
                    low=np.full((self.constraints.product_catelogue_size,), self.constraints.system_min_price, dtype=np.float32),
                    high=np.full((self.constraints.product_catelogue_size,), self.constraints.system_max_price, dtype=np.float32),
                    dtype=np.float32),
                "demand": spaces.Box(
                    low=np.zeros((self.constraints.product_catelogue_size,), dtype=np.float32),
                    high=np.full((self.constraints.product_catelogue_size,), 1e6, dtype=np.float32),
                    dtype=np.float32),
            })
        })
        self.commerce_platform = CommercePlatform(
            product_catelogue_size=self.constraints.product_catelogue_size,
            max_price=self.constraints.system_max_price,
            min_price=self.constraints.system_min_price,
            constraints=self.constraints,
            agent_detector=simple_agent_detector,
            use_defense=use_defense)
        self._rng = np.random.default_rng(self.constraints.seed)
        self.t = 0
        self._prev_prices: Optional[np.ndarray] = None
        self.state: Dict[str, Any] = {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.commerce_platform._rng = np.random.default_rng(seed)
        self.t = 0
        init_prices = self._rng.uniform(low=60.0, high=140.0, size=(self.constraints.product_catelogue_size,)).astype(np.float32)
        self._prev_prices = init_prices.copy()
        self.state = {
            "elasticity": {
                "price": init_prices,
                "demand": np.zeros((self.constraints.product_catelogue_size,), dtype=np.float32),
            }
        }
        return self.state, {}

    def step(self, action: np.ndarray):
        self.t += 1
        base_prices = self.state["elasticity"]["price"].astype(np.float32)
        new_prices = np.clip(base_prices * (1.0 + action.astype(np.float32)),
                           self.constraints.system_min_price,
                           self.constraints.system_max_price).astype(np.float32)
        result = self.commerce_platform.run_pricing_simulation(new_prices)

        if self.commerce_platform.use_defense:
            demand_est = result["q_hat_defended"]
            internal_err = result["internal_error_defended"]
        else:
            demand_est = result["q_hat_naive"]
            internal_err = result["internal_error_naive"]

        self.state["elasticity"]["price"] = new_prices
        self.state["elasticity"]["demand"] = demand_est

        volatility = 0.0 if self._prev_prices is None else \
            float(np.mean(np.abs((new_prices - self._prev_prices) / (self._prev_prices + 1e-6))))
        self._prev_prices = new_prices.copy()

        revenue_observed = float(result["revenue_observed"])
        agent_loss = float(result["agent_loss"])
        err_mean = float(np.mean(internal_err))

        reward = (revenue_observed
                 - self.constraints.w_agent_loss * agent_loss
                 - self.constraints.w_volatility * volatility
                 - self.constraints.w_estimation_error * err_mean)

        terminated = self.t >= self.constraints.episode_length
        info = {
            "t": self.t,
            "revenue_observed": revenue_observed,
            "revenue_oracle": float(result["revenue_oracle"]),
            "agent_loss": agent_loss,
            "ux_volatility": volatility,
            "mean_internal_error": err_mean,
            "look_to_book": float(result["interaction_features"].get("look_to_book", 0.0)),
            "mean_sale_price": float(result["interaction_features"].get("mean_sale_price", 0.0)),
            "true_human_purchases_total": float(np.sum(result["true_human_demand"])),
            "true_agent_purchases_total": float(np.sum(result["true_agent_purchases"])),
        }
        return self.state, float(reward), terminated, False, info


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from collections import defaultdict

    runs = {}
    for use_defense in (False, True):
        env = PHANTOMEnv(use_defense=use_defense)
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

            if info['t'] % 20 == 0 or done:
                print(f"defense={'ON ' if use_defense else 'OFF'} t={info['t']:03d} p={p_mean:6.2f}±{p_std:4.2f} "
                      f"q={q_mean:6.2f} rev={info['revenue_observed']:7.2f} oracle={info['revenue_oracle']:7.2f} "
                      f"loss={info['agent_loss']:6.2f} ux={info['ux_volatility']:.3f} "
                      f"ltb={info['look_to_book']:5.2f} r={reward:7.2f}")

        runs[use_defense] = metrics
        print(f"defense={'ON ' if use_defense else 'OFF'} total_reward={total_reward:.2f}\n")

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('PHANTOM Environment: Defense OFF vs ON', fontsize=14, fontweight='bold')

    plot_configs = [
        ('price_mean', 'Mean Price', 'Price'),
        ('demand_mean', 'Mean Demand Estimate', 'Demand'),
        ('revenue_observed', 'Revenue (Observed)', 'Revenue'),
        ('agent_loss', 'Agent Loss (Oracle - Observed)', 'Loss'),
        ('ux_volatility', 'UX Volatility (Price Change)', 'Volatility'),
        ('look_to_book', 'Look-to-Book Ratio', 'Ratio'),
        ('reward', 'Step Reward', 'Reward'),
        ('human_purchases', 'Human Purchases', 'Count'),
        ('agent_purchases', 'Agent Purchases', 'Count'),
    ]

    for idx, (key, title, ylabel) in enumerate(plot_configs):
        ax = axes[idx // 3, idx % 3]
        for use_defense, label, color in [(False, 'No Defense', 'red'), (True, 'With Defense', 'blue')]:
            m = runs[use_defense]
            ax.plot(m['t'], m[key], label=label, color=color, alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('phantom_env_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved to phantom_env_comparison.png")
    plt.show()
