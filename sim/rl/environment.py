from sys import intern
import gymnasium as gym
from gymnasium import spaces
from matplotlib import interactive
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
    human_price_elasticity: float = -1.2 # assumptions here
    agent_price_elasticity: float = -0.6
    w_agent_loss: float = 1.0
    w_volatility: float = 5.0
    w_estimation_error: float = 0.25
    seed: int = 7


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

class CommercePlatform:
    """
    This is just an extension of the state management for the environment, it does not implement anything dynamic just helps us simulate demand.
    """
    def __init__(self,
                 product_catelogue_size: int,
                 max_price: float,
                 min_price: float,
                 constraints: BusinessLogicConstraints):
        self.product_catelogue_size = product_catelogue_size
        self.product_supply = np.random.uniform(low=10, high=50, size=(self.product_catelogue_size,))
        self.max_price = max_price
        self.min_price = min_price
        self.constraints = constraints
        self.simulation_history: List[Dict[str, Any]] = []
        self._rng = np.random.default_rng(constraints.seed)
        self._last_interaction_df: pd.DataFrame = pd.DataFrame()


    def setup_true_demand(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        # ground truth purchase propensities
        p = np.clip(prices, self.min_price, self.max_price)
        pn = p / self.max_price
        human_prob = self.constraints.base_human_demand * (pn ** self.constraints.human_price_elasticity)
        agent_prob = self.constraints.base_agent_demand * (pn ** self.constraints.agent_price_elasticity)
        return {
            "human_purchase_prob": np.clip(human_prob, 0.0, 0.95),
            "agent_purchase_prob": np.clip(agent_prob, 0.0, 0.95)
        }

    def _load_behavioral_profile(actor : str, demand_forcing):
        """
        This returns a markov chain with average weights which we get from interaction data of our experiments.
        This defines transition probabilities between different events:
        search -> view_item_price_binN: 0.7
        view_item_price_binN -> add_to_cart: 0.2
        we also must reweight with the demand_forcing vector or purchase probabilities per-product
        """


    def _simulate_sessions(self, base_prices: np.ndarray) -> pd.DataFrame:
        demand = self.setup_true_demand(base_prices)
        human_pprob = demand["human_purchase_prob"]
        agent_pprob = demand["agent_purchase_prob"]
        events: List[Dict[str, Any]] = []
        T = self.constraints.sessions_per_step
        n_agent_sessions = int(round(T * self.constraints.agent_share))
        n_human_sessions = T - n_agent_sessions
        n_agent_ids = max(1, n_agent_sessions // 2)
        session_map = {
            'humans': n_human_sessions,
            'agents': n_agent_ids
        }
        pprob_map = {
            'humans': human_pprob,
            'agents': agent_pprob
        }
        joint_events = []
        for actor, n_sessions in session_map.items():
            bp = _load_behavioral_profile(actor, pprob_map[actor])
            counter = 0
            events = []
            while counter < n_sessions:
                session_events = []
                while len(session_events) == 0 or session_events[-1]['action'] == 'checkout':
                    interaction_event = bp.sample(self._rng)
                    interaction_event['session_id'] = f'{actor}_{counter:06d}'
                    # TODO any other assignments
                    session_events.append(interaction_event)
                events.extend(session_events)
                counter += 1
            joint_events.extend(events)

        return pd.DataFrame(joint_events)

    def compute_interaction_features(self, interaction_df: pd.DataFrame) -> Dict[str, float]:
        if interaction_df.empty:
            return {"mean_sale_price": 0.0, "look_to_book": 0.0}
        purchases = interaction_df[interaction_df["action"] == "purchase"]
        mean_sale_price = float(purchases["price_paid"].mean()) if not purchases.empty else 0.0
        views = float((interaction_df["action"] == "view").sum())
        buys = float((interaction_df["action"] == "purchase").sum())
        return {"mean_sale_price": mean_sale_price, "look_to_book": float(views / (buys + 1e-6))}

    def _session_feature_table(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: adapt this
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

    def get_interaction_data(self) -> np.ndarray:
        if self._last_interaction_df.empty:
            return np.array([], dtype=object)
        return self._last_interaction_df.to_dict(orient="records")


class PHANTOMEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, constraints):
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
            # TODO: define more features that we compute from the interaction data
        })
        self.commerce_platform = CommercePlatform(
            product_catelogue_size=self.constraints.product_catelogue_size,
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

        self.state["elasticity"]["price"] = new_prices
        # TODO: use the commerce platform to simulate sessions
        interactions_df = self.commerce_platform._simulate_sessions(new_prices)
        result = self.commerce_platform.compute_interaction_features(interactions_df)
        # TODO: implement COI computation to use in reward
        COI = 0.0

        volatility = 0.0 if self._prev_prices is None else \
            float(np.mean(np.abs((new_prices - self._prev_prices) / (self._prev_prices + 1e-6))))
        self._prev_prices = new_prices.copy()

        revenue_observed = float(result["revenue_observed"])
        agent_loss = float(result["agent_loss"])

        reward = (revenue_observed
                  - COI
                  - self.constraints.w_agent_loss * agent_loss
                  - self.constraints.w_volatility * volatility
                  - self.constraints.w_estimation_error
                  )

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
