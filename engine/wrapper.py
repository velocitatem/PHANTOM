import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from .engine import Limbo, MarketEngine, PricingEngine


class PHANTOM(gym.Env):
    """Gymnasium wrapper for the Limbo pricing-market simulation. Platform sets prices, market responds with demand."""
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self,
                 n_products: int = 10,
                 alpha: float = 0.3,
                 N: int = 100,
                 price_bounds: tuple = (10.0, 150.0),
                 lambda_coi: float = 0.1,  # coi leakage penalty weight
                 render_mode: str = None):
        super().__init__()
        self.n_products = n_products
        self.price_bounds = price_bounds
        self.lambda_coi = lambda_coi
        self.render_mode = render_mode
        self.alpha = alpha
        self.N = N

        self.market = MarketEngine(alpha=alpha, N=N)
        self._platform_stub = PricingEngine()
        self._limbo = Limbo(self._platform_stub, self.market)

        # action: continuous prices for each product
        self.action_space = spaces.Box(
            low=price_bounds[0], high=price_bounds[1],
            shape=(n_products,), dtype=np.float32
        )
        # observation: demand estimate + previous prices
        self.observation_space = spaces.Dict({
            "demand": spaces.Box(low=0.0, high=100.0, shape=(n_products,), dtype=np.float32),
            "prices": spaces.Box(low=price_bounds[0], high=price_bounds[1], shape=(n_products,), dtype=np.float32),
        })

        self._prices = None
        self._demand = None
        self._step_count = 0
        self._demand_history = []
        self._price_history = []
        self._revenue_history = []
        self._fig = None
        self._gs = None
        self._dashboard_colors = {
            'bg': '#f5f0e8', 'panel': '#ebe3d5', 'accent': '#c9b99a',
            'text': '#3d3229', 'green': '#5c7a5c', 'red': '#8b4049',
            'blue': '#5a7384', 'orange': '#b87333', 'purple': '#7d6b7d'
        }

    def _get_obs(self) -> dict:
        demand_arr = np.array([self._demand.get(i, 0.0) for i in range(self.n_products)], dtype=np.float32)
        return {"demand": demand_arr, "prices": self._prices.astype(np.float32)}

    def _compute_reward(self, prices: np.ndarray, demand: dict) -> float:
        demand_arr = np.array([demand.get(i, 0.0) for i in range(self.n_products)])
        revenue = np.sum(prices * demand_arr)  # revenue = price * quantity proxy
        base_price = self.price_bounds[0]
        return float(revenue)# - self.lambda_coi * coi_leak)

    def _record_history(self):
        demand_arr = np.array([self._demand.get(i, 0.0) for i in range(self.n_products)])
        self._demand_history.append(demand_arr)
        self._price_history.append(self._prices.copy())
        revenue = np.sum(self._prices * demand_arr)
        self._revenue_history.append(revenue)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._prices = np.random.uniform(*self.price_bounds, size=self.n_products)
        self._demand = self.market.act(self._prices)
        self._step_count = 0
        self._demand_history, self._price_history, self._revenue_history = [], [], []
        self._record_history()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._prices = np.clip(action, *self.price_bounds)
        self._demand = self.market.act(self._prices)
        self._step_count += 1
        self._record_history()

        reward = self._compute_reward(self._prices, self._demand)
        terminated = self._step_count >= 100
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {"step": self._step_count}

    def _compute_elasticity(self) -> np.ndarray:
        """point elasticity: e = (dQ/dP) * (P/Q) estimated via finite differences, clipped to [-5, 5]"""
        if len(self._price_history) < 2:
            return np.zeros(self.n_products)
        p = np.array(self._price_history)
        q = np.array(self._demand_history)
        dp = np.diff(p, axis=0)
        dq = np.diff(q, axis=0)
        min_dp = 0.5  # ignore tiny price changes to avoid explosions
        valid = np.abs(dp) > min_dp
        with np.errstate(divide='ignore', invalid='ignore'):
            elasticity = np.where(valid, (dq / dp) * (p[:-1] / np.maximum(q[:-1], 1.0)), 0.0)
            elasticity = np.clip(elasticity, -5.0, 5.0)
            elasticity = np.nan_to_num(elasticity, nan=0.0)
        return np.mean(elasticity, axis=0) if len(elasticity) > 0 else np.zeros(self.n_products)

    def _style_axis(self, ax, title: str = None, xlabel: str = None, ylabel: str = None):
        c = self._dashboard_colors
        ax.set_facecolor(c['panel'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(c['accent']); ax.spines['left'].set_color(c['accent'])
        ax.tick_params(colors=c['text'], labelsize=8)
        if title: ax.set_title(title, color=c['text'], fontsize=11, fontweight='bold', pad=8)
        if xlabel: ax.set_xlabel(xlabel, color=c['text'], fontsize=9)
        if ylabel: ax.set_ylabel(ylabel, color=c['text'], fontsize=9)

    def render(self):
        if self.render_mode == "human":
            c = self._dashboard_colors
            if self._fig is None:
                plt.ion()
                self._fig = plt.figure(figsize=(14, 10), facecolor=c['bg'])
                self._gs = GridSpec(3, 3, figure=self._fig, hspace=0.35, wspace=0.3,
                                    left=0.07, right=0.95, top=0.92, bottom=0.08)
                plt.show(block=False)

            self._fig.clear()
            self._fig.suptitle(f'PHANTOM  Market Dynamics  [t={self._step_count}, α={self.alpha:.2f}]',
                              color=c['text'], fontsize=14, fontweight='bold')

            demand_mat = np.array(self._demand_history).T
            price_mat = np.array(self._price_history).T
            elasticity = self._compute_elasticity()
            cmap = mcolors.LinearSegmentedColormap.from_list('phantom', [c['bg'], c['blue'], c['green']])
            cmap_div = mcolors.LinearSegmentedColormap.from_list('elast', [c['red'], c['bg'], c['blue']])

            # price-demand elasticity scatter (all historical data points)
            ax_elast = self._fig.add_subplot(self._gs[0, 0])
            prices_flat = np.array(self._price_history).flatten()
            demands_flat = np.array(self._demand_history).flatten()
            product_ids = np.tile(np.arange(self.n_products), len(self._price_history))
            scatter = ax_elast.scatter(prices_flat, demands_flat, c=product_ids, cmap='plasma',
                                       alpha=0.6, s=15, edgecolors='none')
            if len(prices_flat) > 1:  # fit regression line
                z = np.polyfit(prices_flat, demands_flat, 1)
                p_line = np.linspace(prices_flat.min(), prices_flat.max(), 50)
                ax_elast.plot(p_line, np.polyval(z, p_line), '--', color=c['red'], lw=1.5, alpha=0.8)
            self._style_axis(ax_elast, "Price-Demand Relationship", "Price ($)", "Demand")

            # elasticity coefficients bar
            ax_ebar = self._fig.add_subplot(self._gs[0, 1])
            colors_e = [c['red'] if e < -0.5 else c['blue'] if e > 0.5 else c['accent'] for e in elasticity]
            ax_ebar.barh(range(self.n_products), elasticity, color=colors_e, alpha=0.8, edgecolor=c['bg'])
            ax_ebar.axvline(0, color=c['text'], lw=0.8, alpha=0.5)
            ax_ebar.axvline(-1, color=c['red'], lw=1, ls='--', alpha=0.5)  # unit elastic reference
            ax_ebar.set_yticks(range(self.n_products))
            ax_ebar.set_yticklabels([f'P{i}' for i in range(self.n_products)], fontsize=7)
            self._style_axis(ax_ebar, "Price Elasticity ε", "ε = (ΔQ/ΔP)·(P/Q)", None)

            # session composition pie
            ax_pie = self._fig.add_subplot(self._gs[0, 2])
            n_humans, n_agents = self.market.Nhumans, self.market.Nagents
            ax_pie.set_facecolor(c['panel'])
            wedges, _ = ax_pie.pie([n_humans, n_agents], colors=[c['blue'], c['red']],
                                   startangle=90, wedgeprops={'linewidth': 2, 'edgecolor': c['bg']})
            ax_pie.legend(wedges, [f'H ({n_humans})', f'A ({n_agents})'],
                          loc='lower center', fontsize=8, frameon=False,
                          labelcolor=c['text'], bbox_to_anchor=(0.5, -0.05))
            ax_pie.set_title("Session Mix", color=c['text'], fontsize=11, fontweight='bold')

            # price heatmap over time
            ax_pheat = self._fig.add_subplot(self._gs[1, :2])
            im_p = ax_pheat.imshow(price_mat, aspect='auto', cmap='viridis', origin='lower')
            self._style_axis(ax_pheat, "Price Heatmap P(product, t)", "Step", "Product")
            cbar_p = self._fig.colorbar(im_p, ax=ax_pheat, fraction=0.03, pad=0.02)
            cbar_p.ax.tick_params(colors=c['text'], labelsize=7)
            cbar_p.set_label('$', color=c['text'], fontsize=8)

            # demand heatmap over time
            ax_dheat = self._fig.add_subplot(self._gs[1, 2])
            im_d = ax_dheat.imshow(demand_mat, aspect='auto', cmap=cmap, origin='lower')
            self._style_axis(ax_dheat, "Demand Q(product, t)", "Step", None)
            cbar_d = self._fig.colorbar(im_d, ax=ax_dheat, fraction=0.046, pad=0.02)
            cbar_d.ax.tick_params(colors=c['text'], labelsize=7)

            # cross-correlation matrix (price-demand covariance per product)
            ax_corr = self._fig.add_subplot(self._gs[2, 0])
            if len(self._price_history) > 2:
                corr_mat = np.corrcoef(price_mat, demand_mat)[:self.n_products, self.n_products:]
                im_corr = ax_corr.imshow(corr_mat, cmap=cmap_div, vmin=-1, vmax=1, aspect='auto')
                ax_corr.set_xticks(range(self.n_products))
                ax_corr.set_yticks(range(self.n_products))
                ax_corr.set_xticklabels([f'Q{i}' for i in range(self.n_products)], fontsize=6)
                ax_corr.set_yticklabels([f'P{i}' for i in range(self.n_products)], fontsize=6)
                cbar_c = self._fig.colorbar(im_corr, ax=ax_corr, fraction=0.046, pad=0.02)
                cbar_c.ax.tick_params(colors=c['text'], labelsize=7)
            self._style_axis(ax_corr, "Price-Demand Correlation", None, None)

            # revenue curve with demand dispersion (std dev shows concentration)
            ax_rev = self._fig.add_subplot(self._gs[2, 1:])
            n_steps = len(self._revenue_history)
            demand_std = [np.std(d) for d in self._demand_history]
            ax_rev.fill_between(range(n_steps), self._revenue_history, alpha=0.3, color=c['green'])
            ax_rev.plot(self._revenue_history, color=c['green'], linewidth=2, label='Revenue')
            ax_rev.set_xlim(0, max(n_steps, 1))
            ax_rev.set_ylim(0, max(self._revenue_history) * 1.1 if self._revenue_history else 1)
            ax2 = ax_rev.twinx()
            ax2.plot(range(n_steps), demand_std, color=c['blue'], linewidth=2, ls='-', alpha=0.9, label='σ(Demand)')
            d_min, d_max = min(demand_std), max(demand_std)
            margin = (d_max - d_min) * 0.2 if d_max > d_min else 0.5
            ax2.set_ylim(max(0, d_min - margin), d_max + margin)
            ax2.tick_params(axis='y', colors=c['blue'], labelsize=8)
            ax2.spines['right'].set_color(c['blue'])
            ax2.set_ylabel('Demand σ', color=c['blue'], fontsize=9)
            self._style_axis(ax_rev, "Revenue & Demand Dispersion", "Step", "Revenue ($)")
            ax_rev.legend(loc='upper left', fontsize=7, frameon=False, labelcolor=c['text'])
            ax2.legend(loc='upper right', fontsize=7, frameon=False, labelcolor=c['text'])

            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
            plt.pause(0.05)

        elif self.render_mode == "ansi":
            return f"step={self._step_count}, prices={self._prices}, demand={self._demand}"
        return None

    def close(self):
        if self._fig: plt.close(self._fig)
        self._fig = None


if __name__ == "__main__":
    env = PHANTOM(n_products=15, alpha=0.3, N=100, render_mode="human")
    obs, _ = env.reset()
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        env.render()
        if term: break
    env.close()
