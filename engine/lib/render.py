"""rendering logic for PHANTOM environment dashboard"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def style_axis(ax, title: str = None, xlabel: str = None, ylabel: str = None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title: ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)


class DashboardRenderer:
    """stateful renderer for PHANTOM market dynamics visualization"""

    def __init__(self):
        self.fig = None
        self.gs = None

    def render(self, env) -> None:
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(14, 10))
            self.gs = GridSpec(3, 3, figure=self.fig, hspace=0.35, wspace=0.3,
                               left=0.07, right=0.95, top=0.92, bottom=0.08)
            plt.show(block=False)

        self.fig.clear()
        self.fig.suptitle(f'PHANTOM  Market Dynamics  [t={env._step_count}, a={env.alpha:.2f}]',
                          fontsize=14, fontweight='bold')

        demand_mat = np.array(env._demand_history).T
        price_mat = np.array(env._price_history).T
        elasticity = env._compute_elasticity()

        self._render_scatter(env)
        self._render_elasticity_bar(env, elasticity)
        self._render_session_pie(env)
        self._render_price_heatmap(price_mat)
        self._render_demand_heatmap(demand_mat)
        self._render_correlation(env.n_products, price_mat, demand_mat)
        self._render_revenue(env)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _render_scatter(self, env):
        ax = self.fig.add_subplot(self.gs[0, 0])
        prices_flat = np.array(env._price_history).flatten()
        demands_flat = np.array(env._demand_history).flatten()
        product_ids = np.tile(np.arange(env.n_products), len(env._price_history))
        ax.scatter(prices_flat, demands_flat, c=product_ids, cmap='plasma', alpha=0.6, s=15, edgecolors='none')
        if len(prices_flat) > 1:
            z = np.polyfit(prices_flat, demands_flat, 1)
            p_line = np.linspace(prices_flat.min(), prices_flat.max(), 50)
            ax.plot(p_line, np.polyval(z, p_line), '--', lw=1.5, alpha=0.8)
        style_axis(ax, "Price-Demand Relationship", "Price ($)", "Demand")

    def _render_elasticity_bar(self, env, elasticity):
        ax = self.fig.add_subplot(self.gs[0, 1])
        ax.barh(range(env.n_products), elasticity, alpha=0.8)
        ax.axvline(0, lw=0.8, alpha=0.5)
        ax.axvline(-1, lw=1, ls='--', alpha=0.5)
        ax.set_yticks(range(env.n_products))
        ax.set_yticklabels([f'P{i}' for i in range(env.n_products)], fontsize=7)
        style_axis(ax, "Price Elasticity", "(dQ/dP)(P/Q)", None)

    def _render_session_pie(self, env):
        ax = self.fig.add_subplot(self.gs[0, 2])
        n_h, n_a = env.market.Nhumans, env.market.Nagents
        wedges, _ = ax.pie([n_h, n_a], startangle=90, wedgeprops={'linewidth': 2, 'edgecolor': 'white'})
        ax.legend(wedges, [f'H ({n_h})', f'A ({n_a})'], loc='lower center', fontsize=8,
                  frameon=False, bbox_to_anchor=(0.5, -0.05))
        ax.set_title("Session Mix", fontsize=11, fontweight='bold')

    def _render_price_heatmap(self, price_mat):
        ax = self.fig.add_subplot(self.gs[1, :2])
        im = ax.imshow(price_mat, aspect='auto', cmap='viridis', origin='lower')
        style_axis(ax, "Price Heatmap P(product, t)", "Step", "Product")
        cbar = self.fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('$', fontsize=8)

    def _render_demand_heatmap(self, demand_mat):
        ax = self.fig.add_subplot(self.gs[1, 2])
        im = ax.imshow(demand_mat, aspect='auto', cmap='Blues', origin='lower')
        style_axis(ax, "Demand Q(product, t)", "Step", None)
        self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    def _render_correlation(self, n_products, price_mat, demand_mat):
        ax = self.fig.add_subplot(self.gs[2, 0])
        if price_mat.shape[1] > 2:
            corr = np.corrcoef(price_mat, demand_mat)[:n_products, n_products:]
            im = ax.imshow(corr, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
            ax.set_xticks(range(n_products))
            ax.set_yticks(range(n_products))
            ax.set_xticklabels([f'Q{i}' for i in range(n_products)], fontsize=6)
            ax.set_yticklabels([f'P{i}' for i in range(n_products)], fontsize=6)
            self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        style_axis(ax, "Price-Demand Correlation", None, None)

    def _render_revenue(self, env):
        ax = self.fig.add_subplot(self.gs[2, 1:])
        n_steps = len(env._revenue_history)
        demand_std = [np.std(d) for d in env._demand_history]
        ax.fill_between(range(n_steps), env._revenue_history, alpha=0.3)
        ax.plot(env._revenue_history, linewidth=2, label='Revenue')
        ax.set_xlim(0, max(n_steps, 1))
        ax.set_ylim(0, max(env._revenue_history) * 1.1 if env._revenue_history else 1)

        ax2 = ax.twinx()
        ax2.plot(range(n_steps), demand_std, linewidth=2, ls='-', alpha=0.9, label='sigma(Demand)')
        d_min, d_max = min(demand_std), max(demand_std)
        margin = (d_max - d_min) * 0.2 if d_max > d_min else 0.5
        ax2.set_ylim(max(0, d_min - margin), d_max + margin)
        ax2.set_ylabel('Demand sigma', fontsize=9)

        style_axis(ax, "Revenue & Demand Dispersion", "Step", "Revenue ($)")
        ax.legend(loc='upper left', fontsize=7, frameon=False)
        ax2.legend(loc='upper right', fontsize=7, frameon=False)

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
