from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from process_first_sweep import run as run_first_sweep
from process_ppo_benchmark import run as run_ppo_benchmark


def _output_dir() -> Path:
    return Path(__file__).resolve().parent / "generated"


def _plot_dir() -> Path:
    return _output_dir() / "plots"


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 220,
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def _fmt_thousands(value: float, _: int) -> str:
    return f"{int(value):,}"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def _plot_ppo_alpha_curves(alpha_mode: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.4), constrained_layout=True)
    robust_color = "#C44E52"
    baseline_color = "#4C72B0"
    mode_colors = {"robust": robust_color, "no_robust": baseline_color}
    mode_labels = {"robust": "Robust", "no_robust": "Non-robust"}

    panels = [
        ("eval_revenue_mean", "Mean Episode Revenue", "Revenue"),
        ("eval_reward_mean", "Mean Episode Reward", "Reward"),
        ("eval_coi_leakage_mean", "Mean COI Leakage", "COI Leakage"),
        ("eval_volatility_mean", "Mean Price Volatility", "Volatility"),
    ]

    for ax, (metric_prefix, title, ylabel) in zip(axes.flat, panels):
        mean_col = f"{metric_prefix}_mean"
        std_col = f"{metric_prefix}_std"
        for mode in ("no_robust", "robust"):
            sub = alpha_mode[alpha_mode["mode"] == mode].sort_values("alpha")
            if sub.empty:
                continue
            x = sub["alpha"].to_numpy(dtype=float)
            y = sub[mean_col].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=mode_colors[mode],
                label=mode_labels[mode],
            )
            if std_col in sub.columns:
                sigma = sub[std_col].fillna(0.0).to_numpy(dtype=float)
                ax.fill_between(
                    x,
                    y - sigma,
                    y + sigma,
                    color=mode_colors[mode],
                    alpha=0.14,
                    linewidth=0,
                )

        ax.set_title(title)
        ax.set_xlabel(r"Contamination $\alpha$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(alpha_mode["alpha"].unique()))
        if metric_prefix in {"eval_revenue_mean", "eval_reward_mean"}:
            ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))

    out_path = out_dir / "ppo_alpha_curves.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_ppo_delta_curves(deltas: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.0), constrained_layout=True)
    deltas = deltas.sort_values("alpha")
    x = deltas["alpha"].to_numpy(dtype=float)

    top_metrics = [
        ("eval_revenue_mean_delta_pct", "Revenue", "#4C72B0"),
        ("eval_reward_mean_delta_pct", "Reward", "#8172B3"),
    ]
    for col, label, color in top_metrics:
        axes[0].plot(
            x,
            deltas[col].to_numpy(dtype=float),
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=label,
        )
    axes[0].axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    axes[0].set_title("Robust Minus Non-robust Delta by Contamination")
    axes[0].set_ylabel("Delta (%)")
    axes[0].set_xlabel(r"Contamination $\alpha$")
    axes[0].set_xticks(x)
    axes[0].legend(loc="lower left")

    bottom_metrics = [
        ("eval_coi_leakage_mean_delta_pct", "COI Leakage", "#55A868"),
        ("eval_volatility_mean_delta_pct", "Volatility", "#DD8452"),
    ]
    for col, label, color in bottom_metrics:
        axes[1].plot(
            x,
            deltas[col].to_numpy(dtype=float),
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=label,
        )
    axes[1].axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("Delta (%)")
    axes[1].set_xlabel(r"Contamination $\alpha$")
    axes[1].set_xticks(x)
    axes[1].legend(loc="lower left")

    out_path = out_dir / "ppo_delta_curves.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_ppo_tradeoff_scatter(deltas: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    data = deltas.sort_values("alpha")
    x = data["eval_coi_leakage_mean_delta_pct"].to_numpy(dtype=float)
    y = data["eval_revenue_mean_delta_pct"].to_numpy(dtype=float)
    alphas = data["alpha"].to_numpy(dtype=float)

    scatter = ax.scatter(
        x,
        y,
        c=alphas,
        cmap="viridis",
        s=72,
        edgecolor="#222222",
        linewidth=0.5,
    )
    for x_i, y_i, alpha in zip(x, y, alphas):
        ax.annotate(
            rf"$\alpha={alpha:.2f}$",
            (x_i, y_i),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )

    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax.axvline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax.set_xlabel("COI Leakage Delta (%)")
    ax.set_ylabel("Revenue Delta (%)")
    ax.set_title("PPO Robust Tradeoff Frontier")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(r"Contamination $\alpha$")

    out_path = out_dir / "ppo_tradeoff_scatter.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_first_sweep_tier_revenue(tier_mode: pd.DataFrame, out_dir: Path) -> Path:
    pivot = (
        tier_mode.pivot(index="tier", columns="mode", values="eval_revenue_mean_mean")
        .dropna(subset=["robust", "no_robust"], how="any")
        .copy()
    )
    if pivot.empty:
        raise ValueError("First sweep tier summary missing robust/non-robust pairs")

    order = sorted(pivot.index.tolist())
    pivot = pivot.loc[order]
    delta_pct = 100.0 * (pivot["robust"] - pivot["no_robust"]) / pivot["no_robust"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), constrained_layout=True)
    x = np.arange(len(order))
    width = 0.36

    axes[0].bar(
        x - width / 2,
        pivot["no_robust"].to_numpy(dtype=float),
        width=width,
        label="Non-robust",
        color="#4C72B0",
    )
    axes[0].bar(
        x + width / 2,
        pivot["robust"].to_numpy(dtype=float),
        width=width,
        label="Robust",
        color="#C44E52",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order, rotation=20)
    axes[0].set_ylabel("Mean Revenue")
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
    axes[0].set_title("First Sweep Tier Revenue (log scale)")
    axes[0].legend()

    axes[1].bar(x, delta_pct.to_numpy(dtype=float), color="#55A868", width=0.55)
    axes[1].axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order, rotation=20)
    axes[1].set_ylabel("Revenue Delta (%)")
    axes[1].set_title("Robust Minus Non-robust by Tier")

    out_path = out_dir / "first_sweep_tier_revenue.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_plots(data_dir: Path, out_dir: Path) -> list[Path]:
    alpha_mode = _load_csv(data_dir / "ppo_alpha_mode_summary.csv")
    deltas = _load_csv(data_dir / "ppo_alpha_deltas.csv")
    tier_mode = _load_csv(data_dir / "first_sweep_tier_mode_summary.csv")

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _plot_ppo_alpha_curves(alpha_mode, out_dir),
        _plot_ppo_delta_curves(deltas, out_dir),
        _plot_ppo_tradeoff_scatter(deltas, out_dir),
        _plot_first_sweep_tier_revenue(tier_mode, out_dir),
    ]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create paper-ready plots from result CSVs"
    )
    parser.add_argument("--data-dir", type=Path, default=_output_dir())
    parser.add_argument("--plot-dir", type=Path, default=_plot_dir())
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Regenerate processed CSVs before plotting",
    )
    args = parser.parse_args()

    _configure_style()

    if bool(args.refresh_data):
        run_ppo_benchmark(
            input_path=Path(__file__).resolve().parents[5]
            / "tpu_orchestration"
            / "results"
            / "ppo_benchmark.csv",
            output_dir=args.data_dir,
            include_non_finished=False,
        )
        run_first_sweep(
            input_path=Path(__file__).resolve().parents[5]
            / "tpu_orchestration"
            / "results"
            / "first_sweep.csv",
            output_dir=args.data_dir,
            include_non_finished=False,
            top_n=25,
        )

    outputs = build_plots(data_dir=args.data_dir, out_dir=args.plot_dir)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
