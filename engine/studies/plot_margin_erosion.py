"""plot margin erosion: margin/COI/revenue vs α with thesis-quality formatting"""

import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (7, 4),
        "figure.dpi": 150,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "errorbar.capsize": 3,
        "grid.alpha": 0.3,
    }
)


def plot_margin_erosion(data: dict, out: Path):
    s = data["summary"]
    αs = sorted([float(k.split("_")[1]) for k in s.keys()])

    def get(metric):
        return (
            [s[f"alpha_{α:.1f}"][f"{metric}_mean"] for α in αs],
            [s[f"alpha_{α:.1f}"][f"{metric}_std"] for α in αs],
        )

    margins, margin_e = get("margin")
    cois, coi_e = get("coi_level")
    revs, rev_e = get("revenue")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    axes[0].errorbar(
        αs,
        margins,
        yerr=margin_e,
        marker="o",
        capsize=4,
        label="Standard RL",
        color="#d62728",
    )
    axes[0].axhline(0.05, color="gray", linestyle="--", linewidth=1, label="Floor")
    axes[0].set(
        xlabel="Agent proportion (α)",
        ylabel="Margin",
        title="Margin erosion",
        ylim=(0, max(margins) * 1.2),
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].errorbar(αs, cois, yerr=coi_e, marker="s", capsize=4, color="#ff7f0e")
    axes[1].set(
        xlabel="Agent proportion (α)",
        ylabel="COI",
        title="COI collapse (E[P] - p_min)",
        ylim=(0, None),
    )
    axes[1].grid(alpha=0.3)

    axes[2].errorbar(αs, revs, yerr=rev_e, marker="^", capsize=4, color="#2ca02c")
    axes[2].set(
        xlabel="Agent proportion (α)",
        ylabel="Revenue",
        title="Revenue degradation",
        ylim=(0, None),
    )
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    pdf = out / "margin_erosion_alpha.pdf"
    png = out / "margin_erosion_alpha.png"
    plt.savefig(pdf, bbox_inches="tight", dpi=300)
    plt.savefig(png, bbox_inches="tight", dpi=150)
    print(f"→ {pdf}\n→ {png}")


def print_latex(data: dict):
    s = data["summary"]
    αs = sorted([float(k.split("_")[1]) for k in s.keys()])

    print("\n% LaTeX table for appendix")
    print("\\begin{table}[h]\n\\centering")
    print("\\caption{Margin erosion: standard RL under agent contamination}")
    print("\\label{tab:margin_erosion}")
    print("\\begin{tabular}{cccc}\n\\toprule")
    print("α & Margin & COI & Revenue \\\\\n\\midrule")

    for α in αs:
        d = s[f"alpha_{α:.1f}"]
        print(
            f"{α:.1f} & ${d['margin_mean']:.3f} \\pm {d['margin_std']:.3f}$ & "
            f"${d['coi_level_mean']:.1f} \\pm {d['coi_level_std']:.1f}$ & "
            f"${d['revenue_mean']:.0f} \\pm {d['revenue_std']:.0f}$ \\\\"
        )

    print("\\bottomrule\n\\end{tabular}\n\\end{table}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m engine.studies.plot_margin_erosion <results.json>")

    path = Path(sys.argv[1])
    if not path.exists():
        sys.exit(f"error: {path} not found")

    with open(path) as f:
        data = json.load(f)

    plot_margin_erosion(data, path.parent)
    print_latex(data)
    print(
        f"\n{len(data['results'])} runs, {len(data['summary'])} α levels, "
        f"algos={data['config']['algos']}, seeds={data['config']['seeds']}"
    )
