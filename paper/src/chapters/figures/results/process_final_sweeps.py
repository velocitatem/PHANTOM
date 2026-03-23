from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_bundle_dir() -> Path:
    base = _project_root() / "engine" / "studies" / "results" / "wandb_sweep_bundles"
    bundles = sorted(
        [path for path in base.glob("bundle_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not bundles:
        raise FileNotFoundError(f"No sweep bundle directories found in {base}")
    return bundles[0]


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "generated" / "final"


def _default_plot_dir(output_dir: Path) -> Path:
    return output_dir / "plots"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _mode_of(row: pd.Series) -> str:
    mode_hint = str(row.get("study_mode", "")).strip().lower()
    if mode_hint in {"baseline", "no_robust"}:
        return "baseline"
    if mode_hint in {"defended", "robust"}:
        return "defended"
    if _truthy(row.get("baseline_mode")) or _truthy(row.get("no_robust")):
        return "baseline"
    return "defended"


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


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


def _load_runs(bundle_dir: Path) -> pd.DataFrame:
    path = bundle_dir / "runs_finished.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    frame = pd.read_csv(path)
    frame["mode"] = frame.apply(_mode_of, axis=1)
    _coerce_numeric(
        frame,
        [
            "alpha",
            "n_products",
            "eval_revenue_mean",
            "eval_reward_mean",
            "eval_supra_share_mean",
            "eval_volatility_mean",
            "eval_coi_level_mean",
            "eval_coi_leakage_mean",
            "objective_score",
        ],
    )
    return frame


def _focus_sweep(runs: pd.DataFrame) -> str:
    coverage = (
        runs.groupby("sweep_id", as_index=False)
        .agg(
            n_alpha=("alpha", lambda s: int(pd.Series(s).dropna().nunique())),
            max_alpha=("alpha", "max"),
            run_count=("run_id", "size"),
        )
        .sort_values(
            ["n_alpha", "max_alpha", "run_count"], ascending=[False, False, False]
        )
    )
    if coverage.empty:
        raise ValueError("No sweep rows available in runs_finished.csv")
    return str(coverage.iloc[0]["sweep_id"])


def _alpha_mode_summary(runs: pd.DataFrame) -> pd.DataFrame:
    return (
        runs.groupby(["alpha", "mode"], as_index=False)
        .agg(
            runs=("run_id", "size"),
            revenue_mean=("eval_revenue_mean", "mean"),
            reward_mean=("eval_reward_mean", "mean"),
            supra_mean=("eval_supra_share_mean", "mean"),
            volatility_mean=("eval_volatility_mean", "mean"),
            coi_leakage_mean=("eval_coi_leakage_mean", "mean"),
            coi_level_mean=("eval_coi_level_mean", "mean"),
        )
        .sort_values(["alpha", "mode"])
        .reset_index(drop=True)
    )


def _alpha_deltas(alpha_mode: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for alpha, group in alpha_mode.groupby("alpha", sort=True):
        defended = group[group["mode"] == "defended"]
        baseline = group[group["mode"] == "baseline"]
        if defended.empty or baseline.empty:
            continue
        d_rev = float(defended["revenue_mean"].iloc[0])
        b_rev = float(baseline["revenue_mean"].iloc[0])
        d_reward = float(defended["reward_mean"].iloc[0])
        b_reward = float(baseline["reward_mean"].iloc[0])
        d_vol = float(defended["volatility_mean"].iloc[0])
        b_vol = float(baseline["volatility_mean"].iloc[0])
        d_supra = float(defended["supra_mean"].iloc[0])
        b_supra = float(baseline["supra_mean"].iloc[0])
        d_coi_leak = float(defended["coi_leakage_mean"].iloc[0])
        b_coi_leak = float(baseline["coi_leakage_mean"].iloc[0])
        rows.append(
            {
                "alpha": float(alpha),
                "revenue_delta": d_rev - b_rev,
                "revenue_delta_pct": 0.0
                if b_rev == 0.0
                else 100.0 * (d_rev - b_rev) / b_rev,
                "reward_delta": d_reward - b_reward,
                "reward_delta_pct": 0.0
                if b_reward == 0.0
                else 100.0 * (d_reward - b_reward) / b_reward,
                "volatility_delta": d_vol - b_vol,
                "supra_delta": d_supra - b_supra,
                "coi_leakage_delta": d_coi_leak - b_coi_leak,
            }
        )
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


def _zone_summary(alpha_deltas: pd.DataFrame) -> pd.DataFrame:
    if alpha_deltas.empty:
        return pd.DataFrame()
    data = alpha_deltas.copy()
    data["zone"] = np.where(
        data["alpha"] >= 0.7, "high_alpha_0_7_plus", "low_alpha_below_0_7"
    )
    return (
        data.groupby("zone", as_index=False)
        .agg(
            alpha_cells=("alpha", "size"),
            revenue_delta_pct_mean=("revenue_delta_pct", "mean"),
            reward_delta_pct_mean=("reward_delta_pct", "mean"),
            coi_leakage_delta_mean=("coi_leakage_delta", "mean"),
            volatility_delta_mean=("volatility_delta", "mean"),
        )
        .sort_values("zone")
    )


def _save_plot(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_focus_revenue_by_alpha(alpha_mode: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    for mode, color, label in (
        ("baseline", "#4C72B0", "Baseline"),
        ("defended", "#C44E52", "Defended"),
    ):
        sub = alpha_mode[alpha_mode["mode"] == mode].sort_values("alpha")
        if sub.empty:
            continue
        ax.plot(
            sub["alpha"],
            sub["revenue_mean"],
            marker="o",
            linewidth=1.9,
            markersize=4,
            color=color,
            label=label,
        )
    ax.axvline(0.7, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_xlabel(r"Contamination $\alpha$")
    ax.set_ylabel("Mean episode revenue")
    ax.set_title("Final Cohort Revenue Curves")
    ax.legend(loc="lower left")
    return _save_plot(fig, out_path)


def _plot_focus_revenue_delta(alpha_deltas: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    x = alpha_deltas["alpha"].to_numpy(dtype=float)
    y = alpha_deltas["revenue_delta_pct"].to_numpy(dtype=float)
    ax.plot(x, y, marker="o", linewidth=2.0, markersize=4, color="#C44E52")
    ax.fill_between(x, y, 0.0, color="#C44E52", alpha=0.12)
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.axvline(0.7, color="#666666", linewidth=1.0, linestyle="--")
    high = alpha_deltas[alpha_deltas["alpha"] >= 0.7]
    if not high.empty:
        best = high.reindex(
            high["revenue_delta_pct"].abs().sort_values(ascending=False).index
        ).iloc[0]
        ax.scatter(
            [best["alpha"]],
            [best["revenue_delta_pct"]],
            color="#1f77b4",
            s=45,
            zorder=3,
        )
        ax.annotate(
            f"high-alpha peak {best['revenue_delta_pct']:.2f}%",
            (float(best["alpha"]), float(best["revenue_delta_pct"])),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    ax.set_xlabel(r"Contamination $\alpha$")
    ax.set_ylabel("Defended minus baseline revenue (%)")
    ax.set_title("Revenue Delta by Contamination (Final Cohort)")
    return _save_plot(fig, out_path)


def _plot_focus_risk_deltas(alpha_deltas: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    x = alpha_deltas["alpha"].to_numpy(dtype=float)
    ax.plot(
        x,
        alpha_deltas["coi_leakage_delta"].to_numpy(dtype=float),
        marker="o",
        linewidth=1.8,
        markersize=4,
        color="#55A868",
        label="COI leakage delta",
    )
    ax.plot(
        x,
        alpha_deltas["volatility_delta"].to_numpy(dtype=float),
        marker="s",
        linewidth=1.8,
        markersize=3.8,
        color="#8172B3",
        label="Volatility delta",
    )
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.axvline(0.7, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_xlabel(r"Contamination $\alpha$")
    ax.set_ylabel("Defended minus baseline")
    ax.set_title("Leakage and Stability Deltas (Final Cohort)")
    ax.legend(loc="lower left")
    return _save_plot(fig, out_path)


def _write_include(path: Path, figure_rel_path: str, width: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"\\includegraphics[width={width}]{{{figure_rel_path}}}\n")
    return path


def run(bundle_dir: Path, output_dir: Path, plot_dir: Path) -> list[Path]:
    all_runs = _load_runs(bundle_dir)
    focus_id = _focus_sweep(all_runs)
    focus_runs = all_runs[all_runs["sweep_id"] == focus_id].copy()
    alpha_mode = _alpha_mode_summary(focus_runs)
    deltas = _alpha_deltas(alpha_mode)
    zones = _zone_summary(deltas)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    alpha_mode_path = output_dir / "final_focus_alpha_mode_summary.csv"
    alpha_mode.to_csv(alpha_mode_path, index=False)
    written.append(alpha_mode_path)

    delta_path = output_dir / "final_focus_alpha_deltas.csv"
    deltas.to_csv(delta_path, index=False)
    written.append(delta_path)

    zone_path = output_dir / "final_focus_zone_summary.csv"
    zones.to_csv(zone_path, index=False)
    written.append(zone_path)

    headline = {
        "bundle": str(bundle_dir),
        "focus_cohort": "max_alpha_coverage",
        "alpha_cells": int(deltas["alpha"].nunique()) if not deltas.empty else 0,
        "alpha_min": float(deltas["alpha"].min()) if not deltas.empty else None,
        "alpha_max": float(deltas["alpha"].max()) if not deltas.empty else None,
        "mean_revenue_delta_pct": float(deltas["revenue_delta_pct"].mean())
        if not deltas.empty
        else None,
        "mean_reward_delta_pct": float(deltas["reward_delta_pct"].mean())
        if not deltas.empty
        else None,
        "zone_summary": zones.to_dict(orient="records"),
    }
    headline_path = output_dir / "final_focus_headline_summary.json"
    headline_path.write_text(json.dumps(headline, indent=2) + "\n")
    written.append(headline_path)

    written.append(
        _plot_focus_revenue_by_alpha(
            alpha_mode,
            plot_dir / "final_focus_revenue_by_alpha.pdf",
        )
    )
    written.append(
        _plot_focus_revenue_delta(
            deltas,
            plot_dir / "final_focus_revenue_delta.pdf",
        )
    )
    written.append(
        _plot_focus_risk_deltas(
            deltas,
            plot_dir / "final_focus_risk_deltas.pdf",
        )
    )

    include_dir = Path(__file__).resolve().parent / "includes" / "final"
    written.append(
        _write_include(
            include_dir / "final_focus_revenue_by_alpha.tex",
            "chapters/figures/results/generated/final/plots/final_focus_revenue_by_alpha.pdf",
            "0.98\\linewidth",
        )
    )
    written.append(
        _write_include(
            include_dir / "final_focus_revenue_delta.tex",
            "chapters/figures/results/generated/final/plots/final_focus_revenue_delta.pdf",
            "0.95\\linewidth",
        )
    )
    written.append(
        _write_include(
            include_dir / "final_focus_risk_deltas.tex",
            "chapters/figures/results/generated/final/plots/final_focus_risk_deltas.pdf",
            "0.95\\linewidth",
        )
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate final paper figures/tables from the final sweep cohort"
    )
    parser.add_argument("--bundle-dir", type=Path, default=_default_bundle_dir())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--plot-dir", type=Path, default=None)
    args = parser.parse_args()

    _configure_style()
    plot_dir = (
        args.plot_dir
        if args.plot_dir is not None
        else _default_plot_dir(args.output_dir)
    )
    outputs = run(
        bundle_dir=args.bundle_dir, output_dir=args.output_dir, plot_dir=plot_dir
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
