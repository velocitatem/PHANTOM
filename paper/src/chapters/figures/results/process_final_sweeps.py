from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
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


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
            cwd=_project_root(),
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


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


def _alpha_product_coi_preservation(runs: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        runs.groupby(["alpha", "n_products", "mode"], as_index=False)
        .agg(
            runs=("run_id", "size"),
            coi_level_mean=("eval_coi_level_mean", "mean"),
        )
        .sort_values(["alpha", "n_products", "mode"])
        .reset_index(drop=True)
    )

    rows: list[dict[str, float | int]] = []
    for (alpha, n_products), group in grouped.groupby(
        ["alpha", "n_products"], sort=True
    ):
        defended = group[group["mode"] == "defended"]
        baseline = group[group["mode"] == "baseline"]
        if defended.empty or baseline.empty:
            continue

        d_coi = float(defended["coi_level_mean"].iloc[0])
        b_coi = float(baseline["coi_level_mean"].iloc[0])
        rows.append(
            {
                "alpha": float(alpha),
                "n_products": float(n_products),
                "baseline_runs": int(baseline["runs"].iloc[0]),
                "defended_runs": int(defended["runs"].iloc[0]),
                "baseline_coi_level_mean": b_coi,
                "defended_coi_level_mean": d_coi,
                "coi_preserved": d_coi - b_coi,
                "coi_preserved_pct": 0.0
                if b_coi == 0.0
                else 100.0 * (d_coi - b_coi) / b_coi,
            }
        )

    return (
        pd.DataFrame(rows).sort_values(["alpha", "n_products"]).reset_index(drop=True)
    )


def _save_plot(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _smoothed_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window: int = 5,
    points: int = 320,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[mask]
    y_values = y_values[mask]
    if x_values.size == 0:
        return x_values, y_values

    order = np.argsort(x_values)
    x_values = x_values[order]
    y_values = y_values[order]

    unique_x = np.unique(x_values)
    if unique_x.size != x_values.size:
        dedup = (
            pd.DataFrame({"x": x_values, "y": y_values})
            .groupby("x", as_index=False)
            .agg(y=("y", "mean"))
            .sort_values("x")
        )
        x_values = dedup["x"].to_numpy(dtype=float)
        y_values = dedup["y"].to_numpy(dtype=float)

    if x_values.size < 3:
        return x_values, y_values

    win = int(max(3, window))
    if win % 2 == 0:
        win += 1
    if win > x_values.size:
        win = x_values.size if x_values.size % 2 == 1 else x_values.size - 1
    if win < 3:
        return x_values, y_values

    half = win // 2
    offsets = np.arange(-half, half + 1, dtype=float)
    sigma = max(win / 3.0, 1.0)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / np.sum(kernel)
    y_padded = np.pad(y_values, (half, half), mode="edge")
    y_smooth = np.convolve(y_padded, kernel, mode="valid")

    n_points = max(int(points), x_values.size)
    x_dense = np.linspace(float(np.min(x_values)), float(np.max(x_values)), n_points)
    y_dense = np.interp(x_dense, x_values, y_smooth)
    return x_dense, y_dense


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
    ax.set_xlabel(r"Contamination $\alpha$")
    ax.set_ylabel("Mean episode revenue")
    ax.set_title("Final Cohort Revenue Curves")
    ax.legend(loc="lower left")
    return _save_plot(fig, out_path)


def _plot_focus_coi_by_alpha(alpha_mode: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    for mode, color, label in (
        ("baseline", "#4C72B0", "Baseline"),
        ("defended", "#C44E52", "Defended"),
    ):
        sub = alpha_mode[alpha_mode["mode"] == mode].sort_values("alpha")
        if sub.empty:
            continue
        x_raw = sub["alpha"].to_numpy(dtype=float)
        y_raw = sub["coi_level_mean"].to_numpy(dtype=float)
        x_smooth, y_smooth = _smoothed_curve(x_raw, y_raw)
        ax.plot(
            x_smooth,
            y_smooth,
            linewidth=1.9,
            color=color,
            label=label,
        )
        ax.scatter(
            x_raw,
            y_raw,
            s=18,
            color=color,
            edgecolor="#FFFFFF",
            linewidth=0.45,
            zorder=3,
        )

    paired = alpha_mode.pivot_table(
        index="alpha",
        columns="mode",
        values="coi_level_mean",
        aggfunc="mean",
    ).sort_index()
    if {"baseline", "defended"}.issubset(set(paired.columns)):
        paired = paired.dropna(subset=["baseline", "defended"], how="any")
        if not paired.empty:
            x = paired.index.to_numpy(dtype=float)
            y_baseline = paired["baseline"].to_numpy(dtype=float)
            y_defended = paired["defended"].to_numpy(dtype=float)
            x_fill, y_baseline_smooth = _smoothed_curve(x, y_baseline)
            _, y_defended_smooth = _smoothed_curve(x, y_defended)
            ax.fill_between(
                x_fill,
                y_baseline_smooth,
                y_defended_smooth,
                color="#55A868",
                alpha=0.12,
                label="Gap",
            )

    ax.set_xlabel(r"Contamination $\alpha$")
    ax.set_ylabel("Mean COI level")
    ax.set_title("Final Cohort COI Curves")
    ax.legend(loc="lower left")
    return _save_plot(fig, out_path)


def _plot_focus_coi_preservation_grid(
    coi_preservation: pd.DataFrame, out_path: Path
) -> Path:
    if coi_preservation.empty:
        raise ValueError("COI preservation grid requires at least one paired cell")

    alpha_levels = sorted(coi_preservation["alpha"].dropna().unique().tolist())
    endpoint_targets = (0.0, 1.0)
    endpoint_levels = [
        alpha
        for target in endpoint_targets
        for alpha in alpha_levels
        if np.isclose(alpha, target, atol=1e-9)
    ]
    if len(endpoint_levels) < 2 and alpha_levels:
        endpoint_levels = [alpha_levels[0], alpha_levels[-1]]
    endpoint_levels = sorted(set(endpoint_levels))

    data = coi_preservation[coi_preservation["alpha"].isin(endpoint_levels)].copy()
    if data.empty:
        raise ValueError(
            "COI preservation grid has no rows for selected alpha endpoints"
        )

    alpha_levels = sorted(data["alpha"].dropna().unique().tolist())
    product_levels = sorted(data["n_products"].dropna().unique().tolist())

    bars = data.pivot_table(
        index="n_products",
        columns="alpha",
        values="coi_preserved",
        aggfunc="mean",
    ).reindex(index=product_levels, columns=alpha_levels)

    x = np.arange(len(product_levels), dtype=float)
    n_alpha = max(len(alpha_levels), 1)
    bar_width = min(0.78 / n_alpha, 0.35)
    offsets = (np.arange(n_alpha, dtype=float) - (n_alpha - 1) / 2.0) * bar_width
    palette = ["#4C72B0", "#C44E52", "#55A868", "#8172B3"]

    fig, ax = plt.subplots(figsize=(7.8, 5.0), constrained_layout=True)
    for idx, alpha in enumerate(alpha_levels):
        values = bars[alpha].to_numpy(dtype=float)
        mask = np.isfinite(values)
        if not np.any(mask):
            continue
        xpos = x[mask] + offsets[idx]
        v = values[mask]
        ax.bar(
            xpos,
            v,
            width=bar_width * 0.96,
            color=palette[idx % len(palette)],
            label=rf"$\alpha={alpha:.1f}$",
        )
        for x_i, y_i in zip(xpos, v):
            ax.text(
                float(x_i),
                float(y_i) + (0.035 if y_i >= 0 else -0.035),
                f"{y_i:+.2f}",
                ha="center",
                va="bottom" if y_i >= 0 else "top",
                fontsize=7,
            )

    valid = bars.to_numpy(dtype=float)
    valid = valid[np.isfinite(valid)]
    max_abs = float(np.max(np.abs(valid))) if valid.size else 1.0
    max_abs = max(max_abs * 1.22, 0.4)
    ax.set_ylim(-max_abs, max_abs)

    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in product_levels])
    ax.set_xlabel("Product count")
    ax.set_ylabel("COI preserved (defended minus baseline)")
    ax.set_title("COI Preservation by Product Count at $\\alpha=0.0$ vs $\\alpha=1.0$")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.22)
    return _save_plot(fig, out_path)


def _plot_focus_revenue_delta(alpha_deltas: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    x = alpha_deltas["alpha"].to_numpy(dtype=float)
    y = alpha_deltas["revenue_delta_pct"].to_numpy(dtype=float)
    ax.plot(x, y, marker="o", linewidth=2.0, markersize=4, color="#C44E52")
    ax.fill_between(x, y, 0.0, color="#C44E52", alpha=0.12)
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
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
    ax.set_xlabel(r"Contamination $\alpha$")
    ax.set_ylabel("Defended minus baseline")
    ax.set_title("Leakage and Stability Deltas (Final Cohort)")
    ax.legend(loc="lower left")
    return _save_plot(fig, out_path)


def _write_include(path: Path, figure_rel_path: str, width: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"\\includegraphics[width={width}]{{{figure_rel_path}}}\n")
    return path


def run(
    bundle_dir: Path,
    output_dir: Path,
    plot_dir: Path,
    focus_sweep_id: str | None = None,
) -> list[Path]:
    all_runs = _load_runs(bundle_dir)
    focus_id = str(focus_sweep_id) if focus_sweep_id else _focus_sweep(all_runs)
    if focus_id not in set(all_runs["sweep_id"].astype(str).unique()):
        raise ValueError(f"Requested focus sweep_id not found: {focus_id}")
    focus_runs = all_runs[all_runs["sweep_id"] == focus_id].copy()
    alpha_mode = _alpha_mode_summary(focus_runs)
    deltas = _alpha_deltas(alpha_mode)
    zones = _zone_summary(deltas)
    coi_preservation = _alpha_product_coi_preservation(focus_runs)

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

    coi_grid_path = output_dir / "final_focus_coi_preservation_grid.csv"
    coi_preservation.to_csv(coi_grid_path, index=False)
    written.append(coi_grid_path)

    headline = {
        "bundle": str(bundle_dir),
        "focus_cohort": "max_alpha_coverage",
        "focus_sweep_id": focus_id,
        "focus_run_count": int(len(focus_runs)),
        "git_commit": _git_commit(),
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
        _plot_focus_coi_by_alpha(
            alpha_mode,
            plot_dir / "final_focus_coi_by_alpha.pdf",
        )
    )
    written.append(
        _plot_focus_coi_preservation_grid(
            coi_preservation,
            plot_dir / "final_focus_coi_preservation_grid.pdf",
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

    include_dir = Path(__file__).resolve().parent / "includes"
    written.append(
        _write_include(
            include_dir / "final_focus_revenue_by_alpha.tex",
            "chapters/figures/results/generated/final/plots/final_focus_revenue_by_alpha.pdf",
            "0.98\\linewidth",
        )
    )
    written.append(
        _write_include(
            include_dir / "final_focus_coi_by_alpha.tex",
            "chapters/figures/results/generated/final/plots/final_focus_coi_by_alpha.pdf",
            "0.98\\linewidth",
        )
    )
    written.append(
        _write_include(
            include_dir / "final_focus_coi_preservation_grid.tex",
            "chapters/figures/results/generated/final/plots/final_focus_coi_preservation_grid.pdf",
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
    parser.add_argument("--focus-sweep-id", type=str, default=None)
    args = parser.parse_args()

    _configure_style()
    plot_dir = (
        args.plot_dir
        if args.plot_dir is not None
        else _default_plot_dir(args.output_dir)
    )
    outputs = run(
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
        plot_dir=plot_dir,
        focus_sweep_id=args.focus_sweep_id,
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
