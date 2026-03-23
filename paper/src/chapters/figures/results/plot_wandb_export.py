from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


def _load_tikzplotlib():
    def _patch_webcolors() -> None:
        try:
            import webcolors

            if hasattr(webcolors, "CSS3_HEX_TO_NAMES"):
                return
            css3 = getattr(webcolors, "CSS3", "css3")
            webcolors.CSS3_HEX_TO_NAMES = {
                webcolors.name_to_hex(name, spec=css3): name
                for name in webcolors.names(spec=css3)
            }
        except Exception:
            return

    _patch_webcolors()

    try:
        from matplotlib.legend import Legend

        if not hasattr(Legend, "_ncol") and hasattr(Legend, "_ncols"):
            Legend._ncol = property(lambda self: self._ncols)
    except Exception:
        pass

    try:
        import tikzplotlib as module

        return module, None
    except Exception:
        pass

    try:
        from matplotlib.backends import backend_pgf

        if not hasattr(backend_pgf, "common_texification") and hasattr(
            backend_pgf, "_tex_escape"
        ):
            backend_pgf.common_texification = backend_pgf._tex_escape

        _patch_webcolors()

        import tikzplotlib as module

        return module, None
    except Exception as exc:
        return None, exc


TIKZPLOTLIB, TIKZPLOTLIB_IMPORT_ERROR = _load_tikzplotlib()


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "generated" / "wandb"


def _default_plot_dir(output_dir: Path) -> Path:
    return output_dir / "plots"


def _sanitize(key: str) -> str:
    return key.replace("/", "_").replace("-", "_")


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


def _coerce_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _extract_alpha(frame: pd.DataFrame) -> pd.Series:
    if "study/alpha" in frame.columns:
        return pd.to_numeric(frame["study/alpha"], errors="coerce")
    if "alpha" in frame.columns:
        return pd.to_numeric(frame["alpha"], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _extract_mode(frame: pd.DataFrame) -> pd.Series:
    if "study/mode" in frame.columns:
        mode = frame["study/mode"].astype(str).str.strip().str.lower()
        mapping = {
            "baseline": "baseline",
            "no_robust": "baseline",
            "defended": "defended",
            "robust": "defended",
        }
        return mode.map(mapping).fillna("")

    if "study/no_robust" in frame.columns:
        no_robust = pd.to_numeric(frame["study/no_robust"], errors="coerce").fillna(0.0)
        return pd.Series(
            np.where(no_robust > 0.5, "baseline", "defended"),
            index=frame.index,
            dtype="object",
        )

    if "no_robust" in frame.columns:
        no_robust = (
            frame["no_robust"].astype(str).str.lower().isin({"1", "true", "yes"})
        )
        return pd.Series(
            np.where(no_robust, "baseline", "defended"),
            index=frame.index,
            dtype="object",
        )

    return pd.Series("", index=frame.index, dtype="object")


def _prepare_frame(frame: pd.DataFrame, include_non_finished: bool) -> pd.DataFrame:
    data = frame.copy()
    if not include_non_finished and "State" in data.columns:
        data = data[data["State"].astype(str).str.lower() == "finished"].copy()

    data["alpha"] = _extract_alpha(data)
    data["mode"] = _extract_mode(data)
    data = data[data["mode"].isin({"baseline", "defended"})]
    data = data[data["alpha"].notna()]

    _coerce_numeric(
        data,
        [
            "eval/revenue_mean",
            "eval/reward_mean",
            "eval/coi_level_mean",
            "eval/coi_leakage_mean",
            "eval/volatility_mean",
            "eval/revenue_std",
            "eval/reward_std",
            "eval/margin_mean",
            "train/agent_prob",
            "train/alpha_adv",
            "lambda_coi",
            "ambiguity_radius",
            "n_products",
        ],
    )

    return data.sort_values(["alpha", "mode"]).reset_index(drop=True)


def _summary_by_alpha_mode(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    agg_spec: dict[str, tuple[str, str]] = {"runs": ("mode", "size")}
    for metric in metrics:
        safe = _sanitize(metric)
        agg_spec[f"{safe}_mean"] = (metric, "mean")
        agg_spec[f"{safe}_std"] = (metric, "std")

    return (
        frame.groupby(["alpha", "mode"], as_index=False)
        .agg(**agg_spec)
        .sort_values(["alpha", "mode"])
        .reset_index(drop=True)
    )


def _delta_by_alpha(summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for alpha, alpha_group in summary.groupby("alpha", sort=True):
        defended = alpha_group[alpha_group["mode"] == "defended"]
        baseline = alpha_group[alpha_group["mode"] == "baseline"]
        if defended.empty or baseline.empty:
            continue

        row: dict[str, float] = {
            "alpha": float(alpha),
            "runs_defended": float(defended["runs"].iloc[0]),
            "runs_baseline": float(baseline["runs"].iloc[0]),
        }
        for metric in metrics:
            safe = _sanitize(metric)
            defended_value = float(defended[f"{safe}_mean"].iloc[0])
            baseline_value = float(baseline[f"{safe}_mean"].iloc[0])
            delta = defended_value - baseline_value
            row[f"{safe}_defended"] = defended_value
            row[f"{safe}_baseline"] = baseline_value
            row[f"{safe}_delta"] = delta
            row[f"{safe}_delta_pct"] = (
                np.nan if baseline_value == 0 else 100.0 * delta / baseline_value
            )
        rows.append(row)

    return pd.DataFrame(rows)


def _summary_by_parameter(
    frame: pd.DataFrame, parameter: str, metrics: list[str]
) -> pd.DataFrame:
    defended = frame[frame["mode"] == "defended"].copy()
    defended = defended[defended[parameter].notna()].copy()
    agg_spec: dict[str, tuple[str, str]] = {"runs": ("mode", "size")}
    for metric in metrics:
        safe = _sanitize(metric)
        agg_spec[f"{safe}_mean"] = (metric, "mean")
        agg_spec[f"{safe}_std"] = (metric, "std")

    return (
        defended.groupby(["alpha", parameter], as_index=False)
        .agg(**agg_spec)
        .sort_values(["alpha", parameter])
        .reset_index(drop=True)
    )


def _save_table(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _save_figure(fig: plt.Figure, pdf_path: Path, export_tikz: bool) -> list[Path]:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    written = [pdf_path]

    if export_tikz:
        if TIKZPLOTLIB is None:
            raise RuntimeError(
                "tikzplotlib import failed. Install/upgrade tikzplotlib and matplotlib-compatible dependencies. "
                f"Original error: {TIKZPLOTLIB_IMPORT_ERROR}"
            )

        try:
            from matplotlib.legend import Legend
            from matplotlib.lines import Line2D

            for legend in fig.findobj(Legend):
                if not hasattr(legend, "_ncol") and hasattr(legend, "_ncols"):
                    setattr(legend, "_ncol", legend._ncols)
                if not hasattr(legend, "legendHandles") and hasattr(
                    legend, "legend_handles"
                ):
                    setattr(legend, "legendHandles", legend.legend_handles)

            for line in fig.findobj(Line2D):
                if hasattr(line, "_us_dashSeq"):
                    continue
                if not hasattr(line, "_dash_pattern"):
                    continue
                dash_pattern = getattr(line, "_dash_pattern")
                if not isinstance(dash_pattern, tuple) or len(dash_pattern) != 2:
                    continue
                setattr(line, "_us_dashOffset", dash_pattern[0])
                setattr(line, "_us_dashSeq", dash_pattern[1])
        except Exception:
            pass

        tikz_path = pdf_path.with_suffix(".tikz.tex")
        TIKZPLOTLIB.save(str(tikz_path), figure=fig)
        written.append(tikz_path)

    plt.close(fig)
    return written


def _plot_alpha_curves(
    alpha_mode: pd.DataFrame, out_dir: Path, export_tikz: bool
) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.4), constrained_layout=True)
    mode_colors = {"baseline": "#4C72B0", "defended": "#C44E52"}
    mode_labels = {"baseline": "Baseline", "defended": "Defended"}

    panels = [
        ("eval_revenue_mean", "Mean Episode Revenue", "Revenue"),
        ("eval_reward_mean", "Mean Episode Reward", "Reward"),
        ("eval_coi_leakage_mean", "Mean COI Leakage", "COI Leakage"),
        ("eval_volatility_mean", "Mean Price Volatility", "Volatility"),
    ]

    for ax, (metric_prefix, title, ylabel) in zip(axes.flat, panels):
        mean_col = f"{metric_prefix}_mean"
        std_col = f"{metric_prefix}_std"
        for mode in ("baseline", "defended"):
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
    return _save_figure(fig, out_dir / "wandb_alpha_curves.pdf", export_tikz)


def _plot_delta_curves(
    deltas: pd.DataFrame, out_dir: Path, export_tikz: bool
) -> list[Path]:
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
    axes[0].set_title("Defended Minus Baseline Delta by Contamination")
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

    return _save_figure(fig, out_dir / "wandb_delta_curves.pdf", export_tikz)


def _plot_tradeoff_scatter(
    deltas: pd.DataFrame, out_dir: Path, export_tikz: bool
) -> list[Path]:
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
    ax.set_title("Defended Tradeoff Frontier")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(r"Contamination $\alpha$")

    return _save_figure(fig, out_dir / "wandb_tradeoff_scatter.pdf", export_tikz)


def _plot_reward_robustness(
    alpha_mode: pd.DataFrame, out_dir: Path, export_tikz: bool
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.6, 4.5), constrained_layout=True)
    mode_colors = {"baseline": "#4C72B0", "defended": "#C44E52"}
    mode_labels = {"baseline": "Baseline", "defended": "Defended"}

    for mode in ("baseline", "defended"):
        sub = alpha_mode[alpha_mode["mode"] == mode].sort_values("alpha")
        x = sub["alpha"].to_numpy(dtype=float)
        y = sub["eval_reward_mean_std"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=mode_colors[mode],
            label=mode_labels[mode],
        )

    ax.set_title("Reward Robustness Across Contamination")
    ax.set_xlabel(r"Contamination $\alpha$")
    ax.set_ylabel("Reward Std Across Runs")
    ax.set_xticks(sorted(alpha_mode["alpha"].unique()))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
    ax.legend(loc="upper left")
    return _save_figure(fig, out_dir / "wandb_reward_robustness.pdf", export_tikz)


def _plot_parameter_sensitivity(
    summary: pd.DataFrame,
    parameter: str,
    out_name: str,
    out_dir: Path,
    export_tikz: bool,
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    values = sorted(summary[parameter].dropna().unique())
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(values))]

    panels = [
        ("eval_revenue_mean", "Revenue"),
        ("eval_coi_leakage_mean", "COI Leakage"),
    ]
    for ax, (metric_prefix, ylabel) in zip(axes, panels):
        mean_col = f"{metric_prefix}_mean"
        std_col = f"{metric_prefix}_std"
        for value, color in zip(values, colors):
            sub = summary[summary[parameter] == value].sort_values("alpha")
            if sub.empty:
                continue
            x = sub["alpha"].to_numpy(dtype=float)
            y = sub[mean_col].to_numpy(dtype=float)
            sigma = sub[std_col].fillna(0.0).to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.6,
                markersize=3.6,
                color=color,
                label=f"{parameter}={value:.2f}",
            )
            ax.fill_between(
                x, y - sigma, y + sigma, color=color, alpha=0.10, linewidth=0
            )

        ax.set_xlabel(r"Contamination $\alpha$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(summary["alpha"].unique()))
        if metric_prefix == "eval_revenue_mean":
            ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))

    axes[0].set_title(f"{parameter} Sensitivity (Defended)")
    axes[1].set_title("Leakage Side-Effect")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=max(1, len(values) // 2),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
    )

    return _save_figure(fig, out_dir / f"{out_name}.pdf", export_tikz)


def _plot_delta_summary(
    deltas: pd.DataFrame, out_dir: Path, export_tikz: bool
) -> list[Path]:
    data = deltas.sort_values("alpha")
    x = np.arange(len(data))
    labels = [f"{alpha:.1f}" for alpha in data["alpha"].to_numpy(dtype=float)]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), constrained_layout=True)
    panels = [
        ("eval_revenue_mean_delta_pct", "Revenue Delta (%)", "#4C72B0"),
        ("eval_reward_mean_delta_pct", "Reward Delta (%)", "#8172B3"),
        ("eval_coi_leakage_mean_delta_pct", "COI Leakage Delta (%)", "#55A868"),
    ]
    for ax, (column, title, color) in zip(axes, panels):
        values = data[column].to_numpy(dtype=float)
        ax.bar(x, values, color=color, alpha=0.85)
        ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(r"$\alpha$")
        ax.set_title(title)

    return _save_figure(fig, out_dir / "wandb_delta_summary.pdf", export_tikz)


def build_artifacts(
    input_path: Path,
    output_dir: Path,
    plot_dir: Path,
    include_non_finished: bool,
    export_tikz: bool,
) -> list[Path]:
    raw = pd.read_csv(input_path)
    frame = _prepare_frame(raw, include_non_finished=include_non_finished)

    metrics = [
        metric
        for metric in (
            "eval/revenue_mean",
            "eval/reward_mean",
            "eval/coi_level_mean",
            "eval/coi_leakage_mean",
            "eval/volatility_mean",
            "eval/margin_mean",
            "train/agent_prob",
            "train/alpha_adv",
        )
        if metric in frame.columns
    ]

    alpha_mode = _summary_by_alpha_mode(frame, metrics)
    deltas = _delta_by_alpha(alpha_mode, metrics)
    lambda_summary = _summary_by_parameter(frame, "lambda_coi", metrics)
    radius_summary = _summary_by_parameter(frame, "ambiguity_radius", metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    written.append(_save_table(alpha_mode, output_dir / "wandb_alpha_mode_summary.csv"))
    written.append(_save_table(deltas, output_dir / "wandb_alpha_deltas.csv"))
    written.append(
        _save_table(lambda_summary, output_dir / "wandb_lambda_alpha_summary.csv")
    )
    written.append(
        _save_table(radius_summary, output_dir / "wandb_radius_alpha_summary.csv")
    )

    written.extend(_plot_alpha_curves(alpha_mode, plot_dir, export_tikz))
    written.extend(_plot_delta_curves(deltas, plot_dir, export_tikz))
    written.extend(_plot_tradeoff_scatter(deltas, plot_dir, export_tikz))
    written.extend(_plot_reward_robustness(alpha_mode, plot_dir, export_tikz))
    written.extend(
        _plot_parameter_sensitivity(
            summary=lambda_summary,
            parameter="lambda_coi",
            out_name="wandb_lambda_sensitivity",
            out_dir=plot_dir,
            export_tikz=export_tikz,
        )
    )
    written.extend(
        _plot_parameter_sensitivity(
            summary=radius_summary,
            parameter="ambiguity_radius",
            out_name="wandb_radius_sensitivity",
            out_dir=plot_dir,
            export_tikz=export_tikz,
        )
    )
    written.extend(_plot_delta_summary(deltas, plot_dir, export_tikz))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate W&B sweep visualizations for PHANTOM results"
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Path to W&B export CSV"
    )
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--include-non-finished", action="store_true")
    parser.add_argument(
        "--export-tikz",
        action="store_true",
        help="Export matplotlib figures to TikZ via tikzplotlib",
    )
    args = parser.parse_args()

    _configure_style()
    plot_dir = (
        args.plot_dir
        if args.plot_dir is not None
        else _default_plot_dir(args.output_dir)
    )

    outputs = build_artifacts(
        input_path=args.input,
        output_dir=args.output_dir,
        plot_dir=plot_dir,
        include_non_finished=bool(args.include_non_finished),
        export_tikz=bool(args.export_tikz),
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
