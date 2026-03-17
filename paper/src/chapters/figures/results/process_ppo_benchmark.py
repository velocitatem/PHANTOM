from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_input() -> Path:
    return _project_root() / "tpu_orchestration" / "results" / "ppo_benchmark.csv"


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "generated" / "legacy"


def _sanitize(key: str) -> str:
    return key.replace("/", "_").replace("-", "_")


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
        return frame["study/mode"].astype(str).str.strip().str.lower()
    if "study/no_robust" in frame.columns:
        no_robust = pd.to_numeric(frame["study/no_robust"], errors="coerce").fillna(0.0)
        return pd.Series(
            np.where(no_robust > 0.5, "no_robust", "robust"),
            index=frame.index,
            dtype="object",
        )
    if "no_robust" in frame.columns:
        no_robust = (
            frame["no_robust"].astype(str).str.lower().isin({"1", "true", "yes"})
        )
        return pd.Series(
            np.where(no_robust, "no_robust", "robust"),
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
    data = data[data["mode"].isin({"robust", "no_robust"})]
    data = data[data["alpha"].notna()]

    numeric_cols = [
        "eval/revenue_mean",
        "eval/reward_mean",
        "eval/coi_level_mean",
        "eval/coi_leakage_mean",
        "eval/volatility_mean",
        "eval/margin_mean",
        "train/alpha_adv",
        "train/coi_penalty",
        "train/ux_penalty",
        "train/agent_prob",
    ]
    _coerce_numeric(data, numeric_cols)
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
        robust = alpha_group[alpha_group["mode"] == "robust"]
        no_robust = alpha_group[alpha_group["mode"] == "no_robust"]
        if robust.empty or no_robust.empty:
            continue

        row: dict[str, float] = {
            "alpha": float(alpha),
            "runs_robust": float(robust["runs"].iloc[0]),
            "runs_no_robust": float(no_robust["runs"].iloc[0]),
        }
        for metric in metrics:
            safe = _sanitize(metric)
            robust_value = float(robust[f"{safe}_mean"].iloc[0])
            no_robust_value = float(no_robust[f"{safe}_mean"].iloc[0])
            delta = robust_value - no_robust_value
            row[f"{safe}_robust"] = robust_value
            row[f"{safe}_no_robust"] = no_robust_value
            row[f"{safe}_delta"] = delta
            row[f"{safe}_delta_pct"] = (
                np.nan if no_robust_value == 0 else 100.0 * delta / no_robust_value
            )
        rows.append(row)

    return pd.DataFrame(rows)


def _pairwise_win_rates(frame: pd.DataFrame) -> pd.DataFrame:
    rules = {
        "eval/revenue_mean": "higher",
        "eval/reward_mean": "higher",
        "eval/coi_leakage_mean": "lower",
        "eval/volatility_mean": "lower",
    }
    rows: list[dict[str, float]] = []
    for alpha, alpha_group in frame.groupby("alpha", sort=True):
        robust = alpha_group[alpha_group["mode"] == "robust"]
        no_robust = alpha_group[alpha_group["mode"] == "no_robust"]
        if robust.empty or no_robust.empty:
            continue

        for metric, direction in rules.items():
            if metric not in frame.columns:
                continue
            robust_values = robust[metric].dropna().to_numpy(dtype=float)
            no_robust_values = no_robust[metric].dropna().to_numpy(dtype=float)
            if robust_values.size == 0 or no_robust_values.size == 0:
                continue

            if direction == "higher":
                wins = (robust_values[:, None] > no_robust_values[None, :]).sum()
            else:
                wins = (robust_values[:, None] < no_robust_values[None, :]).sum()
            ties = (robust_values[:, None] == no_robust_values[None, :]).sum()
            total = robust_values.size * no_robust_values.size
            win_prob = (wins + 0.5 * ties) / total
            rows.append(
                {
                    "alpha": float(alpha),
                    "metric": metric,
                    "direction": direction,
                    "wins": int(wins),
                    "ties": int(ties),
                    "total_pairs": int(total),
                    "win_probability": float(win_prob),
                }
            )
    return pd.DataFrame(rows)


def _overall_mode_summary(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    agg_spec: dict[str, tuple[str, str]] = {"runs": ("mode", "size")}
    for metric in metrics:
        safe = _sanitize(metric)
        agg_spec[f"{safe}_mean"] = (metric, "mean")
        agg_spec[f"{safe}_std"] = (metric, "std")
    return frame.groupby("mode", as_index=False).agg(**agg_spec).sort_values("mode")


def _headline_json(overall: pd.DataFrame) -> dict[str, float | str]:
    if {"robust", "no_robust"} - set(overall["mode"].tolist()):
        return {"status": "incomplete_modes"}

    robust = overall[overall["mode"] == "robust"].iloc[0]
    no_robust = overall[overall["mode"] == "no_robust"].iloc[0]

    revenue_delta = float(
        robust["eval_revenue_mean_mean"] - no_robust["eval_revenue_mean_mean"]
    )
    leakage_delta = float(
        robust["eval_coi_leakage_mean_mean"] - no_robust["eval_coi_leakage_mean_mean"]
    )
    return {
        "status": "ok",
        "revenue_delta": revenue_delta,
        "revenue_delta_pct": float(
            100.0 * revenue_delta / no_robust["eval_revenue_mean_mean"]
        ),
        "coi_leakage_delta": leakage_delta,
        "coi_leakage_delta_pct": float(
            100.0 * leakage_delta / no_robust["eval_coi_leakage_mean_mean"]
        ),
    }


def run(input_path: Path, output_dir: Path, include_non_finished: bool) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
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
            "train/alpha_adv",
            "train/coi_penalty",
            "train/ux_penalty",
            "train/agent_prob",
        )
        if metric in frame.columns
    ]

    alpha_mode = _summary_by_alpha_mode(frame, metrics)
    deltas = _delta_by_alpha(alpha_mode, metrics)
    win_rates = _pairwise_win_rates(frame)
    overall = _overall_mode_summary(frame, metrics)
    headline = _headline_json(overall)

    outputs = {
        "ppo_alpha_mode_summary.csv": alpha_mode,
        "ppo_alpha_deltas.csv": deltas,
        "ppo_pairwise_win_rates.csv": win_rates,
        "ppo_overall_mode_summary.csv": overall,
    }
    written_paths: list[Path] = []
    for filename, table in outputs.items():
        path = output_dir / filename
        table.to_csv(path, index=False)
        written_paths.append(path)

    headline_path = output_dir / "ppo_headline_summary.json"
    headline_path.write_text(json.dumps(headline, indent=2))
    written_paths.append(headline_path)
    return written_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process PPO benchmark CSV for paper tables"
    )
    parser.add_argument("--input", type=Path, default=_default_input())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--include-non-finished", action="store_true")
    args = parser.parse_args()

    written = run(
        input_path=args.input,
        output_dir=args.output_dir,
        include_non_finished=bool(args.include_non_finished),
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
