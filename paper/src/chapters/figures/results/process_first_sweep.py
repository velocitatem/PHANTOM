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
    return _project_root() / "tpu_orchestration" / "results" / "first_sweep.csv"


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


def _extract_tier(frame: pd.DataFrame) -> pd.Series:
    for column in ("tiers", "runtime/backend", "algo", "run.backend", "run.algo"):
        if column in frame.columns:
            tier = frame[column].astype(str).str.strip().str.lower()
            if tier.notna().any():
                return tier
    return pd.Series("unknown", index=frame.index, dtype="object")


def _prepare_frame(frame: pd.DataFrame, include_non_finished: bool) -> pd.DataFrame:
    data = frame.copy()
    if not include_non_finished and "State" in data.columns:
        data = data[data["State"].astype(str).str.lower() == "finished"].copy()

    data["alpha"] = _extract_alpha(data)
    data["mode"] = _extract_mode(data)
    data["tier"] = _extract_tier(data)
    data = data[data["mode"].isin({"robust", "no_robust"})]
    data = data[data["alpha"].notna()]

    _coerce_numeric(
        data,
        [
            "eval/revenue_mean",
            "eval/reward_mean",
            "eval/coi_level_mean",
            "eval/coi_leakage_mean",
            "eval/margin_mean",
            "eval/volatility_mean",
            "objective/score",
            "train/alpha_adv",
            "lambda_coi",
            "robust_radius",
            "learning_rate",
            "batch_size",
            "n_steps",
            "total_timesteps",
        ],
    )
    return data.sort_values(["tier", "alpha", "mode"]).reset_index(drop=True)


def _group_summary(
    frame: pd.DataFrame, by: list[str], metrics: list[str]
) -> pd.DataFrame:
    agg_spec: dict[str, tuple[str, str]] = {"runs": ("mode", "size")}
    for metric in metrics:
        safe = _sanitize(metric)
        agg_spec[f"{safe}_mean"] = (metric, "mean")
        agg_spec[f"{safe}_std"] = (metric, "std")
    return frame.groupby(by, as_index=False).agg(**agg_spec).sort_values(by)


def _tier_alpha_deltas(summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (tier, alpha), group in summary.groupby(["tier", "alpha"], sort=True):
        robust = group[group["mode"] == "robust"]
        no_robust = group[group["mode"] == "no_robust"]
        if robust.empty or no_robust.empty:
            continue

        row: dict[str, float | str] = {
            "tier": str(tier),
            "alpha": float(alpha),
            "runs_robust": float(robust["runs"].iloc[0]),
            "runs_no_robust": float(no_robust["runs"].iloc[0]),
        }
        for metric in metrics:
            safe = _sanitize(metric)
            robust_value = float(robust[f"{safe}_mean"].iloc[0])
            no_robust_value = float(no_robust[f"{safe}_mean"].iloc[0])
            delta = robust_value - no_robust_value
            row[f"{safe}_delta"] = delta
            row[f"{safe}_delta_pct"] = (
                np.nan if no_robust_value == 0 else 100.0 * delta / no_robust_value
            )
        rows.append(row)

    return pd.DataFrame(rows)


def _top_runs(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    rank_metric = "objective/score"
    if rank_metric not in frame.columns or frame[rank_metric].notna().sum() == 0:
        rank_metric = "eval/reward_mean"

    keep = [
        "Name",
        "tier",
        "alpha",
        "mode",
        rank_metric,
        "eval/revenue_mean",
        "eval/reward_mean",
        "eval/coi_level_mean",
        "eval/coi_leakage_mean",
        "lambda_coi",
        "robust_radius",
        "learning_rate",
        "batch_size",
        "n_steps",
        "total_timesteps",
    ]
    present = [column for column in keep if column in frame.columns]
    ranked = frame[present].copy().sort_values(rank_metric, ascending=False)
    return ranked.head(max(1, int(n))).reset_index(drop=True)


def _headline_json(
    frame: pd.DataFrame, tier_mode: pd.DataFrame
) -> dict[str, float | str]:
    out: dict[str, float | str] = {
        "runs": int(len(frame)),
        "tiers": int(frame["tier"].nunique()),
        "alphas": int(frame["alpha"].nunique()),
    }

    robust_rows = tier_mode[tier_mode["mode"] == "robust"]
    no_robust_rows = tier_mode[tier_mode["mode"] == "no_robust"]
    if robust_rows.empty or no_robust_rows.empty:
        out["status"] = "incomplete_modes"
        return out

    robust_mean = robust_rows["eval_revenue_mean_mean"].mean()
    no_robust_mean = no_robust_rows["eval_revenue_mean_mean"].mean()
    out.update(
        {
            "status": "ok",
            "mean_tier_revenue_robust": float(robust_mean),
            "mean_tier_revenue_no_robust": float(no_robust_mean),
            "mean_tier_revenue_delta": float(robust_mean - no_robust_mean),
            "mean_tier_revenue_delta_pct": float(
                100.0 * (robust_mean - no_robust_mean) / no_robust_mean
            )
            if no_robust_mean
            else np.nan,
        }
    )
    return out


def run(
    input_path: Path, output_dir: Path, include_non_finished: bool, top_n: int
) -> list[Path]:
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
            "eval/margin_mean",
            "eval/volatility_mean",
            "objective/score",
            "train/alpha_adv",
        )
        if metric in frame.columns
    ]

    tier_mode = _group_summary(frame, ["tier", "mode"], metrics)
    tier_alpha_mode = _group_summary(frame, ["tier", "alpha", "mode"], metrics)
    deltas = _tier_alpha_deltas(tier_alpha_mode, metrics)
    top_configs = _top_runs(frame, n=top_n)
    headline = _headline_json(frame, tier_mode)

    outputs = {
        "first_sweep_tier_mode_summary.csv": tier_mode,
        "first_sweep_tier_alpha_mode_summary.csv": tier_alpha_mode,
        "first_sweep_tier_alpha_deltas.csv": deltas,
        "first_sweep_top_configs.csv": top_configs,
    }
    written_paths: list[Path] = []
    for filename, table in outputs.items():
        path = output_dir / filename
        table.to_csv(path, index=False)
        written_paths.append(path)

    headline_path = output_dir / "first_sweep_headline_summary.json"
    headline_path.write_text(json.dumps(headline, indent=2))
    written_paths.append(headline_path)
    return written_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process first sweep CSV for paper tables"
    )
    parser.add_argument("--input", type=Path, default=_default_input())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--include-non-finished", action="store_true")
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()

    written = run(
        input_path=args.input,
        output_dir=args.output_dir,
        include_non_finished=bool(args.include_non_finished),
        top_n=int(args.top_n),
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
