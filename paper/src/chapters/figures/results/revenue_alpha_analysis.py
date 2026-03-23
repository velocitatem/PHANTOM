from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


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


def _bundle_dir_from_id(bundle_id: str) -> Path:
    token = str(bundle_id).strip()
    name = token if token.startswith("bundle_") else f"bundle_{token}"
    path = (
        _project_root()
        / "engine"
        / "studies"
        / "results"
        / "wandb_sweep_bundles"
        / name
    )
    if not path.exists():
        raise FileNotFoundError(f"Bundle not found: {path}")
    return path


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "generated" / "final"


def _truthy(value: object) -> bool:
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


def _coerce_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


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
            "eta_ux",
            "lambda_coi",
            "eval_revenue_mean",
        ],
    )
    frame = frame[frame["mode"].isin({"baseline", "defended"})].copy()
    return frame


def _get_git_commit() -> str:
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


def _apply_filters(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    data = frame.copy()
    if args.sweep_id:
        allowed = {str(value) for value in args.sweep_id}
        data = data[data["sweep_id"].astype(str).isin(allowed)]
    if args.mode != "all":
        data = data[data["mode"] == args.mode]
    if args.n_products is not None:
        data = data[data["n_products"] == float(args.n_products)]
    if args.eta_ux is not None:
        data = data[data["eta_ux"] == float(args.eta_ux)]
    if args.lambda_coi is not None:
        data = data[data["lambda_coi"] == float(args.lambda_coi)]
    data = data[data["alpha"].notna() & data["eval_revenue_mean"].notna()]
    data = data[data["alpha"] >= float(args.alpha_min)]
    data = data[data["alpha"] <= float(args.alpha_max)]
    return data.reset_index(drop=True)


def _design_matrix(
    frame: pd.DataFrame,
    *,
    include_sweep_fixed_effects: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    y = frame["eval_revenue_mean"].to_numpy(dtype=float)
    x_alpha = frame["alpha"].to_numpy(dtype=float)
    columns = ["intercept", "alpha"]
    blocks = [np.ones_like(x_alpha), x_alpha]
    if include_sweep_fixed_effects:
        dummies = pd.get_dummies(
            frame["sweep_id"].astype(str), prefix="sweep", drop_first=True
        )
        if not dummies.empty:
            blocks.append(dummies.to_numpy(dtype=float).T)
            columns.extend(dummies.columns.tolist())
    X = np.vstack(blocks).T
    return X, y, columns


def _covariance_hc1(X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    xr = X * residuals[:, None]
    meat = xr.T @ xr
    scale = float(n) / max(n - k, 1)
    return scale * (xtx_inv @ meat @ xtx_inv)


def _covariance_cluster(
    X: np.ndarray, residuals: np.ndarray, groups: pd.Series
) -> tuple[np.ndarray, int]:
    xtx_inv = np.linalg.pinv(X.T @ X)
    unique = pd.Series(groups).astype(str).dropna().unique().tolist()
    g = len(unique)
    n, k = X.shape
    if g <= 1:
        return _covariance_hc1(X, residuals), g
    meat = np.zeros((k, k), dtype=float)
    for value in unique:
        mask = pd.Series(groups).astype(str).to_numpy() == value
        Xg = X[mask]
        ug = residuals[mask]
        xu = Xg.T @ ug
        meat += np.outer(xu, xu)
    c = (g / (g - 1.0)) * ((n - 1.0) / max(n - k, 1.0))
    return c * (xtx_inv @ meat @ xtx_inv), g


def _fit_ols(
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    *,
    cov_type: str,
    groups: pd.Series | None = None,
) -> dict[str, object]:
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residuals = y - fitted
    dof = max(n - k, 1)
    sse = float(np.sum(residuals**2))
    y_centered = y - float(np.mean(y))
    sst = float(np.sum(y_centered**2))
    r2 = float(1.0 - sse / sst) if sst > 0 else 0.0
    adj_r2 = float(1.0 - (1.0 - r2) * ((n - 1.0) / max(n - k, 1.0)))

    if cov_type == "iid":
        sigma2 = sse / dof
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        df_t = dof
        clusters = None
    elif cov_type == "hc1":
        cov = _covariance_hc1(X, residuals)
        df_t = dof
        clusters = None
    elif cov_type == "cluster":
        if groups is None:
            raise ValueError("groups are required when cov_type='cluster'")
        cov, clusters = _covariance_cluster(X, residuals, groups)
        df_t = max(clusters - 1, 1)
    else:
        raise ValueError(f"Unsupported cov_type: {cov_type}")

    se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    t_stats = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df=df_t))
    t_crit = float(stats.t.ppf(0.975, df=df_t))
    ci_low = beta - t_crit * se
    ci_high = beta + t_crit * se

    coef_rows = []
    for idx, name in enumerate(columns):
        coef_rows.append(
            {
                "name": name,
                "coef": float(beta[idx]),
                "std_error": float(se[idx]),
                "t_stat": float(t_stats[idx]),
                "p_value": float(p_values[idx]),
                "ci95_low": float(ci_low[idx]),
                "ci95_high": float(ci_high[idx]),
            }
        )

    return {
        "n": int(n),
        "k": int(k),
        "dof": int(dof),
        "df_t": int(df_t),
        "cov_type": cov_type,
        "clusters": int(clusters) if clusters is not None else None,
        "r2": r2,
        "adj_r2": adj_r2,
        "sse": sse,
        "coefficients": coef_rows,
        "residuals": residuals,
        "fitted": fitted,
        "beta": beta,
    }


def _diagnostics(
    X: np.ndarray, y: np.ndarray, fit: dict[str, object]
) -> dict[str, object]:
    residuals = np.asarray(fit["residuals"], dtype=float)
    n, k = X.shape
    if residuals.size < 8:
        normality = {"test": "jarque_bera", "available": False}
    else:
        jb_stat, jb_p = stats.jarque_bera(residuals)
        normality = {
            "test": "jarque_bera",
            "available": True,
            "statistic": float(jb_stat),
            "p_value": float(jb_p),
        }

    if k <= 1:
        hetero = {"test": "breusch_pagan", "available": False}
    else:
        u2 = residuals**2
        aux = _fit_ols(X, u2, [f"x{i}" for i in range(k)], cov_type="iid")
        lm = float(len(u2) * float(aux["r2"]))
        df_bp = k - 1
        p_bp = float(1.0 - stats.chi2.cdf(lm, df_bp))
        hetero = {
            "test": "breusch_pagan",
            "available": True,
            "lm_stat": lm,
            "df": int(df_bp),
            "p_value": p_bp,
        }

    xtx_inv = np.linalg.pinv(X.T @ X)
    leverages = np.sum((X @ xtx_inv) * X, axis=1)
    mse = float(np.sum(residuals**2) / max(n - k, 1))
    if mse <= 0:
        cooks = np.zeros(n, dtype=float)
    else:
        denom = np.clip((1.0 - leverages) ** 2, 1e-10, np.inf)
        cooks = ((residuals**2) / (k * mse)) * (leverages / denom)

    return {
        "normality": normality,
        "heteroskedasticity": hetero,
        "influence": {
            "max_leverage": float(np.max(leverages)) if leverages.size else 0.0,
            "mean_leverage": float(np.mean(leverages)) if leverages.size else 0.0,
            "high_leverage_threshold": float(2.0 * k / max(n, 1)),
            "high_leverage_count": int(np.sum(leverages > (2.0 * k / max(n, 1)))),
            "max_cooks_distance": float(np.max(cooks)) if cooks.size else 0.0,
            "high_cooks_threshold": float(4.0 / max(n, 1)),
            "high_cooks_count": int(np.sum(cooks > (4.0 / max(n, 1)))),
        },
    }


def run(args: argparse.Namespace) -> list[Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(Path(args.bundle_dir))
    filtered = _apply_filters(runs, args)
    if len(filtered) < 3:
        raise ValueError("Filtered cohort must contain at least 3 rows")
    if filtered["alpha"].nunique() < 2:
        raise ValueError("Filtered cohort must contain at least 2 unique alpha values")

    filtered_csv = output_dir / "revenue_alpha_filtered.csv"
    filtered.to_csv(filtered_csv, index=False)

    sample_accounting = {
        "bundle_dir": str(Path(args.bundle_dir)),
        "git_commit": _get_git_commit(),
        "cohort_name": str(args.cohort_name),
        "filters": {
            "sweep_id": args.sweep_id,
            "mode": args.mode,
            "n_products": args.n_products,
            "eta_ux": args.eta_ux,
            "lambda_coi": args.lambda_coi,
            "alpha_min": args.alpha_min,
            "alpha_max": args.alpha_max,
        },
        "n_rows": int(len(filtered)),
        "n_sweeps": int(filtered["sweep_id"].nunique()),
        "alpha_unique": sorted(
            float(v) for v in filtered["alpha"].dropna().unique().tolist()
        ),
        "rows_by_sweep": filtered.groupby("sweep_id").size().astype(int).to_dict(),
        "rows_by_mode": filtered.groupby("mode").size().astype(int).to_dict(),
    }
    sample_path = output_dir / "revenue_alpha_sample_accounting.json"
    sample_path.write_text(json.dumps(sample_accounting, indent=2) + "\n")

    X_simple, y, cols_simple = _design_matrix(
        filtered, include_sweep_fixed_effects=False
    )
    fit_simple = _fit_ols(X_simple, y, cols_simple, cov_type="iid")
    simple_path = output_dir / "revenue_alpha_simple_ols.json"
    simple_path.write_text(
        json.dumps(
            {
                k: v
                for k, v in fit_simple.items()
                if k not in {"residuals", "fitted", "beta"}
            },
            indent=2,
        )
        + "\n"
    )

    X_fe, y_fe, cols_fe = _design_matrix(filtered, include_sweep_fixed_effects=True)
    cov_type = "cluster" if filtered["sweep_id"].nunique() > 1 else "hc1"
    fit_fe = _fit_ols(
        X_fe, y_fe, cols_fe, cov_type=cov_type, groups=filtered["sweep_id"]
    )
    fe_path = output_dir / "revenue_alpha_fixed_effects.json"
    fe_path.write_text(
        json.dumps(
            {
                k: v
                for k, v in fit_fe.items()
                if k not in {"residuals", "fitted", "beta"}
            },
            indent=2,
        )
        + "\n"
    )

    per_sweep_rows: list[dict[str, float | str | int]] = []
    for sweep_id, group in filtered.groupby("sweep_id"):
        if len(group) < 3 or group["alpha"].nunique() < 2:
            continue
        X_sw, y_sw, cols_sw = _design_matrix(group, include_sweep_fixed_effects=False)
        fit_sw = _fit_ols(X_sw, y_sw, cols_sw, cov_type="hc1")
        alpha_row = next(
            row for row in fit_sw["coefficients"] if row["name"] == "alpha"
        )
        per_sweep_rows.append(
            {
                "sweep_id": str(sweep_id),
                "n": int(fit_sw["n"]),
                "alpha_coef": float(alpha_row["coef"]),
                "alpha_std_error": float(alpha_row["std_error"]),
                "alpha_t_stat": float(alpha_row["t_stat"]),
                "alpha_p_value": float(alpha_row["p_value"]),
                "alpha_ci95_low": float(alpha_row["ci95_low"]),
                "alpha_ci95_high": float(alpha_row["ci95_high"]),
                "r2": float(fit_sw["r2"]),
            }
        )
    per_sweep_frame = pd.DataFrame(per_sweep_rows)
    if not per_sweep_frame.empty:
        per_sweep_frame = per_sweep_frame.sort_values("sweep_id").reset_index(drop=True)
    per_sweep_path = output_dir / "revenue_alpha_per_sweep.csv"
    per_sweep_frame.to_csv(per_sweep_path, index=False)

    fit_for_diagnostics = fit_fe if cov_type == "cluster" else fit_simple
    X_for_diagnostics = X_fe if cov_type == "cluster" else X_simple
    diagnostics = _diagnostics(X_for_diagnostics, y, fit_for_diagnostics)
    diagnostics_path = output_dir / "revenue_alpha_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")

    return [
        filtered_csv,
        sample_path,
        simple_path,
        fe_path,
        per_sweep_path,
        diagnostics_path,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproducible contamination-vs-revenue analysis from a sweep bundle"
    )
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--bundle-id", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--cohort-name", type=str, default="custom")
    parser.add_argument("--sweep-id", action="append", default=[])
    parser.add_argument(
        "--mode", choices=["all", "baseline", "defended"], default="all"
    )
    parser.add_argument("--n-products", type=float, default=None)
    parser.add_argument("--eta-ux", type=float, default=None)
    parser.add_argument("--lambda-coi", type=float, default=None)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    args = parser.parse_args()

    if args.bundle_id:
        args.bundle_dir = _bundle_dir_from_id(args.bundle_id)
    elif args.bundle_dir is None:
        args.bundle_dir = _default_bundle_dir()

    outputs = run(args)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
