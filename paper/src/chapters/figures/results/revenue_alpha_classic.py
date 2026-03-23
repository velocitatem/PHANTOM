from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


root = Path(__file__).resolve().parents[5]
runs = (
    root
    / "engine/studies/results/wandb_sweep_bundles/bundle_20260317_122818/runs_finished.csv"
)

df = pd.read_csv(runs)
df = df[
    (df["sweep_id"].astype(str) == "i88nw811")
    & (df["study_mode"].astype(str) == "baseline")
    & (pd.to_numeric(df["n_products"], errors="coerce") == 100.0)
    & (pd.to_numeric(df["eta_ux"], errors="coerce") == 0.0)
].copy()

alpha = pd.to_numeric(df["alpha"], errors="coerce")
revenue = pd.to_numeric(df["eval_revenue_mean"], errors="coerce")
mask = alpha.notna() & revenue.notna()
alpha = alpha[mask].to_numpy(dtype=float)
revenue = revenue[mask].to_numpy(dtype=float)

if len(alpha) < 3 or np.unique(alpha).size < 2:
    raise ValueError("Not enough data for regression")

fit = stats.linregress(alpha, revenue)
n = len(alpha)
dof = n - 2
t_stat = fit.slope / fit.stderr
p_val = 2.0 * stats.t.sf(abs(t_stat), df=dof)
r2 = fit.rvalue**2
t_crit = stats.t.ppf(0.975, dof)
slope_ci = (fit.slope - t_crit * fit.stderr, fit.slope + t_crit * fit.stderr)

x = np.column_stack([np.ones(n), alpha])
beta = np.linalg.lstsq(x, revenue, rcond=None)[0]
resid = revenue - x @ beta
xtx_inv = np.linalg.pinv(x.T @ x)
meat = (x * resid[:, None]).T @ (x * resid[:, None])
cov_hc1 = (n / (n - x.shape[1])) * (xtx_inv @ meat @ xtx_inv)
se_hc1 = np.sqrt(np.diag(cov_hc1))
t_hc1 = beta[1] / se_hc1[1]
p_hc1 = 2.0 * stats.t.sf(abs(t_hc1), df=dof)
slope_ci_hc1 = (beta[1] - t_crit * se_hc1[1], beta[1] + t_crit * se_hc1[1])

print("Contamination-Revenue Slope")
print(
    "cohort: bundle_20260317_122818, sweep=i88nw811, mode=baseline, n_products=100, eta_ux=0.0"
)
print(f"n={n}")
print(f"model: revenue = {fit.intercept:.2f} {fit.slope:+.2f} * alpha")
print(
    f"OLS: t({dof})={t_stat:.2f}, p={p_val:.3e}, R^2={r2:.3f}, slope_95CI=[{slope_ci[0]:.2f}, {slope_ci[1]:.2f}]"
)
print(
    f"HC1: t={t_hc1:.2f}, p={p_hc1:.3e}, slope_95CI=[{slope_ci_hc1[0]:.2f}, {slope_ci_hc1[1]:.2f}]"
)
print(f"effect: +0.1 alpha -> {0.1 * fit.slope:.2f} revenue units")
