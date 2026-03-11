"""validate core thesis problem: margin erosion under agent contamination
trains standard RL (no robust components) across α levels to demonstrate systematic failure
"""

from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from engine.spec import TrainSpec
from engine.orchestrators import run_train_once


def _run_baseline(alpha: float, algo: str, seed: int, steps: int) -> dict:
    spec = TrainSpec.from_flat(
        {
            "algo": algo,
            "seed": seed,
            "alpha": alpha,
            "total_timesteps": steps,
            "lambda_coi": 0.0,
            "robust_radius": 0.0,
            "robust_points": 1,
            "robust_rollouts": 1,
            "no_robust": True,
            "arch": "small",
            "n_products": 10,
            "N": 100,
            "max_steps": 50,
            "eval_freq": 5000,
            "eval_episodes": 10,
            "log_freq": 500,
        }
    )
    result = run_train_once(
        spec,
        project="phantom-margin-erosion",
        offline=True,
        no_wandb=True,
        kind="study",
        scenario=f"alpha{int(alpha * 100):02d}",
        group=f"baseline_{algo}",
        extra_tags=("margin_erosion", "baseline"),
    )
    return {
        "alpha": alpha,
        "algo": algo,
        "seed": seed,
        "eval_reward": result.get("eval/reward_mean", np.nan),
        "eval_revenue": result.get("eval/revenue_mean", np.nan),
        "eval_coi_level": result.get("eval/coi_level_mean", np.nan),
        "eval_margin": result.get("eval/margin_mean", np.nan),
        "eval_agent_prob": result.get("eval/agent_prob_mean", np.nan),
    }


def run_margin_erosion_study(
    alphas: list[float] | None = None,
    algos: list[str] | None = None,
    seeds: int = 3,
    steps: int = 30_000,
) -> dict:
    alphas = alphas or [0.1, 0.3, 0.5, 0.7, 0.9]
    algos = algos or ["ppo", "dqn", "qtable"]
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    results = []
    for α in alphas:
        for algo in algos:
            for si in range(seeds):
                seed = 42 + si
                print(f"α={α:.1f} {algo} seed={seed}")
                m = _run_baseline(α, algo, seed, steps)
                results.append(m)
                print(
                    f"  margin={m['eval_margin']:.3f} rev={m['eval_revenue']:.0f} coi={m['eval_coi_level']:.1f}"
                )

    summary = {}
    for α in alphas:
        runs = [r for r in results if abs(r["alpha"] - α) < 0.01]
        if not runs:
            continue
        s = {}
        for metric in ["margin", "revenue", "coi_level", "agent_prob"]:
            vals = [r[f"eval_{metric}"] for r in runs]
            s[f"{metric}_mean"] = float(np.mean(vals))
            s[f"{metric}_std"] = float(np.std(vals))
        s["n_runs"] = len(runs)
        summary[f"alpha_{α:.1f}"] = s

    output = {
        "timestamp": ts,
        "config": {"alphas": alphas, "algos": algos, "seeds": seeds, "steps": steps},
        "results": results,
        "summary": summary,
    }

    path = output_dir / f"margin_erosion_alpha_{ts}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n→ {path}")
    for α in alphas:
        k = f"alpha_{α:.1f}"
        if k in summary:
            s = summary[k]
            print(
                f"  {k}: margin={s['margin_mean']:.3f}±{s['margin_std']:.3f} "
                f"coi={s['coi_level_mean']:.1f}±{s['coi_level_std']:.1f}"
            )
    return output


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="margin erosion vs α")
    p.add_argument("--quick", action="store_true", help="fast test")
    args = p.parse_args()

    run_margin_erosion_study(
        alphas=[0.1, 0.7] if args.quick else [0.1, 0.3, 0.5, 0.7, 0.9],
        algos=["qtable"] if args.quick else ["ppo", "dqn", "qtable"],
        seeds=1 if args.quick else 3,
        steps=5_000 if args.quick else 30_000,
    )
