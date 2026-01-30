"""mixed design: full factorial on primary factors, latin hypercube on secondary"""
import sys
sys.path.insert(0, "..")
import logging
from itertools import product
import json
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats.qmc import LatinHypercube
from factors import FACTORS, DEMAND_FUNCTIONS, SEEDS_PER_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LH_SAMPLES = 10

def generate_configs(lh_samples: int = LH_SAMPLES):
    primary = [f for f in FACTORS if f.primary]
    secondary = [f for f in FACTORS if not f.primary]

    primary_grid = list(product(*[f.levels for f in primary]))
    lhs = LatinHypercube(d=len(secondary), seed=42)

    configs = []
    for p_combo in primary_grid:
        samples = lhs.random(n=lh_samples)
        for s in samples:
            sec_vals = {
                secondary[i].name: secondary[i].levels[int(s[i] * len(secondary[i].levels))]
                for i in range(len(secondary))
            }
            base = {primary[i].name: p_combo[i] for i in range(len(primary))}
            base.update(sec_vals)

            for seed in range(SEEDS_PER_CONFIG):
                cfg = {**base, "seed": seed}
                cfg["id"] = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
                configs.append(cfg)
    return configs

def run_single(cfg: dict) -> dict:
    from engine.wrapper import PHANTOM
    import numpy as np

    np.random.seed(cfg["seed"])
    demand_fn = DEMAND_FUNCTIONS[cfg["demand_fn"]]

    env = PHANTOM(
        n_products=cfg["n_products"],
        alpha=cfg["alpha"],
        N=cfg["N"],
    )
    env.market.demand = (demand_fn, (cfg["demand_mu"], cfg["demand_sigma"]))

    obs, _ = env.reset()
    total_reward, steps = 0.0, 0

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        steps += 1
        if term: break

    env.close()
    return {
        "id": cfg["id"],
        "config": cfg,
        "total_reward": total_reward,
        "avg_reward": total_reward / steps,
        "steps": steps,
    }

def run_study(max_workers: int = None, output: str = "results_mixed.jsonl", lh_samples: int = LH_SAMPLES):
    configs = generate_configs(lh_samples)
    n_primary_cells = int(np.prod([len(f.levels) for f in FACTORS if f.primary]))
    log.info(f"mixed LH: {len(configs)} configs ({n_primary_cells} primary × {lh_samples} LH × {SEEDS_PER_CONFIG} seeds)")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for i, result in enumerate(ex.map(run_single, configs)):
            results.append(result)
            if (i+1) % 100 == 0: log.info(f"progress: {i+1}/{len(configs)}")

    Path(output).write_text("\n".join(json.dumps(r) for r in results))
    log.info(f"wrote {len(results)} results to {output}")
    return results

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--output", default="results_mixed.jsonl")
    p.add_argument("--lh-samples", type=int, default=10)
    p.add_argument("--dry-run", action="store_true", help="only show design size")
    args = p.parse_args()

    primary = [f for f in FACTORS if f.primary]
    secondary = [f for f in FACTORS if not f.primary]
    configs = generate_configs(args.lh_samples)
    log.info(f"design: {len(configs)} runs | primary: {[f.name for f in primary]} | secondary (LH): {[f.name for f in secondary]}")

    if not args.dry_run:
        run_study(args.workers, args.output, args.lh_samples)
