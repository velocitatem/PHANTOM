"""full factorial design - all factor combinations"""

import sys

sys.path.insert(0, "..")
import logging
from itertools import product
import json
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from .factors import FACTORS, DEMAND_FUNCTIONS, SEEDS_PER_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def generate_configs():
    """generate all factor combinations with seeds"""
    all_levels = [f.levels for f in FACTORS]
    names = [f.name for f in FACTORS]

    configs = []
    for combo in product(*all_levels):
        base = {names[i]: combo[i] for i in range(len(names))}
        for seed in range(SEEDS_PER_CONFIG):
            cfg = {**base, "seed": seed}
            cfg["id"] = hashlib.md5(
                json.dumps(cfg, sort_keys=True).encode()
            ).hexdigest()[:8]
            configs.append(cfg)
    return configs


def run_single(cfg: dict) -> dict:
    """execute one experiment config, return metrics"""
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
        if term:
            break

    env.close()
    return {
        "id": cfg["id"],
        "config": cfg,
        "total_reward": total_reward,
        "avg_reward": total_reward / steps if steps > 0 else 0.0,
        "steps": steps,
    }


def run_study(max_workers: int = None, output: str = "results_full.jsonl"):
    configs = generate_configs()
    log.info(
        f"full factorial: {len(configs)} configs ({len(configs) // SEEDS_PER_CONFIG} unique × {SEEDS_PER_CONFIG} seeds)"
    )

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for i, result in enumerate(ex.map(run_single, configs)):
            results.append(result)
            if (i + 1) % 100 == 0:
                log.info(f"progress: {i + 1}/{len(configs)}")

    Path(output).write_text("\n".join(json.dumps(r) for r in results))
    log.info(f"wrote {len(results)} results to {output}")
    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--output", default="results_full.jsonl")
    p.add_argument("--dry-run", action="store_true", help="only show design size")
    args = p.parse_args()

    configs = generate_configs()
    log.info(
        f"design: {len(configs)} runs | factors: {[f.name for f in FACTORS]} | levels: {[len(f.levels) for f in FACTORS]}"
    )

    if not args.dry_run:
        run_study(args.workers, args.output)
