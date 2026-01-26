"""Summarize TensorBoard logs into comparison tables."""
from __future__ import annotations
import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import pandas as pd

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False


@dataclass
class RunInfo:
    algo: str
    alpha: float
    reward_mode: str
    path: Path


def parse_run_name(name: str) -> RunInfo | None:
    """Extract algo, alpha, reward_mode from run directory name."""
    # patterns: ppo_a0.20_robust, cmp_fixed_a0.20, sac_a0.90_robust
    m = re.match(r'(cmp_)?(\w+)_a([\d.]+)_?(\w+)?', name)
    if not m:
        return None
    prefix, algo, alpha, mode = m.groups()
    return RunInfo(algo=algo, alpha=float(alpha), reward_mode=mode or 'robust', path=Path())


def load_tb_scalars(log_dir: Path, tags: list[str], reduce: str = 'last') -> dict[str, float]:
    """Load scalar values from TensorBoard event files."""
    if not HAS_TB:
        return {}
    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    results = {}
    for tag in tags:
        if tag in ea.Tags().get('scalars', []):
            events = ea.Scalars(tag)
            if not events:
                continue
            vals = [e.value for e in events]
            if reduce == 'last':
                results[tag] = vals[-1]
            elif reduce == 'mean':
                results[tag] = sum(vals) / len(vals)
            elif reduce == 'max':
                results[tag] = max(vals)
            elif reduce == 'min':
                results[tag] = min(vals)
    return results


def load_json_results(log_dir: Path) -> dict[str, float]:
    """Load metrics from results.json if available."""
    results_file = log_dir / 'results.json'
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return {}


def discover_runs(base_dir: Path) -> list[RunInfo]:
    """Find all experiment runs in base directory."""
    runs = []
    for d in base_dir.iterdir():
        if not d.is_dir():
            continue
        info = parse_run_name(d.name)
        if info:
            info.path = d
            runs.append(info)
    return runs


def build_tables(runs: list[RunInfo], metrics: list[str], reduce: str = 'last') -> dict[str, dict[str, pd.DataFrame]]:
    """Build pivot tables: reward_mode -> metric -> DataFrame[alpha x algo]."""
    # collect data: {reward_mode: {metric: {(alpha, algo): value}}}
    data = defaultdict(lambda: defaultdict(dict))

    tb_tags = [f'economics/{m}' if m in ['revenue', 'profit', 'margin'] else f'coi/{m}' if m in ['erosion', 'leakage'] else f'alpha/{m}' for m in metrics]
    tag_map = dict(zip(tb_tags, metrics))

    for run in runs:
        # try json first (final eval metrics)
        jm = load_json_results(run.path)
        tb = load_tb_scalars(run.path, tb_tags, reduce)

        for tag, metric in tag_map.items():
            val = None
            json_key = f'{metric}_mean' if metric != 'reward' else 'reward_mean'
            if json_key in jm:
                val = jm[json_key]
            elif tag in tb:
                val = tb[tag]
            if val is not None:
                data[run.reward_mode][metric][(run.alpha, run.algo)] = val

    # convert to DataFrames
    tables = {}
    for mode, metrics_data in data.items():
        tables[mode] = {}
        for metric, vals in metrics_data.items():
            if not vals:
                continue
            alphas = sorted(set(a for a, _ in vals.keys()))
            algos = sorted(set(al for _, al in vals.keys()))
            df = pd.DataFrame(index=alphas, columns=algos, dtype=float)
            for (a, al), v in vals.items():
                df.loc[a, al] = v
            df.index.name = 'alpha'
            tables[mode][metric] = df
    return tables


def format_table(df: pd.DataFrame, fmt: str = '.3f') -> str:
    """Format DataFrame as markdown table."""
    return df.to_markdown(floatfmt=fmt)


def summarize(base_dir: str = 'sim/case/thesis_simplified/runs',
              metrics: list[str] | None = None,
              reduce: str = 'last',
              output: str | None = None) -> dict:
    """Generate summary tables from experiment runs."""
    base = Path(base_dir)
    metrics = metrics or ['revenue', 'profit', 'margin', 'erosion', 'leakage']

    runs = discover_runs(base)
    if not runs:
        print(f"No runs found in {base}")
        return {}

    print(f"Found {len(runs)} runs")
    tables = build_tables(runs, metrics, reduce)

    lines = []
    for mode, metric_tables in sorted(tables.items()):
        lines.append(f"\n# Reward Mode: {mode}\n")
        for metric, df in sorted(metric_tables.items()):
            lines.append(f"\n## {metric}\n")
            lines.append(format_table(df))
            lines.append("")

    report = '\n'.join(lines)
    print(report)

    if output:
        Path(output).write_text(report)
        print(f"\nSaved to {output}")

    return tables


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='sim/case/thesis_simplified/runs')
    p.add_argument('--metrics', nargs='+', default=['revenue', 'profit', 'margin', 'erosion', 'leakage'])
    p.add_argument('--reduce', default='last', choices=['last', 'mean', 'max', 'min'])
    p.add_argument('--output', '-o', help='save markdown to file')
    args = p.parse_args()
    summarize(args.dir, args.metrics, args.reduce, args.output)
