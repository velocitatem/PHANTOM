from __future__ import annotations

import argparse
import contextlib
import concurrent.futures
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def _has_flag(tokens: list[str], name: str) -> bool:
    return any(tok == name or tok.startswith(f"{name}=") for tok in tokens)


def _entry_tokens(run_kind: str, entry_args: str) -> list[str]:
    tokens = shlex.split(entry_args)
    if run_kind == "benchmark" and not (
        _has_flag(tokens, "--run-kind") or _has_flag(tokens, "--run-mode")
    ):
        return ["--run-kind", "benchmark", *tokens]
    return tokens


def _get_flag_value(tokens: list[str], name: str, default: str = "") -> str:
    for idx, tok in enumerate(tokens):
        if tok == name and idx + 1 < len(tokens):
            return str(tokens[idx + 1])
        if tok.startswith(f"{name}="):
            return str(tok.split("=", 1)[1])
    return str(default)


def _set_flag_value(tokens: list[str], name: str, value: str) -> list[str]:
    updated: list[str] = []
    replaced = False
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if tok == name:
            replaced = True
            updated.extend([name, str(value)])
            idx += 2
            continue
        if tok.startswith(f"{name}="):
            replaced = True
            updated.append(f"{name}={value}")
            idx += 1
            continue
        updated.append(tok)
        idx += 1
    if not replaced:
        updated.extend([name, str(value)])
    return updated


def _remove_flag(tokens: list[str], name: str) -> list[str]:
    updated: list[str] = []
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if tok == name:
            idx += 1
            continue
        if tok.startswith(f"{name}="):
            idx += 1
            continue
        updated.append(tok)
        idx += 1
    return updated


def _csv_values(raw: str) -> list[str]:
    return [piece.strip() for piece in str(raw).split(",") if piece.strip()]


def _alpha_token(alpha: str) -> str:
    return str(alpha).replace(".", "p").replace("-", "m")


def _truthy(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _alive_nodes() -> list[tuple[str, str, bool, float]]:
    seen: set[str] = set()
    nodes: list[tuple[str, str, bool, float]] = []
    for node in ray.nodes():
        if not bool(node.get("Alive", False)):
            continue
        node_id = str(node.get("NodeID", "")).strip()
        ip = str(node.get("NodeManagerAddress", "")).strip()
        if not node_id or not ip or node_id in seen:
            continue
        resources = node.get("Resources", {}) or {}
        is_head = bool(resources.get("node:__internal_head__", 0.0))
        tpu = float(resources.get("TPU", 0.0))
        seen.add(node_id)
        nodes.append((node_id, ip, is_head, tpu))
    return sorted(nodes, key=lambda item: (item[1], item[2], -item[3], item[0]))


def _dedupe_nodes_for_tpu(
    nodes: list[tuple[str, str, bool, float]],
) -> tuple[list[tuple[str, str]], list[dict[str, str | float | bool]]]:
    selected: dict[str, tuple[str, str, bool, float]] = {}
    dropped: list[dict[str, str | float | bool]] = []

    def _score(item: tuple[str, str, bool, float]) -> tuple[int, float, str]:
        node_id, _ip, is_head, tpu = item
        return (1 if bool(is_head) else 0, -float(tpu), str(node_id))

    for item in nodes:
        node_id, ip, is_head, tpu = item
        existing = selected.get(ip)
        if existing is None:
            selected[ip] = item
            continue

        keep, drop = (
            (item, existing) if _score(item) < _score(existing) else (existing, item)
        )
        selected[ip] = keep
        dropped.append(
            {
                "ip": str(ip),
                "dropped_node_id": str(drop[0]),
                "dropped_is_head": bool(drop[2]),
                "dropped_tpu": float(drop[3]),
                "kept_node_id": str(keep[0]),
                "kept_is_head": bool(keep[2]),
                "kept_tpu": float(keep[3]),
            }
        )

    entries = [(node_id, ip) for ip, (node_id, _ip, _is_head, _tpu) in selected.items()]
    entries.sort(key=lambda item: (item[1], item[0]))
    return entries, dropped


def _benchmark_cells(
    tokens: list[str], *, compare_robust: bool
) -> list[tuple[str, str, str, bool]]:
    tiers = _csv_values(
        _get_flag_value(tokens, "--tiers", "static,surge,linear,qtable,ppo")
    )
    alphas = _csv_values(_get_flag_value(tokens, "--alpha-values", "0.0,0.3,0.6"))
    base_no_robust = _has_flag(tokens, "--no-robust")
    if compare_robust:
        modes = [("robust", False), ("no_robust", True)]
    else:
        modes = [("no_robust", True)] if base_no_robust else [("robust", False)]
    return [
        (tier, alpha, mode_label, no_robust)
        for tier in tiers
        for alpha in alphas
        for mode_label, no_robust in modes
    ]


def _thread_limited_env(env: dict[str, str], threads: int) -> dict[str, str]:
    bounded = dict(env)
    n = str(max(1, int(threads)))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        bounded[key] = n
    return bounded


@contextlib.contextmanager
def _semaphore_guard(semaphore: threading.Semaphore | None):
    if semaphore is None:
        yield
        return
    semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


def _run_benchmark_cells_parallel(
    *,
    root: str,
    env: dict[str, str],
    base_tokens: list[str],
    compare_robust: bool,
    inner_workers: int,
    inner_threads: int,
    max_heavy_workers: int,
    rank: int,
) -> int:
    cells = _benchmark_cells(base_tokens, compare_robust=compare_robust)
    if not cells:
        return 0

    cwd = str(Path(root))
    base_out = _get_flag_value(base_tokens, "--output-dir", "engine/studies/results")
    max_workers = max(1, min(int(inner_workers), len(cells)))
    heavy_tiers = {"ppo", "a2c", "dqn"}
    heavy_limit = max(1, int(max_heavy_workers))
    heavy_sem = threading.Semaphore(heavy_limit)
    print(
        {
            "rank": int(rank),
            "benchmark_cells": len(cells),
            "inner_workers": int(max_workers),
            "inner_threads": int(max(1, int(inner_threads))),
            "heavy_limit": int(heavy_limit),
        }
    )

    def _run_cell(
        index: int,
        total: int,
        tier: str,
        alpha: str,
        mode_label: str,
        no_robust: bool,
    ) -> tuple[str, str, str, int]:
        tokens = list(base_tokens)
        tokens = _set_flag_value(tokens, "--tiers", tier)
        tokens = _set_flag_value(tokens, "--alpha-values", alpha)
        if no_robust:
            if not _has_flag(tokens, "--no-robust"):
                tokens.append("--no-robust")
        else:
            tokens = _remove_flag(tokens, "--no-robust")

        cell_out = (
            Path(base_out)
            / f"tier_{tier}"
            / f"mode_{mode_label}"
            / f"alpha_{_alpha_token(alpha)}"
        )
        tokens = _set_flag_value(tokens, "--output-dir", str(cell_out))
        cmd = [sys.executable, "-m", "engine.train", *tokens]
        cell_env = _thread_limited_env(env, int(inner_threads))
        cell_env["PHANTOM_BENCHMARK_COMPARE_ROBUST"] = "0"
        print(
            {
                "rank": int(rank),
                "cell": f"{index}/{total}",
                "tier": tier,
                "mode": mode_label,
                "alpha": alpha,
                "command": " ".join(cmd),
            }
        )
        heavy_guard = heavy_sem if str(tier).lower() in heavy_tiers else None
        with _semaphore_guard(heavy_guard):
            proc = subprocess.run(cmd, cwd=cwd, env=cell_env)
        return tier, alpha, mode_label, int(proc.returncode)

    failures: list[tuple[str, str, str, int]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_run_cell, idx, len(cells), tier, alpha, mode_label, no_robust)
            for idx, (tier, alpha, mode_label, no_robust) in enumerate(cells, start=1)
        ]
        for fut in concurrent.futures.as_completed(futures):
            tier, alpha, mode_label, code = fut.result()
            if code != 0:
                failures.append((tier, alpha, mode_label, code))

    if failures:
        print({"rank": int(rank), "benchmark_failures": failures})
        return 1
    return 0


def _run_sweep_agents_parallel(
    *,
    root: str,
    env: dict[str, str],
    base_tokens: list[str],
    run_kind: str,
    rank: int,
    agents_per_node: int,
    agent_count: int,
    inner_threads: int,
    tpu_agent_slots: int,
) -> int:
    total = max(1, int(agents_per_node))
    cwd = str(Path(root))
    wants_tpu = str(env.get("JAX_PLATFORMS", "")).strip().lower() == "tpu"
    tpu_slots = max(0, int(tpu_agent_slots))
    print(
        {
            "rank": int(rank),
            "sweep_agents": int(total),
            "agent_count": int(agent_count),
            "inner_threads": int(max(1, int(inner_threads))),
            "jax_platform": str(env.get("JAX_PLATFORMS", "")),
            "tpu_agent_slots": int(tpu_slots),
        }
    )

    def _run_agent(slot: int) -> int:
        tokens = list(base_tokens)
        if int(agent_count) > 0 and not _has_flag(tokens, "--count"):
            tokens.extend(["--count", str(int(agent_count))])

        if _has_flag(tokens, "--group"):
            base_group = _get_flag_value(tokens, "--group", "ray-sweep")
            tokens = _set_flag_value(tokens, "--group", f"{base_group}-a{slot}")

        if run_kind == "benchmark":
            out_dir = _get_flag_value(tokens, "--output-dir", "engine/studies/results")
            tokens = _set_flag_value(
                tokens, "--output-dir", str(Path(out_dir) / f"agent_{slot}")
            )
        if run_kind == "train":
            model_dir = _get_flag_value(tokens, "--model-dir", "engine/models")
            tokens = _set_flag_value(
                tokens, "--model-dir", str(Path(model_dir) / f"agent_{slot}")
            )

        cmd = [sys.executable, "-m", "engine.train", *tokens]
        agent_env = _thread_limited_env(env, int(inner_threads))
        if wants_tpu and tpu_slots > 0 and int(slot) > tpu_slots:
            agent_env["JAX_PLATFORMS"] = "cpu"
            agent_env["JAX_PLATFORM_NAME"] = "cpu"
        agent_env["PHANTOM_SWEEP_AGENT_SLOT"] = str(int(slot))
        print(
            {
                "rank": int(rank),
                "agent_slot": int(slot),
                "jax_platform": str(agent_env.get("JAX_PLATFORMS", "")),
                "command": " ".join(cmd),
            }
        )
        proc = subprocess.run(cmd, cwd=cwd, env=agent_env)
        return int(proc.returncode)

    failures: list[tuple[int, int]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=total) as pool:
        future_map = {
            pool.submit(_run_agent, slot): slot for slot in range(1, total + 1)
        }
        for future in concurrent.futures.as_completed(future_map):
            slot = int(future_map[future])
            code = int(future.result())
            if code != 0:
                failures.append((slot, code))

    if failures:
        print({"rank": int(rank), "sweep_failures": failures})
        return 1
    return 0


@ray.remote(max_retries=0)
def _train_on_node(
    *,
    root: str,
    run_kind: str,
    entry_args: str,
    node_id: str,
    node_ip: str,
    rank: int,
    world_size: int,
    coordinator_ip: str,
    coordinator_port: int,
    base_seed: int,
    run_group: str,
    compare_robust: bool,
    output_root: str,
    wandb_entity: str,
    wandb_project: str,
    agents_per_node: int,
    agent_count: int,
    inner_workers: int,
    inner_threads: int,
    max_heavy_workers: int,
    sync_jax: bool,
) -> int:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    requested_platform = str(env.get("PHANTOM_JAX_PLATFORM", "tpu")).strip().lower()
    allow_multi_node_tpu = _truthy(env.get("PHANTOM_ALLOW_MULTI_NODE_TPU"))
    if world_size > 1 and requested_platform == "tpu" and not allow_multi_node_tpu:
        requested_platform = "cpu"
        print(
            "PHANTOM_DISTRIBUTED_NOTE: forcing JAX_PLATFORMS=cpu for multi-node SB3 runs "
            "(set PHANTOM_ALLOW_MULTI_NODE_TPU=1 to keep TPU for JAX workloads)"
        )
    elif world_size > 1 and requested_platform == "tpu" and allow_multi_node_tpu:
        print(
            "PHANTOM_DISTRIBUTED_NOTE: keeping JAX_PLATFORMS=tpu in multi-node mixed mode"
        )
    env["JAX_PLATFORMS"] = requested_platform
    if requested_platform == "cpu":
        env["JAX_PLATFORM_NAME"] = "cpu"
    else:
        env.pop("JAX_PLATFORM_NAME", None)
    if requested_platform == "tpu" and world_size > 1 and allow_multi_node_tpu:
        env["CLOUD_TPU_TASK_ID"] = str(int(rank))
        print(
            {
                "rank": int(rank),
                "node_ip": str(node_ip),
                "jax_platform": "tpu",
                "cloud_tpu_task_id": str(env["CLOUD_TPU_TASK_ID"]),
            }
        )
    else:
        # Keep each process in single-host mode when TPU multi-host is disabled.
        env["CLOUD_TPU_TASK_ID"] = "0"
    if run_kind == "benchmark":
        env["PHANTOM_BENCHMARK_COMPARE_ROBUST"] = "1" if compare_robust else "0"
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity
    if wandb_project:
        env["WANDB_PROJECT"] = wandb_project

    cwd = str(Path(root))

    try:
        subprocess.run(["make", "data.pull"], cwd=cwd, env=env, check=True)
    except (subprocess.SubprocessError, OSError):
        pull_cmd = [sys.executable, "scripts/hf_data.py", "pull"]
        subprocess.run(pull_cmd, cwd=cwd, env=env, check=True)

    if sync_jax and requested_platform == "tpu":
        env_probe = dict(env)
        env_probe["CLOUD_TPU_TASK_ID"] = str(rank)
        probe = (
            "import jax; "
            f"jax.distributed.initialize(coordinator_address='{coordinator_ip}:{coordinator_port}', "
            f"num_processes={world_size}, process_id={rank}); "
            "print('JAX_SYNC', jax.process_index(), jax.device_count(), jax.local_device_count())"
        )
        subprocess.run(
            [sys.executable, "-c", probe], cwd=cwd, env=env_probe, check=True
        )

    tokens = _entry_tokens(run_kind, entry_args)
    is_sweep_agent = _has_flag(tokens, "--sweep-agent")
    seed = int(base_seed + rank)
    if not is_sweep_agent and not _has_flag(tokens, "--seed"):
        tokens.extend(["--seed", str(seed)])

    if run_kind == "train" and not _has_flag(tokens, "--group"):
        tokens.extend(["--group", run_group])

    if is_sweep_agent and int(agent_count) > 0 and not _has_flag(tokens, "--count"):
        tokens.extend(["--count", str(int(agent_count))])

    try:
        tpu_agent_slots = int(
            str(
                env.get(
                    "PHANTOM_TPU_AGENT_SLOTS",
                    "1" if requested_platform == "tpu" else "0",
                )
            ).strip()
        )
    except ValueError:
        tpu_agent_slots = 1 if requested_platform == "tpu" else 0

    if (
        run_kind == "benchmark"
        and output_root
        and not _has_flag(tokens, "--output-dir")
    ):
        out_dir = Path(output_root) / f"rank_{rank}" / f"seed_{seed}"
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        tokens.extend(["--output-dir", str(out_dir)])

    if is_sweep_agent and int(agents_per_node) > 1:
        return _run_sweep_agents_parallel(
            root=root,
            env=env,
            base_tokens=tokens,
            run_kind=run_kind,
            rank=rank,
            agents_per_node=int(agents_per_node),
            agent_count=int(agent_count),
            inner_threads=int(inner_threads),
            tpu_agent_slots=int(max(0, tpu_agent_slots)),
        )

    if run_kind == "benchmark" and int(inner_workers) > 1 and not is_sweep_agent:
        return _run_benchmark_cells_parallel(
            root=root,
            env=env,
            base_tokens=tokens,
            compare_robust=bool(compare_robust),
            inner_workers=int(inner_workers),
            inner_threads=int(inner_threads),
            max_heavy_workers=int(max_heavy_workers),
            rank=rank,
        )

    cmd = [sys.executable, "-m", "engine.train", *tokens]
    print(
        {
            "node_id": node_id,
            "node_ip": node_ip,
            "rank": int(rank),
            "run_kind": run_kind,
            "seed": int(seed),
            "compare_robust": bool(compare_robust),
            "wandb_entity": str(env.get("WANDB_ENTITY", "")),
            "wandb_project": str(env.get("WANDB_PROJECT", "")),
            "command": " ".join(cmd),
        }
    )
    proc = subprocess.run(
        cmd, cwd=cwd, env=_thread_limited_env(env, int(inner_threads))
    )
    return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch one train/benchmark run per Ray TPU node"
    )
    parser.add_argument("--run-kind", choices=["train", "benchmark"], default="train")
    parser.add_argument("--entry-args", type=str, default="")
    parser.add_argument("--train-args", type=str, default="")
    parser.add_argument("--num-nodes", type=int, default=0)
    parser.add_argument("--tpu-per-task", type=float, default=8.0)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--sync-jax", action="store_true")
    parser.add_argument("--coordinator-port", type=int, default=12355)
    parser.add_argument("--run-group", type=str, default="")
    parser.add_argument("--compare-robust", action="store_true")
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--agents-per-node", type=int, default=1)
    parser.add_argument("--agent-count", type=int, default=0)
    parser.add_argument("--inner-workers", type=int, default=1)
    parser.add_argument("--inner-threads", type=int, default=1)
    parser.add_argument("--max-heavy-workers", type=int, default=2)
    parser.add_argument("--worker-cpus", type=float, default=1.0)
    args = parser.parse_args()

    entry_args = str(args.entry_args or args.train_args).strip()
    if not entry_args:
        raise ValueError("--entry-args (or legacy --train-args) is required")

    ray.init(address="auto")

    node_records = _alive_nodes()
    if not node_records:
        raise RuntimeError("No alive Ray nodes found")

    if float(args.tpu_per_task) > 0.0:
        node_entries, dropped = _dedupe_nodes_for_tpu(node_records)
        if dropped:
            print(
                {
                    "tpu_host_dedupe": True,
                    "alive_ray_nodes": len(node_records),
                    "unique_tpu_hosts": len(node_entries),
                    "dropped": dropped,
                }
            )
    else:
        node_entries = [
            (node_id, node_ip) for node_id, node_ip, _is_head, _tpu in node_records
        ]

    requested = int(args.num_nodes)
    if requested > 0:
        if requested > len(node_entries):
            print(
                {
                    "requested_nodes": int(requested),
                    "available_nodes": int(len(node_entries)),
                    "note": "requested nodes exceed available hosts; capping",
                }
            )
        node_entries = node_entries[:requested]

    world_size = len(node_entries)
    coordinator_ip = node_entries[0][1]
    run_group = args.run_group or f"ray-dist-{int(time.time())}"

    print(
        {
            "nodes": [
                {"node_id": node_id, "node_ip": node_ip}
                for node_id, node_ip in node_entries
            ],
            "world_size": world_size,
            "coordinator": f"{coordinator_ip}:{int(args.coordinator_port)}",
            "run_kind": str(args.run_kind),
            "entry_args": entry_args,
            "run_group": run_group,
            "compare_robust": bool(args.compare_robust),
            "output_root": str(args.output_root),
            "agents_per_node": int(args.agents_per_node),
            "agent_count": int(args.agent_count),
            "inner_workers": int(args.inner_workers),
            "inner_threads": int(args.inner_threads),
            "max_heavy_workers": int(args.max_heavy_workers),
        }
    )

    futures = []
    root = str(Path(__file__).resolve().parents[1])
    for rank, (node_id, node_ip) in enumerate(node_entries):
        resources: dict[str, float] = {}
        tpu_per_task = float(args.tpu_per_task)
        if tpu_per_task > 0.0:
            resources["TPU"] = tpu_per_task
        futures.append(
            _train_on_node.options(
                resources=resources,
                num_cpus=float(args.worker_cpus),
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
            ).remote(
                root=root,
                run_kind=str(args.run_kind),
                entry_args=entry_args,
                node_id=node_id,
                node_ip=node_ip,
                rank=rank,
                world_size=world_size,
                coordinator_ip=coordinator_ip,
                coordinator_port=int(args.coordinator_port),
                base_seed=int(args.base_seed),
                run_group=run_group,
                compare_robust=bool(args.compare_robust),
                output_root=str(args.output_root),
                wandb_entity=str(args.wandb_entity),
                wandb_project=str(args.wandb_project),
                agents_per_node=int(args.agents_per_node),
                agent_count=int(args.agent_count),
                inner_workers=int(args.inner_workers),
                inner_threads=int(args.inner_threads),
                max_heavy_workers=int(args.max_heavy_workers),
                sync_jax=bool(args.sync_jax and str(args.run_kind) == "train"),
            )
        )

    results = ray.get(futures)
    failed = [code for code in results if int(code) != 0]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
