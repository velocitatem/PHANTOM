from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import ray


def _has_flag(tokens: list[str], name: str) -> bool:
    return any(tok == name or tok.startswith(f"{name}=") for tok in tokens)


def _alive_node_ips() -> list[str]:
    seen: set[str] = set()
    ips: list[str] = []
    for node in ray.nodes():
        if not bool(node.get("Alive", False)):
            continue
        ip = str(node.get("NodeManagerAddress", "")).strip()
        if not ip or ip in seen:
            continue
        seen.add(ip)
        ips.append(ip)
    return sorted(ips)


@ray.remote(max_retries=0)
def _train_on_node(
    *,
    root: str,
    train_args: str,
    rank: int,
    world_size: int,
    coordinator_ip: str,
    coordinator_port: int,
    base_seed: int,
    run_group: str,
    sync_jax: bool,
) -> int:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    requested_platform = str(env.get("PHANTOM_JAX_PLATFORM", "tpu")).strip().lower()
    if world_size > 1 and requested_platform == "tpu":
        requested_platform = "cpu"
        print(
            "PHANTOM_DISTRIBUTED_NOTE: forcing JAX_PLATFORMS=cpu for multi-node SB3 runs"
        )
    env["JAX_PLATFORMS"] = requested_platform
    # Keep each train process in single-host mode to avoid accidental global stalls.
    env["CLOUD_TPU_TASK_ID"] = "0"

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

    tokens = shlex.split(train_args)
    if not _has_flag(tokens, "--seed"):
        tokens.extend(["--seed", str(base_seed + rank)])
    if not _has_flag(tokens, "--group"):
        tokens.extend(["--group", run_group])

    cmd = [sys.executable, "-m", "engine.train", *tokens]
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch one train run per Ray TPU node"
    )
    parser.add_argument("--train-args", type=str, required=True)
    parser.add_argument("--num-nodes", type=int, default=0)
    parser.add_argument("--tpu-per-task", type=float, default=8.0)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--sync-jax", action="store_true")
    parser.add_argument("--coordinator-port", type=int, default=12355)
    parser.add_argument("--run-group", type=str, default="")
    args = parser.parse_args()

    ray.init(address="auto")

    node_ips = _alive_node_ips()
    if not node_ips:
        raise RuntimeError("No alive Ray nodes found")

    requested = int(args.num_nodes)
    if requested > 0:
        node_ips = node_ips[:requested]

    world_size = len(node_ips)
    coordinator_ip = node_ips[0]
    run_group = args.run_group or f"ray-dist-{int(time.time())}"

    print(
        {
            "nodes": node_ips,
            "world_size": world_size,
            "coordinator": f"{coordinator_ip}:{int(args.coordinator_port)}",
            "train_args": args.train_args,
            "run_group": run_group,
        }
    )

    futures = []
    root = str(Path(__file__).resolve().parents[1])
    for rank, node_ip in enumerate(node_ips):
        resources = {f"node:{node_ip}": 0.01, "TPU": float(args.tpu_per_task)}
        futures.append(
            _train_on_node.options(resources=resources).remote(
                root=root,
                train_args=args.train_args,
                rank=rank,
                world_size=world_size,
                coordinator_ip=coordinator_ip,
                coordinator_port=int(args.coordinator_port),
                base_seed=int(args.base_seed),
                run_group=run_group,
                sync_jax=bool(args.sync_jax),
            )
        )

    results = ray.get(futures)
    failed = [code for code in results if int(code) != 0]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
