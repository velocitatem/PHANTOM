#!/usr/bin/env python3
"""Sync collected behavioral data with HuggingFace Hub.

Usage:
    python scripts/hf_data.py pull   # download from HF to local directories
    python scripts/hf_data.py push   # upload local directories to HF

Expects HF_TOKEN env var (or logged in via `huggingface-cli login`).
Repo id comes from HF_DATASET_REPO env var, default: velocitatem/phantom-collected-data
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HUMAN_DIR = PROJECT_ROOT / "experiments" / "collected_data"
AGENT_DIR = PROJECT_ROOT / "experiments" / "agents" / "collected_data"

DEFAULT_REPO = "velocitatem/phantom-collected-data"

# mapping between local dirs and their prefix inside the HF repo
SLOT_MAP = {"human": HUMAN_DIR, "agent": AGENT_DIR}


def _repo_id() -> str:
    return os.getenv("HF_DATASET_REPO", DEFAULT_REPO)


def _token() -> str | None:
    return os.getenv("HF_TOKEN") or None


def push():
    api = HfApi(token=_token())
    repo = _repo_id()
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True)

    for prefix, local_dir in SLOT_MAP.items():
        if not local_dir.exists():
            print(f"skip {prefix}: {local_dir} does not exist")
            continue
        sessions = [d for d in local_dir.iterdir() if d.is_dir()]
        if not sessions:
            print(f"skip {prefix}: no session directories")
            continue
        print(f"uploading {len(sessions)} sessions from {prefix}/ ...")
        api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=str(local_dir),
            path_in_repo=prefix,
            commit_message=f"update {prefix} data ({len(sessions)} sessions)",
        )
    print("push complete")


def pull():
    repo = _repo_id()
    token = _token()
    cache = snapshot_download(repo, repo_type="dataset", token=token)
    cache = Path(cache)

    for prefix, local_dir in SLOT_MAP.items():
        src = cache / prefix
        if not src.exists():
            print(f"skip {prefix}: not present in remote")
            continue
        local_dir.mkdir(parents=True, exist_ok=True)
        sessions = [d for d in src.iterdir() if d.is_dir()]
        pulled = 0
        for sess in sessions:
            dest = local_dir / sess.name
            dest.mkdir(exist_ok=True)
            for f in sess.iterdir():
                if f.is_file():
                    (dest / f.name).write_bytes(f.read_bytes())
                    pulled += 1
        print(f"{prefix}: pulled {len(sessions)} sessions ({pulled} files)")
    print("pull complete")


def main():
    p = argparse.ArgumentParser(description="Sync collected data with HuggingFace Hub")
    p.add_argument("action", choices=["pull", "push"], help="pull or push data")
    args = p.parse_args()
    {"pull": pull, "push": push}[args.action]()


if __name__ == "__main__":
    main()
