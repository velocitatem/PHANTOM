#!/usr/bin/env python3
"""Build and upload a Hugging Face dataset card for whoclickedit."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "experiments" / "exports" / "whoclicked.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "exports" / "whoclicked_dataset_card.md"
DEFAULT_REPO = os.getenv("HF_WHOCLICKED_REPO", "velocitatem/whoclickedit")


def _token() -> str | None:
    return os.getenv("HF_TOKEN") or None


def _exception_details(exc: Exception) -> str:
    parts = [str(exc).strip()]
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if status is not None:
            parts.append(f"HTTP {status}")
        text = getattr(response, "text", "")
        if text:
            parts.append(text.strip()[:500])
    return " | ".join(p for p in parts if p)


def _size_category(n_rows: int) -> str:
    if n_rows < 1_000:
        return "n<1K"
    if n_rows < 10_000:
        return "1K<n<10K"
    if n_rows < 100_000:
        return "10K<n<100K"
    if n_rows < 1_000_000:
        return "100K<n<1M"
    return "1M<n"


def _series_count(df: pd.DataFrame, col: str) -> dict[str, int]:
    if col not in df.columns:
        return {}
    vc = df[col].fillna("<null>").astype(str).value_counts(dropna=False)
    return {k: int(v) for k, v in vc.items()}


def _group_count(df: pd.DataFrame, left: str, right: str) -> dict[tuple[str, str], int]:
    if left not in df.columns or right not in df.columns:
        return {}
    grouped = (
        df.groupby([left, right], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values([left, right])
    )
    out: dict[tuple[str, str], int] = {}
    for _, row in grouped.iterrows():
        out[(str(row[left]), str(row[right]))] = int(row["count"])
    return out


def _session_count_by_actor(df: pd.DataFrame) -> dict[str, int]:
    if "actor_type" not in df.columns or "sessionId" not in df.columns:
        return {}
    grouped = (
        df[["actor_type", "sessionId"]]
        .dropna(subset=["sessionId"])
        .drop_duplicates()
        .groupby("actor_type")
        .size()
    )
    return {str(k): int(v) for k, v in grouped.items()}


def _time_range(df: pd.DataFrame) -> tuple[str, str]:
    if "ts" not in df.columns:
        return "unknown", "unknown"
    ts = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    ts = ts.dropna()
    if ts.empty:
        return "unknown", "unknown"
    return ts.min().isoformat(), ts.max().isoformat()


def _render_card(df: pd.DataFrame) -> str:
    total_rows = len(df)
    total_cols = len(df.columns)
    size_cat = _size_category(total_rows)

    actor_counts = _series_count(df, "actor_type")
    record_counts = _series_count(df, "record_type")
    by_actor_record = _group_count(df, "actor_type", "record_type")
    store_counts = _series_count(df, "storeMode")
    session_counts = _session_count_by_actor(df)
    t_min, t_max = _time_range(df)

    event_counts: dict[str, int] = {}
    if "record_type" in df.columns and "eventName" in df.columns:
        interactions = df[df["record_type"] == "interaction"]
        event_counts = _series_count(interactions, "eventName")

    metadata_cols = sorted(c for c in df.columns if c.startswith("metadata_"))

    actor_lines = (
        "\n".join(f"- `{k}`: {v}" for k, v in actor_counts.items()) or "- none"
    )
    record_lines = (
        "\n".join(f"- `{k}`: {v}" for k, v in record_counts.items()) or "- none"
    )
    pair_lines = (
        "\n".join(
            f"- `{a}` / `{r}`: {n}"
            for (a, r), n in sorted(
                by_actor_record.items(), key=lambda x: (x[0][0], x[0][1])
            )
        )
        or "- none"
    )
    store_lines = (
        "\n".join(f"- `{k}`: {v}" for k, v in store_counts.items()) or "- none"
    )
    session_lines = (
        "\n".join(f"- `{k}`: {v}" for k, v in session_counts.items()) or "- none"
    )
    top_events = list(event_counts.items())[:10]
    event_lines = "\n".join(f"- `{k}`: {v}" for k, v in top_events) or "- none"
    metadata_lines = "\n".join(f"- `{c}`" for c in metadata_cols) or "- none"

    return f"""---
pretty_name: whoclickedit
license: mit
language:
- en
task_categories:
- tabular-classification
task_ids:
- tabular-multi-class-classification
tags:
- e-commerce
- dynamic-pricing
- behavioral-telemetry
- human-vs-agent
- session-data
size_categories:
- {size_cat}
---

# Dataset Card for whoclickedit

## Dataset Summary
whoclickedit is an event-level behavioral dataset for human versus agent interaction analysis in dynamic pricing experiments.
It merges interaction logs and price quote logs into one flat CSV (`whoclicked.csv`) with explicit labels for actor type.

## Dataset Snapshot
- Rows: `{total_rows}`
- Columns: `{total_cols}`
- Time range (UTC): `{t_min}` to `{t_max}`
- Unique sessions by actor:
{session_lines}
- Rows by actor:
{actor_lines}
- Rows by record type:
{record_lines}
- Rows by actor x record type:
{pair_lines}
- Store modes:
{store_lines}

## Source and Processing
Data is collected from two local roots in the PHANTOM project:
- `experiments/collected_data` (human sessions)
- `experiments/agents/collected_data` (agent sessions)

Each session folder contains:
- `int.json` (interaction events)
- `price.json` (price quote logs)

The ETL does the following:
- Normalizes both Kafka-envelope and flat payload formats
- Flattens nested metadata fields into `metadata_*` columns
- Preserves all raw rows (no deduplication)
- Adds labels:
  - `actor_type` in `{{human, agent}}`
  - `is_agent` in `{{0, 1}}`
  - `record_type` in `{{interaction, price_log}}`

## Data Fields
Core fields used for modeling:
- `actor_type`, `is_agent`, `record_type`
- `sessionId`, `experimentId`, `storeMode`, `ts`
- `eventName`, `page`, `productId`, `price`, `userAgent`

Kafka provenance fields:
- `kafka_partition_id`, `kafka_offset`, `kafka_timestamp_ms`, `kafka_compression`
- `kafka_is_transactional`, `kafka_headers`, `kafka_key_*`, `kafka_value_*`

Flattened metadata fields currently present:
{metadata_lines}

Top interaction events:
{event_lines}

## Intended Uses
- Human-vs-agent traffic classification
- Session-level behavioral modeling
- Dynamic pricing robustness analysis under agent-mediated reconnaissance

## Out-of-Scope Uses
- Identity inference or user-level profiling
- Credit, employment, insurance, or legal decision making

## Data Splits
No official train/validation/test split is provided in the current release.
Users should create time-aware or session-aware splits to avoid leakage.

## Privacy and Sensitive Content
- `userAgent` and referrer metadata can be quasi-identifying in small samples.
- Use care before publishing derived artifacts that can re-identify participants.

## Limitations
- Data is generated in a controlled experiment platform, not a full production marketplace.
- Agent traffic currently reflects the configured tasking and browser automation setup.
- Coverage is stronger for `hotel` than `airline` in the current release.

## Citation
If you use this dataset, cite the PHANTOM thesis project and link this dataset page.
"""


def build_card(input_csv: Path, output_md: Path) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)
    card = _render_card(df)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(card)
    print(f"wrote dataset card to {output_md}")


def upload_card(
    card_path: Path, repo_id: str, path_in_repo: str, commit_message: str
) -> None:
    if not card_path.exists():
        raise FileNotFoundError(f"Card file not found: {card_path}")

    api = HfApi(token=_token())
    try:
        me = api.whoami(token=_token())
    except Exception as exc:
        detail = _exception_details(exc)
        raise RuntimeError(f"Hugging Face auth failed. Details: {detail}") from exc

    user_name = me.get("name") or me.get("fullname") or "unknown"
    print(f"authenticated to HF as: {user_name}")

    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:
        detail = _exception_details(exc)
        raise RuntimeError(
            f"Dataset repo '{repo_id}' is not accessible. Details: {detail}"
        ) from exc

    try:
        commit = api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )
    except Exception as exc:
        detail = _exception_details(exc)
        raise RuntimeError(
            f"Card upload failed for '{repo_id}'. Details: {detail}"
        ) from exc

    print(f"uploaded dataset card to https://huggingface.co/datasets/{repo_id}")
    print(f"commit: {commit}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or upload whoclickedit dataset card"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build card markdown from CSV")
    build.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    upload = sub.add_parser("upload", help="upload existing card as dataset README.md")
    upload.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    upload.add_argument("--repo", default=DEFAULT_REPO)
    upload.add_argument("--path-in-repo", default="README.md")
    upload.add_argument("--message", default="Add dataset card for whoclickedit")

    both = sub.add_parser("build-upload", help="build card and upload to dataset repo")
    both.add_argument("--csv", type=Path, default=DEFAULT_INPUT)
    both.add_argument("--card", type=Path, default=DEFAULT_OUTPUT)
    both.add_argument("--repo", default=DEFAULT_REPO)
    both.add_argument("--path-in-repo", default="README.md")
    both.add_argument("--message", default="Add dataset card for whoclickedit")

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.command == "build":
            build_card(args.input, args.output)
            return 0

        if args.command == "upload":
            upload_card(args.input, args.repo, args.path_in_repo, args.message)
            return 0

        if args.command == "build-upload":
            build_card(args.csv, args.card)
            upload_card(args.card, args.repo, args.path_in_repo, args.message)
            return 0

        raise ValueError(f"Unknown command: {args.command}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
