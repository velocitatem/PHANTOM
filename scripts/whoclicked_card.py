#!/usr/bin/env python3
"""Build and upload a Hugging Face dataset card for whoclickedit."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

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


def _badge(label: str, value: str, color: str, logo: str | None = None) -> str:
    encoded_label = quote(label, safe="")
    encoded_value = quote(value, safe="")
    base = (
        "https://img.shields.io/badge/"
        f"{encoded_label}-{encoded_value}-{color}?style=flat-square"
    )
    if logo:
        base = f"{base}&logo={quote(logo, safe='')}&logoColor=white"
    return f"![{label}]({base})"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    header = f"| {' | '.join(headers)} |"
    separator = f"| {' | '.join('---' for _ in headers)} |"
    if not rows:
        empty = f"| {' | '.join('n/a' for _ in headers)} |"
        return "\n".join([header, separator, empty])
    body = "\n".join(f"| {' | '.join(row)} |" for row in rows)
    return "\n".join([header, separator, body])


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

    total_sessions = sum(session_counts.values())
    human_rows = actor_counts.get("human", 0)
    agent_rows = actor_counts.get("agent", 0)

    top_events = list(event_counts.items())[:10]

    snapshot_table = _md_table(
        ["Metric", "Value"],
        [
            ["Rows", f"`{total_rows}`"],
            ["Columns", f"`{total_cols}`"],
            ["Time range (UTC)", f"`{t_min}` -> `{t_max}`"],
            ["Unique sessions", f"`{total_sessions}`"],
        ],
    )

    actor_table = _md_table(
        ["Actor", "Rows", "Share"],
        [
            [
                "`human`",
                str(human_rows),
                f"{(human_rows / total_rows * 100):.1f}%" if total_rows else "0.0%",
            ],
            [
                "`agent`",
                str(agent_rows),
                f"{(agent_rows / total_rows * 100):.1f}%" if total_rows else "0.0%",
            ],
        ],
    )

    pair_table = _md_table(
        ["Actor", "Record type", "Rows"],
        [
            [f"`{actor}`", f"`{record}`", str(n)]
            for (actor, record), n in sorted(
                by_actor_record.items(), key=lambda x: (x[0][0], x[0][1])
            )
        ],
    )

    store_table = _md_table(
        ["Store mode", "Rows"],
        [
            [f"`{mode}`", str(n)]
            for mode, n in sorted(
                store_counts.items(), key=lambda x: x[1], reverse=True
            )
        ],
    )

    event_table = _md_table(
        ["Interaction event", "Count"],
        [[f"`{name}`", str(n)] for name, n in top_events],
    )

    metadata_lines = "\n".join(f"- `{c}`" for c in metadata_cols) or "- none"

    dataset_badge = (
        "[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/"
        "dataset-on-hf-sm.svg)](https://huggingface.co/datasets/velocitatem/whoclickedit)"
    )
    rows_badge = _badge("Rows", str(total_rows), "0A9396")
    cols_badge = _badge("Columns", str(total_cols), "005F73")
    sessions_badge = _badge("Sessions", str(total_sessions), "1D3557")
    human_badge = _badge("Human rows", str(human_rows), "2A9D8F")
    agent_badge = _badge("Agent rows", str(agent_rows), "E76F51")
    license_badge = _badge("License", "MIT", "111827")

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

<img align="right" width="280" src="https://raw.githubusercontent.com/velocitatem/PHANTOM/main/docs/static/images/banner.svg" alt="PHANTOM research banner" />

# [whoclickedit](https://huggingface.co/datasets/velocitatem/whoclickedit)

{dataset_badge}
{rows_badge}
{cols_badge}
{sessions_badge}
{human_badge}
{agent_badge}
{license_badge}

> **Event-level behavior data for dynamic pricing research.**
> This dataset captures how humans and automated agents browse, query prices, and move through the PHANTOM storefronts during controlled experiments.

## What this dataset gives you

- A single flat file (`whoclicked.csv`) with both interaction and price-log events.
- Explicit labels for actor origin: `actor_type` and `is_agent`.
- Provenance fields from Kafka envelopes when available.
- Metadata flattened into feature-ready `metadata_*` columns.

## Snapshot

{snapshot_table}

## Composition

### Rows by actor
{actor_table}

### Rows by actor and record type
{pair_table}

### Store mode coverage
{store_table}

### Top interaction events
{event_table}

## Collection pipeline

Data is sourced from two roots inside PHANTOM:

- `experiments/collected_data` (human sessions)
- `experiments/agents/collected_data` (agent sessions)

Each session directory contains:

- `int.json`: user interaction events
- `price.json`: price quote observations

ETL behavior:

1. Accepts both Kafka-envelope records and flat payload records.
2. Flattens nested JSON to a tabular schema.
3. Preserves row-level provenance (`source_session_dir`, `source_row_index`, topic fields).
4. Adds modeling labels (`actor_type`, `is_agent`, `record_type`).

## Schema highlights

Core modeling fields:

- `actor_type`, `is_agent`, `record_type`
- `sessionId`, `experimentId`, `storeMode`, `ts`
- `eventName`, `page`, `productId`, `price`, `userAgent`

Kafka provenance fields:

- `kafka_partition_id`, `kafka_offset`, `kafka_timestamp_ms`, `kafka_compression`
- `kafka_is_transactional`, `kafka_headers`, `kafka_key_*`, `kafka_value_*`

<details>
<summary>Metadata columns in this release</summary>

{metadata_lines}

</details>

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("velocitatem/whoclickedit")
```

Recommended split strategy:

- Prefer session-aware or time-aware splits.
- Do not split rows from the same `sessionId` across train and test.

## Intended use

- Human-vs-agent behavior classification.
- Session-level telemetry modeling for dynamic pricing defenses.
- Robustness experiments under agent-mediated reconnaissance.

## Safety and limitations

- `userAgent` and referrer metadata can be quasi-identifying in very small samples.
- Data comes from a controlled research platform, not a full production marketplace.
- Current release has stronger coverage for `hotel` flows than `airline` flows.

## Citation

If you use this dataset, cite the PHANTOM thesis project and link this page:
`https://huggingface.co/datasets/velocitatem/whoclickedit`
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
