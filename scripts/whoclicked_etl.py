#!/usr/bin/env python3
"""Build and upload a flattened who-clicked dataset from local collected_data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HUMAN_DIR = PROJECT_ROOT / "experiments" / "collected_data"
DEFAULT_AGENT_DIR = PROJECT_ROOT / "experiments" / "agents" / "collected_data"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "exports" / "whoclicked.csv"
DEFAULT_REPO = os.getenv("HF_WHOCLICKED_REPO", "velocitatem/whoclickedit")

BASE_COLUMNS = [
    "actor_type",
    "is_agent",
    "record_type",
    "topic",
    "source_session_dir",
    "source_file",
    "source_row_index",
    "ingest_format",
    "sessionId",
    "experimentId",
    "storeMode",
    "ts",
    "eventName",
    "page",
    "productId",
    "price",
    "userAgent",
    "kafka_partition_id",
    "kafka_offset",
    "kafka_timestamp_ms",
    "kafka_compression",
    "kafka_is_transactional",
    "kafka_headers",
    "kafka_key_payload",
    "kafka_key_encoding",
    "kafka_key_schema_id",
    "kafka_value_encoding",
    "kafka_value_schema_id",
    "kafka_value_size",
]


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
            text = text.strip()
            if text:
                parts.append(text[:500])
    return " | ".join(p for p in parts if p)


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        normalized_key = str(key).strip().replace(" ", "_")
        next_key = f"{prefix}_{normalized_key}" if prefix else normalized_key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, next_key))
        else:
            flat[next_key] = value
    return flat


def _as_scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return value


def _empty_envelope() -> dict[str, Any]:
    return {
        "kafka_partition_id": None,
        "kafka_offset": None,
        "kafka_timestamp_ms": None,
        "kafka_compression": None,
        "kafka_is_transactional": None,
        "kafka_headers": None,
        "kafka_key_payload": None,
        "kafka_key_encoding": None,
        "kafka_key_schema_id": None,
        "kafka_value_encoding": None,
        "kafka_value_schema_id": None,
        "kafka_value_size": None,
    }


def _extract_payload_and_envelope(
    record: Any,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if (
        isinstance(record, dict)
        and isinstance(record.get("value"), dict)
        and isinstance(record["value"].get("payload"), dict)
    ):
        key = record.get("key") if isinstance(record.get("key"), dict) else {}
        value = record["value"]
        envelope = {
            "kafka_partition_id": record.get("partitionID"),
            "kafka_offset": record.get("offset"),
            "kafka_timestamp_ms": record.get("timestamp"),
            "kafka_compression": record.get("compression"),
            "kafka_is_transactional": record.get("isTransactional"),
            "kafka_headers": _as_scalar(record.get("headers")),
            "kafka_key_payload": key.get("payload"),
            "kafka_key_encoding": key.get("encoding"),
            "kafka_key_schema_id": key.get("schemaId"),
            "kafka_value_encoding": value.get("encoding"),
            "kafka_value_schema_id": value.get("schemaId"),
            "kafka_value_size": value.get("size"),
        }
        return dict(value["payload"]), envelope, "kafka_envelope"

    if isinstance(record, dict):
        return dict(record), _empty_envelope(), "flat_payload"

    return {}, _empty_envelope(), "unknown"


def _load_json_list(path: Path) -> list[Any]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in {path}, got {type(raw).__name__}")
    return raw


def _normalize_file_rows(
    actor_type: str,
    is_agent: int,
    session_dir_name: str,
    source_file: str,
    records: list[Any],
) -> list[dict[str, Any]]:
    record_type = "interaction" if source_file == "int.json" else "price_log"
    topic = "user-interactions" if record_type == "interaction" else "price-logs"

    rows: list[dict[str, Any]] = []
    for idx, raw_record in enumerate(records):
        payload, envelope, ingest_format = _extract_payload_and_envelope(raw_record)
        metadata = payload.pop("metadata", None)

        payload_flat = _flatten_dict(payload)
        row: dict[str, Any] = {
            "actor_type": actor_type,
            "is_agent": is_agent,
            "record_type": record_type,
            "topic": topic,
            "source_session_dir": session_dir_name,
            "source_file": source_file,
            "source_row_index": idx,
            "ingest_format": ingest_format,
            **envelope,
        }
        row.update({k: _as_scalar(v) for k, v in payload_flat.items()})

        if isinstance(metadata, dict):
            metadata_flat = _flatten_dict(metadata, "metadata")
            row.update({k: _as_scalar(v) for k, v in metadata_flat.items()})
        elif metadata is not None:
            row["metadata_raw"] = _as_scalar(metadata)

        rows.append(row)

    return rows


def _collect_rows_for_actor(
    actor_type: str, is_agent: int, base_dir: Path
) -> list[dict[str, Any]]:
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    rows: list[dict[str, Any]] = []
    for session_dir in sorted(
        (p for p in base_dir.iterdir() if p.is_dir()), key=lambda p: p.name
    ):
        for source_file in ("int.json", "price.json"):
            file_path = session_dir / source_file
            if not file_path.exists():
                continue
            records = _load_json_list(file_path)
            rows.extend(
                _normalize_file_rows(
                    actor_type=actor_type,
                    is_agent=is_agent,
                    session_dir_name=session_dir.name,
                    source_file=source_file,
                    records=records,
                )
            )
    return rows


def build_dataframe(human_dir: Path, agent_dir: Path) -> pd.DataFrame:
    rows = [
        *_collect_rows_for_actor("human", 0, human_dir),
        *_collect_rows_for_actor("agent", 1, agent_dir),
    ]
    if not rows:
        return pd.DataFrame(columns=BASE_COLUMNS)

    df = pd.DataFrame(rows)
    ordered_columns = [
        *BASE_COLUMNS,
        *sorted(c for c in df.columns if c not in BASE_COLUMNS),
    ]
    return df[ordered_columns]


def _print_summary(df: pd.DataFrame, output_path: Path) -> None:
    print(f"wrote {len(df)} rows and {len(df.columns)} columns to {output_path}")
    if df.empty:
        return

    print("rows by actor/record_type:")
    grouped = (
        df.groupby(["actor_type", "record_type"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["actor_type", "record_type"])
    )
    for _, row in grouped.iterrows():
        print(f"  - {row['actor_type']} / {row['record_type']}: {int(row['count'])}")

    required = ["actor_type", "is_agent", "record_type", "sessionId", "ts"]
    missing = {col: int(df[col].isna().sum()) for col in required if col in df.columns}
    print(f"missing in required columns: {missing}")


def build_csv(human_dir: Path, agent_dir: Path, output: Path) -> pd.DataFrame:
    df = build_dataframe(human_dir=human_dir, agent_dir=agent_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    _print_summary(df, output)
    return df


def _resolve_repo_id(api: HfApi, repo_id: str) -> str:
    if "/" in repo_id:
        return repo_id
    try:
        me = api.whoami(token=_token())
        username = me.get("name")
        if username:
            return f"{username}/{repo_id}"
    except Exception:
        pass
    return repo_id


def upload_csv(
    input_path: Path,
    repo_id: str,
    path_in_repo: str,
    commit_message: str,
    create_if_missing: bool = False,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    api = HfApi(token=_token())

    try:
        me = api.whoami(token=_token())
    except Exception as exc:
        detail = _exception_details(exc)
        hint = "Set HF_TOKEN with write access or run huggingface-cli login."
        raise RuntimeError(
            f"Hugging Face auth failed. {hint} Details: {detail}"
        ) from exc

    user_name = me.get("name") or me.get("fullname") or "unknown"
    print(f"authenticated to HF as: {user_name}")

    resolved_repo_id = _resolve_repo_id(api, repo_id)
    if create_if_missing:
        api.create_repo(repo_id=resolved_repo_id, repo_type="dataset", exist_ok=True)
    else:
        try:
            api.repo_info(repo_id=resolved_repo_id, repo_type="dataset")
        except Exception as exc:
            detail = _exception_details(exc)
            hint = (
                "Check owner/repo spelling, ensure it is a dataset repo, "
                "or pass --create-if-missing."
            )
            raise RuntimeError(
                f"Dataset repo '{resolved_repo_id}' is not accessible. {hint} Details: {detail}"
            ) from exc

    try:
        commit = api.upload_file(
            path_or_fileobj=str(input_path),
            path_in_repo=path_in_repo,
            repo_id=resolved_repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )
    except Exception as exc:
        detail = _exception_details(exc)
        hint = (
            "Pass --repo <owner>/whoclickedit and ensure HF_TOKEN is set "
            "(or run huggingface-cli login)."
        )
        raise RuntimeError(
            f"Upload failed for '{resolved_repo_id}'. {hint} Details: {detail}"
        ) from exc

    print(
        f"uploaded {input_path} to https://huggingface.co/datasets/{resolved_repo_id}"
    )
    print(f"commit: {commit}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ETL for whoclickedit: flatten local collected_data and upload to HF"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build flattened CSV locally")
    build.add_argument("--human-dir", type=Path, default=DEFAULT_HUMAN_DIR)
    build.add_argument("--agent-dir", type=Path, default=DEFAULT_AGENT_DIR)
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    upload = sub.add_parser("upload", help="upload an existing CSV to HF dataset")
    upload.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    upload.add_argument("--repo", default=DEFAULT_REPO)
    upload.add_argument("--path-in-repo", default="whoclicked.csv")
    upload.add_argument("--message", default="Update flattened whoclickedit dataset")
    upload.add_argument("--create-if-missing", action="store_true")

    build_upload = sub.add_parser(
        "build-upload", help="build CSV and upload to HF dataset"
    )
    build_upload.add_argument("--human-dir", type=Path, default=DEFAULT_HUMAN_DIR)
    build_upload.add_argument("--agent-dir", type=Path, default=DEFAULT_AGENT_DIR)
    build_upload.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build_upload.add_argument("--repo", default=DEFAULT_REPO)
    build_upload.add_argument("--path-in-repo", default="whoclicked.csv")
    build_upload.add_argument(
        "--message", default="Update flattened whoclickedit dataset"
    )
    build_upload.add_argument("--create-if-missing", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    try:
        if args.command == "build":
            build_csv(
                human_dir=args.human_dir, agent_dir=args.agent_dir, output=args.output
            )
            return 0

        if args.command == "upload":
            upload_csv(
                input_path=args.input,
                repo_id=args.repo,
                path_in_repo=args.path_in_repo,
                commit_message=args.message,
                create_if_missing=args.create_if_missing,
            )
            return 0

        if args.command == "build-upload":
            build_csv(
                human_dir=args.human_dir, agent_dir=args.agent_dir, output=args.output
            )
            upload_csv(
                input_path=args.output,
                repo_id=args.repo,
                path_in_repo=args.path_in_repo,
                commit_message=args.message,
                create_if_missing=args.create_if_missing,
            )
            return 0

        raise ValueError(f"Unknown command: {args.command}")

    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
