from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

try:
    import wandb
    from wandb.errors import CommError

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None  # type: ignore[assignment]
    CommError = RuntimeError  # type: ignore[assignment]


def _safe_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_value(value[k]) for k in sorted(value)}
    return str(value)


def _safe_scope(scope: str | None) -> str:
    raw = "manual" if scope in (None, "") else str(scope)
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-")
    return cleaned or "manual"


def checkpoint_artifact_name(
    cfg: Mapping[str, Any], *, backend: str, sweep_id: str | None = None
) -> str:
    payload = {k: _safe_value(cfg[k]) for k in sorted(cfg)}
    scope = _safe_scope(sweep_id)
    canonical = json.dumps(
        {"backend": backend, "scope": scope, "cfg": payload},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:14]
    return f"phantom-{backend}-ckpt-{scope}-{digest}"[:128]


def _is_missing_artifact_error(exc: Exception) -> bool:
    if isinstance(exc, CommError):
        msg = str(exc).lower()
        return "not found" in msg or "does not exist" in msg
    return False


def download_latest_checkpoint(
    artifact_name: str, *, file_name: str
) -> tuple[Path, dict[str, Any]] | None:
    if not HAS_WANDB or wandb.run is None:
        return None
    try:
        artifact = wandb.run.use_artifact(f"{artifact_name}:latest")
    except Exception as exc:
        if _is_missing_artifact_error(exc):
            return None
        raise
    directory = Path(artifact.download())
    checkpoint_path = directory / file_name
    if not checkpoint_path.exists():
        return None
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    return checkpoint_path, metadata


def _aliases_from_metadata(metadata: dict[str, Any] | None) -> list[str]:
    aliases = ["latest"]
    if metadata is None:
        return aliases
    if "step" in metadata:
        try:
            aliases.append(f"step-{int(metadata['step'])}")
        except (TypeError, ValueError):
            pass
    return aliases


def log_checkpoint_bytes(
    artifact_name: str,
    *,
    file_name: str,
    payload: bytes,
    metadata: dict[str, Any] | None = None,
) -> bool:
    if not HAS_WANDB or wandb.run is None:
        return False
    with TemporaryDirectory(prefix="phantom-ckpt-") as tmpdir:
        path = Path(tmpdir) / file_name
        path.write_bytes(payload)
        artifact = wandb.Artifact(
            name=artifact_name,
            type="checkpoint",
            metadata=metadata or {},
        )
        artifact.add_file(path.as_posix(), name=file_name)
        wandb.log_artifact(artifact, aliases=_aliases_from_metadata(metadata))
    return True


def log_checkpoint_file(
    artifact_name: str,
    *,
    file_path: str | Path,
    artifact_file_name: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    if not HAS_WANDB or wandb.run is None:
        return False
    src = Path(file_path)
    if not src.exists():
        return False
    artifact = wandb.Artifact(
        name=artifact_name,
        type="checkpoint",
        metadata=metadata or {},
    )
    artifact.add_file(src.as_posix(), name=artifact_file_name)
    wandb.log_artifact(artifact, aliases=_aliases_from_metadata(metadata))
    return True
