"""Utilities for loading separability artifacts and scoring interaction sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from lib.agent_probability import DEFAULT_AGENT_PRIOR, estimate_agent_probability


DEFAULT_ARTIFACT_DIR = Path("data/separability")


@dataclass
class SeparabilityArtifacts:
    event_transitions: Dict[str, Dict[str, float]]


def _normalize_events(raw_events: Sequence[object]) -> List[object]:
    events: List[object] = []
    for evt in raw_events:
        if hasattr(evt, "value") and hasattr(evt.value, "payload"):
            events.append(evt.value.payload)
        else:
            events.append(evt)
    events.sort(key=lambda e: getattr(e, "ts", ""))
    return events


def _event_transition_distribution(
    events: Sequence[object],
) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Dict[str, int]] = {}
    for src_evt, dst_evt in zip(events, events[1:]):
        src_name = getattr(src_evt, "eventName", "unknown")
        dst_name = getattr(dst_evt, "eventName", "unknown")
        counts.setdefault(src_name, {})
        counts[src_name][dst_name] = counts[src_name].get(dst_name, 0) + 1

    distribution: Dict[str, Dict[str, float]] = {}
    for src, dsts in counts.items():
        total = float(sum(dsts.values()))
        distribution[src] = (
            {dst: val / total for dst, val in dsts.items()} if total else {}
        )
    return distribution


def _kl_divergence(
    p: Dict[str, Dict[str, float]], q: Dict[str, Dict[str, float]]
) -> float:
    eps = 1e-10
    total = 0.0
    for src, dsts in p.items():
        for dst, prob in dsts.items():
            ref = q.get(src, {}).get(dst, 0.0)
            total += (prob + eps) * np.log((prob + eps) / (ref + eps))
    return float(total)


def load_artifacts(
    artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
) -> SeparabilityArtifacts:
    artifact_dir = Path(artifact_dir)
    metadata_path = artifact_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Separability metadata not found in {artifact_dir}. Provide metadata.json with event transitions."
        )

    with open(metadata_path, "r", encoding="utf-8") as fin:
        metadata = json.load(fin)

    transitions = metadata.get("event_transitions")
    if not isinstance(transitions, dict):
        raise ValueError(
            "metadata.json must contain an 'event_transitions' object with 'human' and 'agent' kernels"
        )

    return SeparabilityArtifacts(
        event_transitions=transitions,
    )


def score_session(
    raw_events: Sequence[object],
    artifacts: SeparabilityArtifacts,
) -> dict:
    events = _normalize_events(raw_events)
    if not events:
        return {
            "prob_agent": float(DEFAULT_AGENT_PRIOR),
            "delta_h": 0.0,
            "delta_a": 0.0,
            "gap": 0.0,
        }

    session_dist = _event_transition_distribution(events)
    delta_h = _kl_divergence(session_dist, artifacts.event_transitions.get("human", {}))
    delta_a = _kl_divergence(session_dist, artifacts.event_transitions.get("agent", {}))
    gap = float(delta_h - delta_a)
    prob_agent = estimate_agent_probability(delta_h=delta_h, delta_a=delta_a)

    return {
        "prob_agent": prob_agent,
        "delta_h": delta_h,
        "delta_a": delta_a,
        "gap": gap,
    }


def estimate_alpha(
    prob_agent: float,
    delta_h: float,
    delta_a: float,
    temperature: float = 1.0,
    prior_agent: float = DEFAULT_AGENT_PRIOR,
) -> float:
    _ = prob_agent
    return estimate_agent_probability(
        delta_h=delta_h,
        delta_a=delta_a,
        temperature=temperature,
        prior_agent=prior_agent,
    )


def score_sessions(
    raw_sessions: Iterable[Sequence[object]], artifacts: SeparabilityArtifacts
) -> List[dict]:
    return [score_session(events, artifacts) for events in raw_sessions]
