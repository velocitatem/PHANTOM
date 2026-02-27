"""Utilities for loading separability artifacts and scoring interaction sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import joblib
import numpy as np

from experiments.ml.arch import featurize_trajectory


DEFAULT_ARTIFACT_DIR = Path("data/separability")


@dataclass
class SeparabilityArtifacts:
    scaler: object
    classifier: object
    states: List[str]
    event_transitions: Dict[str, Dict[str, float]]
    feature_dim: int


def _normalize_events(raw_events: Sequence[object]) -> List[object]:
    events: List[object] = []
    for evt in raw_events:
        if hasattr(evt, "value") and hasattr(evt.value, "payload"):
            events.append(evt.value.payload)
        else:
            events.append(evt)
    events.sort(key=lambda e: getattr(e, "ts", ""))
    return events


def _event_transition_distribution(events: Sequence[object]) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Dict[str, int]] = {}
    for src_evt, dst_evt in zip(events, events[1:]):
        src_name = getattr(src_evt, "eventName", "unknown")
        dst_name = getattr(dst_evt, "eventName", "unknown")
        counts.setdefault(src_name, {})
        counts[src_name][dst_name] = counts[src_name].get(dst_name, 0) + 1

    distribution: Dict[str, Dict[str, float]] = {}
    for src, dsts in counts.items():
        total = float(sum(dsts.values()))
        distribution[src] = {dst: val / total for dst, val in dsts.items()} if total else {}
    return distribution


def _kl_divergence(p: Dict[str, Dict[str, float]], q: Dict[str, Dict[str, float]]) -> float:
    eps = 1e-10
    total = 0.0
    for src, dsts in p.items():
        for dst, prob in dsts.items():
            ref = q.get(src, {}).get(dst, 0.0)
            total += (prob + eps) * np.log((prob + eps) / (ref + eps))
    return float(total)


def load_artifacts(artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR) -> SeparabilityArtifacts:
    artifact_dir = Path(artifact_dir)
    scaler_path = artifact_dir / "scaler.joblib"
    model_path = artifact_dir / "classifier.joblib"
    metadata_path = artifact_dir / "metadata.json"

    if not (scaler_path.exists() and model_path.exists() and metadata_path.exists()):
        raise FileNotFoundError(
            f"Separability artifacts not found in {artifact_dir}. Run sim.strong_learner.train first."
        )

    scaler = joblib.load(scaler_path)
    classifier = joblib.load(model_path)
    with open(metadata_path, "r", encoding="utf-8") as fin:
        metadata = json.load(fin)

    return SeparabilityArtifacts(
        scaler=scaler,
        classifier=classifier,
        states=list(metadata["reference_states"]),
        event_transitions=metadata["event_transitions"],
        feature_dim=int(metadata["feature_dim"]),
    )


def score_session(
    raw_events: Sequence[object],
    artifacts: SeparabilityArtifacts,
) -> dict:
    events = _normalize_events(raw_events)
    if not events:
        return {"prob_agent": 0.0, "delta_h": 0.0, "delta_a": 0.0}

    reference_mdp = {"states": artifacts.states}
    features = featurize_trajectory(events, mdp=reference_mdp, input_dim=artifacts.feature_dim)
    scaled = artifacts.scaler.transform(features.reshape(1, -1))
    prob_agent = float(artifacts.classifier.predict_proba(scaled)[0, 1])

    session_dist = _event_transition_distribution(events)
    delta_h = _kl_divergence(session_dist, artifacts.event_transitions.get("human", {}))
    delta_a = _kl_divergence(session_dist, artifacts.event_transitions.get("agent", {}))

    return {
        "prob_agent": prob_agent,
        "delta_h": delta_h,
        "delta_a": delta_a,
    }


def estimate_alpha(prob_agent: float, delta_h: float, delta_a: float, temperature: float = 1.0) -> float:
    divergence_mass = delta_h + delta_a
    if divergence_mass <= 1e-8:
        return float(prob_agent)

    ratio = delta_a / divergence_mass
    blended = 0.5 * prob_agent + 0.5 * ratio
    if temperature <= 0:
        return float(np.clip(blended, 0.0, 1.0))

    scaled = 1.0 / (1.0 + np.exp(-temperature * (blended - 0.5)))
    return float(np.clip(scaled, 0.0, 1.0))


def score_sessions(raw_sessions: Iterable[Sequence[object]], artifacts: SeparabilityArtifacts) -> List[dict]:
    return [score_session(events, artifacts) for events in raw_sessions]
