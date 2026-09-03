"""Triplet diagnostics and notion-specific similarity fingerprints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .statistics import bootstrap_ci


def _field(triplet: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in triplet:
            return triplet[name]
    raise KeyError(f"triplet is missing one of {names}")


def triplet_accuracy(
    triplets: Sequence[Mapping[str, Any]],
    distances: Mapping[tuple[Any, Any], float] | Any,
    *,
    tie_tolerance: float = 0.0,
    tie_aware: bool = False,
) -> dict[str, Any]:
    """Evaluate triplets whose expectation is ``a_closer``, ``b_closer``, or ``tie``."""

    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    outcomes: list[float] = []
    failures: list[dict[str, Any]] = []
    for index, triplet in enumerate(triplets):
        anchor = _field(triplet, "anchor_id", "query_id", "source_id")
        a = _field(triplet, "a_id", "positive_id", "left_id")
        b = _field(triplet, "b_id", "negative_id", "right_id")
        expectation = str(
            triplet.get("expectation", triplet.get("expected", "unspecified"))
        ).lower()
        if expectation not in {"a_closer", "b_closer", "tie"}:
            continue
        if callable(distances):
            da = float(distances(anchor, a))
            db = float(distances(anchor, b))
        else:
            da = float(distances[(anchor, a)])
            db = float(distances[(anchor, b)])
        delta = da - db
        predicted = (
            "tie" if abs(delta) <= tie_tolerance else ("a_closer" if delta < 0 else "b_closer")
        )
        if tie_aware and expectation in {"a_closer", "b_closer"} and predicted == "tie":
            correct = 1.0
        else:
            correct = float(predicted == expectation)
        outcomes.append(correct)
        if not correct:
            failures.append(
                {
                    "triplet_index": index,
                    "anchor_id": anchor,
                    "a_id": a,
                    "b_id": b,
                    "expected": expectation,
                    "predicted": predicted,
                    "distance_a": da,
                    "distance_b": db,
                }
            )
    valid_count = len(outcomes)
    return {
        "accuracy": float(np.mean(outcomes)) if outcomes else float("nan"),
        "valid_triplet_count": valid_count,
        "coverage": valid_count / len(triplets) if triplets else 0.0,
        "tie_aware": bool(tie_aware),
        "failures": failures,
        "outcomes": outcomes,
    }


def evaluate_triplets(
    triplets: Sequence[Mapping[str, Any]],
    distances: Mapping[tuple[Any, Any], float] | Any,
    *,
    tie_tolerance: float = 0.0,
    bootstrap_resamples: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    strict = triplet_accuracy(triplets, distances, tie_tolerance=tie_tolerance, tie_aware=False)
    tie_aware = triplet_accuracy(triplets, distances, tie_tolerance=tie_tolerance, tie_aware=True)
    if bootstrap_resamples > 0 and strict["outcomes"]:
        low, high = bootstrap_ci(strict["outcomes"], resamples=bootstrap_resamples, seed=seed)
    else:
        low = high = float("nan")
    return {
        "strict": strict,
        "tie_aware": tie_aware,
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
    }


FINGERPRINT_DIMENSIONS = (
    "sampling_invariance",
    "gps_noise_robustness",
    "location_sensitivity",
    "shape_sensitivity",
    "direction_sensitivity",
    "temporal_sensitivity",
    "detour_sensitivity",
    "same_od_hard_negative_accuracy",
)


def build_similarity_fingerprint(
    scores: Mapping[str, float],
    *,
    method: str | None = None,
    notion: str | None = None,
    version: str = "1.0",
) -> dict[str, Any]:
    """Return named dimensions, preserving absent values as explicit NaN."""

    unknown = set(scores) - set(FINGERPRINT_DIMENSIONS)
    if unknown:
        raise ValueError(f"unknown fingerprint dimensions: {sorted(unknown)}")
    result: dict[str, Any] = {"fingerprint_version": version}
    if method is not None:
        result["method"] = method
    if notion is not None:
        result["notion"] = notion
    result.update(
        {
            dimension: float(scores[dimension]) if dimension in scores else float("nan")
            for dimension in FINGERPRINT_DIMENSIONS
        }
    )
    return result


triplet_metrics = evaluate_triplets
similarity_fingerprint = build_similarity_fingerprint
