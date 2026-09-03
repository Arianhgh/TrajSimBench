"""Deterministic, ID-based split and scale selection helpers."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

SPLIT_ALGORITHM_VERSION = "sha256-id-v1"
SCALE_LIMITS: dict[str, tuple[int, int]] = {
    "tiny": (1_000, 100),
    "standard": (10_000, 1_000),
    "medium": (50_000, 500),
    "large": (100_000, 200),
}


def _rank(seed: int, value: str, algorithm: str = SPLIT_ALGORITHM_VERSION) -> bytes:
    return hashlib.sha256(f"{algorithm}:{seed}:{value}".encode()).digest()


def stable_id_order(ids: Sequence[str], *, seed: int = 0) -> list[str]:
    """Return IDs in a deterministic pseudo-random order without replacement."""

    return sorted((str(value) for value in ids), key=lambda value: (_rank(seed, value), value))


def standard_split(
    ids: Sequence[str],
    *,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> dict[str, list[str]]:
    if len(ratios) != 3 or any(value < 0 for value in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must be three non-negative values summing to one")
    ordered = stable_id_order(ids, seed=seed)
    n = len(ordered)
    train_end = int(round(n * ratios[0]))
    val_end = train_end + int(round(n * ratios[1]))
    # Correct rounding drift while preserving deterministic ordering.
    val_end = min(n, val_end)
    train_end = min(train_end, val_end)
    return {
        "train": ordered[:train_end],
        "val": ordered[train_end:val_end],
        "test": ordered[val_end:],
    }


def user_held_out_split(
    records: Sequence[Any],
    *,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> dict[str, list[str]]:
    """Split complete user groups, guaranteeing zero user overlap."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if isinstance(record, Mapping):
            trajectory_id, user_id = record["trajectory_id"], record.get("user_id")
        else:
            trajectory_id, user_id = record.trajectory_id, record.user_id
        if user_id is None or str(user_id) == "None":
            raise ValueError("user-held-out split requires a user_id for every trajectory")
        grouped[str(user_id)].append(str(trajectory_id))
    user_split = standard_split(list(grouped), seed=seed, ratios=ratios)
    return {
        partition: [trajectory_id for user in users for trajectory_id in sorted(grouped[user])]
        for partition, users in user_split.items()
    }


def temporal_split(
    records: Sequence[Any], *, ratios: tuple[float, float, float] = (0.7, 0.1, 0.2)
) -> dict[str, list[str]]:
    """Split by earliest start times, rejecting trajectories without timestamps."""

    sortable: list[tuple[float, str]] = []
    for record in records:
        if isinstance(record, Mapping):
            trajectory_id, start = record["trajectory_id"], record.get("start_time_s")
        else:
            trajectory_id, start = record.trajectory_id, getattr(record, "start_time_s", None)
        if start is None:
            metadata = getattr(record, "metadata", {})
            start = metadata.get("start_time_s") if isinstance(metadata, Mapping) else None
        if start is None or not np.isfinite(float(start)):
            raise ValueError("temporal split requires finite start_time_s for every trajectory")
        sortable.append((float(start), str(trajectory_id)))
    ordered = [trajectory_id for _, trajectory_id in sorted(sortable)]
    n = len(ordered)
    train_end = int(round(n * ratios[0]))
    val_end = min(n, train_end + int(round(n * ratios[1])))
    return {
        "train": ordered[:train_end],
        "val": ordered[train_end:val_end],
        "test": ordered[val_end:],
    }


def make_split_bundle(
    ids: Sequence[str],
    *,
    seed: int = 0,
    records: Sequence[Any] | None = None,
    include_temporal: bool = False,
    include_user_held_out: bool = False,
) -> dict[str, dict[str, list[str]]]:
    bundle: dict[str, dict[str, list[str]]] = {"standard": standard_split(ids, seed=seed)}
    if include_user_held_out:
        if records is None:
            raise ValueError("records are required for user-held-out splits")
        bundle["user_held_out"] = user_held_out_split(records, seed=seed)
    if include_temporal:
        if records is None:
            raise ValueError("records are required for temporal splits")
        bundle["temporal"] = temporal_split(records)
    return bundle


@dataclass(frozen=True, slots=True)
class ScaleSelection:
    tier: str
    database_ids: tuple[str, ...]
    query_ids: tuple[str, ...]


def select_scale(
    ids: Sequence[str],
    tier: str,
    *,
    seed: int = 0,
    query_count: int | None = None,
    database_count: int | None = None,
    allow_reduced: bool = False,
) -> ScaleSelection:
    """Select a tier without replacement and make reductions explicit."""

    if tier not in SCALE_LIMITS:
        raise ValueError(
            f"unknown scale tier {tier!r}; choose one of {', '.join(sorted(SCALE_LIMITS))}"
        )
    ordered = stable_id_order(ids, seed=seed)
    requested_database, requested_query = SCALE_LIMITS[tier]
    database_count = requested_database if database_count is None else database_count
    query_count = requested_query if query_count is None else query_count
    if database_count < 0 or query_count < 0:
        raise ValueError("scale counts must be non-negative")
    if database_count + query_count > len(ordered):
        if not allow_reduced:
            raise ValueError(
                f"scale tier {tier!r} needs {database_count + query_count} trajectories but only "
                f"{len(ordered)} are available without replacement; set allow_reduced=true or "
                "choose a smaller tier"
            )
        database_count = min(database_count, len(ordered))
        query_count = min(query_count, max(0, len(ordered) - database_count))
    return ScaleSelection(
        tier,
        tuple(ordered[:database_count]),
        tuple(ordered[database_count : database_count + query_count]),
    )
