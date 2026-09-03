"""Leakage-safe generalization task contracts.

The builder creates task artifacts but deliberately does not fit models.  It
keeps in-domain, held-out, temporal, and cross-dataset evaluations as separate
run dimensions so their results cannot be pooled accidentally.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .base import TaskArtifact, TaskConstructionError, make_quality

GeneralizationMode = Literal[
    "in_domain",
    "user_held_out",
    "temporal_hold_out",
    "zero_shot_cross_dataset",
    "limited_adaptation",
    "full_retraining",
]


def validate_generalization_partitions(
    train_ids: Sequence[str], val_ids: Sequence[str], test_ids: Sequence[str], *, mode: str
) -> dict[str, Any]:
    """Validate disjoint ID partitions and return a serializable summary."""

    allowed = {
        "in_domain",
        "user_held_out",
        "temporal_hold_out",
        "zero_shot_cross_dataset",
        "limited_adaptation",
        "full_retraining",
    }
    if mode not in allowed:
        raise TaskConstructionError(f"unknown generalization mode {mode!r}")
    partitions = {
        "train": tuple(map(str, train_ids)),
        "val": tuple(map(str, val_ids)),
        "test": tuple(map(str, test_ids)),
    }
    duplicates = {name: len(values) - len(set(values)) for name, values in partitions.items()}
    if any(duplicates.values()):
        raise TaskConstructionError(
            f"generalization partitions contain duplicate IDs: {duplicates}"
        )
    overlap = {
        f"{left}:{right}": sorted(set(partitions[left]).intersection(partitions[right]))
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    overlap = {key: value for key, value in overlap.items() if value}
    if overlap:
        raise TaskConstructionError(f"generalization partitions overlap: {overlap}")
    return {
        "mode": mode,
        "train_count": len(partitions["train"]),
        "val_count": len(partitions["val"]),
        "test_count": len(partitions["test"]),
        "id_overlap": overlap,
    }


def build_generalization_task(
    *,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    mode: GeneralizationMode | str = "in_domain",
    seed: int = 0,
    dataset: str = "unknown",
    source_dataset: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> TaskArtifact:
    summary = validate_generalization_partitions(train_ids, val_ids, test_ids, mode=str(mode))
    records = tuple(
        {"task_id": f"generalization:{query_id}", "query_id": query_id, "partition": "test"}
        for query_id in map(str, test_ids)
    )
    return TaskArtifact(
        task_type="generalization",
        schema_version="1.0",
        records=records,
        generator="generalization_partitions",
        generator_version="1.0",
        seed=int(seed),
        config={
            "mode": str(mode),
            "dataset": dataset,
            "source_dataset": source_dataset,
            **dict(config or {}),
        },
        quality=make_quality(len(records), len(records), required_count=len(records)),
        metadata={
            **summary,
            "train_ids": list(map(str, train_ids)),
            "val_ids": list(map(str, val_ids)),
        },
    )


__all__ = ["GeneralizationMode", "build_generalization_task", "validate_generalization_partitions"]
