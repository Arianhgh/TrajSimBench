"""Resume and force-stage helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cache import cache_valid, load_stage_cache, save_stage_cache

DEPENDENTS: dict[str, tuple[str, ...]] = {
    "validate": (
        "load_data",
        "materialize_tasks",
        "fit_methods",
        "build_index",
        "evaluate",
        "metrics",
        "commit",
        "analyze",
        "finalize",
    ),
    "load_data": (
        "materialize_tasks",
        "fit_methods",
        "build_index",
        "evaluate",
        "metrics",
        "commit",
        "analyze",
        "finalize",
    ),
    "materialize_tasks": (
        "fit_methods",
        "build_index",
        "evaluate",
        "metrics",
        "commit",
        "analyze",
        "finalize",
    ),
    "fit_methods": ("build_index", "evaluate", "metrics", "commit", "analyze", "finalize"),
    "build_index": ("evaluate", "metrics", "commit", "analyze", "finalize"),
    "evaluate": ("metrics", "commit", "analyze", "finalize"),
    "metrics": ("commit", "analyze", "finalize"),
    "commit": ("analyze", "finalize"),
    "analyze": ("finalize",),
}


def invalidate_from(records: dict[str, Any], stage: str) -> dict[str, Any]:
    names = {stage, *DEPENDENTS.get(stage, ())}
    for name in names:
        if name in records:
            records[name]["status"] = "pending"
    return records


def resume_stage(
    records: dict[str, Any], stage: str, *, input_fingerprint: str, root: Path
) -> bool:
    return cache_valid(records.get(stage, {}), input_fingerprint=input_fingerprint, root=root)


__all__ = ["DEPENDENTS", "invalidate_from", "resume_stage", "load_stage_cache", "save_stage_cache"]
