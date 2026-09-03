"""Systems-workload task records separate from quality evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .base import TaskArtifact, make_quality


@dataclass(frozen=True, slots=True)
class SystemsWorkload:
    """Resolved dimensions that must accompany a timing result."""

    database_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    warmup_count: int = 1
    repetitions: int = 3
    worker_count: int = 1

    def __post_init__(self) -> None:
        if not self.database_ids or not self.query_ids:
            raise ValueError("systems workloads require non-empty query and database IDs")
        if min(self.warmup_count, self.repetitions, self.worker_count) < 0:
            raise ValueError("systems workload counts must be non-negative")
        if self.repetitions < 1 or self.worker_count < 1:
            raise ValueError("repetitions and worker_count must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "database_ids": list(self.database_ids),
            "query_ids": list(self.query_ids),
            "database_size": len(self.database_ids),
            "query_count": len(self.query_ids),
            "warmup_count": self.warmup_count,
            "repetitions": self.repetitions,
            "worker_count": self.worker_count,
        }


def build_systems_task(
    database_ids: Sequence[str],
    query_ids: Sequence[str],
    *,
    seed: int = 0,
    config: Mapping[str, Any] | None = None,
) -> TaskArtifact:
    resolved = dict(config or {})
    workload = SystemsWorkload(
        tuple(map(str, database_ids)),
        tuple(map(str, query_ids)),
        warmup_count=int(resolved.pop("warmup_count", resolved.pop("warmup", 1))),
        repetitions=int(resolved.pop("repetitions", 3)),
        worker_count=int(resolved.pop("worker_count", 1)),
    )
    records = tuple(
        {
            "task_id": f"systems:{query_id}",
            "query_id": query_id,
            "database_size": len(workload.database_ids),
        }
        for query_id in workload.query_ids
    )
    return TaskArtifact(
        task_type="systems",
        schema_version="1.0",
        records=records,
        generator="systems_workload",
        generator_version="1.0",
        seed=int(seed),
        config={**workload.to_dict(), **resolved},
        quality=make_quality(len(records), len(records), required_count=len(records)),
    )


__all__ = ["SystemsWorkload", "build_systems_task"]
