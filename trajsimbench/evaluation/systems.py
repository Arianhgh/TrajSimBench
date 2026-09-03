"""CPU-safe timing and workload metadata."""

from __future__ import annotations

import os
import platform
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from time import perf_counter_ns
from typing import Any

import numpy as np


def system_snapshot() -> dict[str, Any]:
    memory = None
    try:
        import psutil

        memory = int(psutil.virtual_memory().total)
    except ImportError:
        memory = None
    return {
        "cpu_model": platform.processor() or platform.machine(),
        "logical_cores": os.cpu_count(),
        "physical_cores": _physical_cores(),
        "ram_bytes": memory,
        "os": platform.platform(),
        "python": platform.python_version(),
        "gpu": None,
    }


def _physical_cores() -> int | None:
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except ImportError:
        return None


@dataclass(frozen=True, slots=True)
class TimingSummary:
    stage: str
    samples_ns: tuple[int, ...]
    median_ns: float
    p95_ns: float
    throughput: float | None = None
    metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_timings(
    stage: str,
    samples_ns: list[int] | tuple[int, ...],
    *,
    workload: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> TimingSummary:
    if not samples_ns:
        raise ValueError("timing samples cannot be empty")
    values = np.asarray(samples_ns, dtype=float)
    throughput = (
        float(workload / (np.median(values) / 1e9))
        if workload is not None and np.median(values) > 0
        else None
    )
    return TimingSummary(
        stage,
        tuple(int(x) for x in samples_ns),
        float(np.median(values)),
        float(np.quantile(values, 0.95)),
        throughput,
        metadata or system_snapshot(),
    )


@contextmanager
def timed_stage(stage: str, *, metadata: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    started = perf_counter_ns()
    record: dict[str, Any] = {
        "stage": stage,
        "start_ns": started,
        "metadata": metadata or system_snapshot(),
    }
    try:
        yield record
    finally:
        record["end_ns"] = perf_counter_ns()
        record["duration_ns"] = record["end_ns"] - started
