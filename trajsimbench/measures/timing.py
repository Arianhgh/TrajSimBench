"""Small CPU pairwise timing harness for Phase 2 smoke checks.

This is intentionally not the formal systems benchmark.  It reports wall-clock
samples for a warmed, single-process pairwise call and keeps the workload
dimensions explicit for later systems instrumentation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import median
from time import perf_counter_ns
from typing import Any

import numpy as np

from .base import TrajectoryMeasure


@dataclass(frozen=True, slots=True)
class PairwiseTiming:
    measure: str
    candidate_count: int
    warmup_count: int
    repetitions: int
    samples_ns: tuple[int, ...]

    @property
    def median_ns(self) -> float:
        return float(median(self.samples_ns))

    @property
    def p95_ns(self) -> float:
        values = np.asarray(self.samples_ns, dtype=np.float64)
        return float(np.percentile(values, 95, method="linear"))

    @property
    def throughput_pairs_per_s(self) -> float:
        if self.median_ns <= 0:
            return float("inf")
        return self.candidate_count * 1_000_000_000.0 / self.median_ns

    def as_dict(self) -> dict[str, Any]:
        return {
            "measure": self.measure,
            "candidate_count": self.candidate_count,
            "warmup_count": self.warmup_count,
            "repetitions": self.repetitions,
            "samples_ns": list(self.samples_ns),
            "median_ns": self.median_ns,
            "p95_ns": self.p95_ns,
            "throughput_pairs_per_s": self.throughput_pairs_per_s,
        }


def time_pairwise(
    measure: TrajectoryMeasure,
    query: Any,
    candidates: Iterable[Any],
    *,
    warmup: int = 1,
    repetitions: int = 3,
) -> PairwiseTiming:
    """Time deterministic pairwise scoring after explicit warm-up calls."""

    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("cannot time an empty candidate collection")
    for _ in range(warmup):
        measure.pairwise(query, candidate_list)
    samples: list[int] = []
    for _ in range(repetitions):
        started = perf_counter_ns()
        measure.pairwise(query, candidate_list)
        samples.append(max(0, perf_counter_ns() - started))
    return PairwiseTiming(measure.name, len(candidate_list), warmup, repetitions, tuple(samples))


__all__ = ["PairwiseTiming", "time_pairwise"]
