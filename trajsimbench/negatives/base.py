"""Common negative-generator protocol and free-space geometry helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from trajsimbench.perturbations.base import spatial_columns, timestamp_column
from trajsimbench.perturbations.result import hash_payload


class NegativeConstructionError(ValueError):
    """Malformed negative-generator inputs."""


def _id(value: Any, fallback: str = "trajectory") -> str:
    return (
        str(getattr(value, "trajectory_id", getattr(value, "id", fallback)))
        if not isinstance(value, str)
        else value
    )


def _view_for(value: Any, dataset: Any | None) -> Any:
    if not isinstance(value, str):
        return value
    if dataset is None:
        raise NegativeConstructionError(
            f"candidate {value!r} requires a dataset to resolve its points"
        )
    if hasattr(dataset, "by_id"):
        return dataset.by_id(value)
    if isinstance(dataset, Mapping):
        return dataset[value]
    for index in range(len(dataset)):
        view = dataset[index]
        if _id(view, str(index)) == value:
            return view
    raise NegativeConstructionError(f"candidate {value!r} is not in the dataset")


def xy_points(view: Any) -> np.ndarray:
    points = np.asarray(getattr(view, "points", view), dtype=np.float64)
    metadata = getattr(view, "metadata", {}) or {}
    x_col, y_col = spatial_columns(points, metadata)
    return np.array(points[:, [x_col, y_col]], copy=True, dtype=np.float64)


def trajectory_length(view: Any) -> float:
    xy = xy_points(view)
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0


def endpoints(view: Any) -> tuple[np.ndarray, np.ndarray]:
    xy = xy_points(view)
    if not len(xy):
        raise NegativeConstructionError("empty trajectories have no endpoints")
    return xy[0], xy[-1]


def resample_polyline(view: Any, sample_count: int = 32) -> np.ndarray:
    xy = xy_points(view)
    if len(xy) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if len(xy) == 1:
        return np.repeat(xy, sample_count, axis=0)
    segment = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    total = float(cumulative[-1])
    if total <= 0:
        return np.repeat(xy[:1], sample_count, axis=0)
    positions = np.linspace(0.0, total, sample_count)
    output = np.empty((sample_count, 2), dtype=np.float64)
    for index, position in enumerate(positions):
        right = int(np.searchsorted(cumulative, position, side="right"))
        right = min(max(right, 1), len(xy) - 1)
        left = right - 1
        span = cumulative[right] - cumulative[left]
        ratio = 0.0 if span == 0 else (position - cumulative[left]) / span
        output[index] = xy[left] + ratio * (xy[right] - xy[left])
    return output


def shape_distance(first: Any, second: Any, *, sample_count: int = 32) -> float:
    """Translation- and scale-normalized free-space shape distance."""

    a = resample_polyline(first, sample_count)
    b = resample_polyline(second, sample_count)
    if not len(a) or not len(b):
        return float("inf")
    a = a - a[0]
    b = b - b[0]
    a_scale = max(float(np.linalg.norm(a[-1] - a[0])), trajectory_length(first), 1e-12)
    b_scale = max(float(np.linalg.norm(b[-1] - b[0])), trajectory_length(second), 1e-12)
    a /= a_scale
    b /= b_scale
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def _point_coverage(first: np.ndarray, second: np.ndarray, tolerance_m: float) -> float:
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1) <= tolerance_m)) if len(first) else 0.0


def route_overlap(
    first: Any, second: Any, *, tolerance_m: float = 25.0, sample_count: int = 64
) -> float:
    """Versioned free-space route overlap based on resampled path coverage."""

    if tolerance_m < 0:
        raise NegativeConstructionError("route-overlap tolerance must be non-negative")
    a, b = resample_polyline(first, sample_count), resample_polyline(second, sample_count)
    if not len(a) or not len(b):
        return 0.0
    return 0.5 * (_point_coverage(a, b, tolerance_m) + _point_coverage(b, a, tolerance_m))


def temporal_behavior_distance(first: Any, second: Any) -> float:
    def profile(view: Any) -> tuple[float, np.ndarray]:
        points = np.asarray(getattr(view, "points", view), dtype=np.float64)
        time_col = timestamp_column(points, getattr(view, "metadata", {}) or {})
        if time_col is None or len(points) < 2:
            return float("nan"), np.empty(0, dtype=np.float64)
        times = points[:, time_col]
        if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
            return float("nan"), np.empty(0, dtype=np.float64)
        duration = float(times[-1] - times[0])
        intervals = np.diff(times)
        total = float(intervals.sum())
        return duration, intervals / total if total > 0 else np.zeros_like(intervals)

    duration_a, profile_a = profile(first)
    duration_b, profile_b = profile(second)
    if not np.isfinite(duration_a) or not np.isfinite(duration_b):
        return float("nan")
    size = max(len(profile_a), len(profile_b), 2)
    a = np.interp(np.linspace(0, 1, size), np.linspace(0, 1, len(profile_a)), profile_a)
    b = np.interp(np.linspace(0, 1, size), np.linspace(0, 1, len(profile_b)), profile_b)
    duration_term = abs(np.log((duration_a + 1e-12) / (duration_b + 1e-12)))
    return float(duration_term + np.mean(np.abs(a - b)))


@dataclass(frozen=True, slots=True)
class NegativeCandidate:
    candidate_id: str
    negative_type: str
    achieved_constraints: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "achieved_constraints", MappingProxyType(dict(self.achieved_constraints))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "negative_type": self.negative_type,
            "achieved_constraints": dict(self.achieved_constraints),
        }


@dataclass(frozen=True, slots=True)
class NegativeGenerationReport:
    generator: str
    generator_version: str
    query_id: str
    attempted: int
    accepted: int
    max_candidates_examined: int
    rejection_reasons: Mapping[str, int] = field(default_factory=dict)
    required_count: int = 1
    minimum_yield: float = 0.0
    quality_gate_passed: bool = True
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.attempted < 0 or self.accepted < 0 or self.max_candidates_examined < 0:
            raise NegativeConstructionError("negative report counts must be non-negative")
        object.__setattr__(
            self, "rejection_reasons", MappingProxyType(dict(self.rejection_reasons))
        )
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))

    @property
    def yield_rate(self) -> float:
        return self.accepted / self.attempted if self.attempted else 0.0

    @property
    def rejection_rate(self) -> float:
        return 1.0 - self.yield_rate if self.attempted else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator": self.generator,
            "generator_version": self.generator_version,
            "query_id": self.query_id,
            "attempted": self.attempted,
            "accepted": self.accepted,
            "yield_rate": self.yield_rate,
            "rejection_rate": self.rejection_rate,
            "max_candidates_examined": self.max_candidates_examined,
            "rejection_reasons": dict(self.rejection_reasons),
            "required_count": self.required_count,
            "minimum_yield": self.minimum_yield,
            "quality_gate_passed": self.quality_gate_passed,
            "config": dict(self.config),
        }


@dataclass(frozen=True, slots=True)
class NegativeGenerationResult:
    query_id: str
    candidates: tuple[NegativeCandidate, ...]
    report: NegativeGenerationReport

    def __post_init__(self) -> None:
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise NegativeConstructionError("negative result contains duplicate candidates")
        if self.report.accepted != len(self.candidates):
            raise NegativeConstructionError(
                "negative report accepted count does not match candidates"
            )

    @property
    def generated(self) -> bool:
        return bool(self.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "candidate_ids": [candidate.candidate_id for candidate in self.candidates],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "report": self.report.to_dict(),
            "content_hash": hash_payload(
                {
                    "query_id": self.query_id,
                    "candidates": [candidate.to_dict() for candidate in self.candidates],
                    "report": self.report.to_dict(),
                }
            ),
        }


class NegativeGenerator(ABC):
    name = "negative"
    version = "1.0"

    def __init__(self, **defaults: Any) -> None:
        self.defaults = dict(defaults)

    def generate(
        self,
        query: Any,
        candidate_ids: Sequence[Any],
        *,
        dataset: Any | None = None,
        count: int = 1,
        seed: int = 0,
        config: Mapping[str, Any] | None = None,
    ) -> NegativeGenerationResult:
        if count < 1:
            raise NegativeConstructionError("negative count must be positive")
        query_id = _id(query)
        query_view = _view_for(query, dataset)
        resolved = dict(self.defaults)
        resolved.update(config or {})
        max_examined = int(resolved.get("max_candidates_examined", len(candidate_ids)))
        if max_examined < 0:
            raise NegativeConstructionError("max_candidates_examined must be non-negative")
        minimum_yield = float(resolved.get("minimum_yield", 0.0))
        if not 0 <= minimum_yield <= 1:
            raise NegativeConstructionError("minimum_yield must be in [0, 1]")
        if len(candidate_ids) == 0:
            report = self._report(query_id, 0, (), max_examined, count, minimum_yield, resolved)
            return NegativeGenerationResult(query_id, (), report)
        # Candidate order is canonicalized before RNG use so input container
        # order and method iteration order cannot change the task.
        unique: dict[str, Any] = {}
        for candidate in candidate_ids:
            candidate_id = _id(candidate)
            if candidate_id != query_id:
                unique.setdefault(candidate_id, candidate)
        ordered_ids = sorted(unique)
        rng = np.random.default_rng(int(seed))
        if ordered_ids:
            order = rng.permutation(len(ordered_ids))
            ordered_ids = [ordered_ids[index] for index in order]
        accepted: list[NegativeCandidate] = []
        reasons: list[str] = []
        examined = 0
        for candidate_id in ordered_ids[:max_examined]:
            examined += 1
            candidate_view = _view_for(unique[candidate_id], dataset)
            try:
                qualifies, achieved, reason = self._qualify(query_view, candidate_view, resolved)
            except (NegativeConstructionError, ValueError, IndexError) as exc:
                qualifies, achieved, reason = False, {}, str(exc)
            if qualifies:
                accepted.append(NegativeCandidate(candidate_id, self.name, achieved))
                if len(accepted) >= count:
                    break
            else:
                reasons.append(reason or "constraint_not_met")
        report = self._report(
            query_id,
            examined,
            reasons,
            max_examined,
            count,
            minimum_yield,
            resolved,
            accepted=len(accepted),
        )
        return NegativeGenerationResult(query_id, tuple(accepted), report)

    def _report(
        self,
        query_id: str,
        examined: int,
        reasons: Sequence[str],
        max_examined: int,
        required: int,
        minimum_yield: float,
        config: Mapping[str, Any],
        *,
        accepted: int = 0,
    ) -> NegativeGenerationReport:
        yield_rate = accepted / examined if examined else 0.0
        return NegativeGenerationReport(
            generator=self.name,
            generator_version=self.version,
            query_id=query_id,
            attempted=examined,
            accepted=accepted,
            max_candidates_examined=max_examined,
            rejection_reasons=Counter(reasons),
            required_count=required,
            minimum_yield=minimum_yield,
            quality_gate_passed=accepted >= required and yield_rate >= minimum_yield,
            config=config,
        )

    @abstractmethod
    def _qualify(
        self, query: Any, candidate: Any, config: Mapping[str, Any]
    ) -> tuple[bool, Mapping[str, Any], str | None]:
        raise NotImplementedError
