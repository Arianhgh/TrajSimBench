"""Shared projected-coordinate and validation helpers for classical measures."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import DistanceResult, TrajectoryView


def projected_points(trajectory: TrajectoryView) -> np.ndarray:
    """Return finite ``float64`` projected ``(x_m, y_m)`` points.

    The canonical foundation layout is ``lon, lat, x_m, y_m, timestamp``.
    For lightweight callers, two-column arrays are already interpreted as
    projected coordinates.  A metadata ``projected_points`` array takes
    precedence and is useful for foundation views with separate point arrays.
    """

    metadata = trajectory.metadata
    if "projected_points" in metadata:
        raw = np.asarray(metadata["projected_points"], dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] < 2:
            raise ValueError("metadata['projected_points'] must be shaped (n, >=2)")
        points = raw[:, :2]
    else:
        try:
            raw = np.asarray(trajectory.points, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("trajectory points must be numeric") from exc
        if raw.ndim != 2 or raw.shape[1] < 2:
            raise ValueError("trajectory points must be shaped (n, >=2)")
        if raw.shape[1] >= 4:
            points = raw[:, 2:4]
        else:
            points = raw[:, :2]
    if points.shape[0] == 0:
        raise ValueError("empty trajectories are not supported by classical measures")
    if not np.isfinite(points).all():
        raise ValueError("projected trajectory coordinates must be finite")
    return np.ascontiguousarray(points, dtype=np.float64)


def timestamps(trajectory: TrajectoryView) -> np.ndarray | None:
    """Return timestamps in seconds, or ``None`` when they are unavailable."""

    for key in ("timestamps", "timestamps_s", "timestamp_s"):
        if key in trajectory.metadata and trajectory.metadata[key] is not None:
            raw = np.asarray(trajectory.metadata[key], dtype=np.float64)
            if raw.ndim != 1 or raw.shape[0] != trajectory.points.shape[0]:
                raise ValueError(f"metadata[{key!r}] must align one-to-one with points")
            if not np.isfinite(raw).all():
                return None
            return raw
    points = np.asarray(trajectory.points)
    if points.ndim == 2 and points.shape[1] >= 5:
        raw = np.asarray(points[:, 4], dtype=np.float64)
        if np.isfinite(raw).all():
            return raw
    # Compact standalone time-aware examples may use [x, y, timestamp_s].
    # The canonical foundation representation remains the five-column form.
    if points.ndim == 2 and points.shape[1] == 3:
        raw = np.asarray(points[:, 2], dtype=np.float64)
        if np.isfinite(raw).all():
            return raw
    return None


def validate_pair(a: TrajectoryView, b: TrajectoryView) -> tuple[np.ndarray, np.ndarray]:
    """Validate and project one pair of trajectories."""

    return projected_points(a), projected_points(b)


def point_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dense pairwise Euclidean distances between two projected paths."""

    delta = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", delta, delta, dtype=np.float64))


def path_length(points: np.ndarray) -> float:
    """Polyline length in projected coordinate units (normally metres)."""

    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def normalization_denominator(
    policy: str,
    a: np.ndarray,
    b: np.ndarray,
    raw_score: float,
) -> float:
    """Return a positive denominator for the frozen normalization policies."""

    if policy == "none":
        return 1.0
    if policy == "max_input_length":
        return float(max(len(a), len(b), 1))
    if policy == "path_length":
        # A symmetric geometric scale; the max avoids a direction-dependent
        # result and remains well-defined for a point/zero-length path.
        return float(max(path_length(a), path_length(b), 1.0))
    raise ValueError(f"unknown normalization policy {policy!r}")


def result(
    raw_score: float,
    *,
    distance: float | None = None,
    details: dict[str, Any] | None = None,
) -> DistanceResult:
    """Construct a finite canonical result with a consistent raw score."""

    raw = float(raw_score)
    canonical = raw if distance is None else float(distance)
    if not np.isfinite(canonical) or canonical < 0:
        raise ValueError("classical measure produced an invalid distance")
    return DistanceResult(canonical, raw, details=details or {})


__all__ = [
    "normalization_denominator",
    "path_length",
    "point_distances",
    "projected_points",
    "result",
    "timestamps",
    "validate_pair",
]
