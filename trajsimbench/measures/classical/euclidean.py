"""Arc-length-resampled pointwise Euclidean trajectory distance."""

from __future__ import annotations

import numpy as np

from .._geometry import path_length, result, validate_pair
from ..base import DistanceResult, MeasureCapabilities, TrajectoryMeasure, TrajectoryView
from ..config import EuclideanConfig


def resample_by_arc_length(points: np.ndarray, count: int) -> np.ndarray:
    """Resample a projected polyline at equally spaced normalized arc lengths."""

    if count < 1:
        raise ValueError("count must be positive")
    if points.shape[0] == 1 or count == 1:
        return np.repeat(points[:1], count, axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    total = float(cumulative[-1])
    if total == 0.0:
        return np.repeat(points[:1], count, axis=0)
    # Duplicate cumulative positions arise from repeated points.  Keeping the
    # first occurrence preserves the path's left endpoint and is equivalent to
    # interpolation over a zero-length segment.
    positions, indices = np.unique(cumulative, return_index=True)
    targets = np.linspace(0.0, total, count)
    return np.column_stack(
        [np.interp(targets, positions, points[indices, dimension]) for dimension in range(2)]
    )


class EuclideanMeasure(TrajectoryMeasure):
    """Mean pointwise distance after common normalized arc-length resampling."""

    name = "euclidean"
    version = "1.0.0"
    capabilities = MeasureCapabilities(supports_batch=True, symmetric=True)
    config: EuclideanConfig

    @property
    def config_model(self) -> type[EuclideanConfig]:
        return EuclideanConfig

    def _distance_impl(self, a: TrajectoryView, b: TrajectoryView) -> DistanceResult:
        points_a, points_b = validate_pair(a, b)
        count = self.config.n_samples
        sampled_a = resample_by_arc_length(points_a, count)
        sampled_b = resample_by_arc_length(points_b, count)
        distances = np.linalg.norm(sampled_a - sampled_b, axis=1)
        score = float(np.mean(distances))
        return result(
            score,
            details={
                "resample_count": count,
                "path_length_a": path_length(points_a),
                "path_length_b": path_length(points_b),
            },
        )


ResampledEuclideanMeasure = EuclideanMeasure
Euclidean = EuclideanMeasure


__all__ = ["Euclidean", "EuclideanMeasure", "ResampledEuclideanMeasure", "resample_by_arc_length"]
