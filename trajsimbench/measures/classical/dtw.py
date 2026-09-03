"""Dynamic time warping over projected 2-D Euclidean point costs."""

from __future__ import annotations

import numpy as np

from .._geometry import (
    normalization_denominator,
    path_length,
    point_distances,
    result,
    validate_pair,
)
from ..base import DistanceResult, MeasureCapabilities, TrajectoryMeasure, TrajectoryView
from ..config import DTWConfig


class DTWMeasure(TrajectoryMeasure):
    """Exact DTW with optional Sakoe--Chiba band and global normalization."""

    name = "dtw"
    version = "1.0.0"
    capabilities = MeasureCapabilities(supports_batch=True, symmetric=True)
    config: DTWConfig

    @property
    def config_model(self) -> type[DTWConfig]:
        return DTWConfig

    @staticmethod
    def _warping_path_length(cumulative: np.ndarray) -> int:
        """Count cells in a deterministic optimal backtrace."""

        i, j = cumulative.shape[0] - 1, cumulative.shape[1] - 1
        count = 1
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                # Fixed tie order makes path-length normalization reproducible.
                predecessors = np.asarray(
                    [cumulative[i - 1, j - 1], cumulative[i - 1, j], cumulative[i, j - 1]]
                )
                step = int(np.argmin(predecessors))
                if step == 0:
                    i -= 1
                    j -= 1
                elif step == 1:
                    i -= 1
                else:
                    j -= 1
            count += 1
        return count

    def _distance_impl(self, a: TrajectoryView, b: TrajectoryView) -> DistanceResult:
        points_a, points_b = validate_pair(a, b)
        n, m = len(points_a), len(points_b)
        window = self.config.window
        if window is not None and abs(n - m) > window:
            raise ValueError(
                f"Sakoe-Chiba window {window} cannot connect trajectories of lengths {n} and {m}"
            )
        costs = point_distances(points_a, points_b)
        cumulative = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        cumulative[0, 0] = 0.0
        for i in range(1, n + 1):
            first = 1 if window is None else max(1, i - window)
            last = m if window is None else min(m, i + window)
            for j in range(first, last + 1):
                cumulative[i, j] = costs[i - 1, j - 1] + min(
                    cumulative[i - 1, j], cumulative[i, j - 1], cumulative[i - 1, j - 1]
                )
        raw_score = float(cumulative[n, m])
        if not np.isfinite(raw_score):
            raise ValueError("DTW window leaves no valid warping path")
        if self.config.normalization == "path_length":
            denominator = float(self._warping_path_length(cumulative))
        else:
            denominator = normalization_denominator(
                self.config.normalization, points_a, points_b, raw_score
            )
        distance = raw_score / denominator
        return result(
            raw_score,
            distance=distance,
            details={
                "normalization": self.config.normalization,
                "normalization_denominator": denominator,
                "window": window,
                "path_length_a": path_length(points_a),
                "path_length_b": path_length(points_b),
                "dp_cells": n * m,
            },
        )


DynamicTimeWarpingMeasure = DTWMeasure
DTW = DTWMeasure


__all__ = ["DTW", "DTWMeasure", "DynamicTimeWarpingMeasure"]
