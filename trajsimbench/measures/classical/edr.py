"""Edit distance on real sequences (EDR)."""

from __future__ import annotations

import numpy as np

from .._geometry import point_distances, result, validate_pair
from ..base import DistanceResult, MeasureCapabilities, TrajectoryMeasure, TrajectoryView
from ..config import EDRConfig


class EDRMeasure(TrajectoryMeasure):
    """EDR with zero substitution cost inside epsilon and unit gap costs."""

    name = "edr"
    version = "1.0.0"
    capabilities = MeasureCapabilities(supports_batch=True, symmetric=True)
    config: EDRConfig

    @property
    def config_model(self) -> type[EDRConfig]:
        return EDRConfig

    def _distance_impl(self, a: TrajectoryView, b: TrajectoryView) -> DistanceResult:
        points_a, points_b = validate_pair(a, b)
        distances = point_distances(points_a, points_b)
        n, m = len(points_a), len(points_b)
        edit = np.zeros((n + 1, m + 1), dtype=np.int32)
        edit[:, 0] = np.arange(n + 1)
        edit[0, :] = np.arange(m + 1)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                substitution = 0 if distances[i - 1, j - 1] <= self.config.epsilon else 1
                edit[i, j] = min(
                    edit[i - 1, j] + 1,
                    edit[i, j - 1] + 1,
                    edit[i - 1, j - 1] + substitution,
                )
        raw_score = int(edit[n, m])
        denominator = max(n, m)
        return result(
            float(raw_score),
            distance=raw_score / denominator,
            details={
                "edit_cost": raw_score,
                "epsilon": self.config.epsilon,
                "normalization_denominator": denominator,
                "dp_cells": n * m,
            },
        )


EditDistanceOnRealSequencesMeasure = EDRMeasure
EDR = EDRMeasure


__all__ = ["EDR", "EDRMeasure", "EditDistanceOnRealSequencesMeasure"]
