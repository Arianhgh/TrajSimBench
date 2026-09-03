"""Edit distance with real penalty (ERP)."""

from __future__ import annotations

import numpy as np

from .._geometry import result, validate_pair
from ..base import DistanceResult, MeasureCapabilities, TrajectoryMeasure, TrajectoryView
from ..config import ERPConfig


class ERPMeasure(TrajectoryMeasure):
    """Projected-vector ERP with a configured finite gap point."""

    name = "erp"
    version = "1.0.0"
    capabilities = MeasureCapabilities(supports_batch=True, symmetric=True)
    config: ERPConfig

    @property
    def config_model(self) -> type[ERPConfig]:
        return ERPConfig

    def _distance_impl(self, a: TrajectoryView, b: TrajectoryView) -> DistanceResult:
        points_a, points_b = validate_pair(a, b)
        gap = np.asarray(self.config.gap_point, dtype=np.float64)
        n, m = len(points_a), len(points_b)
        dp = np.zeros((n + 1, m + 1), dtype=np.float64)
        gap_a = np.linalg.norm(points_a - gap, axis=1)
        gap_b = np.linalg.norm(points_b - gap, axis=1)
        dp[1:, 0] = np.cumsum(gap_a)
        dp[0, 1:] = np.cumsum(gap_b)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                substitution = dp[i - 1, j - 1] + float(
                    np.linalg.norm(points_a[i - 1] - points_b[j - 1])
                )
                deletion = dp[i - 1, j] + gap_a[i - 1]
                insertion = dp[i, j - 1] + gap_b[j - 1]
                dp[i, j] = min(substitution, deletion, insertion)
        raw_score = float(dp[n, m])
        denominator = max(n, m, 1)
        distance = raw_score if self.config.normalization == "none" else raw_score / denominator
        return result(
            raw_score,
            distance=distance,
            details={
                "gap_point": tuple(self.config.gap_point),
                "normalization": self.config.normalization,
                "normalization_denominator": denominator,
                "dp_cells": n * m,
            },
        )


EditDistanceWithRealPenaltyMeasure = ERPMeasure
ERP = ERPMeasure


__all__ = ["ERP", "ERPMeasure", "EditDistanceWithRealPenaltyMeasure"]
