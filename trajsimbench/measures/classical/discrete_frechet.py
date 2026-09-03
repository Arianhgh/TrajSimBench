"""Standard discrete Fréchet distance."""

from __future__ import annotations

import numpy as np

from .._geometry import point_distances, result, validate_pair
from ..base import DistanceResult, MeasureCapabilities, TrajectoryMeasure, TrajectoryView
from ..config import DiscreteFrechetConfig


class DiscreteFrechetMeasure(TrajectoryMeasure):
    """Minimum leash length over monotone vertex couplings."""

    name = "discrete_frechet"
    version = "1.0.0"
    capabilities = MeasureCapabilities(supports_batch=True, symmetric=True)
    config: DiscreteFrechetConfig

    @property
    def config_model(self) -> type[DiscreteFrechetConfig]:
        return DiscreteFrechetConfig

    def _distance_impl(self, a: TrajectoryView, b: TrajectoryView) -> DistanceResult:
        points_a, points_b = validate_pair(a, b)
        costs = point_distances(points_a, points_b)
        coupling = np.empty_like(costs)
        coupling[0, 0] = costs[0, 0]
        for i in range(1, len(points_a)):
            coupling[i, 0] = max(coupling[i - 1, 0], costs[i, 0])
        for j in range(1, len(points_b)):
            coupling[0, j] = max(coupling[0, j - 1], costs[0, j])
        for i in range(1, len(points_a)):
            for j in range(1, len(points_b)):
                coupling[i, j] = max(
                    costs[i, j],
                    min(coupling[i - 1, j], coupling[i - 1, j - 1], coupling[i, j - 1]),
                )
        return result(float(coupling[-1, -1]), details={"dp_cells": costs.size})


FrechetMeasure = DiscreteFrechetMeasure
DiscreteFrechet = DiscreteFrechetMeasure


__all__ = ["DiscreteFrechet", "DiscreteFrechetMeasure", "FrechetMeasure"]
